import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


app = FastAPI()


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_ROW_KEYS = {
    "id",
    "entity",
    "eventTime",
    "revision",
    "text",
}

MAX_SAFE_INTEGER = 9007199254740991

TIME_RE = re.compile(
    r"^"
    r"(\d{4})-(\d{2})-(\d{2})"
    r"T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})"
    r"$"
)

GENERATION_RE = re.compile(r"^[0-9]+$")

CRC_RE = re.compile(r"^[0-9a-f]{8}$")

# gs://bucket/object
#
# Bucket = one or more non-slash, non-whitespace characters.
# Object = everything after the first slash, as long as it
# contains at least one character.
URI_RE = re.compile(r"^gs://[^/\s]+/.+$")


# ============================================================
# BASIC HELPERS
# ============================================================

def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def utf8(value: str) -> bytes:
    return value.encode("utf-8")


def add_reason(reasons: list[str], code: str):
    if code not in reasons:
        reasons.append(code)


def sorted_reasons(reasons: list[str]) -> list[str]:
    return sorted(
        set(reasons),
        key=lambda x: x.encode("utf-8"),
    )


# ============================================================
# TIMESTAMP
# ============================================================

def parse_timestamp(value: Any):
    if not isinstance(value, str):
        return None

    match = TIME_RE.fullmatch(value)

    if match is None:
        return None

    (
        year_s,
        month_s,
        day_s,
        hour_s,
        minute_s,
        second_s,
        fraction,
        offset,
    ) = match.groups()

    year = int(year_s)
    month = int(month_s)
    day = int(day_s)
    hour = int(hour_s)
    minute = int(minute_s)
    second = int(second_s)

    # Explicit clock validation.
    if hour > 23:
        return None

    if minute > 59:
        return None

    if second > 59:
        return None

    # Timezone.
    if offset == "Z":
        tz = timezone.utc
    else:
        offset_hour = int(offset[1:3])
        offset_minute = int(offset[4:6])

        # Offset magnitude <= 14:00.
        if offset_hour > 14:
            return None

        if offset_minute > 59:
            return None

        # Hour 14 requires minute 00.
        if offset_hour == 14 and offset_minute != 0:
            return None

        sign = 1 if offset[0] == "+" else -1

        tz = timezone(
            sign * timedelta(
                hours=offset_hour,
                minutes=offset_minute,
            )
        )

    # Fractional seconds.
    if fraction is None:
        milliseconds = 0
    else:
        milliseconds = int(fraction.ljust(3, "0"))

    try:
        dt = datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            milliseconds * 1000,
            tzinfo=tz,
        )
    except ValueError:
        return None

    return dt.astimezone(timezone.utc)


def canonical_timestamp(value: str) -> str:
    dt = parse_timestamp(value)

    if dt is None:
        raise ValueError("invalid timestamp")

    return (
        dt.strftime("%Y-%m-%dT%H:%M:%S.")
        + f"{dt.microsecond // 1000:03d}Z"
    )


# ============================================================
# TEXT CANONICALIZATION
# ============================================================

def canonical_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.lower()

    # Python split() recognizes Unicode whitespace.
    return " ".join(value.split())


# ============================================================
# REVISION
# ============================================================

def valid_revision(value: Any) -> bool:
    return (
        type(value) is int
        and value >= 0
        and value <= MAX_SAFE_INTEGER
    )


# ============================================================
# CRC32C - CASTAGNOLI
# ============================================================

def crc32c(data: bytes) -> int:
    """
    CRC32C / Castagnoli.

    Reflected polynomial:
        0x82F63B78
    """

    crc = 0xFFFFFFFF

    for byte in data:
        crc ^= byte

        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0x82F63B78
            else:
                crc >>= 1

    return crc ^ 0xFFFFFFFF


def crc32c_hex(data: bytes) -> str:
    return f"{crc32c(data):08x}"


# Known CRC32C test vector:
# CRC32C("123456789") == e3069283
assert crc32c_hex(b"123456789") == "e3069283"


# ============================================================
# SHA-256
# ============================================================

def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ============================================================
# WORD SET / JACCARD
# ============================================================

def word_set(value: str) -> set[str]:
    """
    Lowercase Unicode letter/number word set.
    """

    value = value.lower()

    result = set()
    current = []

    for char in value:
        category = unicodedata.category(char)

        if category.startswith("L") or category.startswith("N"):
            current.append(char)
        else:
            if current:
                result.add("".join(current))
                current = []

    if current:
        result.add("".join(current))

    return result


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


# ============================================================
# ROW VALIDATION
# ============================================================

def valid_row_shape(row: Any) -> bool:
    if not isinstance(row, dict):
        return False

    # Exactly these keys.
    if set(row.keys()) != EXPECTED_ROW_KEYS:
        return False

    # Four string fields.
    if not isinstance(row["id"], str):
        return False

    if not isinstance(row["entity"], str):
        return False

    if not isinstance(row["eventTime"], str):
        return False

    if not isinstance(row["text"], str):
        return False

    # Non-negative safe integer.
    if not valid_revision(row["revision"]):
        return False

    # Timestamp must itself be valid.
    if parse_timestamp(row["eventTime"]) is None:
        return False

    return True


# ============================================================
# OUTPUT ROW
# ============================================================

def output_row(row: dict) -> dict:
    # Explicit insertion order required by assignment.
    return {
        "id": row["id"],
        "entity": row["entity"],
        "eventTime": row["eventTime"],
        "revision": row["revision"],
        "text": row["text"],
    }


def row_json(row: dict) -> str:
    return compact_json(output_row(row))


# ============================================================
# POLICY
# ============================================================

def validate_policy(policy: Any):
    if not isinstance(policy, dict):
        return False, None, None, None

    min_time = parse_timestamp(policy.get("minTime"))
    max_time = parse_timestamp(policy.get("maxTime"))

    threshold = policy.get("contaminationThreshold")

    threshold_valid = (
        isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and math.isfinite(float(threshold))
        and 0 <= threshold <= 1
    )

    if min_time is None:
        return False, min_time, max_time, threshold

    if max_time is None:
        return False, min_time, max_time, threshold

    if not threshold_valid:
        return False, min_time, max_time, threshold

    if min_time > max_time:
        return False, min_time, max_time, threshold

    return True, min_time, max_time, float(threshold)


# ============================================================
# JSONL PARSING
# ============================================================

def parse_jsonl(content: str):
    """
    Returns:

        parsed_rows,
        jsonl_invalid,
        schema_invalid

    Blank lines are ignored.

    Only LF is a record separator.
    CRLF is supported by removing the CR before LF.
    """

    # Do not use splitlines(), because it treats Unicode line
    # separators as line boundaries.
    lines = content.split("\n")

    cleaned_lines = []

    for line in lines:
        if line.endswith("\r"):
            line = line[:-1]

        cleaned_lines.append(line)

    non_blank = [
        line
        for line in cleaned_lines
        if line.strip() != ""
    ]

    if not non_blank:
        return [], False, True

    parsed_rows = []
    jsonl_invalid = False
    schema_invalid = False

    for line in non_blank:

        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError, ValueError):
            jsonl_invalid = True
            continue

        if not valid_row_shape(value):
            schema_invalid = True
            continue

        parsed_rows.append(value)

    return (
        parsed_rows,
        jsonl_invalid,
        schema_invalid,
    )


# ============================================================
# OBJECT VALIDATION
# ============================================================

def validate_object(obj: Any):

    # If object isn't JSON object.
    if not isinstance(obj, dict):
        return {
            "accepted": False,
            "rejection": {
                "uri": None,
                "reasonCodes": [
                    "URI_INVALID",
                    "GENERATION_INVALID",
                    "CRC32C_INVALID",
                    "SCHEMA_INVALID",
                ],
            },
        }

    reasons = []

    # --------------------------------------------------------
    # URI
    # --------------------------------------------------------

    uri = obj.get("uri")

    if not isinstance(uri, str):
        reasons.append("URI_INVALID")

    elif URI_RE.fullmatch(uri) is None:
        reasons.append("URI_INVALID")

    # --------------------------------------------------------
    # GENERATIONS
    # --------------------------------------------------------

    generation = obj.get("generation")
    fetched_generation = obj.get("fetchedGeneration")

    generation_valid = (
        isinstance(generation, str)
        and GENERATION_RE.fullmatch(generation) is not None
    )

    fetched_generation_valid = (
        isinstance(fetched_generation, str)
        and GENERATION_RE.fullmatch(fetched_generation) is not None
    )

    if not generation_valid or not fetched_generation_valid:
        reasons.append("GENERATION_INVALID")

    # Only compare when both supplied generations are valid
    # decimal strings.
    if (
        generation_valid
        and fetched_generation_valid
        and generation != fetched_generation
    ):
        reasons.append("GENERATION_MISMATCH")

    # --------------------------------------------------------
    # CRC32C
    # --------------------------------------------------------

    supplied_crc = obj.get("crc32c")

    crc_valid = (
        isinstance(supplied_crc, str)
        and CRC_RE.fullmatch(supplied_crc) is not None
    )

    if not crc_valid:
        reasons.append("CRC32C_INVALID")

    # CRC mismatch is ONLY checked when:
    #   content is string
    #   CRC syntax is valid
    content = obj.get("content")

    if isinstance(content, str) and crc_valid:

        actual_crc = crc32c_hex(
            content.encode("utf-8")
        )

        if actual_crc != supplied_crc:
            reasons.append("CRC32C_MISMATCH")

    # --------------------------------------------------------
    # SCHEMA ID / CONTENT
    # --------------------------------------------------------

    schema_id = obj.get("schemaId")

    if schema_id != "training-v1":
        reasons.append("SCHEMA_INVALID")

    if not isinstance(content, str):
        reasons.append("SCHEMA_INVALID")

    # --------------------------------------------------------
    # JSONL
    # --------------------------------------------------------

    parsed_rows = []

    if isinstance(content, str):

        (
            parsed_rows,
            jsonl_invalid,
            schema_invalid,
        ) = parse_jsonl(content)

        if jsonl_invalid:
            reasons.append("JSONL_INVALID")

        if schema_invalid:
            reasons.append("SCHEMA_INVALID")

    # --------------------------------------------------------
    # OBJECT RESULT
    # --------------------------------------------------------

    reasons = sorted_reasons(reasons)

    if reasons:

        return {
            "accepted": False,
            "rejection": {
                "uri": uri if isinstance(uri, str) else None,
                "reasonCodes": reasons,
            },
        }

    return {
        "accepted": True,
        "uri": uri,
        "generation": generation,
        "crc32c": supplied_crc,
        "schemaId": schema_id,
        "rows": parsed_rows,
    }


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate(objects):

    groups = {}

    for obj in objects:

        for original in obj["rows"]:

            canonical_row = {
                "id": original["id"],
                "entity": canonical_text(
                    original["entity"]
                ),
                "eventTime": canonical_timestamp(
                    original["eventTime"]
                ),
                "revision": original["revision"],
                "text": canonical_text(
                    original["text"]
                ),
            }

            key = (
                canonical_row["entity"],
                canonical_row["eventTime"],
                canonical_row["text"],
            )

            groups.setdefault(key, []).append(
                canonical_row
            )

    retained = []
    rejected = []

    for group in groups.values():

        ordered = sorted(
            group,
            key=lambda row: (
                -row["revision"],
                row["id"].encode("utf-8"),
            ),
        )

        retained.append(ordered[0])

        for loser in ordered[1:]:

            rejected.append({
                "id": loser["id"],
                "reasonCodes": [
                    "DUPLICATE"
                ],
            })

    return retained, rejected


# ============================================================
# SPLIT
# ============================================================

def split_row(row):

    entity_hash = hashlib.sha256(
        row["entity"].encode("utf-8")
    ).digest()

    bucket = entity_hash[0] % 10

    if bucket <= 5:
        return "train"

    if bucket <= 7:
        return "validation"

    return "test"


# ============================================================
# ARTIFACT SERIALIZATION
# ============================================================

def serialize_rows(rows) -> bytes:

    result = bytearray()

    for row in rows:

        result.extend(
            row_json(row).encode("utf-8")
        )

        result.extend(b"\n")

    return bytes(result)


def sort_split(rows):

    return sorted(
        rows,
        key=lambda row: (
            row["id"].encode("utf-8"),
            row_json(row).encode("utf-8"),
        ),
    )


# ============================================================
# ENDPOINT
# ============================================================

@app.post("/build-corpus")
async def build_corpus(request: Request):

    # --------------------------------------------------------
    # REQUEST PARSING
    # --------------------------------------------------------

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            },
        )

    # Request must itself be an object.
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            },
        )

    # Policy must exist.
    if "policy" not in body:
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            },
        )

    # Objects must be an array.
    if not isinstance(body.get("objects"), list):
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            },
        )

    policy = body["policy"]
    objects = body["objects"]

    # --------------------------------------------------------
    # POLICY
    # --------------------------------------------------------

    (
        policy_valid,
        min_time,
        max_time,
        threshold,
    ) = validate_policy(policy)

    # --------------------------------------------------------
    # OBJECTS
    # --------------------------------------------------------

    accepted_objects = []
    rejected_objects = []

    for obj in objects:

        result = validate_object(obj)

        if result["accepted"]:
            accepted_objects.append(result)

        else:
            rejected_objects.append(
                result["rejection"]
            )

    # --------------------------------------------------------
    # LINEAGE
    # --------------------------------------------------------

    lineage = []

    for obj in accepted_objects:

        lineage.append({
            "uri": obj["uri"],
            "generation": obj["generation"],
            "crc32c": obj["crc32c"],
            "schemaId": obj["schemaId"],
        })

    # --------------------------------------------------------
    # DEDUPLICATE
    # --------------------------------------------------------

    retained, rejected_rows = deduplicate(
        accepted_objects
    )

    # --------------------------------------------------------
    # POLICY
    # --------------------------------------------------------

    if not policy_valid:

        for row in retained:

            rejected_rows.append({
                "id": row["id"],
                "reasonCodes": [
                    "POLICY_INVALID"
                ],
            })

        retained = []

    else:

        kept = []

        for row in retained:

            event_dt = parse_timestamp(
                row["eventTime"]
            )

            if (
                event_dt < min_time
                or event_dt > max_time
            ):

                rejected_rows.append({
                    "id": row["id"],
                    "reasonCodes": [
                        "OUT_OF_WINDOW"
                    ],
                })

            else:
                kept.append(row)

        retained = kept

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    train = []
    validation = []
    test = []

    for row in retained:

        split = split_row(row)

        if split == "train":
            train.append(row)

        elif split == "validation":
            validation.append(row)

        else:
            test.append(row)

    # --------------------------------------------------------
    # CONTAMINATION
    # --------------------------------------------------------

    train_sets = [
        word_set(row["text"])
        for row in train
    ]

    def is_contaminated(row):

        target = word_set(row["text"])

        for train_words in train_sets:

            if (
                jaccard(
                    target,
                    train_words,
                )
                >= threshold
            ):
                return True

        return False

    clean_validation = []

    for row in validation:

        if is_contaminated(row):

            rejected_rows.append({
                "id": row["id"],
                "reasonCodes": [
                    "TRAIN_CONTAMINATION"
                ],
            })

        else:
            clean_validation.append(row)

    clean_test = []

    for row in test:

        if is_contaminated(row):

            rejected_rows.append({
                "id": row["id"],
                "reasonCodes": [
                    "TRAIN_CONTAMINATION"
                ],
            })

        else:
            clean_test.append(row)

    validation = clean_validation
    test = clean_test

    # --------------------------------------------------------
    # SORT SPLITS
    # --------------------------------------------------------

    train = sort_split(train)
    validation = sort_split(validation)
    test = sort_split(test)

    # --------------------------------------------------------
    # DIGESTS
    # --------------------------------------------------------

    train_bytes = serialize_rows(train)
    validation_bytes = serialize_rows(validation)
    test_bytes = serialize_rows(test)

    digests = {
        "train": sha256_hex(train_bytes),
        "validation": sha256_hex(validation_bytes),
        "test": sha256_hex(test_bytes),
    }

    # --------------------------------------------------------
    # MERGE REJECTED ROWS
    # --------------------------------------------------------

    by_id = {}

    for rejection in rejected_rows:

        rid = rejection["id"]

        if rid not in by_id:
            by_id[rid] = []

        by_id[rid].extend(
            rejection["reasonCodes"]
        )

    rejected_rows_final = []

    for rid, reasons in by_id.items():

        rejected_rows_final.append({
            "id": rid,
            "reasonCodes": sorted_reasons(
                reasons
            ),
        })

    # --------------------------------------------------------
    # SORT FINAL ARRAYS
    # --------------------------------------------------------

    rejected_objects.sort(
        key=lambda item: (
            (
                item["uri"].encode("utf-8")
                if isinstance(item["uri"], str)
                else b""
            ),
            compact_json(item).encode("utf-8"),
        )
    )

    rejected_rows_final.sort(
        key=lambda item: (
            item["id"].encode("utf-8"),
            compact_json(item).encode("utf-8"),
        )
    )

    lineage.sort(
        key=lambda item: (
            item["uri"].encode("utf-8"),
            compact_json(item).encode("utf-8"),
        )
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "splits": {
            "train": [
                output_row(row)
                for row in train
            ],
            "validation": [
                output_row(row)
                for row in validation
            ],
            "test": [
                output_row(row)
                for row in test
            ],
        },
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows_final,
        "digests": digests,
        "lineage": lineage,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}
