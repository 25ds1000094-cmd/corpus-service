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

URI_RE = re.compile(
    r"^gs://[^/\s]+/.+$"
)


# ============================================================
# JSON HELPERS
# ============================================================

def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


class DuplicateKeyError(ValueError):
    pass


def reject_duplicate_keys(pairs):
    """
    Make JSON object member names deterministic.

    Python's normal json parser keeps the last duplicate key.
    We reject duplicates instead so that the same JSONL input
    cannot acquire different meanings in different parsers.
    """

    result = {}

    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(
                f"duplicate key: {key}"
            )

        result[key] = value

    return result


def parse_json_line(line: str):
    return json.loads(
        line,
        object_pairs_hook=reject_duplicate_keys,
    )


# ============================================================
# UTF-8
# ============================================================

def utf8(value: str) -> bytes:
    return value.encode("utf-8")


# ============================================================
# REASON CODES
# ============================================================

def sorted_reasons(reasons):
    return sorted(
        set(reasons),
        key=lambda x: x.encode("utf-8"),
    )


# ============================================================
# TIMESTAMP
# ============================================================

def parse_timestamp(value):
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

    # Clock validation.
    if hour > 23:
        return None

    if minute > 59:
        return None

    if second > 59:
        return None

    # Timezone validation.
    if offset == "Z":

        tz = timezone.utc

    else:

        offset_hour = int(offset[1:3])
        offset_minute = int(offset[4:6])

        if offset_hour > 14:
            return None

        if offset_minute > 59:
            return None

        # Hour 14 must have exactly 00 minutes.
        if offset_hour == 14 and offset_minute != 0:
            return None

        sign = 1 if offset[0] == "+" else -1

        tz = timezone(
            sign * timedelta(
                hours=offset_hour,
                minutes=offset_minute,
            )
        )

    if fraction is None:
        milliseconds = 0
    else:
        milliseconds = int(
            fraction.ljust(3, "0")
        )

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


def canonical_timestamp(value):
    dt = parse_timestamp(value)

    if dt is None:
        raise ValueError("invalid timestamp")

    return (
        dt.strftime("%Y-%m-%dT%H:%M:%S.")
        + f"{dt.microsecond // 1000:03d}Z"
    )


# ============================================================
# CANONICAL TEXT
# ============================================================

def canonical_text(value):
    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    value = value.lower()

    # Unicode whitespace -> one ASCII space.
    return " ".join(value.split())


# ============================================================
# REVISION
# ============================================================

def valid_revision(value):
    return (
        type(value) is int
        and value >= 0
        and value <= MAX_SAFE_INTEGER
    )


# ============================================================
# CRC32C
# ============================================================

def crc32c(data: bytes) -> int:

    crc = 0xFFFFFFFF

    for byte in data:

        crc ^= byte

        for _ in range(8):

            if crc & 1:
                crc = (
                    crc >> 1
                ) ^ 0x82F63B78

            else:
                crc >>= 1

    return crc ^ 0xFFFFFFFF


def crc32c_hex(data: bytes) -> str:
    return f"{crc32c(data):08x}"


# Known CRC32C Castagnoli test vector.
assert (
    crc32c_hex(b"123456789")
    == "e3069283"
)


# ============================================================
# SHA-256
# ============================================================

def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ============================================================
# WORD SET
# ============================================================

def word_set(value):

    value = value.lower()

    result = set()
    current = []

    for char in value:

        category = unicodedata.category(char)

        if (
            category.startswith("L")
            or category.startswith("N")
        ):

            current.append(char)

        else:

            if current:
                result.add(
                    "".join(current)
                )
                current = []

    if current:
        result.add(
            "".join(current)
        )

    return result


def jaccard(a, b):

    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


# ============================================================
# ROW VALIDATION
# ============================================================

def valid_row_shape(row):

    if not isinstance(row, dict):
        return False

    if set(row.keys()) != EXPECTED_ROW_KEYS:
        return False

    if not isinstance(row["id"], str):
        return False

    if not isinstance(row["entity"], str):
        return False

    if not isinstance(row["eventTime"], str):
        return False

    if not isinstance(row["text"], str):
        return False

    if not valid_revision(row["revision"]):
        return False

    if parse_timestamp(
        row["eventTime"]
    ) is None:
        return False

    return True


# ============================================================
# ROW SERIALIZATION
# ============================================================

def output_row(row):

    return {
        "id": row["id"],
        "entity": row["entity"],
        "eventTime": row["eventTime"],
        "revision": row["revision"],
        "text": row["text"],
    }


def row_json(row):

    return compact_json(
        output_row(row)
    )


# ============================================================
# POLICY
# ============================================================

def validate_policy(policy):

    if not isinstance(policy, dict):
        return (
            False,
            None,
            None,
            None,
        )

    min_time = parse_timestamp(
        policy.get("minTime")
    )

    max_time = parse_timestamp(
        policy.get("maxTime")
    )

    threshold = policy.get(
        "contaminationThreshold"
    )

    threshold_valid = (
        isinstance(
            threshold,
            (int, float),
        )
        and not isinstance(
            threshold,
            bool,
        )
        and math.isfinite(
            float(threshold)
        )
        and 0 <= threshold <= 1
    )

    if min_time is None:
        return False, None, None, None

    if max_time is None:
        return False, None, None, None

    if not threshold_valid:
        return False, None, None, None

    if min_time > max_time:
        return False, None, None, None

    return (
        True,
        min_time,
        max_time,
        float(threshold),
    )


# ============================================================
# JSONL
# ============================================================

def parse_jsonl(content):

    # JSONL uses LF. CRLF is supported.
    lines = content.split("\n")

    cleaned = []

    for line in lines:

        if line.endswith("\r"):
            line = line[:-1]

        cleaned.append(line)

    # Blank lines are ignored.
    non_blank = [
        line
        for line in cleaned
        if line.strip() != ""
    ]

    # File must contain at least one row.
    if not non_blank:
        return [], False, True

    rows = []

    jsonl_invalid = False
    schema_invalid = False

    for line in non_blank:

        try:

            value = parse_json_line(line)

        except (
            json.JSONDecodeError,
            DuplicateKeyError,
            TypeError,
            ValueError,
        ):

            jsonl_invalid = True
            continue

        if not valid_row_shape(value):

            schema_invalid = True
            continue

        rows.append(value)

    return (
        rows,
        jsonl_invalid,
        schema_invalid,
    )


# ============================================================
# OBJECT VALIDATION
# ============================================================

def validate_object(obj):

    # A non-object cannot have valid object fields.
    if not isinstance(obj, dict):

        return {
            "accepted": False,
            "rejection": {
                "uri": None,
                "reasonCodes": sorted_reasons([
                    "URI_INVALID",
                    "GENERATION_INVALID",
                    "CRC32C_INVALID",
                    "SCHEMA_INVALID",
                ]),
            },
        }

    reasons = []

    # --------------------------------------------------------
    # URI
    # --------------------------------------------------------

    uri = obj.get("uri")

    if not isinstance(uri, str):

        reasons.append(
            "URI_INVALID"
        )

    elif URI_RE.fullmatch(uri) is None:

        reasons.append(
            "URI_INVALID"
        )

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    generation = obj.get(
        "generation"
    )

    fetched_generation = obj.get(
        "fetchedGeneration"
    )

    generation_valid = (
        isinstance(
            generation,
            str,
        )
        and GENERATION_RE.fullmatch(
            generation
        ) is not None
    )

    fetched_generation_valid = (
        isinstance(
            fetched_generation,
            str,
        )
        and GENERATION_RE.fullmatch(
            fetched_generation
        ) is not None
    )

    if (
        not generation_valid
        or not fetched_generation_valid
    ):

        reasons.append(
            "GENERATION_INVALID"
        )

    # Mismatch only applies when BOTH are valid
    # decimal strings.
    if (
        generation_valid
        and fetched_generation_valid
        and generation != fetched_generation
    ):

        reasons.append(
            "GENERATION_MISMATCH"
        )

    # --------------------------------------------------------
    # CONTENT
    # --------------------------------------------------------

    content = obj.get(
        "content"
    )

    # --------------------------------------------------------
    # CRC32C
    # --------------------------------------------------------

    supplied_crc = obj.get(
        "crc32c"
    )

    crc_valid = (
        isinstance(
            supplied_crc,
            str,
        )
        and CRC_RE.fullmatch(
            supplied_crc
        ) is not None
    )

    if not crc_valid:

        reasons.append(
            "CRC32C_INVALID"
        )

    # Only calculate mismatch when:
    # 1. content is string
    # 2. CRC syntax is valid
    if (
        isinstance(content, str)
        and crc_valid
    ):

        actual_crc = crc32c_hex(
            content.encode("utf-8")
        )

        if actual_crc != supplied_crc:

            reasons.append(
                "CRC32C_MISMATCH"
            )

    # --------------------------------------------------------
    # SCHEMA
    # --------------------------------------------------------

    schema_id = obj.get(
        "schemaId"
    )

    if schema_id != "training-v1":

        reasons.append(
            "SCHEMA_INVALID"
        )

    if not isinstance(
        content,
        str,
    ):

        reasons.append(
            "SCHEMA_INVALID"
        )

    # --------------------------------------------------------
    # JSONL
    # --------------------------------------------------------

    parsed_rows = []

    if isinstance(
        content,
        str,
    ):

        (
            parsed_rows,
            jsonl_invalid,
            schema_invalid,
        ) = parse_jsonl(
            content
        )

        if jsonl_invalid:

            reasons.append(
                "JSONL_INVALID"
            )

        if schema_invalid:

            reasons.append(
                "SCHEMA_INVALID"
            )

    reasons = sorted_reasons(
        reasons
    )

    # --------------------------------------------------------
    # REJECTED OBJECT
    # --------------------------------------------------------

    if reasons:

        return {
            "accepted": False,
            "rejection": {
                "uri": (
                    uri
                    if isinstance(
                        uri,
                        str,
                    )
                    else None
                ),
                "reasonCodes": reasons,
            },
        }

    # --------------------------------------------------------
    # ACCEPTED OBJECT
    # --------------------------------------------------------

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

            row = {
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
                row["entity"],
                row["eventTime"],
                row["text"],
            )

            groups.setdefault(
                key,
                [],
            ).append(row)

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

        retained.append(
            ordered[0]
        )

        for loser in ordered[1:]:

            rejected.append({
                "id": loser["id"],
                "reasonCodes": [
                    "DUPLICATE"
                ],
            })

    return retained, rejected


# ============================================================
# SPLITTING
# ============================================================

def split_name(row):

    digest = hashlib.sha256(
        row["entity"].encode("utf-8")
    ).digest()

    bucket = digest[0] % 10

    if bucket <= 5:
        return "train"

    if bucket <= 7:
        return "validation"

    return "test"


# ============================================================
# SORTING
# ============================================================

def sort_split(rows):

    return sorted(
        rows,
        key=lambda row: (
            row["id"].encode("utf-8"),
            row_json(row).encode("utf-8"),
        ),
    )


def sort_rejected_objects(items):

    return sorted(
        items,
        key=lambda item: (
            (
                item["uri"].encode("utf-8")
                if isinstance(
                    item["uri"],
                    str,
                )
                else b""
            ),
            compact_json(
                item
            ).encode("utf-8"),
        ),
    )


def sort_rejected_rows(items):

    return sorted(
        items,
        key=lambda item: (
            item["id"].encode("utf-8"),
            compact_json(
                item
            ).encode("utf-8"),
        ),
    )


def sort_lineage(items):

    return sorted(
        items,
        key=lambda item: (
            item["uri"].encode("utf-8"),
            compact_json(
                item
            ).encode("utf-8"),
        ),
    )


# ============================================================
# SERIALIZATION
# ============================================================

def serialize_rows(rows):

    result = bytearray()

    for row in rows:

        result.extend(
            row_json(row).encode(
                "utf-8"
            )
        )

        result.extend(
            b"\n"
        )

    return bytes(result)


# ============================================================
# ENDPOINT
# ============================================================

@app.post("/build-corpus")
async def build_corpus(
    request: Request
):

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

    # Top-level request must be object.
    if not isinstance(
        body,
        dict,
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            },
        )

    # policy must be present.
    if "policy" not in body:

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            },
        )

    # objects must be an array.
    if not isinstance(
        body.get("objects"),
        list,
    ):

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
    ) = validate_policy(
        policy
    )

    # --------------------------------------------------------
    # OBJECTS
    # --------------------------------------------------------

    accepted_objects = []
    rejected_objects = []

    for obj in objects:

        result = validate_object(
            obj
        )

        if result["accepted"]:

            accepted_objects.append(
                result
            )

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
            "generation": obj[
                "generation"
            ],
            "crc32c": obj[
                "crc32c"
            ],
            "schemaId": obj[
                "schemaId"
            ],
        })

    # --------------------------------------------------------
    # DEDUPLICATION
    # --------------------------------------------------------

    retained, rejected_rows = (
        deduplicate(
            accepted_objects
        )
    )

    # --------------------------------------------------------
    # POLICY / WINDOW
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

            event_time = parse_timestamp(
                row["eventTime"]
            )

            if (
                event_time < min_time
                or event_time > max_time
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

        split = split_name(row)

        if split == "train":

            train.append(row)

        elif split == "validation":

            validation.append(row)

        else:

            test.append(row)

    # --------------------------------------------------------
    # CONTAMINATION
    # --------------------------------------------------------

    train_words = [
        word_set(row["text"])
        for row in train
    ]

    def contaminated(row):

        target = word_set(
            row["text"]
        )

        for train_word_set in train_words:

            similarity = jaccard(
                target,
                train_word_set,
            )

            if similarity >= threshold:
                return True

        return False

    clean_validation = []

    for row in validation:

        if contaminated(row):

            rejected_rows.append({
                "id": row["id"],
                "reasonCodes": [
                    "TRAIN_CONTAMINATION"
                ],
            })

        else:

            clean_validation.append(
                row
            )

    clean_test = []

    for row in test:

        if contaminated(row):

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
    validation = sort_split(
        validation
    )
    test = sort_split(test)

    # --------------------------------------------------------
    # DIGESTS
    # --------------------------------------------------------

    train_bytes = serialize_rows(
        train
    )

    validation_bytes = serialize_rows(
        validation
    )

    test_bytes = serialize_rows(
        test
    )

    digests = {
        "train": sha256_hex(
            train_bytes
        ),
        "validation": sha256_hex(
            validation_bytes
        ),
        "test": sha256_hex(
            test_bytes
        ),
    }

    # --------------------------------------------------------
    # MERGE REJECTED ROWS
    # --------------------------------------------------------

    row_map = {}

    for rejection in rejected_rows:

        rid = rejection["id"]

        if rid not in row_map:

            row_map[rid] = []

        row_map[rid].extend(
            rejection[
                "reasonCodes"
            ]
        )

    final_rejected_rows = []

    for rid, reasons in row_map.items():

        final_rejected_rows.append({
            "id": rid,
            "reasonCodes": sorted_reasons(
                reasons
            ),
        })

    # --------------------------------------------------------
    # FINAL SORTING
    # --------------------------------------------------------

    rejected_objects = (
        sort_rejected_objects(
            rejected_objects
        )
    )

    final_rejected_rows = (
        sort_rejected_rows(
            final_rejected_rows
        )
    )

    lineage = sort_lineage(
        lineage
    )

    # --------------------------------------------------------
    # EXACT RESPONSE SHAPE
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
        "rejectedObjects": (
            rejected_objects
        ),
        "rejectedRows": (
            final_rejected_rows
        ),
        "digests": digests,
        "lineage": lineage,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "ok"
    }


@app.get("/health")
async def health():
    return {
        "status": "ok"
    }
