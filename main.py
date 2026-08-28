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
# Regular expressions
# ============================================================

# YYYY-MM-DDTHH:mm:ss[.sss](Z|+HH:mm|-HH:mm)
TIME_RE = re.compile(
    r"^"
    r"(\d{4})-(\d{2})-(\d{2})"
    r"T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})"
    r"$"
)

# Decimal string. No sign, decimal point, whitespace, etc.
GENERATION_RE = re.compile(r"^[0-9]+$")

# Exactly 8 lowercase hexadecimal characters.
CRC_RE = re.compile(r"^[0-9a-f]{8}$")

# gs://bucket/object
#
# The object part may contain additional "/" characters.
URI_RE = re.compile(r"^gs://[^/\s]+/.+$")


# ============================================================
# Generic helpers
# ============================================================

def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def utf8(value: str) -> bytes:
    return value.encode("utf-8")


def utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


# ============================================================
# Timestamp handling
# ============================================================

def parse_timestamp(value: Any):
    """
    Validate an assignment timestamp and return a UTC datetime.

    Accepted:
      YYYY-MM-DDTHH:mm:ssZ
      YYYY-MM-DDTHH:mm:ss.sZ
      YYYY-MM-DDTHH:mm:ss.ssZ
      YYYY-MM-DDTHH:mm:ss.sssZ

    Or the same with +/-HH:mm offset.

    Returns None when invalid.
    """

    if not isinstance(value, str):
        return None

    match = TIME_RE.fullmatch(value)

    if match is None:
        return None

    (
        year,
        month,
        day,
        hour,
        minute,
        second,
        fraction,
        offset,
    ) = match.groups()

    year = int(year)
    month = int(month)
    day = int(day)
    hour = int(hour)
    minute = int(minute)
    second = int(second)

    # Validate timezone offset.
    if offset == "Z":
        tz = timezone.utc
    else:
        offset_hour = int(offset[1:3])
        offset_minute = int(offset[4:6])

        # Maximum magnitude is 14:00.
        if offset_hour > 14:
            return None

        # +14:30 and -14:30 are invalid.
        if offset_hour == 14 and offset_minute != 0:
            return None

        sign = 1 if offset[0] == "+" else -1

        tz = timezone(
            sign * timedelta(
                hours=offset_hour,
                minutes=offset_minute,
            )
        )

    # Convert .1 -> .100
    #        .12 -> .120
    #        .123 -> .123
    if fraction is None:
        milliseconds = 0
    else:
        milliseconds = int(fraction.ljust(3, "0"))

    try:
        dt = datetime(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
            microsecond=milliseconds * 1000,
            tzinfo=tz,
        )
    except ValueError:
        return None

    return dt.astimezone(timezone.utc)


def canonical_timestamp(value: str) -> str:
    """
    Convert a valid timestamp to:
    YYYY-MM-DDTHH:mm:ss.sssZ
    """

    dt = parse_timestamp(value)

    if dt is None:
        raise ValueError("invalid timestamp")

    return (
        dt.strftime("%Y-%m-%dT%H:%M:%S.")
        + f"{dt.microsecond // 1000:03d}Z"
    )


# ============================================================
# Canonicalization
# ============================================================

def canonical_text(value: str) -> str:
    """
    NFKC -> lowercase -> trim -> collapse Unicode whitespace.
    """

    value = unicodedata.normalize("NFKC", value)
    value = value.lower()

    # str.split() recognizes Unicode whitespace.
    return " ".join(value.split())


# ============================================================
# Revision validation
# ============================================================

MAX_SAFE_INTEGER = 9007199254740991


def valid_revision(value: Any) -> bool:
    return (
        type(value) is int
        and 0 <= value <= MAX_SAFE_INTEGER
    )


# ============================================================
# CRC32C
# ============================================================

def crc32c(data: bytes) -> int:
    """
    CRC32C / Castagnoli.

    Polynomial in reflected form:
    0x82F63B78
    """

    crc = 0xFFFFFFFF
    polynomial = 0x82F63B78

    for byte in data:
        crc ^= byte

        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ polynomial
            else:
                crc >>= 1

    return crc ^ 0xFFFFFFFF


def crc32c_hex(data: bytes) -> str:
    return f"{crc32c(data):08x}"


# ============================================================
# SHA-256
# ============================================================

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ============================================================
# Word sets / contamination
# ============================================================

def word_set(value: str) -> set[str]:
    """
    Build a lowercase Unicode letter/number word-set.

    A word is a maximal sequence of Unicode characters whose
    Unicode category begins with L (letter) or N (number).
    """

    value = value.lower()

    result = set()
    current = []

    for char in value:
        category = unicodedata.category(char)

        if category.startswith(("L", "N")):
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
# Row helpers
# ============================================================

EXPECTED_ROW_KEYS = {
    "id",
    "entity",
    "eventTime",
    "revision",
    "text",
}


def row_has_valid_shape(row: Any) -> bool:
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

    if parse_timestamp(row["eventTime"]) is None:
        return False

    return True


def output_row(row: dict) -> dict:
    """
    Explicitly create the required output key order.
    """

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
# Sorting helpers
# ============================================================

def sort_reason_codes(codes: list[str]) -> list[str]:
    return sorted(
        set(codes),
        key=utf8_key,
    )


def sort_rejected_objects(items: list[dict]) -> list[dict]:
    return sorted(
        items,
        key=lambda item: (
            utf8_key(item["uri"])
            if isinstance(item["uri"], str)
            else b"",
            utf8_key(compact_json(item)),
        ),
    )


def sort_rejected_rows(items: list[dict]) -> list[dict]:
    return sorted(
        items,
        key=lambda item: (
            utf8_key(item["id"]),
            utf8_key(compact_json(item)),
        ),
    )


def sort_lineage(items: list[dict]) -> list[dict]:
    return sorted(
        items,
        key=lambda item: (
            utf8_key(item["uri"]),
            utf8_key(compact_json(item)),
        ),
    )


# ============================================================
# Policy
# ============================================================

def validate_policy(policy: Any):
    """
    Returns:
        (valid, min_datetime, max_datetime, threshold)
    """

    if not isinstance(policy, dict):
        return False, None, None, None

    min_time = parse_timestamp(policy.get("minTime"))
    max_time = parse_timestamp(policy.get("maxTime"))

    threshold = policy.get("contaminationThreshold")

    # bool is technically an int in Python, so explicitly reject it.
    threshold_valid = (
        isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and math.isfinite(float(threshold))
        and 0 <= threshold <= 1
    )

    if min_time is None or max_time is None or not threshold_valid:
        return False, min_time, max_time, threshold

    if min_time > max_time:
        return False, min_time, max_time, threshold

    return True, min_time, max_time, float(threshold)


# ============================================================
# Object processing
# ============================================================

def process_object(obj: Any):
    """
    Validate one source object.

    Returns either:
        {
            "accepted": True,
            ...
        }

    or:
        {
            "accepted": False,
            "rejected": {...}
        }
    """

    # If the object isn't an object, every relevant object-level
    # identity/schema condition is independently applicable.
    if not isinstance(obj, dict):
        return {
            "accepted": False,
            "rejected": {
                "uri": None,
                "reasonCodes": sort_reason_codes([
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

    if not isinstance(uri, str) or URI_RE.fullmatch(uri) is None:
        reasons.append("URI_INVALID")

    # --------------------------------------------------------
    # Generation
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

    crc_syntax_valid = (
        isinstance(supplied_crc, str)
        and CRC_RE.fullmatch(supplied_crc) is not None
    )

    if not crc_syntax_valid:
        reasons.append("CRC32C_INVALID")

    # IMPORTANT:
    # CRC mismatch is checked ONLY when:
    #   1. content is a string
    #   2. CRC syntax is valid
    content = obj.get("content")

    if isinstance(content, str) and crc_syntax_valid:
        actual_crc = crc32c_hex(content.encode("utf-8"))

        if actual_crc != supplied_crc:
            reasons.append("CRC32C_MISMATCH")

    # --------------------------------------------------------
    # Schema/content
    # --------------------------------------------------------

    schema_id = obj.get("schemaId")

    if not isinstance(content, str):
        reasons.append("SCHEMA_INVALID")

    if schema_id != "training-v1":
        reasons.append("SCHEMA_INVALID")

    parsed_rows = []

    # --------------------------------------------------------
    # JSONL
    # --------------------------------------------------------

    if isinstance(content, str):

        lines = content.splitlines()

        # Empty / blank-only file.
        if not any(line.strip() for line in lines):
            reasons.append("SCHEMA_INVALID")

        else:
            jsonl_error = False
            row_schema_error = False

            for line in lines:

                # Blank lines are ignored.
                if not line.strip():
                    continue

                try:
                    parsed = json.loads(line)
                except (json.JSONDecodeError, TypeError, ValueError):
                    jsonl_error = True
                    continue

                if not row_has_valid_shape(parsed):
                    row_schema_error = True
                    continue

                parsed_rows.append(parsed)

            if jsonl_error:
                reasons.append("JSONL_INVALID")

            if row_schema_error:
                reasons.append("SCHEMA_INVALID")

    # If any object-level error occurred, reject the entire object.
    if reasons:
        return {
            "accepted": False,
            "rejected": {
                "uri": uri if isinstance(uri, str) else None,
                "reasonCodes": sort_reason_codes(reasons),
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
# Endpoint
# ============================================================

@app.post("/build-corpus")
async def build_corpus(request: Request):

    # ========================================================
    # REQUEST PARSING
    # ========================================================
    #
    # Do NOT use a Pydantic request model here.
    #
    # We need complete control over malformed/missing fields
    # so the assignment's exact 400 response can be returned.
    # ========================================================

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    # Request itself must be a JSON object.
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    # Missing policy -> exact INVALID_INPUT.
    if "policy" not in body:
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    # Missing objects -> non-array -> exact INVALID_INPUT.
    if not isinstance(body.get("objects"), list):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    policy = body["policy"]
    objects = body["objects"]

    # ========================================================
    # POLICY
    # ========================================================

    (
        policy_valid,
        min_time,
        max_time,
        threshold,
    ) = validate_policy(policy)

    # ========================================================
    # OBJECT VALIDATION
    # ========================================================

    rejected_objects = []
    accepted_objects = []

    for obj in objects:

        result = process_object(obj)

        if result["accepted"]:
            accepted_objects.append(result)
        else:
            rejected_objects.append(result["rejected"])

    # ========================================================
    # LINEAGE
    # ========================================================
    #
    # Lineage represents successfully validated source objects.
    # It is independent of whether their rows later survive
    # deduplication, policy filtering, or contamination.
    # ========================================================

    lineage = []

    for obj in accepted_objects:
        lineage.append({
            "uri": obj["uri"],
            "generation": obj["generation"],
            "crc32c": obj["crc32c"],
            "schemaId": obj["schemaId"],
        })

    # ========================================================
    # COLLECT CANONICAL ROWS
    # ========================================================

    candidates = []

    for obj in accepted_objects:

        for original in obj["rows"]:

            canonical = {
                "id": original["id"],
                "entity": canonical_text(original["entity"]),
                "eventTime": canonical_timestamp(
                    original["eventTime"]
                ),
                "revision": original["revision"],
                "text": canonical_text(original["text"]),
            }

            dedup_key = (
                canonical["entity"],
                canonical["eventTime"],
                canonical["text"],
            )

            candidates.append({
                "row": canonical,
                "dedup_key": dedup_key,
            })

    # ========================================================
    # DEDUPLICATION
    # ========================================================

    groups = {}

    for candidate in candidates:
        groups.setdefault(
            candidate["dedup_key"],
            []
        ).append(candidate)

    retained = []
    rejected_rows = []

    for group in groups.values():

        # Winner:
        #   1. highest revision
        #   2. UTF-8-byte-smallest ID
        ordered = sorted(
            group,
            key=lambda item: (
                -item["row"]["revision"],
                utf8_key(item["row"]["id"]),
            ),
        )

        winner = ordered[0]
        retained.append(winner)

        for loser in ordered[1:]:
            rejected_rows.append({
                "id": loser["row"]["id"],
                "reasonCodes": ["DUPLICATE"],
            })

    # ========================================================
    # POLICY / TIME WINDOW
    # ========================================================

    if not policy_valid:

        for item in retained:
            rejected_rows.append({
                "id": item["row"]["id"],
                "reasonCodes": ["POLICY_INVALID"],
            })

        retained = []

    else:

        kept = []

        for item in retained:

            event_dt = parse_timestamp(
                item["row"]["eventTime"]
            )

            if event_dt < min_time or event_dt > max_time:

                rejected_rows.append({
                    "id": item["row"]["id"],
                    "reasonCodes": ["OUT_OF_WINDOW"],
                })

            else:
                kept.append(item)

        retained = kept

    # ========================================================
    # SPLIT
    # ========================================================

    train = []
    validation = []
    test = []

    for item in retained:

        entity_bytes = utf8(item["row"]["entity"])

        first_byte = sha256(entity_bytes)[0]

        bucket = first_byte % 10

        if 0 <= bucket <= 5:
            train.append(item)

        elif 6 <= bucket <= 7:
            validation.append(item)

        else:
            test.append(item)

    # ========================================================
    # CONTAMINATION
    # ========================================================

    train_word_sets = [
        word_set(item["row"]["text"])
        for item in train
    ]

    def contaminated(item):
        target = word_set(item["row"]["text"])

        for train_words in train_word_sets:

            similarity = jaccard(
                target,
                train_words,
            )

            if similarity >= threshold:
                return True

        return False

    clean_validation = []

    for item in validation:

        if contaminated(item):
            rejected_rows.append({
                "id": item["row"]["id"],
                "reasonCodes": [
                    "TRAIN_CONTAMINATION"
                ],
            })
        else:
            clean_validation.append(item)

    clean_test = []

    for item in test:

        if contaminated(item):
            rejected_rows.append({
                "id": item["row"]["id"],
                "reasonCodes": [
                    "TRAIN_CONTAMINATION"
                ],
            })
        else:
            clean_test.append(item)

    validation = clean_validation
    test = clean_test

    # ========================================================
    # SORT SPLITS
    # ========================================================

    def sort_split(items):

        return sorted(
            items,
            key=lambda item: (
                utf8_key(item["row"]["id"]),
                utf8_key(row_json(item["row"])),
            ),
        )

    train = sort_split(train)
    validation = sort_split(validation)
    test = sort_split(test)

    # ========================================================
    # JSONL ARTIFACTS
    # ========================================================

    def serialize_split(items) -> bytes:

        pieces = []

        for item in items:
            pieces.append(
                utf8(row_json(item["row"]))
            )
            pieces.append(b"\n")

        return b"".join(pieces)

    train_bytes = serialize_split(train)
    validation_bytes = serialize_split(validation)
    test_bytes = serialize_split(test)

    # ========================================================
    # DIGESTS
    # ========================================================

    digests = {
        "train": sha256_hex(train_bytes),
        "validation": sha256_hex(validation_bytes),
        "test": sha256_hex(test_bytes),
    }

    # ========================================================
    # MERGE REJECTED ROW REASONS
    # ========================================================

    row_reasons = {}

    for rejection in rejected_rows:

        rid = rejection["id"]

        row_reasons.setdefault(rid, [])

        row_reasons[rid].extend(
            rejection["reasonCodes"]
        )

    rejected_rows = []

    for rid, reasons in row_reasons.items():

        rejected_rows.append({
            "id": rid,
            "reasonCodes": sort_reason_codes(reasons),
        })

    # ========================================================
    # FINAL SORTING
    # ========================================================

    rejected_objects = sort_rejected_objects(
        rejected_objects
    )

    rejected_rows = sort_rejected_rows(
        rejected_rows
    )

    lineage = sort_lineage(lineage)

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {
        "splits": {
            "train": [
                output_row(item["row"])
                for item in train
            ],
            "validation": [
                output_row(item["row"])
                for item in validation
            ],
            "test": [
                output_row(item["row"])
                for item in test
            ],
        },
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows,
        "digests": digests,
        "lineage": lineage,
    }


# ============================================================
# Health endpoints
# ============================================================

@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}
