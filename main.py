import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


app = FastAPI()


# ============================================================
# CONSTANTS
# ============================================================

MAX_SAFE_INTEGER = 9007199254740991

ROW_KEYS = {
    "id",
    "entity",
    "eventTime",
    "revision",
    "text",
}

GENERATION_RE = re.compile(r"^[0-9]+$")

CRC32C_RE = re.compile(r"^[0-9a-f]{8}$")

# Assignment form:
# gs://bucket/object
#
# bucket = one or more characters other than /
# object = one or more characters after /
URI_RE = re.compile(
    r"^gs://[^/]+/[^/]+$"
)

TIME_RE = re.compile(
    r"^"
    r"(\d{4})"
    r"-"
    r"(\d{2})"
    r"-"
    r"(\d{2})"
    r"T"
    r"(\d{2})"
    r":"
    r"(\d{2})"
    r":"
    r"(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})"
    r"$"
)


# ============================================================
# JSON HELPERS
# ============================================================

def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def utf8(value):
    return value.encode("utf-8")


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

    m = TIME_RE.fullmatch(value)

    if m is None:
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
    ) = m.groups()

    year = int(year_s)
    month = int(month_s)
    day = int(day_s)
    hour = int(hour_s)
    minute = int(minute_s)
    second = int(second_s)

    if hour > 23:
        return None

    if minute > 59:
        return None

    if second > 59:
        return None

    # --------------------------------------------------------
    # OFFSET
    # --------------------------------------------------------

    if offset == "Z":

        tz = timezone.utc

    else:

        offset_hour = int(
            offset[1:3]
        )

        offset_minute = int(
            offset[4:6]
        )

        if offset_hour > 14:
            return None

        if offset_minute > 59:
            return None

        if (
            offset_hour == 14
            and offset_minute != 0
        ):
            return None

        sign = (
            1
            if offset[0] == "+"
            else -1
        )

        tz = timezone(
            sign * timedelta(
                hours=offset_hour,
                minutes=offset_minute,
            )
        )

    # --------------------------------------------------------
    # FRACTION
    # --------------------------------------------------------

    if fraction is None:

        milliseconds = 0

    else:

        milliseconds = int(
            fraction.ljust(3, "0")
        )

    # --------------------------------------------------------
    # CALENDAR VALIDATION
    # --------------------------------------------------------

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

    return dt.astimezone(
        timezone.utc
    )


def canonical_timestamp(value):

    dt = parse_timestamp(value)

    if dt is None:
        raise ValueError(
            "invalid timestamp"
        )

    return (
        dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        + "."
        + f"{dt.microsecond // 1000:03d}"
        + "Z"
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

    # Python split() handles Unicode whitespace.
    # Joining with ASCII space performs trim + collapse.
    return " ".join(
        value.split()
    )


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

def crc32c(data):

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


def crc32c_hex(data):

    return f"{crc32c(data):08x}"


# Castagnoli standard test vector.
assert (
    crc32c_hex(
        b"123456789"
    )
    == "e3069283"
)


# ============================================================
# SHA256
# ============================================================

def sha256_hex(data):

    return hashlib.sha256(
        data
    ).hexdigest()


# ============================================================
# ROW VALIDATION
# ============================================================

def row_is_valid(row):

    if not isinstance(
        row,
        dict,
    ):
        return False

    # EXACTLY the five fields.
    if set(row.keys()) != ROW_KEYS:
        return False

    # Four text fields.
    if not isinstance(
        row["id"],
        str,
    ):
        return False

    if not isinstance(
        row["entity"],
        str,
    ):
        return False

    if not isinstance(
        row["eventTime"],
        str,
    ):
        return False

    if not isinstance(
        row["text"],
        str,
    ):
        return False

    # Revision.
    if not valid_revision(
        row["revision"]
    ):
        return False

    # Event time.
    if parse_timestamp(
        row["eventTime"]
    ) is None:
        return False

    return True


# ============================================================
# JSONL PARSER
# ============================================================

def parse_jsonl(content):

    """
    Returns:

        rows
        jsonl_invalid
        schema_invalid

    IMPORTANT:
    If even one non-blank line is malformed, the object
    receives JSONL_INVALID and is rejected.

    If even one successfully parsed line has the wrong row
    shape, the object receives SCHEMA_INVALID and is rejected.
    """

    lines = content.split("\n")

    non_blank = []

    for line in lines:

        # JSONL commonly arrives as CRLF.
        if line.endswith("\r"):
            line = line[:-1]

        # Blank lines are ignored.
        if line.strip() == "":
            continue

        non_blank.append(line)

    # At least one row required.
    if len(non_blank) == 0:

        return (
            [],
            False,
            True,
        )

    rows = []

    jsonl_invalid = False
    schema_invalid = False

    for line in non_blank:

        try:

            value = json.loads(
                line
            )

        except Exception:

            jsonl_invalid = True

            continue

        if not row_is_valid(
            value
        ):

            schema_invalid = True

            continue

        rows.append(
            value
        )

    return (
        rows,
        jsonl_invalid,
        schema_invalid,
    )


# ============================================================
# OBJECT VALIDATION
# ============================================================

def validate_object(obj):

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # If an item isn't an object, the only sensible object-level
    # classification available is SCHEMA_INVALID.
    #
    # We do NOT manufacture URI/GENERATION/CRC failures for
    # fields that don't exist.
    # --------------------------------------------------------

    if not isinstance(
        obj,
        dict,
    ):

        return {
            "accepted": False,
            "rejection": {
                "uri": None,
                "reasonCodes": [
                    "SCHEMA_INVALID"
                ],
            },
        }

    reasons = []

    # --------------------------------------------------------
    # URI
    # --------------------------------------------------------

    uri = obj.get(
        "uri"
    )

    if (
        not isinstance(uri, str)
        or URI_RE.fullmatch(uri) is None
    ):

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

    if not generation_valid:

        reasons.append(
            "GENERATION_INVALID"
        )

    if not fetched_generation_valid:

        reasons.append(
            "GENERATION_INVALID"
        )

    # Mismatch only makes sense when both values
    # are valid decimal strings.
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

    content_valid = isinstance(
        content,
        str,
    )

    # --------------------------------------------------------
    # CRC32C
    # --------------------------------------------------------

    supplied_crc = obj.get(
        "crc32c"
    )

    crc_syntax_valid = (
        isinstance(
            supplied_crc,
            str,
        )
        and CRC32C_RE.fullmatch(
            supplied_crc
        ) is not None
    )

    if not crc_syntax_valid:

        reasons.append(
            "CRC32C_INVALID"
        )

    # EXACTLY as assignment specifies:
    #
    # check mismatch only if:
    # - content is string
    # - CRC syntax is valid
    if (
        content_valid
        and crc_syntax_valid
    ):

        actual_crc = crc32c_hex(
            content.encode("utf-8")
        )

        if actual_crc != supplied_crc:

            reasons.append(
                "CRC32C_MISMATCH"
            )

    # --------------------------------------------------------
    # SCHEMA ID
    # --------------------------------------------------------

    schema_id = obj.get(
        "schemaId"
    )

    if schema_id != "training-v1":

        reasons.append(
            "SCHEMA_INVALID"
        )

    # Non-string content.
    if not content_valid:

        reasons.append(
            "SCHEMA_INVALID"
        )

    # --------------------------------------------------------
    # JSONL
    # --------------------------------------------------------

    rows = []

    if content_valid:

        (
            rows,
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

    # --------------------------------------------------------
    # OBJECT DECISION
    # --------------------------------------------------------

    reasons = sorted_reasons(
        reasons
    )

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
    # VALID OBJECT
    # --------------------------------------------------------

    return {
        "accepted": True,
        "uri": uri,
        "generation": generation,
        "crc32c": supplied_crc,
        "schemaId": schema_id,
        "rows": rows,
    }


# ============================================================
# CANONICAL ROW
# ============================================================

def canonicalize_row(row):

    return {
        "id": row["id"],
        "entity": canonical_text(
            row["entity"]
        ),
        "eventTime": canonical_timestamp(
            row["eventTime"]
        ),
        "revision": row["revision"],
        "text": canonical_text(
            row["text"]
        ),
    }


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate(objects):

    groups = {}

    for obj in objects:

        for original in obj["rows"]:

            row = canonicalize_row(
                original
            )

            key = (
                row["entity"],
                row["eventTime"],
                row["text"],
            )

            groups.setdefault(
                key,
                [],
            ).append(
                row
            )

    retained = []
    rejected = []

    for group in groups.values():

        # Highest revision first.
        #
        # For equal revision:
        # smallest UTF-8 ID first.
        ordered = sorted(
            group,
            key=lambda row: (
                -row["revision"],
                row["id"].encode(
                    "utf-8"
                ),
            ),
        )

        winner = ordered[0]

        retained.append(
            winner
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
# SPLIT
# ============================================================

def determine_split(row):

    digest = hashlib.sha256(
        row["entity"].encode(
            "utf-8"
        )
    ).digest()

    bucket = digest[0] % 10

    if bucket <= 5:
        return "train"

    if bucket <= 7:
        return "validation"

    return "test"


# ============================================================
# WORD SET / JACCARD
# ============================================================

def word_set(value):

    value = value.lower()

    result = set()
    current = []

    for char in value:

        category = unicodedata.category(
            char
        )

        if (
            category.startswith("L")
            or category.startswith("N")
        ):

            current.append(
                char
            )

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
# ROW SERIALIZATION
# ============================================================

def row_json(row):

    return compact_json({
        "id": row["id"],
        "entity": row["entity"],
        "eventTime": row["eventTime"],
        "revision": row["revision"],
        "text": row["text"],
    })


def sort_split(rows):

    return sorted(
        rows,
        key=lambda row: (
            row["id"].encode("utf-8"),
            row_json(row).encode(
                "utf-8"
            ),
        ),
    )


def serialize_split(rows):

    output = bytearray()

    for row in rows:

        output.extend(
            row_json(row).encode(
                "utf-8"
            )
        )

        output.extend(
            b"\n"
        )

    return bytes(output)


# ============================================================
# POLICY
# ============================================================

def validate_policy(policy):

    if not isinstance(
        policy,
        dict,
    ):

        return (
            False,
            None,
            None,
            None,
        )

    min_time = parse_timestamp(
        policy.get(
            "minTime"
        )
    )

    max_time = parse_timestamp(
        policy.get(
            "maxTime"
        )
    )

    threshold = policy.get(
        "contaminationThreshold"
    )

    # JSON booleans are not contamination thresholds.
    threshold_valid = (
        type(threshold) in (
            int,
            float,
        )
        and math.isfinite(
            float(threshold)
        )
        and 0 <= threshold <= 1
    )

    if min_time is None:
        return (
            False,
            None,
            None,
            None,
        )

    if max_time is None:
        return (
            False,
            None,
            None,
            None,
        )

    if not threshold_valid:
        return (
            False,
            None,
            None,
            None,
        )

    if min_time > max_time:
        return (
            False,
            None,
            None,
            None,
        )

    return (
        True,
        min_time,
        max_time,
        float(threshold),
    )


# ============================================================
# REJECTED ROW MERGE
# ============================================================

def merge_rejected_rows(
    rejected
):

    by_id = {}

    for item in rejected:

        row_id = item["id"]

        by_id.setdefault(
            row_id,
            [],
        ).extend(
            item["reasonCodes"]
        )

    result = []

    for row_id, reasons in by_id.items():

        result.append({
            "id": row_id,
            "reasonCodes": sorted_reasons(
                reasons
            ),
        })

    return sorted(
        result,
        key=lambda item: (
            item["id"].encode(
                "utf-8"
            ),
            compact_json(item).encode(
                "utf-8"
            ),
        ),
    )


# ============================================================
# MAIN ENDPOINT
# ============================================================

@app.post("/build-corpus")
async def build_corpus(
    request: Request
):

    # ========================================================
    # REQUEST PARSING
    # ========================================================

    try:

        body = await request.json()

    except Exception:

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            },
        )

    # Top-level request must be an object.
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

    # Missing policy.
    if "policy" not in body:

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            },
        )

    # Missing objects.
    if "objects" not in body:

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            },
        )

    # objects must be an array.
    if not isinstance(
        body["objects"],
        list,
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            },
        )

    policy = body["policy"]
    supplied_objects = body["objects"]

    # ========================================================
    # POLICY
    # ========================================================

    (
        policy_valid,
        min_time,
        max_time,
        threshold,
    ) = validate_policy(
        policy
    )

    # ========================================================
    # OBJECT VALIDATION
    # ========================================================

    accepted_objects = []
    rejected_objects = []

    for supplied_object in supplied_objects:

        result = validate_object(
            supplied_object
        )

        if result["accepted"]:

            accepted_objects.append(
                result
            )

        else:

            rejected_objects.append(
                result["rejection"]
            )

    # ========================================================
    # LINEAGE
    #
    # Only fully valid objects appear here.
    #
    # Row-level rejection later does NOT remove lineage.
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
    # DEDUPLICATION
    # ========================================================

    (
        retained,
        rejected_rows,
    ) = deduplicate(
        accepted_objects
    )

    # ========================================================
    # POLICY
    # ========================================================

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

        inside = []

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

                inside.append(
                    row
                )

        retained = inside

    # ========================================================
    # SPLIT
    # ========================================================

    train = []
    validation = []
    test = []

    for row in retained:

        split = determine_split(
            row
        )

        if split == "train":

            train.append(
                row
            )

        elif split == "validation":

            validation.append(
                row
            )

        else:

            test.append(
                row
            )

    # ========================================================
    # CONTAMINATION
    # ========================================================

    train_words = [
        word_set(row["text"])
        for row in train
    ]

    def contaminated(row):

        target = word_set(
            row["text"]
        )

        for other in train_words:

            if (
                jaccard(
                    target,
                    other,
                )
                >= threshold
            ):

                return True

        return False

    # Validation.
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

    validation = clean_validation

    # Test.
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

            clean_test.append(
                row
            )

    test = clean_test

    # ========================================================
    # SORT SPLITS
    # ========================================================

    train = sort_split(
        train
    )

    validation = sort_split(
        validation
    )

    test = sort_split(
        test
    )

    # ========================================================
    # DIGESTS
    # ========================================================

    train_digest = sha256_hex(
        serialize_split(train)
    )

    validation_digest = sha256_hex(
        serialize_split(validation)
    )

    test_digest = sha256_hex(
        serialize_split(test)
    )

    # ========================================================
    # REJECTED OBJECTS
    # ========================================================

    rejected_objects.sort(
        key=lambda item: (
            (
                item["uri"].encode(
                    "utf-8"
                )
                if isinstance(
                    item["uri"],
                    str,
                )
                else b""
            ),
            compact_json(item).encode(
                "utf-8"
            ),
        )
    )

    # ========================================================
    # REJECTED ROWS
    # ========================================================

    rejected_rows = merge_rejected_rows(
        rejected_rows
    )

    # ========================================================
    # LINEAGE
    # ========================================================

    lineage.sort(
        key=lambda item: (
            item["uri"].encode(
                "utf-8"
            ),
            compact_json(item).encode(
                "utf-8"
            ),
        )
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "splits": {
            "train": train,
            "validation": validation,
            "test": test,
        },
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows,
        "digests": {
            "train": train_digest,
            "validation": validation_digest,
            "test": test_digest,
        },
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
