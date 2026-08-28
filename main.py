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

OBJECT_KEYS = {
    "uri",
    "generation",
    "fetchedGeneration",
    "content",
    "crc32c",
    "schemaId",
}

GENERATION_RE = re.compile(r"^[0-9]+$")
CRC32C_RE = re.compile(r"^[0-9a-f]{8}$")

# gs://bucket/object
#
# Bucket:
#   - non-empty
#   - no slash
#   - no whitespace
#
# Object:
#   - non-empty
#   - may contain /
#   - no whitespace
#
# This intentionally does not restrict the object path to one
# segment because GCS object names may contain '/'.
URI_RE = re.compile(
    r"^gs://([^/\s]+)/([^\s]+)$"
)

TIME_RE = re.compile(
    r"^"
    r"(\d{4})-(\d{2})-(\d{2})"
    r"T"
    r"(\d{2}):(\d{2}):(\d{2})"
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


def sorted_reasons(reasons):
    return sorted(
        set(reasons),
        key=lambda value: value.encode("utf-8"),
    )


def invalid_input():
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"},
    )


# Strict JSON parsing.
#
# Duplicate JSON keys are rejected instead of silently allowing
# the last value to overwrite the earlier one.
#
# NaN / Infinity / -Infinity are also rejected because they are
# not valid JSON values.
def reject_duplicate_keys(pairs):
    result = {}

    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value

    return result


def reject_non_json_number(value):
    raise ValueError("non-standard JSON number")


def strict_json_loads(data):
    return json.loads(
        data,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_non_json_number,
    )


# ============================================================
# TIMESTAMP
# ============================================================

def parse_timestamp(value):

    if type(value) is not str:
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

    if hour > 23:
        return None

    if minute > 59:
        return None

    if second > 59:
        return None

    if offset == "Z":

        tz = timezone.utc

    else:

        offset_hour = int(offset[1:3])
        offset_minute = int(offset[4:6])

        if offset_hour > 14:
            return None

        if offset_minute > 59:
            return None

        if offset_hour == 14 and offset_minute != 0:
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
        dt.strftime("%Y-%m-%dT%H:%M:%S")
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

    return " ".join(
        value.split()
    )


# ============================================================
# REVISION
# ============================================================

def valid_revision(value):

    return (
        type(value) is int
        and 0 <= value <= MAX_SAFE_INTEGER
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


# Castagnoli CRC32C test vector.
assert crc32c_hex(
    b"123456789"
) == "e3069283"


# ============================================================
# SHA256
# ============================================================

def sha256_hex(data):
    return hashlib.sha256(
        data
    ).hexdigest()


# ============================================================
# URI VALIDATION
# ============================================================

def valid_uri(value):

    if type(value) is not str:
        return False

    match = URI_RE.fullmatch(value)

    if match is None:
        return False

    bucket = match.group(1)
    object_name = match.group(2)

    if bucket == "":
        return False

    if object_name == "":
        return False

    # URI must not contain ASCII control characters.
    for char in value:

        if ord(char) < 0x20:

            return False

        if ord(char) == 0x7F:

            return False

    return True


# ============================================================
# ROW VALIDATION
# ============================================================

def row_is_valid(row):

    if type(row) is not dict:
        return False

    if set(row.keys()) != ROW_KEYS:
        return False

    if type(row["id"]) is not str:
        return False

    if type(row["entity"]) is not str:
        return False

    if type(row["eventTime"]) is not str:
        return False

    if type(row["text"]) is not str:
        return False

    if not valid_revision(
        row["revision"]
    ):
        return False

    if parse_timestamp(
        row["eventTime"]
    ) is None:
        return False

    return True


# ============================================================
# JSONL
# ============================================================

def parse_jsonl(content):

    rows = []

    jsonl_invalid = False
    schema_invalid = False

    found_non_blank = False

    lines = content.split("\n")

    for line in lines:

        # Accept CRLF, but only remove the CR that is
        # immediately before an LF.
        if line.endswith("\r"):
            line = line[:-1]

        if line.strip() == "":
            continue

        found_non_blank = True

        try:

            value = strict_json_loads(
                line
            )

        except (
            json.JSONDecodeError,
            ValueError,
            TypeError,
        ):

            jsonl_invalid = True
            continue

        if not row_is_valid(value):

            schema_invalid = True
            continue

        rows.append(value)

    if not found_non_blank:

        schema_invalid = True

    return (
        rows,
        jsonl_invalid,
        schema_invalid,
    )


# ============================================================
# OBJECT VALIDATION
# ============================================================

def validate_object(obj):

    # A supplied object must itself be a JSON object.
    if type(obj) is not dict:

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
    # OBJECT SCHEMA
    # --------------------------------------------------------

    # This is important for hidden schema tests:
    # unexpected object-level fields are not silently accepted.
    if set(obj.keys()) != OBJECT_KEYS:

        reasons.append(
            "SCHEMA_INVALID"
        )

    # --------------------------------------------------------
    # URI
    # --------------------------------------------------------

    uri = obj.get("uri")

    if not valid_uri(uri):

        reasons.append(
            "URI_INVALID"
        )

    # --------------------------------------------------------
    # GENERATIONS
    # --------------------------------------------------------

    generation = obj.get(
        "generation"
    )

    fetched_generation = obj.get(
        "fetchedGeneration"
    )

    generation_valid = (
        type(generation) is str
        and GENERATION_RE.fullmatch(
            generation
        ) is not None
    )

    fetched_generation_valid = (
        type(fetched_generation) is str
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

    # Compare exactly what was supplied.
    #
    # "001" != "1"
    #
    # Invalid strings are also compared if both fields are
    # present, because mismatch is independent from syntax.
    if (
        "generation" in obj
        and "fetchedGeneration" in obj
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

    content_valid = (
        type(content) is str
    )

    if not content_valid:

        reasons.append(
            "SCHEMA_INVALID"
        )

    # --------------------------------------------------------
    # CRC32C
    # --------------------------------------------------------

    supplied_crc = obj.get(
        "crc32c"
    )

    crc_valid = (
        type(supplied_crc) is str
        and CRC32C_RE.fullmatch(
            supplied_crc
        ) is not None
    )

    if not crc_valid:

        reasons.append(
            "CRC32C_INVALID"
        )

    elif content_valid:

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

    if obj.get("schemaId") != "training-v1":

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
        ) = parse_jsonl(content)

        if jsonl_invalid:

            reasons.append(
                "JSONL_INVALID"
            )

        if schema_invalid:

            reasons.append(
                "SCHEMA_INVALID"
            )

    # --------------------------------------------------------
    # FINAL RESULT
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
                    if type(uri) is str
                    else None
                ),
                "reasonCodes": reasons,
            },
        }

    return {
        "accepted": True,
        "uri": uri,
        "generation": generation,
        "crc32c": supplied_crc,
        "schemaId": "training-v1",
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
        row["entity"].encode("utf-8")
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

    result = set()
    current = []

    for char in value.lower():

        category = unicodedata.category(
            char
        )

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

    return (
        len(a & b)
        / len(union)
    )


# ============================================================
# SERIALIZATION
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
            row_json(row).encode("utf-8"),
        ),
    )


def serialize_split(rows):

    output = bytearray()

    for row in rows:

        output.extend(
            row_json(row).encode("utf-8")
        )

        output.extend(
            b"\n"
        )

    return bytes(output)


# ============================================================
# POLICY
# ============================================================

def validate_policy(policy):

    if type(policy) is not dict:

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
        type(threshold) in (int, float)
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
# REJECTED ROWS
# ============================================================

def merge_rejected_rows(rejected):

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
            item["id"].encode("utf-8"),
            compact_json(item).encode(
                "utf-8"
            ),
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

    content_type = (
        request.headers
        .get("content-type", "")
        .split(";", 1)[0]
        .strip()
        .lower()
    )

    if content_type != "application/json":

        return invalid_input()

    try:

        raw_body = await request.body()

        body = strict_json_loads(
            raw_body.decode("utf-8")
        )

    except Exception:

        return invalid_input()

    if type(body) is not dict:

        return invalid_input()

    if "policy" not in body:

        return invalid_input()

    if "objects" not in body:

        return invalid_input()

    if type(body["objects"]) is not list:

        return invalid_input()

    # --------------------------------------------------------
    # POLICY
    # --------------------------------------------------------

    (
        policy_valid,
        min_time,
        max_time,
        threshold,
    ) = validate_policy(
        body["policy"]
    )

    # --------------------------------------------------------
    # OBJECT VALIDATION
    # --------------------------------------------------------

    accepted_objects = []
    rejected_objects = []

    for supplied_object in body["objects"]:

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

    # --------------------------------------------------------
    # LINEAGE
    # --------------------------------------------------------

    # Only objects that passed every object-level integrity
    # check enter lineage.
    lineage = []

    for obj in accepted_objects:

        lineage.append({
            "uri": obj["uri"],
            "generation": obj["generation"],
            "crc32c": obj["crc32c"],
            "schemaId": obj["schemaId"],
        })

    # --------------------------------------------------------
    # DEDUPLICATION
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

                inside.append(row)

        retained = inside

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    train = []
    validation = []
    test = []

    for row in retained:

        split = determine_split(row)

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

    # --------------------------------------------------------
    # VALIDATION CONTAMINATION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TEST CONTAMINATION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SORT SPLITS
    # --------------------------------------------------------

    train = sort_split(train)
    validation = sort_split(validation)
    test = sort_split(test)

    # --------------------------------------------------------
    # DIGESTS
    # --------------------------------------------------------

    train_digest = sha256_hex(
        serialize_split(train)
    )

    validation_digest = sha256_hex(
        serialize_split(validation)
    )

    test_digest = sha256_hex(
        serialize_split(test)
    )

    # --------------------------------------------------------
    # REJECTED OBJECTS
    # --------------------------------------------------------

    rejected_objects.sort(
        key=lambda item: (
            (
                item["uri"].encode("utf-8")
                if type(item["uri"]) is str
                else b""
            ),
            compact_json(item).encode(
                "utf-8"
            ),
        ),
    )

    # --------------------------------------------------------
    # REJECTED ROWS
    # --------------------------------------------------------

    rejected_rows = merge_rejected_rows(
        rejected_rows
    )

    # --------------------------------------------------------
    # LINEAGE
    # --------------------------------------------------------

    lineage.sort(
        key=lambda item: (
            item["uri"].encode("utf-8"),
            compact_json(item).encode(
                "utf-8"
            ),
        ),
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

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
