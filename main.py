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

# Generation = decimal string, one or more digits.
GENERATION_RE = re.compile(
    r"^[0-9]+$"
)

# CRC32C = exactly 8 lowercase hexadecimal characters.
CRC32C_RE = re.compile(
    r"^[0-9a-f]{8}$"
)

# The assignment specifies gs://bucket/object:
# - gs:// is required
# - bucket must be non-empty
# - / is required
# - object must be non-empty
#
# We intentionally do NOT add Google's additional bucket-name
# restrictions because the assignment only specifies this pattern.
URI_RE = re.compile(
    r"^gs://[^/]+/.+$"
)

# Required timestamp grammar:
#
# YYYY-MM-DDTHH:mm:ss
# optionally .s, .ss, or .sss
# followed by Z or +/-HH:mm
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
# BASIC HELPERS
# ============================================================

def compact_json(value):
    """
    Compact JSON:
    - no spaces
    - non-ASCII emitted directly
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def utf8(value):
    return value.encode("utf-8")


def sort_reason_codes(codes):
    """
    Deduplicate and sort reason codes by UTF-8 bytes.
    """
    return sorted(
        set(codes),
        key=lambda x: x.encode("utf-8"),
    )


# ============================================================
# TIMESTAMP VALIDATION
# ============================================================

def parse_timestamp(value):
    """
    Validate an assignment timestamp and return
    a UTC datetime.

    Returns None when invalid.
    """

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

    # --------------------------------------------------------
    # TIMEZONE VALIDATION
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

        # Offset magnitude <= 14:00.
        if offset_hour > 14:
            return None

        if offset_minute > 59:
            return None

        # If hour is 14, minutes must be 00.
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

        # .1   -> 100 ms
        # .12  -> 120 ms
        # .123 -> 123 ms
        milliseconds = int(
            fraction.ljust(3, "0")
        )

    # datetime catches invalid calendar dates,
    # such as February 30.
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
    """
    Convert a valid timestamp to:

    YYYY-MM-DDTHH:mm:ss.sssZ
    """

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
# TEXT CANONICALIZATION
# ============================================================

def canonical_text(value):
    """
    NFKC
    -> lowercase
    -> trim
    -> collapse Unicode whitespace
       to one ASCII space
    """

    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    value = value.lower()

    # Python's split() recognizes Unicode whitespace.
    # Joining with " " gives exactly one ASCII space
    # between remaining pieces and trims the ends.
    return " ".join(
        value.split()
    )


# ============================================================
# REVISION
# ============================================================

def valid_revision(value):
    """
    Revision must be:
    - an actual JSON integer
    - >= 0
    - <= JavaScript's safe integer limit
    """

    return (
        type(value) is int
        and value >= 0
        and value <= MAX_SAFE_INTEGER
    )


# ============================================================
# CRC32C - CASTAGNOLI
# ============================================================

def crc32c(data):
    """
    CRC32C using the Castagnoli polynomial.
    """

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


# Standard CRC32C test vector.
assert (
    crc32c_hex(b"123456789")
    == "e3069283"
)


# ============================================================
# SHA-256
# ============================================================

def sha256_hex(data):
    return hashlib.sha256(
        data
    ).hexdigest()


# ============================================================
# ROW VALIDATION
# ============================================================

def valid_row_shape(row):
    """
    A valid parsed JSONL row must:
    - be an object
    - have exactly the five required keys
    - have four string fields
    - have a safe non-negative integer revision
    - have a valid eventTime
    """

    if not isinstance(
        row,
        dict,
    ):
        return False

    if set(row.keys()) != ROW_KEYS:
        return False

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
# JSONL PARSING
# ============================================================

def parse_jsonl(content):
    """
    Returns:

        (
            parsed_valid_rows,
            has_jsonl_invalid,
            has_schema_invalid
        )

    Important distinction:

    JSON cannot be parsed
        -> JSONL_INVALID

    JSON parses successfully but isn't
    the required row shape
        -> SCHEMA_INVALID
    """

    lines = content.split("\n")

    non_blank = []

    for line in lines:

        # Allow normal CRLF files.
        if line.endswith("\r"):
            line = line[:-1]

        # Blank lines are ignored.
        #
        # Using strip here only decides whether the line
        # is blank. It does not modify a real JSON line.
        if line.strip() == "":
            continue

        non_blank.append(line)

    # File must contain at least one row.
    if not non_blank:

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

            parsed = json.loads(
                line
            )

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):

            jsonl_invalid = True

            continue

        # Parsed JSON, but wrong row schema.
        if not valid_row_shape(
            parsed
        ):

            schema_invalid = True

            continue

        rows.append(
            parsed
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
    """
    Validate all object-level requirements.

    We intentionally collect ALL independently applicable
    object reason codes before deciding whether the object
    is accepted.
    """

    reasons = []

    # --------------------------------------------------------
    # NON-OBJECT
    # --------------------------------------------------------

    if not isinstance(
        obj,
        dict,
    ):

        return {
            "accepted": False,
            "rejection": {
                "uri": None,
                "reasonCodes": sort_reason_codes([
                    "URI_INVALID",
                    "GENERATION_INVALID",
                    "CRC32C_INVALID",
                    "SCHEMA_INVALID",
                ]),
            },
        }

    # --------------------------------------------------------
    # URI
    # --------------------------------------------------------

    uri = obj.get(
        "uri"
    )

    uri_valid = (
        isinstance(uri, str)
        and URI_RE.fullmatch(
            uri
        ) is not None
    )

    if not uri_valid:

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

    # Each invalid generation independently contributes
    # GENERATION_INVALID, but the code itself is deduplicated
    # in the final reason array.
    if not generation_valid:

        reasons.append(
            "GENERATION_INVALID"
        )

    if not fetched_generation_valid:

        reasons.append(
            "GENERATION_INVALID"
        )

    # Mismatch is ONLY checked when both are valid
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

    content_is_string = isinstance(
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

    # The assignment explicitly says:
    #
    # CRC32C_MISMATCH is checked only for:
    # - string content
    # - syntactically valid CRC
    #
    if (
        content_is_string
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

    # Non-string content is explicitly SCHEMA_INVALID.
    if not content_is_string:

        reasons.append(
            "SCHEMA_INVALID"
        )

    # --------------------------------------------------------
    # JSONL
    # --------------------------------------------------------

    parsed_rows = []

    if content_is_string:

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

    # --------------------------------------------------------
    # FINAL OBJECT REASONS
    # --------------------------------------------------------

    reasons = sort_reason_codes(
        reasons
    )

    # --------------------------------------------------------
    # OBJECT REJECTED
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
    # OBJECT ACCEPTED
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
# ROW CANONICALIZATION
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

        for original_row in obj["rows"]:

            row = canonicalize_row(
                original_row
            )

            # Exact JSON tuple conceptually:
            #
            # [entity, eventTime, text]
            #
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

        # Highest revision wins.
        #
        # If revisions tie, smallest UTF-8 ID wins.
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

    return (
        retained,
        rejected,
    )


# ============================================================
# SPLIT ASSIGNMENT
# ============================================================

def determine_split(row):

    digest = hashlib.sha256(
        row["entity"].encode("utf-8")
    ).digest()

    first_byte = digest[0]

    bucket = first_byte % 10

    if bucket <= 5:

        return "train"

    if bucket <= 7:

        return "validation"

    return "test"


# ============================================================
# WORD SET / JACCARD
# ============================================================

def unicode_letter_number_word_set(
    value
):
    """
    Lowercase Unicode text.

    A word is a maximal sequence of Unicode
    letters and/or Unicode numbers.
    """

    value = value.lower()

    words = set()
    current = []

    for char in value:

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

                words.add(
                    "".join(current)
                )

                current = []

    if current:

        words.add(
            "".join(current)
        )

    return words


def jaccard_similarity(a, b):

    # Assignment explicitly says empty/empty = 1.
    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


# ============================================================
# ROW JSON
# ============================================================

def row_json(row):

    # IMPORTANT:
    #
    # The assignment requires this exact key order:
    #
    # id, entity, eventTime, revision, text
    #
    output = {
        "id": row["id"],
        "entity": row["entity"],
        "eventTime": row["eventTime"],
        "revision": row["revision"],
        "text": row["text"],
    }

    return compact_json(
        output
    )


# ============================================================
# SPLIT SORTING
# ============================================================

def sort_split(rows):

    return sorted(
        rows,
        key=lambda row: (
            row["id"].encode("utf-8"),
            row_json(row).encode("utf-8"),
        ),
    )


# ============================================================
# SPLIT SERIALIZATION
# ============================================================

def serialize_split(rows):

    output = bytearray()

    for row in rows:

        # Compact JSON.
        output.extend(
            row_json(row).encode(
                "utf-8"
            )
        )

        # Exactly one newline.
        output.extend(
            b"\n"
        )

    return bytes(output)


# ============================================================
# POLICY VALIDATION
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

    # Must be a finite JSON number in [0, 1].
    #
    # bool is deliberately excluded because Python's bool
    # is technically an int.
    threshold_valid = (
        (
            type(threshold) is int
            or type(threshold) is float
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

    # A backwards window is invalid.
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
# REJECTED ROW MERGING
# ============================================================

def merge_rejected_rows(
    rejected_rows
):
    """
    If the same ID receives multiple rejection reasons,
    combine them into one response entry.

    Reason codes are then deduplicated and UTF-8 sorted.
    """

    by_id = {}

    for item in rejected_rows:

        row_id = item["id"]

        if row_id not in by_id:

            by_id[row_id] = []

        by_id[row_id].extend(
            item["reasonCodes"]
        )

    result = []

    for row_id, reasons in by_id.items():

        result.append({
            "id": row_id,
            "reasonCodes": sort_reason_codes(
                reasons
            ),
        })

    return result


# ============================================================
# SORT REJECTED OBJECTS
# ============================================================

def sort_rejected_objects(
    items
):

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


# ============================================================
# SORT REJECTED ROWS
# ============================================================

def sort_rejected_rows(
    items
):

    return sorted(
        items,
        key=lambda item: (
            item["id"].encode("utf-8"),
            compact_json(
                item
            ).encode("utf-8"),
        ),
    )


# ============================================================
# SORT LINEAGE
# ============================================================

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
# REQUEST ERROR
# ============================================================

def invalid_input_response():

    return JSONResponse(
        status_code=400,
        content={
            "error": "INVALID_INPUT"
        },
    )


# ============================================================
# POST /build-corpus
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

        return invalid_input_response()

    # Top-level JSON must be an object.
    if not isinstance(
        body,
        dict,
    ):

        return invalid_input_response()

    # Assignment explicitly specifies that a missing
    # policy is INVALID_INPUT.
    if "policy" not in body:

        return invalid_input_response()

    # objects must exist and be an array.
    if (
        "objects" not in body
        or not isinstance(
            body["objects"],
            list,
        )
    ):

        return invalid_input_response()

    policy = body["policy"]
    objects = body["objects"]

    # ========================================================
    # POLICY
    # ========================================================

    (
        policy_valid,
        min_time,
        max_time,
        contamination_threshold,
    ) = validate_policy(
        policy
    )

    # ========================================================
    # OBJECT VALIDATION
    # ========================================================

    accepted_objects = []
    rejected_objects = []

    for supplied_object in objects:

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
    # ========================================================
    #
    # IMPORTANT:
    #
    # Lineage is created immediately after object validation.
    #
    # Therefore:
    #
    # accepted object + rows later rejected
    #     -> STILL IN LINEAGE
    #
    # invalid/rejected object
    #     -> NOT IN LINEAGE
    #
    # Identity fields are copied exactly from the request.
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

        # Invalid policy rejects EVERY retained row.
        for row in retained:

            rejected_rows.append({
                "id": row["id"],
                "reasonCodes": [
                    "POLICY_INVALID"
                ],
            })

        retained = []

    else:

        in_window = []

        for row in retained:

            event_time = parse_timestamp(
                row["eventTime"]
            )

            # Inclusive window:
            #
            # eventTime < min -> reject
            # eventTime > max -> reject
            #
            # equal to either boundary -> accepted
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

                in_window.append(
                    row
                )

        retained = in_window

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
    # TRAIN CONTAMINATION
    # ========================================================

    train_word_sets = [
        unicode_letter_number_word_set(
            row["text"]
        )
        for row in train
    ]

    def is_contaminated(row):

        target_words = (
            unicode_letter_number_word_set(
                row["text"]
            )
        )

        for train_words in train_word_sets:

            similarity = (
                jaccard_similarity(
                    target_words,
                    train_words,
                )
            )

            if (
                similarity
                >= contamination_threshold
            ):

                return True

        return False

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

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

            clean_validation.append(
                row
            )

    validation = clean_validation

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

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

            clean_test.append(
                row
            )

    test = clean_test

    # ========================================================
    # DETERMINISTIC SPLIT ORDER
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
    # EXACT SPLIT BYTES
    # ========================================================

    train_bytes = serialize_split(
        train
    )

    validation_bytes = serialize_split(
        validation
    )

    test_bytes = serialize_split(
        test
    )

    # ========================================================
    # DIGESTS
    # ========================================================

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

    # ========================================================
    # REJECTED ROWS
    # ========================================================

    rejected_rows = merge_rejected_rows(
        rejected_rows
    )

    rejected_rows = sort_rejected_rows(
        rejected_rows
    )

    # ========================================================
    # REJECTED OBJECTS
    # ========================================================

    rejected_objects = (
        sort_rejected_objects(
            rejected_objects
        )
    )

    # ========================================================
    # LINEAGE SORT
    # ========================================================

    lineage = sort_lineage(
        lineage
    )

    # ========================================================
    # EXACT RESPONSE SHAPE
    # ========================================================

    return {
        "splits": {
            "train": [
                {
                    "id": row["id"],
                    "entity": row["entity"],
                    "eventTime": row["eventTime"],
                    "revision": row["revision"],
                    "text": row["text"],
                }
                for row in train
            ],
            "validation": [
                {
                    "id": row["id"],
                    "entity": row["entity"],
                    "eventTime": row["eventTime"],
                    "revision": row["revision"],
                    "text": row["text"],
                }
                for row in validation
            ],
            "test": [
                {
                    "id": row["id"],
                    "entity": row["entity"],
                    "eventTime": row["eventTime"],
                    "revision": row["revision"],
                    "text": row["text"],
                }
                for row in test
            ],
        },
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows,
        "digests": digests,
        "lineage": lineage,
    }


# ============================================================
# HEALTH ENDPOINTS
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
