import hashlib
import json
import re
import unicodedata
import zlib
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel


app = FastAPI()


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

TIME_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})$"
)

GENERATION_RE = re.compile(r"^[0-9]+$")
CRC_RE = re.compile(r"^[0-9a-f]{8}$")
URI_RE = re.compile(r"^gs://[^/\s]+/[^/\s]+$")


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def utf8(value: str) -> bytes:
    return value.encode("utf-8")


def utf8_key(value: str) -> bytes:
    return utf8(value)


def canonical_text(value: str) -> str:
    """
    Unicode NFKC, lowercase, trim, and collapse
    Unicode whitespace to one ASCII space.
    """
    value = unicodedata.normalize("NFKC", value)
    value = value.lower()

    parts = value.split()
    return " ".join(parts)


def parse_time(value: Any):
    if not isinstance(value, str):
        return None

    match = TIME_RE.fullmatch(value)
    if not match:
        return None

    year, month, day, hour, minute, second, fraction, offset = match.groups()

    year = int(year)
    month = int(month)
    day = int(day)
    hour = int(hour)
    minute = int(minute)
    second = int(second)

    # Validate offset
    if offset == "Z":
        tz = timezone.utc
    else:
        sign = 1 if offset[0] == "+" else -1
        off_hour = int(offset[1:3])
        off_minute = int(offset[4:6])

        if off_hour > 14:
            return None

        if off_hour == 14 and off_minute != 0:
            return None

        from datetime import timedelta

        tz = timezone(
            sign * timedelta(
                hours=off_hour,
                minutes=off_minute,
            )
        )

    # Fraction -> milliseconds
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

    dt = dt.astimezone(timezone.utc)

    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + \
        f"{dt.microsecond // 1000:03d}Z"


def valid_safe_revision(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        and value <= 9007199254740991
    )


def crc32c(data: bytes) -> int:
    """
    CRC32C (Castagnoli).

    Python's zlib.crc32 is NOT CRC32C, so we implement
    the Castagnoli polynomial directly.
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


def word_set(value: str) -> set[str]:
    """
    Lowercase Unicode letter/number word-set.

    A word is a maximal sequence of Unicode letters/numbers.
    """
    value = value.lower()

    words = []
    current = []

    for char in value:
        category = unicodedata.category(char)

        if category[0] in ("L", "N"):
            current.append(char)
        else:
            if current:
                words.append("".join(current))
                current = []

    if current:
        words.append("".join(current))

    return set(words)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bucket_for_entity(entity: str) -> int:
    digest = hashlib.sha256(utf8(entity)).digest()
    return digest[0] % 10


def row_json(row: dict) -> str:
    """
    Exact required key order.
    """
    ordered = {
        "id": row["id"],
        "entity": row["entity"],
        "eventTime": row["eventTime"],
        "revision": row["revision"],
        "text": row["text"],
    }

    return compact_json(ordered)


def add_reason(reasons: list[str], reason: str):
    if reason not in reasons:
        reasons.append(reason)


def sort_reason_codes(reasons: list[str]) -> list[str]:
    return sorted(set(reasons), key=utf8_key)


# ------------------------------------------------------------
# Input validation
# ------------------------------------------------------------

class BuildRequest(BaseModel):
    policy: dict | None = None
    objects: list | None = None


# ------------------------------------------------------------
# Main endpoint
# ------------------------------------------------------------

@app.post("/build-corpus")
async def build_corpus(request: BuildRequest):

    # Exact INVALID_INPUT response
    if not isinstance(request.policy, dict) or not isinstance(
        request.objects, list
    ):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    policy = request.policy

    min_time = parse_time(policy.get("minTime"))
    max_time = parse_time(policy.get("maxTime"))

    threshold = policy.get("contaminationThreshold")

    policy_valid = (
        min_time is not None
        and max_time is not None
        and isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and threshold >= 0
        and threshold <= 1
        and threshold != float("inf")
        and threshold != float("-inf")
    )

    if policy_valid and min_time > max_time:
        policy_valid = False

    accepted_objects = []
    rejected_objects = []

    # --------------------------------------------------------
    # Validate each object
    # --------------------------------------------------------

    for obj in request.objects:

        reasons = []

        uri = obj.get("uri") if isinstance(obj, dict) else None

        # URI
        if not isinstance(uri, str) or not URI_RE.fullmatch(uri):
            add_reason(reasons, "URI_INVALID")

        # Generations
        generation = obj.get("generation") if isinstance(obj, dict) else None
        fetched_generation = (
            obj.get("fetchedGeneration")
            if isinstance(obj, dict)
            else None
        )

        generation_valid = (
            isinstance(generation, str)
            and GENERATION_RE.fullmatch(generation) is not None
        )

        fetched_generation_valid = (
            isinstance(fetched_generation, str)
            and GENERATION_RE.fullmatch(fetched_generation) is not None
        )

        if not generation_valid or not fetched_generation_valid:
            add_reason(reasons, "GENERATION_INVALID")

        if (
            generation_valid
            and fetched_generation_valid
            and generation != fetched_generation
        ):
            add_reason(reasons, "GENERATION_MISMATCH")

        # CRC
        supplied_crc = obj.get("crc32c") if isinstance(obj, dict) else None

        crc_valid = (
            isinstance(supplied_crc, str)
            and CRC_RE.fullmatch(supplied_crc) is not None
        )

        if not crc_valid:
            add_reason(reasons, "CRC32C_INVALID")

        content = obj.get("content") if isinstance(obj, dict) else None

        if (
            isinstance(content, str)
            and crc_valid
        ):
            actual_crc = crc32c_hex(utf8(content))

            if actual_crc != supplied_crc:
                add_reason(reasons, "CRC32C_MISMATCH")

        # Schema/content
        schema_id = obj.get("schemaId") if isinstance(obj, dict) else None

        if not isinstance(content, str) or schema_id != "training-v1":
            add_reason(reasons, "SCHEMA_INVALID")

        parsed_rows = []

        if isinstance(content, str):
            lines = content.splitlines()

            if not any(line.strip() for line in lines):
                add_reason(reasons, "SCHEMA_INVALID")
            else:
                for line in lines:
                    if not line.strip():
                        continue

                    try:
                        parsed = json.loads(line)
                    except Exception:
                        add_reason(reasons, "JSONL_INVALID")
                        parsed_rows = []
                        break

                    if not isinstance(parsed, dict):
                        add_reason(reasons, "SCHEMA_INVALID")
                        continue

                    expected_keys = {
                        "id",
                        "entity",
                        "eventTime",
                        "revision",
                        "text",
                    }

                    if set(parsed.keys()) != expected_keys:
                        add_reason(reasons, "SCHEMA_INVALID")
                        continue

                    if (
                        not isinstance(parsed["id"], str)
                        or not isinstance(parsed["entity"], str)
                        or not isinstance(parsed["eventTime"], str)
                        or not isinstance(parsed["text"], str)
                        or not valid_safe_revision(parsed["revision"])
                    ):
                        add_reason(reasons, "SCHEMA_INVALID")
                        continue

                    if parse_time(parsed["eventTime"]) is None:
                        add_reason(reasons, "SCHEMA_INVALID")
                        continue

                    parsed_rows.append(parsed)

        if reasons:
            rejected_objects.append({
                "uri": uri if isinstance(uri, str) else None,
                "reasonCodes": sort_reason_codes(reasons),
            })
            continue

        accepted_objects.append({
            "uri": uri,
            "generation": generation,
            "fetchedGeneration": fetched_generation,
            "crc32c": supplied_crc,
            "schemaId": schema_id,
            "rows": parsed_rows,
        })

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    candidates = []

    for obj in accepted_objects:
        for original in obj["rows"]:

            row = {
                "id": original["id"],
                "entity": canonical_text(original["entity"]),
                "eventTime": parse_time(original["eventTime"]),
                "revision": original["revision"],
                "text": canonical_text(original["text"]),
            }

            key = (
                row["entity"],
                row["eventTime"],
                row["text"],
            )

            candidates.append({
                "row": row,
                "key": key,
                "uri": obj["uri"],
                "generation": obj["generation"],
                "crc32c": obj["crc32c"],
                "schemaId": obj["schemaId"],
            })

    grouped = {}

    for candidate in candidates:
        grouped.setdefault(candidate["key"], []).append(candidate)

    retained = []
    rejected_rows = []

    for _, group in grouped.items():

        # Highest revision first.
        # If tied, UTF-8-byte-smallest ID wins.
        winner = sorted(
            group,
            key=lambda x: (
                -x["row"]["revision"],
                utf8_key(x["row"]["id"]),
            ),
        )[0]

        retained.append(winner)

        for loser in group:
            if loser is winner:
                continue

            rejected_rows.append({
                "id": loser["row"]["id"],
                "reasonCodes": ["DUPLICATE"],
            })

    # --------------------------------------------------------
    # Policy and window
    # --------------------------------------------------------

    if not policy_valid:
        for item in retained:
            rejected_rows.append({
                "id": item["row"]["id"],
                "reasonCodes": ["POLICY_INVALID"],
            })

        retained = []

    else:
        kept_after_window = []

        for item in retained:
            event_time = item["row"]["eventTime"]

            if event_time < min_time or event_time > max_time:
                rejected_rows.append({
                    "id": item["row"]["id"],
                    "reasonCodes": ["OUT_OF_WINDOW"],
                })
            else:
                kept_after_window.append(item)

        retained = kept_after_window

    # --------------------------------------------------------
    # Split into train / validation / test
    # --------------------------------------------------------

    train = []
    validation = []
    test = []

    for item in retained:

        bucket = bucket_for_entity(item["row"]["entity"])

        if bucket <= 5:
            item["split"] = "train"
            train.append(item)

        elif bucket <= 7:
            item["split"] = "validation"
            validation.append(item)

        else:
            item["split"] = "test"
            test.append(item)

    # --------------------------------------------------------
    # Train contamination
    # --------------------------------------------------------

    train_word_sets = [
        word_set(item["row"]["text"])
        for item in train
    ]

    def contamination(item):
        target_words = word_set(item["row"]["text"])

        for train_words in train_word_sets:
            if jaccard(target_words, train_words) >= threshold:
                return True

        return False

    final_validation = []

    for item in validation:
        if contamination(item):
            rejected_rows.append({
                "id": item["row"]["id"],
                "reasonCodes": ["TRAIN_CONTAMINATION"],
            })
        else:
            final_validation.append(item)

    final_test = []

    for item in test:
        if contamination(item):
            rejected_rows.append({
                "id": item["row"]["id"],
                "reasonCodes": ["TRAIN_CONTAMINATION"],
            })
        else:
            final_test.append(item)

    validation = final_validation
    test = final_test

    # --------------------------------------------------------
    # Sorting
    # --------------------------------------------------------

    def sort_items(items):
        return sorted(
            items,
            key=lambda x: (
                utf8_key(x["row"]["id"]),
                utf8_key(row_json(x["row"])),
            ),
        )

    train = sort_items(train)
    validation = sort_items(validation)
    test = sort_items(test)

    # --------------------------------------------------------
    # Artifact serialization + digest
    # --------------------------------------------------------

    def serialize_split(items):
        output = b""

        for item in items:
            output += utf8(row_json(item["row"]))
            output += b"\n"

        return output

    train_bytes = serialize_split(train)
    validation_bytes = serialize_split(validation)
    test_bytes = serialize_split(test)

    digests = {
        "train": sha256_hex(train_bytes),
        "validation": sha256_hex(validation_bytes),
        "test": sha256_hex(test_bytes),
    }

    # --------------------------------------------------------
    # Rejected rows
    # --------------------------------------------------------

    # Merge reason codes for the same ID.
    merged_rejections = {}

    for rejection in rejected_rows:
        rid = rejection["id"]

        if rid not in merged_rejections:
            merged_rejections[rid] = []

        merged_rejections[rid].extend(
            rejection["reasonCodes"]
        )

    rejected_rows_final = []

    for rid, reasons in merged_rejections.items():
        rejected_rows_final.append({
            "id": rid,
            "reasonCodes": sort_reason_codes(reasons),
        })

    rejected_rows_final.sort(
        key=lambda x: (
            utf8_key(x["id"]),
            utf8_key(compact_json(x)),
        )
    )

    # --------------------------------------------------------
    # Rejected objects
    # --------------------------------------------------------

    rejected_objects.sort(
        key=lambda x: (
            utf8_key(x["uri"]) if isinstance(x["uri"], str) else b"",
            utf8_key(compact_json(x)),
        )
    )

    # --------------------------------------------------------
    # Lineage
    # --------------------------------------------------------

    lineage = []

    for obj in accepted_objects:
        lineage.append({
            "uri": obj["uri"],
            "generation": obj["generation"],
            "crc32c": obj["crc32c"],
            "schemaId": obj["schemaId"],
        })

    lineage.sort(
        key=lambda x: (
            utf8_key(x["uri"]),
            utf8_key(compact_json(x)),
        )
    )

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {
        "splits": {
            "train": [item["row"] for item in train],
            "validation": [item["row"] for item in validation],
            "test": [item["row"] for item in test],
        },
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows_final,
        "digests": digests,
        "lineage": lineage,
    }


# ------------------------------------------------------------
# Render health check
# ------------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}
