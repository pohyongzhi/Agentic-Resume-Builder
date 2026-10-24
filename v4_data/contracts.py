"""Small, dependency-free validator for the checked-in schema vocabulary.

The schema uses only local $ref, type, enum, const, required, properties,
additionalProperties, items, minItems, uniqueItems, minLength, pattern,
minimum and maximum. Unknown schema keywords fail closed.
"""

import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas/v4/records.schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


class DataError(ValueError):
    """An invalid or unsafe dataset operation."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    raw = value if isinstance(value, str) else canonical(value)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate(record, definition):
    """Validate a named record using the same schema shipped to consumers."""
    def check(value, rule, path):
        allowed = {"$ref", "type", "enum", "const", "required", "properties",
                   "additionalProperties", "items", "minItems", "uniqueItems",
                   "minLength", "pattern", "minimum", "maximum", "description"}
        if set(rule) - allowed:
            raise DataError(f"Unsupported schema keywords at {path}")
        if "$ref" in rule:
            check(value, SCHEMA["$defs"][rule["$ref"].removeprefix("#/$defs/")], path)
            return
        types = {"object": lambda v: isinstance(v, dict),
                 "array": lambda v: isinstance(v, list),
                 "string": lambda v: isinstance(v, str),
                 "integer": lambda v: type(v) is int,
                 "number": lambda v: type(v) in (int, float) and math.isfinite(v),
                 "boolean": lambda v: type(v) is bool,
                 "null": lambda v: v is None}
        if "type" in rule and not types[rule["type"]](value):
            raise DataError(f"{path}: expected {rule['type']}")
        if "const" in rule and value != rule["const"]:
            raise DataError(f"{path}: invalid constant")
        if "enum" in rule and value not in rule["enum"]:
            raise DataError(f"{path}: invalid enum")
        if isinstance(value, dict):
            if set(rule.get("required", [])) - value.keys():
                raise DataError(f"{path}: missing required fields")
            props = rule.get("properties", {})
            if rule.get("additionalProperties") is False and value.keys() - props.keys():
                raise DataError(f"{path}: unknown fields")
            for key, item in value.items():
                if key in props:
                    check(item, props[key], f"{path}.{key}")
        if isinstance(value, list):
            if len(value) < rule.get("minItems", 0):
                raise DataError(f"{path}: too few items")
            if rule.get("uniqueItems") and len({canonical(v) for v in value}) != len(value):
                raise DataError(f"{path}: duplicate items")
            for i, item in enumerate(value):
                check(item, rule.get("items", {}), f"{path}[{i}]")
        if isinstance(value, str):
            if len(value) < rule.get("minLength", 0):
                raise DataError(f"{path}: empty string")
            if "pattern" in rule and not re.search(rule["pattern"], value):
                raise DataError(f"{path}: invalid string format")
        if type(value) in (int, float):
            if not math.isfinite(value):
                raise DataError(f"{path}: non-finite number")
            if "minimum" in rule and value < rule["minimum"]:
                raise DataError(f"{path}: below minimum")
            if "maximum" in rule and value > rule["maximum"]:
                raise DataError(f"{path}: above maximum")
    check(record, SCHEMA["$defs"][definition], definition)
    return record


def read_jsonl(path, definition):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise DataError(f"{path}:{number}: blank JSONL record")
            try:
                rows.append(validate(json.loads(line), definition))
            except (ValueError, TypeError) as exc:
                raise DataError(f"{path}:{number}: {exc}") from exc
    return rows


def write_jsonl(path, rows, definition):
    """Never overwrite an earlier run, and validate before opening output."""
    lines = [canonical(validate(row, definition)) + "\n" for row in rows]
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.writelines(lines)


def index_unique(rows, key="id"):
    result = {}
    for row in rows:
        if row[key] in result:
            raise DataError(f"Duplicate {key}: {row[key]}")
        result[row[key]] = row
    return result
