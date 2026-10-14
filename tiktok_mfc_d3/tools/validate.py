from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .common import iter_jsonl, read_json

SCHEMA_TO_RECORD = {
    "video.schema.json": "videos.jsonl",
    "claim.schema.json": "claims.jsonl",
    "evidence.schema.json": "evidence.jsonl",
    "verification.schema.json": "verifications.jsonl",
    "explanation.schema.json": "explanations.jsonl",
}


def validate_dataset(dataset_root: Path) -> dict[str, Any]:
    errors = []
    for schema_file, record_file in SCHEMA_TO_RECORD.items():
        schema = read_json(dataset_root / "schemas" / schema_file)
        validator = Draft202012Validator(schema)
        for idx, record in enumerate(iter_jsonl(dataset_root / "records" / record_file), start=1):
            for error in validator.iter_errors(record):
                errors.append({
                    "record_file": record_file,
                    "line": idx,
                    "path": list(error.absolute_path),
                    "message": error.message,
                })
    return {"valid": not errors, "errors": errors}
