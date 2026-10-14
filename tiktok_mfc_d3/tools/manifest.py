from __future__ import annotations

from pathlib import Path
from .common import iter_jsonl, read_json, write_json


def refresh_manifest(dataset_root: Path) -> dict:
    manifest_path = dataset_root / "manifest.json"
    manifest = read_json(manifest_path)

    videos = list(iter_jsonl(dataset_root / "records" / "videos.jsonl"))
    claims = list(iter_jsonl(dataset_root / "records" / "claims.jsonl"))
    evidence = list(iter_jsonl(dataset_root / "records" / "evidence.jsonl"))
    verifications = list(iter_jsonl(dataset_root / "records" / "verifications.jsonl"))

    gold_claim_ids = {
        v.get("claim_id") for v in verifications
        if v.get("gold_verdict") in {"SUPPORTED", "REFUTED", "NEI"}
    }
    manifest["statistics"] = {
        "videos": len(videos),
        "claims": len(claims),
        "gold_claims": len(gold_claim_ids),
        "text_evidence": sum(1 for e in evidence if e.get("type") == "text"),
        "image_evidence": sum(1 for e in evidence if e.get("type") == "image"),
    }
    write_json(manifest_path, manifest)
    return manifest
