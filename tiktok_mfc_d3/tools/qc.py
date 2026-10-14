from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .common import iter_jsonl

SEMANTIC_VERDICTS = {"SUPPORTED", "REFUTED", "NEI"}


def run_qc(dataset_root: Path) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    claims = list(iter_jsonl(dataset_root / "records" / "claims.jsonl"))
    evidence = list(iter_jsonl(dataset_root / "records" / "evidence.jsonl"))
    verifications = list(iter_jsonl(dataset_root / "records" / "verifications.jsonl"))
    explanations = list(iter_jsonl(dataset_root / "records" / "explanations.jsonl"))
    multimedia = list(iter_jsonl(dataset_root / "records" / "multimedia.jsonl"))

    claim_ids = {c.get("claim_id") for c in claims}
    evidence_by_id = {e.get("evidence_id"): e for e in evidence}


    # Distinguish a real no-claim sample from technical claim-extraction failure.
    latest_multimedia_by_video: dict[str, dict[str, Any]] = {}
    for record in multimedia:
        video_id = str(record.get("video_id") or "")
        if video_id:
            latest_multimedia_by_video[video_id] = record

    claim_count_by_video = Counter(str(c.get("video_id") or "") for c in claims)
    for video_id, record in latest_multimedia_by_video.items():
        extraction = record.get("claim_extraction") or {}
        completed = extraction.get("completed")
        no_claim_detected = extraction.get("no_claim_detected")
        extraction_status = str(extraction.get("status") or "").upper()

        if completed is False and no_claim_detected is not None:
            errors.append({
                "code": "TECHNICAL_FAILURE_MISLABELED_AS_NO_CLAIM",
                "video_id": video_id,
                "claim_extraction_status": extraction_status,
            })
        elif completed is False and claim_count_by_video.get(video_id, 0) == 0:
            warnings.append({
                "code": "CLAIM_EXTRACTION_INCOMPLETE",
                "video_id": video_id,
                "claim_extraction_status": extraction_status,
                "message": "Video source is registered but claim extraction did not complete; do not use it as a negative/no-claim sample.",
            })

    for v in verifications:
        claim_id = v.get("claim_id")
        if claim_id not in claim_ids:
            errors.append({"code": "ORPHAN_VERIFICATION", "claim_id": claim_id})

        gold_verdict = v.get("gold_verdict")
        status = v.get("annotation_status")
        gold_eids = v.get("gold_evidence_ids") or []

        if gold_verdict is not None and gold_verdict not in SEMANTIC_VERDICTS:
            errors.append({"code": "INVALID_GOLD_VERDICT", "claim_id": claim_id, "value": gold_verdict})
        if gold_verdict in {"SUPPORTED", "REFUTED"} and not gold_eids:
            errors.append({"code": "GOLD_VERDICT_WITHOUT_EVIDENCE", "claim_id": claim_id})
        if status in {"FAILED", "INSUFFICIENT_COLLECTION"} and gold_verdict is not None:
            errors.append({"code": "FAILURE_HAS_GOLD_VERDICT", "claim_id": claim_id})
        if v.get("system", {}).get("contract_violations"):
            warnings.append({
                "code": "BACKEND_CONTRACT_VIOLATION",
                "claim_id": claim_id,
                "details": v["system"]["contract_violations"],
            })
        for eid in gold_eids:
            ev = evidence_by_id.get(eid)
            if ev is None:
                errors.append({"code": "MISSING_GOLD_EVIDENCE", "claim_id": claim_id, "evidence_id": eid})
            elif not (ev.get("annotation") or {}).get("is_gold"):
                errors.append({"code": "NON_GOLD_EVIDENCE_REFERENCED", "claim_id": claim_id, "evidence_id": eid})

    for ex in explanations:
        gold = ex.get("gold") or {}
        if gold.get("text") and not gold.get("evidence_ids"):
            errors.append({"code": "EXPLANATION_WITHOUT_EVIDENCE", "claim_id": ex.get("claim_id")})
        quality = ex.get("quality") or {}
        if quality.get("review_status") == "APPROVED" and quality.get("contains_unsupported_claim") is True:
            errors.append({"code": "APPROVED_EXPLANATION_UNSUPPORTED", "claim_id": ex.get("claim_id")})

    verdict_dist = Counter(v.get("gold_verdict") for v in verifications if v.get("gold_verdict"))
    claim_type_dist = Counter((c.get("annotation") or {}).get("claim_type") for c in claims)

    return {
        "summary": {
            "claims": len(claims),
            "evidence": len(evidence),
            "verifications": len(verifications),
            "explanations": len(explanations),
            "multimedia_runs": len(multimedia),
            "errors": len(errors),
            "warnings": len(warnings),
            "gold_verdict_distribution": dict(verdict_dist),
            "claim_type_distribution": dict(claim_type_dist),
        },
        "errors": errors,
        "warnings": warnings,
    }
