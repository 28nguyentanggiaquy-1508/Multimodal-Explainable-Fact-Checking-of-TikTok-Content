from __future__ import annotations

from copy import deepcopy
from typing import Any

SEMANTIC_VERDICTS = {"SUPPORTED", "REFUTED", "NEI"}
FAILURE_MARKERS = (
    "no usable text or image evidence",
    "không có bằng chứng truy xuất đủ",
    "evidence retrieval returned no usable",
    "evidence retrieval không trả về",
    "timeout",
    "timed out",
    "unavailable",
    "api error",
    "failed",
)
NON_SUCCESS_STATUSES = {"FAILED", "UNAVAILABLE", "SKIPPED", "INSUFFICIENT_COLLECTION"}


def _contains_failure_marker(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        s = value.casefold()
        return any(m.casefold() in s for m in FAILURE_MARKERS)
    if isinstance(value, list):
        return any(_contains_failure_marker(x) for x in value)
    if isinstance(value, dict):
        return any(_contains_failure_marker(v) for v in value.values())
    return False


def detect_verdict_contract_violation(claim_item: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    evidence = claim_item.get("evidence") or []
    verification = claim_item.get("verification") or {}
    verdict = str(verification.get("verdict") or "").upper()
    status = str(verification.get("status") or verification.get("system_status") or "").upper()

    no_evidence = len(evidence) == 0 and len(verification.get("evidence_ids") or []) == 0
    if verdict == "NEI" and no_evidence and (_contains_failure_marker(verification) or status in NON_SUCCESS_STATUSES):
        errors.append(
            "SYSTEM_FAILURE_AS_NEI: backend returned NEI although evidence retrieval/verification failed or produced no usable evidence."
        )
    if verdict in SEMANTIC_VERDICTS and status in NON_SUCCESS_STATUSES:
        errors.append(
            f"NON_SUCCESS_WITH_VERDICT: verification status={status} must not emit semantic verdict={verdict}."
        )
    return errors


def normalize_system_response(response: dict[str, Any]) -> dict[str, Any]:
    """Protect dataset ingestion from technical failure being converted into a semantic verdict.

    It never changes Gold annotations because backend outputs are SYSTEM data only.
    """
    out = deepcopy(response)
    for item in out.get("claims") or []:
        violations = detect_verdict_contract_violation(item)
        if not violations:
            continue
        verification = item.setdefault("verification", {})
        verification["raw_verdict"] = verification.get("verdict")
        verification["verdict"] = None
        verification["system_status"] = verification.get("status") or "INSUFFICIENT_COLLECTION"
        verification["contract_violations"] = violations
        verification["confidence"] = None
    return out

