"""Runtime-only verdict safety guard for TikTok fact-checking.

No Gold data, no model training, no online calls. Unverified evidence-collection
completeness is NEVER converted into NEI. This is a conservative engineering
safety check, not a truthfulness guarantee for supported/refuted predictions.
"""
from __future__ import annotations

from copy import deepcopy
from urllib.parse import urlsplit

FINAL_VERDICTS = {"SUPPORTED", "REFUTED", "NEI"}


def _external_url(url: object) -> bool:
    """Only public-looking HTTP(S) source URLs outside TikTok can support verdicts.

    This is an evidence *origin* filter, NOT a guarantee of publisher quality,
    factual correctness, DNS safety, or independence among publishers.
    """
    if not isinstance(url, str):
        return False
    try:
        p = urlsplit(url.strip())
        host = (p.hostname or "").casefold().rstrip(".")
        if p.scheme not in {"http", "https"} or not host or p.username or p.password:
            return False
        if host == "tiktok.com" or host.endswith(".tiktok.com"):
            return False
        if host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}:
            return False
        return "." in host
    except (ValueError, TypeError):
        return False


def _independent_evidence_ids(row: dict) -> set[str]:
    ids: set[str] = set()
    for raw in row.get("evidence") or []:
        if not isinstance(raw, dict):
            continue
        source_id = str(raw.get("id") or "").strip()
        if source_id and _external_url(raw.get("source_url")):
            ids.add(source_id)
    return ids


def _collection_complete(retrieval: dict) -> bool:
    """Require explicitly audited collection completion, not mere search success.

    Legacy v2.3.7 retrieval has no `collection_complete` provenance field.
    Therefore its NEI must remain unverified by default. A later version must
    define and audit completion criteria before setting this flag to True.
    """
    if retrieval.get("collection_complete") is not True:
        return False
    if str(retrieval.get("status") or "").upper() != "SUCCESS":
        return False
    if retrieval.get("errors"):
        return False
    for channel in ("text", "image"):
        info = retrieval.get(channel)
        if not isinstance(info, dict) or str(info.get("status") or "").upper() != "SUCCESS":
            return False
        if info.get("errors"):
            return False
    return True


def _mark_unavailable(row: dict, verification: dict, reason: str) -> None:
    verification["status"] = "UNAVAILABLE"
    verification["verdict"] = None
    verification["confidence"] = None
    verification["evidence_coverage"] = None
    verification["evidence_ids"] = []
    verification["evidence_assessments"] = []
    verification["rationale"] = "Chưa đủ điều kiện đưa ra kết luận kiểm chứng đáng tin cậy."
    verification["uncertainty"] = reason
    limits = verification.get("limitations")
    verification["limitations"] = (limits if isinstance(limits, list) else []) + [reason]
    row["explanation"] = None
    row["runtime_decision"] = "WITHHELD"
    row["runtime_limit_reason"] = reason


def enforce_runtime_verdict_policy(response: dict) -> dict:
    """Apply safety checks to one **runtime** response, with no Gold writes.

    Does not change Phase-2 observations, source claims, or stored human labels.
    Returned dict is a deep copy. May WITHHOLD an AI verdict but never invents
    a new substantive SUPPORTED/REFUTED/NEI label.
    """
    if not isinstance(response, dict):
        raise TypeError("Fact-check pipeline result must be a dict")
    result = deepcopy(response)
    rows = result.get("claims") or []
    withheld = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        verification = row.get("verification")
        if not isinstance(verification, dict):
            continue
        status = str(verification.get("status") or "").upper()
        verdict = str(verification.get("verdict") or "").upper()
        retrieval = row.get("retrieval") if isinstance(row.get("retrieval"), dict) else {}
        citations = verification.get("evidence_ids") or []
        if not isinstance(citations, list):
            citations = []
        citation_ids = [str(v).strip() for v in citations if str(v).strip()]
        available_ids = _independent_evidence_ids(row)
        reason = ""

        if status != "SUCCESS":
            if verdict in FINAL_VERDICTS:
                reason = "TECHNICAL_STATUS_CANNOT_HAVE_VERDICT"
            else:
                row["explanation"] = None
                row["runtime_decision"] = "NOT_VERIFIED"
                continue
        elif verdict == "NEI":
            if not _collection_complete(retrieval):
                reason = "EVIDENCE_COLLECTION_COMPLETENESS_NOT_CERTIFIED"
        elif verdict in {"SUPPORTED", "REFUTED"}:
            if not citation_ids:
                reason = "MISSING_EVIDENCE_CITATION"
            elif not set(citation_ids).issubset(available_ids):
                reason = "CITATION_NOT_IN_RETRIEVED_EXTERNAL_EVIDENCE"
            elif str(retrieval.get("status") or "").upper() in {"FAILED", "UNAVAILABLE"}:
                reason = "RETRIEVAL_TECHNICAL_FAILURE"
        else:
            reason = "INVALID_OR_MISSING_VERDICT"

        if reason:
            _mark_unavailable(row, verification, reason)
            withheld += 1
        else:
            row["runtime_decision"] = "VERDICT_ELIGIBLE_FOR_DISPLAY_NOT_GOLD"
            row["runtime_limit_reason"] = None

    if withheld:
        reports = result.get("phase_status")
        if isinstance(reports, dict):
            phase4 = reports.get("phase_4")
            if isinstance(phase4, dict):
                phase4["status"] = "DEGRADED"
                detail = phase4.get("details")
                if not isinstance(detail, dict):
                    detail = {}
                    phase4["details"] = detail
                detail["runtime_verdicts_withheld"] = withheld
        if str(result.get("status") or "").upper() == "SUCCESS":
            result["status"] = "DEGRADED"

    checked_rows = [r for r in rows if isinstance(r, dict)]
    result["verdict_ready"] = bool(checked_rows) and all(
        isinstance(r.get("verification"), dict)
        and r["verification"].get("status") == "SUCCESS"
        and r["verification"].get("verdict") in FINAL_VERDICTS
        for r in checked_rows
    )
    result["explanation_ready"] = bool(checked_rows) and all(
        isinstance(r.get("explanation"), dict)
        and r["explanation"].get("status") == "SUCCESS"
        and bool(str(r["explanation"].get("text") or "").strip())
        for r in checked_rows
    )
    result["runtime_policy"] = {
        "version": "v0.5.0",
        "scope": "AUTOMATED_PREDICTION_NOT_GOLD",
        "withheld_verdicts": withheld,
        "total_claims": len(checked_rows),
        "nei_requires_explicit_collection_certificate": True,
        "safe_to_freeze_gold": False,
    }
    return result
