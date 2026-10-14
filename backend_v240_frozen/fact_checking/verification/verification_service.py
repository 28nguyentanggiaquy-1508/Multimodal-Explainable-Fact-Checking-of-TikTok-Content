from __future__ import annotations

import asyncio

from config import GEMINI_ANALYST_MODEL, VERIFICATION_BATCH_ENABLED, VERIFICATION_BATCH_MAX_CLAIMS
from fact_checking.evidence.evidence_schema import EvidenceItem
from fact_checking.schemas import CandidateClaim, MultimediaAnalysis
from fact_checking.verification.verification_schema import (
    EvidenceAssessment,
    VerificationResult,
)


ALLOWED_VERDICTS = {"SUPPORTED", "REFUTED", "NEI"}
ALLOWED_STANCES = {"support", "refute", "neutral"}


def _clamp_score(value, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return max(0.0, min(1.0, score))


def nei_no_evidence_result(reason: str) -> VerificationResult:
    """Semantic NEI: retrieval ran, but usable evidence is insufficient."""
    return VerificationResult(
        status="SUCCESS",
        verdict="NEI",
        confidence=0.0,
        rationale=reason,
        uncertainty=reason,
        limitations=["No usable retrieved evidence was available after a completed retrieval stage."],
        multimodal_used=False,
        model="",
    )


def unavailable_verification_result(reason: str, limitations: list[str] | None = None) -> VerificationResult:
    """Operational failure: no truthfulness label is emitted."""
    return VerificationResult(
        status="UNAVAILABLE",
        verdict=None,
        confidence=None,
        rationale="Không thể hoàn tất bước xác minh do thành phần phụ thuộc không khả dụng.",
        uncertainty=reason,
        limitations=limitations or [reason],
        multimodal_used=False,
        model="",
    )


def failed_verification_result(reason: str) -> VerificationResult:
    return VerificationResult(
        status="FAILED",
        verdict=None,
        confidence=None,
        rationale="Bước xác minh gặp lỗi xử lý nội bộ và không phát sinh verdict.",
        uncertainty=reason,
        limitations=[reason],
        multimodal_used=False,
        model="",
    )


def _normalize_result(raw: dict, evidence: list[EvidenceItem]) -> VerificationResult:
    valid_ids = {item.id for item in evidence}
    verdict = str(raw.get("verdict") or "NEI").upper().strip()
    if verdict not in ALLOWED_VERDICTS:
        verdict = "NEI"

    confidence = _clamp_score(raw.get("confidence"))
    evidence_ids = []

    if isinstance(raw.get("evidence_ids"), list):
        for value in raw["evidence_ids"]:
            evidence_id = str(value).strip()
            if evidence_id in valid_ids and evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)

    assessments = []
    if isinstance(raw.get("evidence_assessments"), list):
        for row in raw["evidence_assessments"]:
            if not isinstance(row, dict):
                continue
            evidence_id = str(row.get("evidence_id") or "").strip()
            if evidence_id not in valid_ids:
                continue
            stance = str(row.get("stance") or "neutral").lower().strip()
            if stance not in ALLOWED_STANCES:
                stance = "neutral"
            assessments.append(
                EvidenceAssessment(
                    evidence_id=evidence_id,
                    stance=stance,
                    confidence=_clamp_score(row.get("confidence")),
                    rationale=str(row.get("rationale") or "").strip(),
                )
            )

    if not evidence_ids:
        for assessment in assessments:
            if assessment.stance != "neutral" and assessment.evidence_id not in evidence_ids:
                evidence_ids.append(assessment.evidence_id)

    rationale = str(raw.get("rationale") or "").strip()
    uncertainty = str(raw.get("uncertainty") or "").strip()
    limitations = []
    if isinstance(raw.get("limitations"), list):
        limitations = [
            str(item).strip()
            for item in raw["limitations"]
            if str(item).strip()
        ][:10]

    attached_image_count = int(raw.get("_attached_image_count") or 0)
    multimodal_used = bool(raw.get("multimodal_used")) and attached_image_count > 0

    # Evidence-grounding guardrail: decisive labels require attributable evidence.
    if verdict in {"SUPPORTED", "REFUTED"} and not evidence_ids:
        verdict = "NEI"
        confidence = min(confidence, 0.35)
        limitations.append("Decisive verdict had no valid cited evidence ID and was downgraded to NEI.")
        uncertainty = uncertainty or "Không có evidence ID hợp lệ cho kết luận quyết định."

    expected_stance = {"SUPPORTED": "support", "REFUTED": "refute"}.get(verdict)
    if expected_stance:
        has_matching_stance = any(
            assessment.evidence_id in evidence_ids
            and assessment.stance == expected_stance
            for assessment in assessments
        )
        if not has_matching_stance:
            verdict = "NEI"
            confidence = min(confidence, 0.35)
            limitations.append("Cited evidence assessments did not contain a stance matching the decisive verdict.")
            uncertainty = uncertainty or "Evidence stance chưa đủ nhất quán cho kết luận quyết định."

    if verdict == "NEI" and not uncertainty:
        uncertainty = "Bằng chứng hiện có chưa đủ để support hoặc refute claim một cách chắc chắn."

    return VerificationResult(
        status="SUCCESS",
        verdict=verdict,
        confidence=confidence,
        evidence_ids=evidence_ids,
        evidence_assessments=assessments,
        rationale=rationale,
        uncertainty=uncertainty,
        limitations=limitations,
        multimodal_used=multimodal_used,
        model=str(raw.get("_model") or GEMINI_ANALYST_MODEL),
    )


def _looks_operationally_unavailable(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = (
        "429",
        "resource_exhausted",
        "503",
        "unavailable",
        "high demand",
        "timeout",
        "deadline exceeded",
        "connection",
        "circuit open",
        "geminicircuitopenerror",
    )
    return any(marker in text for marker in markers)


async def verify_claim(
    claim: CandidateClaim,
    multimedia: MultimediaAnalysis,
    evidence: list[EvidenceItem],
    reasoning_support: dict,
) -> VerificationResult:
    """
    Phase 4: claim + retrieved multimodal evidence -> SUPPORTED/REFUTED/NEI.

    The reasoning-support text is an intermediate evidence integration aid;
    the final verdict remains constrained to retrieved evidence IDs.
    """
    if not evidence:
        return nei_no_evidence_result(
            "Evidence Retrieval hoàn tất nhưng không thu được bằng chứng đủ để xác minh claim."
        )

    try:
        from fact_checking.verification.vlm_verifier import verify_with_vlm_sync

        raw = await asyncio.to_thread(
            verify_with_vlm_sync,
            claim,
            multimedia,
            evidence,
            reasoning_support,
        )
    except Exception as exc:
        reason = f"Verifier error: {type(exc).__name__}: {exc}"
        if _looks_operationally_unavailable(exc):
            return unavailable_verification_result(reason)
        return failed_verification_result(reason)

    return _normalize_result(raw, evidence)



async def verify_claims_batch(
    jobs: list[tuple[CandidateClaim, list[EvidenceItem], dict]],
    multimedia: MultimediaAnalysis,
) -> dict[str, VerificationResult]:
    """Quota-aware batch verification. No sequential retry after batch failure."""
    if not jobs:
        return {}

    # Preserve the simple path for one claim and for explicitly disabled batch mode.
    if not VERIFICATION_BATCH_ENABLED or len(jobs) == 1:
        results: dict[str, VerificationResult] = {}
        for claim, evidence, reasoning_support in jobs:
            results[claim.id] = await verify_claim(
                claim=claim,
                multimedia=multimedia,
                evidence=evidence,
                reasoning_support=reasoning_support,
            )
        return results

    if len(jobs) > VERIFICATION_BATCH_MAX_CLAIMS:
        # Split into bounded batches, not one request per claim.
        results: dict[str, VerificationResult] = {}
        for start in range(0, len(jobs), VERIFICATION_BATCH_MAX_CLAIMS):
            part = jobs[start:start + VERIFICATION_BATCH_MAX_CLAIMS]
            results.update(await verify_claims_batch(part, multimedia))
        return results

    try:
        from fact_checking.verification.vlm_verifier import verify_batch_with_vlm_sync
        raw_by_id = await asyncio.to_thread(
            verify_batch_with_vlm_sync,
            jobs,
            multimedia,
        )
    except Exception as exc:
        reason = f"Batch verifier error: {type(exc).__name__}: {exc}"
        factory = unavailable_verification_result if _looks_operationally_unavailable(exc) else failed_verification_result
        return {claim.id: factory(reason) for claim, _, _ in jobs}

    results: dict[str, VerificationResult] = {}
    for claim, evidence, _ in jobs:
        raw = raw_by_id.get(claim.id)
        if not raw:
            results[claim.id] = failed_verification_result(
                f"Batch verifier omitted claim_id={claim.id}"
            )
            continue
        results[claim.id] = _normalize_result(raw, evidence)
    return results
