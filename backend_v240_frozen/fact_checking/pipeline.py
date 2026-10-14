from __future__ import annotations

import time
import uuid

from core.vlm.checkpoint import save_checkpoint
from fact_checking.claims.claim_service import extract_claims
from fact_checking.evidence.evidence_service import retrieve_evidence
from fact_checking.multimedia.multimedia_service import analyze_multimedia
from fact_checking.reasoning.reasoning_support import build_reasoning_support
from fact_checking.schemas import CONTRACT_VERSION, StageReport
from fact_checking.verification.verification_service_multivlm import (
    failed_verification_result,
    nei_no_evidence_result,
    unavailable_verification_result,
    verify_claims_batch,
)

PIPELINE_VERSION = "v2.4.0-hybrid-quota-independent"


def _duration_ms(started: float) -> int:
    return int(round((time.perf_counter() - started) * 1000))


def _aggregate_status(statuses: list[str], *, empty: str = "SKIPPED") -> str:
    if not statuses:
        return empty
    if all(status == "SUCCESS" for status in statuses):
        return "SUCCESS"
    if all(status in {"UNAVAILABLE", "FAILED"} for status in statuses):
        return "UNAVAILABLE" if "UNAVAILABLE" in statuses else "FAILED"
    if all(status == "SKIPPED" for status in statuses):
        return "SKIPPED"
    return "DEGRADED"


def _phase_message(stage: str, status: str, *, has_claims: bool) -> str:
    if not has_claims:
        return f"{stage} skipped because no selected claim was available."
    if stage == "phase_3":
        return {
            "SUCCESS": "Evidence retrieval and local conceptual reasoning completed.",
            "DEGRADED": "Evidence retrieval/reasoning completed with one or more degraded modalities/providers.",
            "UNAVAILABLE": "Evidence retrieval was unavailable for the selected claims.",
            "FAILED": "Evidence retrieval/reasoning failed due to an internal error.",
        }.get(status, "Evidence retrieval/reasoning finished.")
    return {
        "SUCCESS": "Evidence-grounded verification completed.",
        "DEGRADED": "Verification completed with degraded dependencies.",
        "UNAVAILABLE": "Verification could not run because the verifier/dependency was unavailable.",
        "FAILED": "Verification failed due to an internal error.",
    }.get(status, "Verification finished.")



def _claim_extraction_semantics(multimedia_status: str, claim_count: int) -> tuple[str, bool, bool | None]:
    """Expose dataset-safe claim-extraction semantics at the backend boundary.

    A provider/technical failure must never be encoded as a true no-claim sample.
    DEGRADED with zero selected claims is therefore incomplete/unknown until the
    dataset semantic guard can inspect the detailed provider reason.
    """
    status = str(multimedia_status or "").upper() or "UNAVAILABLE"
    if claim_count > 0:
        return status, True, False
    if status == "SUCCESS":
        return "SUCCESS", True, True
    if status in {"FAILED", "UNAVAILABLE"}:
        return status, False, None
    return status, False, None


def _build_system_explanation(verification: dict) -> dict | None:
    """Project-defined explainability projection from the evidence-grounded verifier.

    This is SYSTEM output, never Gold. It intentionally reuses the verifier's
    concise evidence-grounded rationale instead of asking another model to invent
    a second explanation.
    """
    if str(verification.get("status") or "").upper() != "SUCCESS":
        return None
    verdict = str(verification.get("verdict") or "").upper()
    if verdict not in {"SUPPORTED", "REFUTED", "NEI"}:
        return None
    rationale = str(verification.get("rationale") or "").strip()
    if not rationale:
        return None
    return {
        "status": "SUCCESS",
        "generator": "phase4_verification_rationale_projection",
        "origin": "SYSTEM",
        "contract_version": CONTRACT_VERSION,
        "is_gold": False,
        "verdict": verdict,
        "text": rationale,
        "evidence_ids": list(verification.get("evidence_ids") or []),
        "evidence_coverage": verification.get("evidence_coverage"),
        "multimodal_used": bool(verification.get("multimodal_used")),
        "provider": verification.get("provider"),
        "model": verification.get("model"),
    }


async def run_fact_check_pipeline(video_url: str) -> dict:
    trace_id = str(uuid.uuid4())
    pipeline_started = time.perf_counter()

    phase_reports: dict[str, StageReport] = {
        "phase_1": StageReport(
            status="SUCCESS",
            message="TikTok URL accepted and fact-check trace initialized.",
            duration_ms=0,
            details={"trace_id": trace_id},
        )
    }

    # PHASE 2
    phase2_started = time.perf_counter()
    multimedia = await analyze_multimedia(video_url)
    claims = await extract_claims(multimedia)
    phase_reports["phase_2"] = StageReport(
        status=multimedia.status,
        message=(
            "Multimedia observations and candidate claims extracted."
            if multimedia.status == "SUCCESS"
            else (multimedia.degraded_reason or "Phase 2 did not complete normally.")
        ),
        duration_ms=_duration_ms(phase2_started),
        details={
            "available_modalities": multimedia.available_modalities,
            "candidate_claim_count": len(multimedia.candidate_claims),
            "selected_claim_count": len(claims),
            "source": multimedia.source,
            "model": multimedia.model,
        },
    )

    claim_results: list[dict] = []
    retrieval_statuses: list[str] = []
    reasoning_statuses: list[str] = []
    verification_statuses: list[str] = []
    pending_verification: list[tuple] = []

    # PHASE 3: complete retrieval + reasoning for all claims without requiring Gemini.
    phase3_started_total = time.perf_counter()
    for claim in claims:
        try:
            retrieval = await retrieve_evidence(claim=claim, multimedia=multimedia)
        except Exception as exc:
            retrieval = None
            retrieval_status = "FAILED"
            retrieval_error = f"{type(exc).__name__}: {exc}"
        else:
            retrieval_status = retrieval.status
            retrieval_error = ""

        retrieval_statuses.append(retrieval_status)

        if retrieval is None:
            evidence = []
            retrieval_payload = {
                "status": "FAILED", "text": None, "image": None,
                "reranker_status": "SKIPPED", "errors": [retrieval_error],
            }
            reasoning_support = {
                "status": "SKIPPED", "generator": "none", "key_concepts": [],
                "reasoning_support": "",
                "semantic_consistency": {"text": None, "image": None, "overall": None},
                "errors": [retrieval_error],
            }
            verification = failed_verification_result(
                f"Evidence Retrieval failed: {retrieval_error}"
            )
        else:
            evidence = retrieval.evidence
            retrieval_payload = retrieval.model_dump(exclude={"evidence"})

            if retrieval.status in {"UNAVAILABLE", "FAILED"} and not evidence:
                reasoning_support = {
                    "status": "SKIPPED", "generator": "none", "key_concepts": [],
                    "reasoning_support": "",
                    "semantic_consistency": {"text": None, "image": None, "overall": None},
                    "errors": retrieval.errors,
                }
                verification = unavailable_verification_result(
                    "Evidence Retrieval provider/model không khả dụng.",
                    limitations=retrieval.errors or ["No evidence provider completed successfully."],
                )
            elif not evidence:
                reasoning_support = {
                    "status": "SKIPPED", "generator": "none", "key_concepts": [],
                    "reasoning_support": "",
                    "semantic_consistency": {"text": None, "image": None, "overall": None},
                    "errors": [],
                }
                verification = nei_no_evidence_result(
                    "Evidence Retrieval hoàn tất nhưng không tìm thấy evidence đủ để support/refute claim."
                )
            else:
                reasoning_support = await build_reasoning_support(
                    claim=claim.text,
                    evidence=evidence,
                )
                verification = None
                pending_verification.append((claim, evidence, reasoning_support))

        reasoning_statuses.append(str(reasoning_support.get("status") or "SKIPPED"))
        claim_results.append(
            {
                "claim": claim.model_dump(),
                "retrieval": retrieval_payload,
                "evidence": [item.model_dump() for item in evidence],
                "reasoning_support": reasoning_support,
                "verification": verification.model_dump() if verification else None,
                "explanation": None,
                "error": retrieval_error or None,
            }
        )

    phase3_total_ms = _duration_ms(phase3_started_total)
    phase3_status = _aggregate_status(retrieval_statuses)
    if phase3_status == "SUCCESS" and any(status == "DEGRADED" for status in reasoning_statuses):
        phase3_status = "DEGRADED"
    phase_reports["phase_3"] = StageReport(
        status=phase3_status,
        message=_phase_message("phase_3", phase3_status, has_claims=bool(claims)),
        duration_ms=phase3_total_ms,
        details={
            "retrieval_statuses": retrieval_statuses,
            "reasoning_statuses": reasoning_statuses,
            "claims_with_evidence": sum(bool(item["evidence"]) for item in claim_results),
        },
    )

    # Durable Phase-3 checkpoint: provider/quota failure in Phase 4 must never force
    # rerunning download, multimedia parsing, retrieval or reasoning.
    try:
        save_checkpoint(
            trace_id=trace_id,
            phase="phase_3_ready",
            payload={
                "contract_version": CONTRACT_VERSION,
                "pipeline_version": PIPELINE_VERSION,
                "trace_id": trace_id,
                "video_url": video_url,
                "phase_status": {key: report.model_dump() for key, report in phase_reports.items()},
                "multimedia": multimedia.model_dump(),
                "claims": claim_results,
                "pending_claim_ids": [item[0].id for item in pending_verification],
            },
        )
    except Exception as exc:
        print(f"[Checkpoint] phase_3_ready skipped: {type(exc).__name__}: {exc}")

    # PHASE 4: provider-agnostic quota-safe verification.
    phase4_started = time.perf_counter()
    verification_by_id = await verify_claims_batch(pending_verification, multimedia)

    for row in claim_results:
        if row["verification"] is not None:
            verification_statuses.append(row["verification"]["status"])
            continue
        claim_id = row["claim"]["id"]
        verification = verification_by_id.get(claim_id)
        if verification is None:
            verification = failed_verification_result(
                f"No verification result returned for {claim_id}"
            )
        row["verification"] = verification.model_dump()
        verification_statuses.append(verification.status)

    # Produce a concise SYSTEM explanation only from the final evidence-grounded
    # verification result. This is not a Gold explanation and does not expose CoT.
    for row in claim_results:
        row["explanation"] = _build_system_explanation(row["verification"] or {})

    phase4_total_ms = _duration_ms(phase4_started)

    phase4_status = _aggregate_status(verification_statuses)
    phase_reports["phase_4"] = StageReport(
        status=phase4_status,
        message=_phase_message("phase_4", phase4_status, has_claims=bool(claims)),
        duration_ms=phase4_total_ms,
        details={
            "claim_statuses": verification_statuses,
            "batch_candidate_count": len(pending_verification),
        },
    )

    verdict_ready = bool(claim_results) and all(
        item["verification"].get("status") == "SUCCESS"
        and item["verification"].get("verdict") in {"SUPPORTED", "REFUTED", "NEI"}
        for item in claim_results
    )
    explanation_ready = bool(claim_results) and all(
        isinstance(item.get("explanation"), dict)
        and item["explanation"].get("status") == "SUCCESS"
        and bool(str(item["explanation"].get("text") or "").strip())
        for item in claim_results
    )

    skip_reason = None
    if not claims:
        if multimedia.content_type == "fictional":
            skip_reason = "Nội dung chủ yếu là hư cấu và không có factual claim ngoài đời thực đủ điều kiện kiểm chứng."
        elif multimedia.status in {"UNAVAILABLE", "FAILED"}:
            skip_reason = "Phase 2 không khả dụng nên chưa thể xác định claim đủ tin cậy để tiếp tục."
        else:
            skip_reason = "Không phát hiện checkable/check-worthy claim để tiếp tục fact-check."
    elif not verdict_ready:
        skip_reason = "Ít nhất một claim chưa hoàn tất verification do retrieval/verifier bị degraded hoặc unavailable."

    overall_status = _aggregate_status(
        [report.status for report in phase_reports.values()], empty="FAILED"
    )

    claim_extraction_status, claim_extraction_completed, no_claim_detected = (
        _claim_extraction_semantics(multimedia.status, len(claims))
    )

    result = {
        "contract_version": CONTRACT_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "implemented_through": "phase_4",
        "trace_id": trace_id,
        "status": overall_status,
        "video_url": video_url,
        "phase_status": {key: report.model_dump() for key, report in phase_reports.items()},
        "multimedia": multimedia.model_dump(),
        "claims": claim_results,
        "claim_count": len(claim_results),
        "claim_extraction_status": claim_extraction_status,
        "claim_extraction_completed": claim_extraction_completed,
        "no_claim_detected": no_claim_detected,
        "provenance": {
            "contract_version": CONTRACT_VERSION,
            "origin": "SYSTEM",
            "pipeline_version": PIPELINE_VERSION,
            "trace_id": trace_id,
            "multimedia_provider": multimedia.source or None,
            "multimedia_model": multimedia.model or None,
        },
        "verdict_ready": verdict_ready,
        "explanation_ready": explanation_ready,
        "skip_reason": skip_reason,
        "duration_ms": _duration_ms(pipeline_started),
    }

    try:
        save_checkpoint(trace_id=trace_id, phase="final", payload=result)
    except Exception as exc:
        print(f"[Checkpoint] final skipped: {type(exc).__name__}: {exc}")
    return result

