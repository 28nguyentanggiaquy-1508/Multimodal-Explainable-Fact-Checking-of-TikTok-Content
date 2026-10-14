from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from core.vlm.router import csv_order
from fact_checking.verification.multivlm_verifier import verify_gemini, verify_local, verify_openai
from fact_checking.verification.quota_safe import prepare_verification_inputs


class VerificationResultV23(BaseModel):
    status: str = "SUCCESS"
    verdict: str | None = None
    confidence: float | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_assessments: list[dict[str, Any]] = Field(default_factory=list)
    evidence_coverage: str | None = None
    rationale: str = ""
    uncertainty: str = ""
    limitations: list[str] = Field(default_factory=list)
    multimodal_used: bool = False
    model: str = ""
    provider: str | None = None
    operational_metadata: dict[str, Any] = Field(default_factory=dict)


def failed_verification_result(reason: str) -> VerificationResultV23:
    return VerificationResultV23(
        status="FAILED",
        verdict=None,
        confidence=None,
        evidence_coverage=None,
        rationale="Không thể hoàn tất bước xác minh do lỗi kỹ thuật nội bộ.",
        uncertainty=str(reason or "INTERNAL_FAILURE"),
        limitations=[str(reason or "Internal verification failure.")],
        multimodal_used=False,
        provider=None,
    )


def unavailable_verification_result(
    reason: str,
    *,
    limitations: list[str] | None = None,
) -> VerificationResultV23:
    return VerificationResultV23(
        status="UNAVAILABLE",
        verdict=None,
        confidence=None,
        evidence_coverage=None,
        rationale="Không thể hoàn tất bước xác minh do dependency/provider không khả dụng.",
        uncertainty=str(reason or "UNAVAILABLE"),
        limitations=list(limitations or [str(reason or "Verification unavailable.")]),
        multimodal_used=False,
        provider=None,
    )


def nei_no_evidence_result(reason: str) -> VerificationResultV23:
    return VerificationResultV23(
        status="SUCCESS",
        verdict="NEI",
        confidence=0.5,
        evidence_ids=[],
        evidence_assessments=[],
        evidence_coverage="INSUFFICIENT",
        rationale=str(reason or "External evidence is insufficient to support or refute the claim."),
        uncertainty="INSUFFICIENT_EXTERNAL_EVIDENCE",
        limitations=["Evidence retrieval completed but did not provide sufficient external evidence."],
        multimodal_used=False,
        model="deterministic-guard",
        provider="system",
    )


def _provider_order() -> list[str]:
    raw = os.getenv("VLM_VERIFICATION_PROVIDER_ORDER", os.getenv("VLM_PROVIDER_ORDER", "local,openai,gemini"))
    return csv_order(raw, "local,openai,gemini")


async def verify_claim(
    *,
    claim: Any,
    multimedia: Any,
    evidence: list[Any],
    reasoning_support: Any,
) -> VerificationResultV23:
    """Provider-agnostic Phase 4 verifier.

    Invariant: provider/technical failure never becomes NEI. NEI is only emitted by
    successful evidence-grounded verification or the deterministic no-evidence guard.
    """
    if not evidence:
        return unavailable_verification_result(
            "INSUFFICIENT_COLLECTION",
            limitations=["No usable retrieved evidence."],
        )

    compact_evidence, compact_multimedia, compact_reasoning, compact_stats = prepare_verification_inputs(
        claim=claim,
        multimedia=multimedia,
        evidence=evidence,
        reasoning_support=reasoning_support,
    )
    if not compact_evidence:
        return unavailable_verification_result(
            "VERIFICATION_REFINEMENT_EMPTY",
            limitations=["Evidence refinement left no usable candidate."],
        )

    failures: list[str] = []
    provider_attempts: list[dict[str, Any]] = []
    dispatch = {
        "local": verify_local,
        "openai": verify_openai,
        "gemini": verify_gemini,
    }

    for provider in _provider_order():
        fn = dispatch.get(provider)
        if fn is None:
            failures.append(f"{provider}: unsupported provider")
            provider_attempts.append({"provider": provider, "status": "SKIPPED", "reason": "unsupported"})
            continue
        try:
            data = fn(
                claim=claim,
                multimedia=compact_multimedia,
                evidence=compact_evidence,
                reasoning_support=compact_reasoning,
            )
            data = dict(data)
            data["operational_metadata"] = {
                "quota_safe": compact_stats,
                "provider_attempts": provider_attempts + [{"provider": provider, "status": "SUCCESS"}],
            }
            return VerificationResultV23(**data)
        except Exception as exc:
            message = f"{provider}: {type(exc).__name__}: {exc}"
            failures.append(message)
            provider_attempts.append({"provider": provider, "status": "UNAVAILABLE", "reason": message[:1000]})
            continue

    return VerificationResultV23(
        status="UNAVAILABLE",
        verdict=None,
        confidence=None,
        evidence_ids=[],
        evidence_assessments=[],
        evidence_coverage=None,
        rationale="Không thể hoàn tất bước xác minh do tất cả VLM provider đều không khả dụng.",
        uncertainty=" | ".join(failures),
        limitations=failures,
        multimodal_used=False,
        model="",
        provider=None,
        operational_metadata={
            "quota_safe": compact_stats,
            "provider_attempts": provider_attempts,
        },
    )


def _candidate_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    if hasattr(item, key):
        return getattr(item, key, default)
    if hasattr(item, "model_dump"):
        try:
            data = item.model_dump()
            if isinstance(data, dict):
                return data.get(key, default)
        except Exception:
            pass
    return default


def _extract_batch_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[list[Any], Any]:
    multimedia = kwargs.get("multimedia")
    candidates = None
    for key in (
        "candidates", "verification_candidates", "items", "batch", "claim_items", "claim_results"
    ):
        value = kwargs.get(key)
        if isinstance(value, list):
            candidates = value
            break
    if candidates is None:
        for arg in args:
            if isinstance(arg, list):
                candidates = arg
                break
    if multimedia is None:
        for arg in args:
            if not isinstance(arg, list):
                multimedia = arg
                break
    if candidates is None and isinstance(kwargs.get("claims"), list):
        claims = kwargs.get("claims") or []
        evidences = kwargs.get("evidence") or kwargs.get("evidences") or kwargs.get("evidence_list") or []
        reasonings = kwargs.get("reasoning_support") or kwargs.get("reasoning_supports") or []
        candidates = []
        for idx, claim in enumerate(claims):
            candidates.append(
                {
                    "claim": claim,
                    "evidence": evidences[idx] if idx < len(evidences) else [],
                    "reasoning_support": reasonings[idx] if idx < len(reasonings) else {},
                }
            )
    return list(candidates or []), multimedia


async def verify_claims_batch(*args: Any, **kwargs: Any) -> dict[str, VerificationResultV23]:
    candidates, multimedia = _extract_batch_inputs(args, kwargs)
    results: dict[str, VerificationResultV23] = {}
    for index, item in enumerate(candidates, start=1):
        claim = _candidate_value(item, "claim")
        evidence = _candidate_value(item, "evidence", []) or []
        reasoning_support = _candidate_value(item, "reasoning_support", {}) or {}
        if claim is None and isinstance(item, (tuple, list)):
            claim = item[0] if len(item) >= 1 else None
            evidence = item[1] if len(item) >= 2 else []
            reasoning_support = item[2] if len(item) >= 3 else {}
            if multimedia is None and len(item) >= 4:
                multimedia = item[3]

        claim_id = str(_candidate_value(claim, "id", "") or _candidate_value(item, "claim_id", "") or f"claim_{index:02d}").strip()
        if claim is None:
            results[claim_id] = unavailable_verification_result("INVALID_BATCH_CANDIDATE")
            continue

        results[claim_id] = await verify_claim(
            claim=claim,
            multimedia=multimedia,
            evidence=list(evidence) if isinstance(evidence, (list, tuple)) else [],
            reasoning_support=reasoning_support,
        )
    return results
