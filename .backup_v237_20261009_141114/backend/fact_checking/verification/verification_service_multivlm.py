from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from core.vlm import circuit
from fact_checking.verification.multivlm_verifier import verify_openai
from fact_checking.verification.verification_service import verify_claim as _legacy_verify_claim


class VerificationResultV23(BaseModel):
    status: str = "SUCCESS"
    verdict: str | None = None
    confidence: float | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_assessments: list[dict[str, Any]] = Field(default_factory=list)
    rationale: str = ""
    uncertainty: str = ""
    limitations: list[str] = Field(default_factory=list)
    multimodal_used: bool = False
    model: str = ""
    provider: str | None = None


def _provider_order() -> list[str]:
    raw = os.getenv("VLM_VERIFICATION_PROVIDER_ORDER", os.getenv("VLM_PROVIDER_ORDER", "openai,gemini"))
    order = [x.strip().lower() for x in raw.split(",") if x.strip()]
    return order or ["openai", "gemini"]


def _remember_gemini_quota(text: str) -> None:
    if "429" not in text and "RESOURCE_EXHAUSTED" not in text:
        return
    retry = circuit.parse_retry_after_seconds(text, 300)
    for model in {
        os.getenv("GEMINI_VISION_MODEL", "gemini-3.8-flash"),
        os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.7-flash"),
    }:
        if model:
            circuit.open_circuit("gemini", model, seconds=retry, reason=text)


async def verify_claim(
    *,
    claim: Any,
    multimedia: Any,
    evidence: list[Any],
    reasoning_support: Any,
) -> VerificationResultV23:
    """Drop-in replacement for Phase 4 verification_service.verify_claim."""
    if not evidence:
        return VerificationResultV23(
            status="UNAVAILABLE",
            verdict=None,
            confidence=None,
            rationale=(
                "Không thể hoàn tất verification vì Evidence Retrieval "
                "không cung cấp evidence khả dụng."
            ),
            uncertainty="INSUFFICIENT_COLLECTION",
            limitations=["No usable retrieved evidence."],
            multimodal_used=False,
            provider=None,
        )

    failures: list[str] = []
    for provider in _provider_order():
        if provider == "openai":
            try:
                return VerificationResultV23(
                    **verify_openai(
                        claim=claim,
                        multimedia=multimedia,
                        evidence=evidence,
                        reasoning_support=reasoning_support,
                    )
                )
            except Exception as exc:
                failures.append(f"openai: {type(exc).__name__}: {exc}")
                continue

        if provider == "gemini":
            gemini_models = [
                os.getenv("GEMINI_VISION_MODEL", "gemini-3.8-flash"),
                os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.7-flash"),
            ]
            if all(circuit.is_open("gemini", m) for m in gemini_models if m):
                failures.append("gemini: persistent circuit open")
                continue
            try:
                legacy = await _legacy_verify_claim(
                    claim=claim,
                    multimedia=multimedia,
                    evidence=evidence,
                    reasoning_support=reasoning_support,
                )
                data = legacy.model_dump() if hasattr(legacy, "model_dump") else dict(legacy)
                if str(data.get("status") or "").upper() in {"FAILED", "UNAVAILABLE"}:
                    text = " | ".join(
                        [
                            str(data.get("uncertainty") or ""),
                            " ".join(str(x) for x in (data.get("limitations") or [])),
                        ]
                    )
                    _remember_gemini_quota(text)
                    failures.append(f"gemini: {text}")
                    continue
                data["provider"] = "gemini"
                return VerificationResultV23(**data)
            except Exception as exc:
                text = f"{type(exc).__name__}: {exc}"
                _remember_gemini_quota(text)
                failures.append(f"gemini: {text}")
                continue

    return VerificationResultV23(
        status="UNAVAILABLE",
        verdict=None,
        confidence=None,
        evidence_ids=[],
        evidence_assessments=[],
        rationale="Không thể hoàn tất bước xác minh do tất cả VLM provider đều không khả dụng.",
        uncertainty=" | ".join(failures),
        limitations=failures,
        multimodal_used=False,
        model="",
        provider=None,
    )


# ============================================================
# v2.3.1 - BATCH COMPATIBILITY
# ============================================================

def _candidate_value(item: Any, key: str, default: Any = None) -> Any:
    """Read a field from dict/Pydantic/object candidates defensively."""
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
    """
    Compatibility layer for the v2.2 pipeline.

    Supports common call styles such as:
      verify_claims_batch(candidates, multimedia=multimedia)
      verify_claims_batch(multimedia, candidates)
      verify_claims_batch(candidates=..., multimedia=...)
      verify_claims_batch(items=..., multimedia=...)
    """
    multimedia = kwargs.get("multimedia")

    candidates = None
    for key in (
        "candidates",
        "verification_candidates",
        "items",
        "batch",
        "claim_items",
        "claim_results",
    ):
        value = kwargs.get(key)
        if isinstance(value, list):
            candidates = value
            break

    # Positional compatibility: locate the first list-like batch.
    if candidates is None:
        for arg in args:
            if isinstance(arg, list):
                candidates = arg
                break

    # If multimedia wasn't passed by name, use the first non-list positional value.
    if multimedia is None:
        for arg in args:
            if not isinstance(arg, list):
                multimedia = arg
                break

    # Alternate call style: separate claims/evidence/reasoning arrays.
    if candidates is None and isinstance(kwargs.get("claims"), list):
        claims = kwargs.get("claims") or []
        evidences = (
            kwargs.get("evidence")
            or kwargs.get("evidences")
            or kwargs.get("evidence_list")
            or kwargs.get("evidence_by_claim")
            or []
        )
        reasonings = (
            kwargs.get("reasoning_support")
            or kwargs.get("reasoning_supports")
            or kwargs.get("reasoning_list")
            or kwargs.get("reasoning_by_claim")
            or []
        )
        candidates = []
        for idx, claim in enumerate(claims):
            ev = evidences[idx] if isinstance(evidences, list) and idx < len(evidences) else []
            rs = reasonings[idx] if isinstance(reasonings, list) and idx < len(reasonings) else {}
            candidates.append({
                "claim": claim,
                "evidence": ev,
                "reasoning_support": rs,
            })

    return list(candidates or []), multimedia


async def verify_claims_batch(
    *args: Any,
    **kwargs: Any,
) -> dict[str, VerificationResultV23]:
    """
    Batch Multi-VLM verifier tương thích pipeline v2.2.

    Trả về:
        {
            "claim_01": VerificationResultV23(...),
            "claim_02": VerificationResultV23(...),
            ...
        }

    Technical/provider failure:
        status = UNAVAILABLE
        verdict = None
    """
    candidates, multimedia = _extract_batch_inputs(args, kwargs)

    results: dict[str, VerificationResultV23] = {}

    for index, item in enumerate(candidates, start=1):
        claim = _candidate_value(item, "claim")
        evidence = _candidate_value(item, "evidence", []) or []
        reasoning_support = (
            _candidate_value(item, "reasoning_support", {})
            or {}
        )

        # Hỗ trợ tuple/list:
        # (claim, evidence, reasoning_support[, multimedia])
        if claim is None and isinstance(item, (tuple, list)):
            if len(item) >= 1:
                claim = item[0]

            if len(item) >= 2:
                evidence = item[1] or []

            if len(item) >= 3:
                reasoning_support = item[2] or {}

            if multimedia is None and len(item) >= 4:
                multimedia = item[3]

        # ----------------------------------------------------
        # XÁC ĐỊNH CLAIM ID
        # ----------------------------------------------------
        claim_id = ""

        if claim is not None:
            claim_id = str(
                _candidate_value(
                    claim,
                    "id",
                    "",
                )
                or ""
            ).strip()

        # Một số batch candidate có claim_id trực tiếp
        if not claim_id:
            claim_id = str(
                _candidate_value(
                    item,
                    "claim_id",
                    "",
                )
                or ""
            ).strip()

        # Fallback cuối cùng
        if not claim_id:
            claim_id = f"claim_{index:02d}"

        # ----------------------------------------------------
        # INVALID CANDIDATE
        # ----------------------------------------------------
        if claim is None:
            results[claim_id] = VerificationResultV23(
                status="UNAVAILABLE",
                verdict=None,
                confidence=None,
                rationale=(
                    "Không thể xác minh vì batch candidate thiếu claim."
                ),
                uncertainty="INVALID_BATCH_CANDIDATE",
                limitations=[
                    "Missing claim in verification batch candidate."
                ],
                multimodal_used=False,
                model="",
                provider=None,
            )
            continue

        # ----------------------------------------------------
        # VERIFY
        # ----------------------------------------------------
        result = await verify_claim(
            claim=claim,
            multimedia=multimedia,
            evidence=(
                list(evidence)
                if isinstance(evidence, (list, tuple))
                else []
            ),
            reasoning_support=reasoning_support,
        )

        results[claim_id] = result

    return results
