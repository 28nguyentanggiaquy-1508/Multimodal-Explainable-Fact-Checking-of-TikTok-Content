from __future__ import annotations

import asyncio
from datetime import date
import re

from fact_checking.evidence.evidence_ranker import rerank_evidence
from fact_checking.evidence.evidence_schema import (
    EvidenceItem,
    EvidenceRetrievalResult,
    RetrievalChannelResult,
)
from fact_checking.evidence.image_retriever import retrieve_image_evidence
from fact_checking.evidence.text_retriever import retrieve_text_evidence


_CURRENT_STATE_MARKERS = (
    "hiện có",
    "hiện nay",
    "tính đến",
    "đến nay",
    "đang công tác",
    "hiện tại",
    "currently",
    "as of",
)

_RELATIVE_EVENT_MARKERS = (
    "vừa được",
    "mới được",
    "gần đây",
    "cuối năm",
    "những tháng cuối năm",
    "nửa cuối năm",
    "đầu năm",
    "quý cuối",
    "cuối quý",
    "recently",
)


def _claim_text(claim) -> str:
    if isinstance(claim, str):
        return claim.strip()
    if isinstance(claim, dict):
        return str(claim.get("text") or "").strip()
    if hasattr(claim, "text"):
        return str(claim.text or "").strip()
    return str(claim or "").strip()


def _failed_channel(modality: str, exc: Exception) -> RetrievalChannelResult:
    return RetrievalChannelResult(
        modality=modality,
        status="FAILED",
        errors=[f"{type(exc).__name__}: {exc}"],
    )




def _collection_complete_certificate(
    *,
    overall_status: str,
    text_result: RetrievalChannelResult,
    image_result: RetrievalChannelResult,
    errors: list[str],
) -> bool:
    """Issue an explicit completion certificate only for fully healthy Phase 3.

    This mirrors (and does not weaken) runtime_safety_v050._collection_complete:
    overall SUCCESS, both text/image SUCCESS, and no retrieval/channel errors.
    """
    return (
        str(overall_status or "").upper() == "SUCCESS"
        and str(text_result.status or "").upper() == "SUCCESS"
        and str(image_result.status or "").upper() == "SUCCESS"
        and not errors
        and not text_result.errors
        and not image_result.errors
    )

def _temporal_anchor(multimedia) -> date | None:
    metadata = getattr(multimedia, "metadata", None) or {}
    raw = str(metadata.get("upload_date") or "").strip()
    match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", raw)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _temporal_mode(claim_text: str) -> str:
    value = claim_text.casefold()
    if any(marker in value for marker in _CURRENT_STATE_MARKERS):
        return "current_state"
    if any(marker in value for marker in _RELATIVE_EVENT_MARKERS):
        return "relative_event"
    return "none"


def _time_sensitive_claim(claim_text: str) -> bool:
    return _temporal_mode(claim_text) != "none"


async def retrieve_evidence(claim, multimedia=None) -> EvidenceRetrievalResult:
    """Claim -> discovery -> local text/image retrieval -> diversity-aware Top-K."""
    claim_text = _claim_text(claim)

    if not claim_text:
        empty_text = RetrievalChannelResult(modality="text", status="FAILED")
        empty_image = RetrievalChannelResult(modality="image", status="FAILED")
        return EvidenceRetrievalResult(
            status="FAILED",
            evidence=[],
            text=empty_text,
            image=empty_image,
            collection_complete=False,
            errors=["Empty claim text"],
        )

    anchor = _temporal_anchor(multimedia)
    temporal_mode = _temporal_mode(claim_text)
    time_sensitive = temporal_mode != "none"

    print("\n[Evidence] ================================")
    print(f"[Evidence] claim={claim_text!r}")
    if anchor and time_sensitive:
        print(
            f"[Evidence] time_sensitive=True mode={temporal_mode} "
            f"anchor={anchor.isoformat()}"
        )

    text_result, image_result = await asyncio.gather(
        asyncio.to_thread(
            retrieve_text_evidence,
            claim_text,
            None,
            temporal_anchor=anchor,
            time_sensitive=time_sensitive,
            temporal_mode=temporal_mode,
        ),
        asyncio.to_thread(
            retrieve_image_evidence,
            claim_text,
            None,
            temporal_anchor=anchor,
            time_sensitive=time_sensitive,
            temporal_mode=temporal_mode,
        ),
        return_exceptions=True,
    )

    if isinstance(text_result, Exception):
        text_result = _failed_channel("text", text_result)
    if isinstance(image_result, Exception):
        image_result = _failed_channel("image", image_result)

    all_evidence: list[EvidenceItem] = [
        *text_result.evidence,
        *image_result.evidence,
    ]
    errors = [*text_result.errors, *image_result.errors]

    print(
        "[Evidence] "
        f"text={len(text_result.evidence)}({text_result.status}) "
        f"image={len(image_result.evidence)}({image_result.status})"
    )

    channel_statuses = {text_result.status, image_result.status}
    any_available_channel = any(
        status in {"SUCCESS", "DEGRADED"}
        for status in channel_statuses
    )

    if not all_evidence:
        if "SUCCESS" in channel_statuses:
            overall_status = "SUCCESS"
        elif "DEGRADED" in channel_statuses:
            overall_status = "DEGRADED"
        else:
            overall_status = "UNAVAILABLE"

        if "FAILED" in channel_statuses and not any_available_channel:
            overall_status = "FAILED"

        return EvidenceRetrievalResult(
            status=overall_status,
            evidence=[],
            text=text_result,
            image=image_result,
            reranker_status="SKIPPED",
            collection_complete=_collection_complete_certificate(
                overall_status=overall_status, text_result=text_result, image_result=image_result, errors=errors
            ),
            errors=errors,
        )

    try:
        ranked, reranker_status = await rerank_evidence(
            claim=claim_text,
            evidence=all_evidence,
        )
    except Exception as exc:
        print(f"[Evidence/Rank] error={type(exc).__name__}: {exc}")
        ranked = sorted(
            all_evidence,
            key=lambda item: item.relevance_score,
            reverse=True,
        )[:8]
        reranker_status = "DEGRADED"
        errors.append(f"Reranker: {type(exc).__name__}: {exc}")

    overall_status = "SUCCESS"
    if (
        text_result.status in {"DEGRADED", "UNAVAILABLE", "FAILED"}
        or image_result.status in {"DEGRADED", "UNAVAILABLE", "FAILED"}
        or reranker_status == "DEGRADED"
    ):
        overall_status = "DEGRADED"

    print(
        "[Evidence] "
        f"selected={len(ranked)} status={overall_status} "
        f"reranker={reranker_status}"
    )

    return EvidenceRetrievalResult(
        status=overall_status,
        evidence=ranked,
        text=text_result,
        image=image_result,
        reranker_status=reranker_status,
        collection_complete=_collection_complete_certificate(
            overall_status=overall_status, text_result=text_result, image_result=image_result, errors=errors
        ),
        errors=errors,
    )
