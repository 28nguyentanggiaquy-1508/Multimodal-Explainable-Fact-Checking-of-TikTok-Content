from __future__ import annotations

import asyncio

from pydantic import ValidationError

from fact_checking.schemas import (
    CandidateClaim,
    MultimediaAnalysis,
    OCRSegment,
    SpeechSegment,
    VisualObservation,
)


def _validate_items(model_type, raw_items):
    valid = []
    if not isinstance(raw_items, list):
        return valid
    for item in raw_items:
        try:
            valid.append(model_type.model_validate(item))
        except (ValidationError, TypeError, ValueError):
            continue
    return valid


def _normalize_content_type(value) -> str:
    content_type = str(value or "uncertain").strip().lower()
    if content_type not in {"real_world", "fictional", "mixed", "uncertain"}:
        return "uncertain"
    return content_type


def _phase2_status(
    *,
    raw: dict,
    visual_observations,
    speech_segments,
    speech_transcript: str,
    ocr_segments,
    ocr_text: str,
    caption: str | None,
    candidate_claims,
) -> tuple[str, str | None, list[str]]:
    available = []
    if visual_observations:
        available.append("visual")
    if speech_segments or speech_transcript:
        available.append("speech")
    if ocr_segments or ocr_text:
        available.append("ocr")
    if caption:
        available.append("caption")
    if raw.get("cross_modal_notes"):
        available.append("cross_modal")

    error = str(raw.get("error") or "").strip()
    has_primary_multimedia = bool(
        visual_observations
        or speech_segments
        or speech_transcript
        or ocr_segments
        or ocr_text
    )
    has_any_output = bool(available or candidate_claims)

    if error:
        if has_any_output:
            return "DEGRADED", error, available
        return "UNAVAILABLE", error, available

    if has_primary_multimedia:
        return "SUCCESS", None, available

    if caption or candidate_claims:
        return (
            "DEGRADED",
            "Full visual/speech/OCR extraction was unavailable; the stage used text metadata/fallback output.",
            available,
        )

    return "FAILED", "No usable multimedia observation was produced.", available


async def analyze_multimedia(video_url: str) -> MultimediaAnalysis:
    """
    Phase 2: TikTok multimedia -> observations + candidate claims.

    This stage explicitly distinguishes a successful multimodal analysis from a
    caption-only degraded fallback. That distinction is required for research
    evaluation and for honest downstream reporting.
    """
    from core.multivlm_phase2 import analyze_tiktok_multimedia

    raw = await asyncio.to_thread(analyze_tiktok_multimedia, video_url)
    if not isinstance(raw, dict):
        raw = {}

    visual_observations = _validate_items(
        VisualObservation,
        raw.get("visual_observations", []),
    )
    speech_segments = _validate_items(
        SpeechSegment,
        raw.get("speech_segments", []),
    )
    ocr_segments = _validate_items(
        OCRSegment,
        raw.get("ocr_segments", []),
    )
    candidate_claims = _validate_items(
        CandidateClaim,
        raw.get("candidate_claims", []),
    )

    normalized_claims = []
    seen_ids = set()
    for index, claim in enumerate(candidate_claims, start=1):
        claim_id = (claim.id or "").strip()
        if not claim_id or claim_id in seen_ids:
            claim_id = f"claim_{index:02d}"
        seen_ids.add(claim_id)
        normalized_claims.append(claim.model_copy(update={"id": claim_id}))

    speech_transcript = str(raw.get("speech_transcript") or "").strip()
    ocr_text = str(raw.get("ocr_text") or "").strip()
    caption = (
        str(raw.get("caption")).strip()
        if raw.get("caption") not in (None, "")
        else None
    )

    status, degraded_reason, available_modalities = _phase2_status(
        raw=raw,
        visual_observations=visual_observations,
        speech_segments=speech_segments,
        speech_transcript=speech_transcript,
        ocr_segments=ocr_segments,
        ocr_text=ocr_text,
        caption=caption,
        candidate_claims=normalized_claims,
    )

    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    source = str(raw.get("source") or "")

    if (
        status == "DEGRADED"
        and metadata.get("caption_claim_fallback_used")
        and not visual_observations
        and not speech_segments
        and not ocr_segments
    ):
        source = "gemini_multimodal_caption_fallback"

    return MultimediaAnalysis(
        video_url=video_url,
        content_type=_normalize_content_type(raw.get("content_type")),
        status=status,
        degraded_reason=degraded_reason,
        available_modalities=available_modalities,
        visual_observations=visual_observations,
        speech_segments=speech_segments,
        speech_transcript=speech_transcript,
        ocr_segments=ocr_segments,
        ocr_text=ocr_text,
        caption=caption,
        candidate_claims=normalized_claims,
        cross_modal_notes=[
            str(item).strip()
            for item in raw.get("cross_modal_notes", [])
            if str(item).strip()
        ]
        if isinstance(raw.get("cross_modal_notes"), list)
        else [],
        metadata=metadata,
        source=source,
        model=str(raw.get("model") or ""),
        error=str(raw.get("error")) if raw.get("error") else None,
    )
