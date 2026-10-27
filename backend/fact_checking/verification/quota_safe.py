from __future__ import annotations

import os
import re
import unicodedata
from copy import deepcopy
from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    if hasattr(value, "model_dump"):
        try:
            data = value.model_dump()
            return deepcopy(data) if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _float_score(item: dict[str, Any]) -> float:
    for key in ("relevance_score", "rerank_score", "retrieval_score"):
        try:
            value = item.get(key)
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            pass
    return 0.0


def _dedup_key(item: dict[str, Any]) -> str:
    url = str(item.get("source_url") or "").strip().lower().split("#", 1)[0]
    if url:
        return url
    domain = str(item.get("source_domain") or "").strip().lower()
    content = " ".join(str(item.get("content") or "").lower().split())[:180]
    return f"{domain}|{content}"


def _norm_for_match(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _numeric_tokens(value: Any) -> set[str]:
    return {x.replace(",", ".") for x in re.findall(r"\d+(?:[.,]\d+)*", _norm_for_match(value))}


def _word_tokens(value: Any) -> set[str]:
    return set(re.findall(r"[0-9a-zA-ZÀ-ỹ]+", _norm_for_match(value)))


def _direct_claim_coverage(item: dict[str, Any], claim_text: str) -> tuple[float, float, float]:
    content = str(item.get("content") or "")
    claim_nums = _numeric_tokens(claim_text)
    content_nums = _numeric_tokens(content)
    numeric = 0.0 if not claim_nums else len(claim_nums & content_nums) / len(claim_nums)
    claim_words = _word_tokens(claim_text)
    content_words = _word_tokens(content)
    lexical = 0.0 if not claim_words else len(claim_words & content_words) / len(claim_words)
    return (numeric, lexical, _float_score(item))


def refine_evidence(evidence: list[Any], *, claim_text: str = "") -> list[dict[str, Any]]:
    max_total = max(1, int(os.getenv("VERIFICATION_REFINED_MAX_EVIDENCE", "5")))
    max_text = max(0, int(os.getenv("VERIFICATION_REFINED_MAX_TEXT", "4")))
    max_images = max(0, int(os.getenv("VERIFICATION_MAX_IMAGES", "1")))

    rows = [_as_dict(item) for item in evidence]
    rows = [item for item in rows if item]
    text_rows = sorted(
        [item for item in rows if str(item.get("modality") or "text").lower() != "image"],
        key=_float_score,
        reverse=True,
    )
    image_rows = sorted(
        [item for item in rows if str(item.get("modality") or "").lower() == "image"],
        key=_float_score,
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    seen_domains: set[str] = set()

    # Pass 0: reserve one directly claim-covering text item before diversity.
    # This prevents source-diversity compaction from dropping the only passage
    # that contains a critical exact number/qualifier needed by Phase 4.
    if text_rows and claim_text and max_text > 0:
        direct = max(text_rows, key=lambda item: _direct_claim_coverage(item, claim_text))
        numeric, lexical, _ = _direct_claim_coverage(direct, claim_text)
        if numeric > 0.0 or lexical >= 0.55:
            selected.append(direct)
            key = _dedup_key(direct)
            seen_keys.add(key)
            domain = str(direct.get("source_domain") or "").strip().lower()
            if domain:
                seen_domains.add(domain)

    # Pass 1: maximize text-source diversity.
    for item in text_rows:
        if len(selected) >= max_text:
            break
        key = _dedup_key(item)
        domain = str(item.get("source_domain") or "").strip().lower()
        if key in seen_keys or (domain and domain in seen_domains):
            continue
        selected.append(item)
        seen_keys.add(key)
        if domain:
            seen_domains.add(domain)

    # Pass 2: fill text slots even if a domain repeats, still no duplicate URL/content.
    for item in text_rows:
        if len(selected) >= max_text:
            break
        key = _dedup_key(item)
        if key in seen_keys:
            continue
        selected.append(item)
        seen_keys.add(key)

    # Keep a small amount of image evidence only when available.
    image_added = 0
    for item in image_rows:
        if len(selected) >= max_total or image_added >= max_images:
            break
        key = _dedup_key(item)
        if key in seen_keys:
            continue
        selected.append(item)
        seen_keys.add(key)
        image_added += 1

    # Fill remaining capacity with text before adding additional images.
    for item in text_rows:
        if len(selected) >= max_total:
            break
        key = _dedup_key(item)
        if key in seen_keys:
            continue
        selected.append(item)
        seen_keys.add(key)

    return selected[:max_total]


def _parse_clock(value: str | None) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parts = [float(piece) for piece in text.split(":")]
    except ValueError:
        return None
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


def _timestamp_span(value: Any) -> tuple[float, float] | None:
    text = str(value or "").replace("–", "-").replace("—", "-").strip()
    if not text:
        return None
    left, sep, right = text.partition("-")
    start = _parse_clock(left)
    end = _parse_clock(right) if sep else start
    if start is None or end is None:
        return None
    return (min(start, end), max(start, end))


def _overlaps(item_span: tuple[float, float], claim_span: tuple[float, float], pad_seconds: float) -> bool:
    return not (
        item_span[1] < claim_span[0] - pad_seconds
        or item_span[0] > claim_span[1] + pad_seconds
    )


def _local_items(
    items: list[Any],
    claim_span: tuple[float, float] | None,
    *,
    pad_seconds: float,
    max_items: int,
) -> list[dict[str, Any]]:
    rows = [_as_dict(item) for item in items]
    rows = [item for item in rows if item]
    if claim_span is None:
        return rows[:max_items]
    selected: list[dict[str, Any]] = []
    for item in rows:
        span = _timestamp_span(item.get("timestamp"))
        if span is not None and _overlaps(span, claim_span, pad_seconds):
            selected.append(item)
            if len(selected) >= max_items:
                break
    return selected


def _joined(items: list[dict[str, Any]], key: str, max_chars: int) -> str:
    return " ".join(
        str(item.get(key) or "").strip()
        for item in items
        if str(item.get(key) or "").strip()
    )[:max_chars]


def compact_multimedia(claim: Any, multimedia: Any) -> dict[str, Any]:
    claim_dict = _as_dict(claim)
    source = _as_dict(multimedia)
    result = deepcopy(source)
    claim_span = _timestamp_span(claim_dict.get("timestamp"))
    pad = max(0.0, float(os.getenv("VERIFICATION_CONTEXT_PAD_SECONDS", "20")))
    speech_chars = max(0, int(os.getenv("VERIFICATION_SPEECH_CHARS", "1600")))
    ocr_chars = max(0, int(os.getenv("VERIFICATION_OCR_CHARS", "1000")))
    max_visual = max(0, int(os.getenv("VERIFICATION_MAX_VISUAL_OBSERVATIONS", "4")))

    speech_local = _local_items(
        list(source.get("speech_segments") or []),
        claim_span,
        pad_seconds=pad,
        max_items=12,
    )
    ocr_local = _local_items(
        list(source.get("ocr_segments") or []),
        claim_span,
        pad_seconds=pad,
        max_items=8,
    )
    visual_local = _local_items(
        list(source.get("visual_observations") or []),
        claim_span,
        pad_seconds=pad,
        max_items=max_visual,
    )

    speech_text = _joined(speech_local, "text", speech_chars)
    if not speech_text:
        speech_text = str(source.get("speech_transcript") or "")[:speech_chars]
    ocr_text = _joined(ocr_local, "text", ocr_chars)
    if not ocr_text:
        ocr_text = str(source.get("ocr_text") or "")[:ocr_chars]

    result["speech_segments"] = speech_local
    result["speech_transcript"] = speech_text
    result["ocr_segments"] = ocr_local
    result["ocr_text"] = ocr_text
    result["visual_observations"] = visual_local
    metadata = deepcopy(result.get("metadata") or {}) if isinstance(result.get("metadata"), dict) else {}
    metadata["quota_safe_context"] = {
        "claim_timestamp": claim_dict.get("timestamp"),
        "pad_seconds": pad,
        "speech_segments": len(speech_local),
        "ocr_segments": len(ocr_local),
        "visual_observations": len(visual_local),
    }
    result["metadata"] = metadata
    return result


def compact_reasoning(reasoning_support: Any) -> dict[str, Any]:
    data = _as_dict(reasoning_support)
    max_chars = max(0, int(os.getenv("VERIFICATION_REASONING_CHARS", "1600")))
    data["reasoning_support"] = str(data.get("reasoning_support") or "")[:max_chars]
    return data


def prepare_verification_inputs(
    *,
    claim: Any,
    multimedia: Any,
    evidence: list[Any],
    reasoning_support: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    claim_dict = _as_dict(claim)
    refined = refine_evidence(evidence, claim_text=str(claim_dict.get("text") or ""))
    compact_mm = compact_multimedia(claim, multimedia)
    compact_reasoning_support = compact_reasoning(reasoning_support)
    stats = {
        "evidence_before": len(evidence or []),
        "evidence_after": len(refined),
        "selected_evidence_ids": [str(item.get("id") or "") for item in refined],
        "source_context_timestamp": claim_dict.get("timestamp"),
    }
    return refined, compact_mm, compact_reasoning_support, stats
