from __future__ import annotations

import asyncio
import json
from collections import Counter

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from config import (
    EVIDENCE_MAX_PER_DOMAIN,
    EVIDENCE_MAX_PER_URL,
    EVIDENCE_USE_LLM_RERANKER,
    GEMINI_ANALYST_MODEL,
    GEMINI_API_KEY,
    IMAGE_EVIDENCE_TOP_K,
    TEXT_EVIDENCE_TOP_K,
)
from core.gemini_utils import generate_content_with_retry
from fact_checking.evidence.evidence_schema import EvidenceItem
from fact_checking.schemas import StageStatus


_client = genai.Client(api_key=GEMINI_API_KEY) if (genai is not None and GEMINI_API_KEY) else None


def _extract_json(text: str) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return {}


def _response_text(response) -> str:
    value = getattr(response, "text", None)
    if value:
        return value
    chunks = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
    return "".join(chunks)


def _rerank_text_sync(claim: str, candidates: list[EvidenceItem]) -> dict[str, float]:
    if _client is None or not candidates:
        return {}

    payload = [
        {
            "id": item.id,
            "title": item.title,
            "content": item.content[:1200],
            "source_domain": item.source_domain,
            "base_relevance": item.relevance_score,
        }
        for item in candidates
    ]

    prompt = f"""
Bạn là Evidence Relevance Reranker.

CLAIM:
{claim}

CANDIDATE TEXT EVIDENCE:
{json.dumps(payload, ensure_ascii=False)}

Chỉ đánh giá mức LIÊN QUAN giữa candidate và claim. Không xác định đúng/sai.
Không thêm kiến thức ngoài candidate. Trả duy nhất JSON:
{{"scores":[{{"id":"text_...","relevance_score":0.0}}]}}
""".strip()

    response, _ = generate_content_with_retry(
        _client,
        model=GEMINI_ANALYST_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    data = _extract_json(_response_text(response))
    score_map: dict[str, float] = {}
    for row in data.get("scores", []):
        if not isinstance(row, dict) or not row.get("id"):
            continue
        try:
            score = float(row.get("relevance_score", 0.0))
        except (TypeError, ValueError):
            continue
        score_map[str(row["id"])] = max(0.0, min(1.0, score))
    return score_map


def _diverse_select(items: list[EvidenceItem], k: int) -> list[EvidenceItem]:
    """Prefer source diversity, then fill remaining slots by score."""
    if k <= 0:
        return []
    ranked = sorted(
        items,
        key=lambda item: (
            bool(item.metadata.get("temporal_eligible", True)),
            item.relevance_score,
        ),
        reverse=True,
    )
    selected: list[EvidenceItem] = []
    url_count: Counter[str] = Counter()
    domain_count: Counter[str] = Counter()

    for item in ranked:
        if not item.metadata.get("temporal_eligible", True):
            continue
        url = item.source_url or item.id
        domain = item.source_domain or "unknown"
        if url_count[url] >= max(1, EVIDENCE_MAX_PER_URL):
            continue
        if domain_count[domain] >= max(1, EVIDENCE_MAX_PER_DOMAIN):
            continue
        selected.append(item)
        url_count[url] += 1
        domain_count[domain] += 1
        if len(selected) >= k:
            return selected

    # If diversity constraints leave empty slots, fill by score. This keeps the
    # system usable for niche claims with only one or two available sources.
    selected_ids = {item.id for item in selected}
    for item in ranked:
        if item.id in selected_ids:
            continue
        selected.append(item)
        if len(selected) >= k:
            break
    return selected


async def rerank_evidence(
    claim: str,
    evidence: list[EvidenceItem],
) -> tuple[list[EvidenceItem], StageStatus]:
    """MOCHEG-inspired modality-aware ranking with deterministic diversity."""
    text_candidates = sorted(
        [item for item in evidence if item.modality == "text"],
        key=lambda item: item.relevance_score,
        reverse=True,
    )
    image_candidates = sorted(
        [item for item in evidence if item.modality == "image"],
        key=lambda item: item.relevance_score,
        reverse=True,
    )

    score_map: dict[str, float] = {}
    reranker_status: StageStatus = "SKIPPED"

    if EVIDENCE_USE_LLM_RERANKER and _client is not None and text_candidates:
        try:
            score_map = await asyncio.to_thread(
                _rerank_text_sync,
                claim,
                text_candidates[:20],
            )
            reranker_status = "SUCCESS"
        except Exception as exc:
            print(
                "[Evidence/Rank] optional LLM reranker unavailable: "
                f"{type(exc).__name__}"
            )
            reranker_status = "DEGRADED"

    reranked_text = []
    for item in text_candidates:
        llm_score = score_map.get(item.id)
        metadata = dict(item.metadata)

        if llm_score is None:
            metadata["reranker"] = "deterministic_temporal_source_diversity"
            reranked_text.append(item.model_copy(update={"metadata": metadata}))
            continue

        combined = 0.45 * item.relevance_score + 0.55 * llm_score
        metadata["reranker"] = "optional_llm_relevance"
        reranked_text.append(
            item.model_copy(
                update={
                    "rerank_score": llm_score,
                    "relevance_score": max(0.0, min(1.0, combined)),
                    "metadata": metadata,
                }
            )
        )

    reranked_text = _diverse_select(reranked_text, TEXT_EVIDENCE_TOP_K)
    image_candidates = [
        item.model_copy(update={"relevance_score": item.retrieval_score})
        for item in image_candidates
    ]
    image_candidates = _diverse_select(image_candidates, IMAGE_EVIDENCE_TOP_K)

    return reranked_text + image_candidates, reranker_status
