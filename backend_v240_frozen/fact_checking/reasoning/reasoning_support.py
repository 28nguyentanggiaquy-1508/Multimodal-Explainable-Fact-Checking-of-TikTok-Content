from __future__ import annotations

import asyncio
import html
import json
import re
from datetime import date, datetime

import torch
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from config import (
    CONCEPT_CLAIM_THRESHOLD,
    CONCEPT_CLUSTER_THRESHOLD,
    CONCEPT_MIN_FREQ_RATIO,
    GEMINI_ANALYST_MODEL,
    GEMINI_API_KEY,
    REASONING_USE_GEMINI,
)
from core.gemini_utils import generate_content_with_retry
from fact_checking.evidence.evidence_schema import EvidenceItem
from fact_checking.evidence.image_retriever import get_image_encoder, load_remote_image
from fact_checking.evidence.text_retriever import get_text_encoder


_client = genai.Client(api_key=GEMINI_API_KEY) if (genai is not None and GEMINI_API_KEY) else None

_STOP_EDGE = {
    "và", "hoặc", "của", "là", "có", "đã", "đang", "được", "theo", "thì",
    "này", "đó", "một", "những", "các", "trong", "với", "tại", "vào", "cho",
    "từ", "đến", "hiện", "nay", "vừa", "rằng", "cũng", "sẽ", "bị", "cùng", "không",
}
_ORG_PREFIX = {
    "bộ", "quân", "tổng", "chủ", "ủy", "ban", "cục", "luật", "điều", "hội",
    "trường", "viện", "sở", "tòa", "toà", "đảng", "chính", "ủy ban",
}
_INCOMPLETE_LAST = {"đại", "bộ", "tổng", "chủ", "quân", "luật", "điều"}


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
    text = getattr(response, "text", None)
    if text:
        return text
    chunks = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            value = getattr(part, "text", None)
            if value:
                chunks.append(value)
    return "".join(chunks)


def _strip_html(value: str) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(value: str) -> list[str]:
    return re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+(?:[/-][0-9A-Za-zÀ-ỹĐđ]+)*", value or "")


def _clean_concept(value: str) -> str:
    tokens = _tokenize(value)
    while tokens and tokens[0].casefold() in _STOP_EDGE:
        tokens.pop(0)
    while tokens and tokens[-1].casefold() in _STOP_EDGE:
        tokens.pop()
    return " ".join(tokens).strip()


def _good_concept(value: str) -> bool:
    tokens = _tokenize(value)
    if not tokens or len(tokens) > 7:
        return False
    if len(tokens) == 1 and not re.search(r"\d", tokens[0]) and len(tokens[0]) < 4:
        return False
    if all(token.casefold() in _STOP_EDGE for token in tokens):
        return False
    if tokens[-1].casefold() in _INCOMPLETE_LAST:
        return False
    return len(value) >= 3


def _capitalized_sequences(text: str) -> list[str]:
    # Strict proper-name extraction. We deliberately do not greedily consume
    # lowercase organization words here; claim-guided n-grams handle relevant
    # role/organization phrases without producing fragments such as "Bộ Quốc".
    found: list[str] = []
    chunks = re.split(r"[,;:.!?()\[\]{}\n]+", text or "")

    for chunk in chunks:
        tokens = _tokenize(chunk)
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if not token or not token[0].isupper() or token.casefold() in _ORG_PREFIX:
                i += 1
                continue

            seq = [token]
            j = i + 1
            while j < len(tokens) and len(seq) < 5:
                nxt = tokens[j]
                if nxt and nxt[0].isupper() and nxt.casefold() not in _ORG_PREFIX:
                    seq.append(nxt)
                    j += 1
                    continue
                break

            cleaned = _clean_concept(" ".join(seq))
            if len(_tokenize(cleaned)) >= 2 and _good_concept(cleaned):
                found.append(cleaned)
            i = max(i + 1, j if j > i + 1 else i + 1)

    return found


def _claim_guided_ngrams(claim: str, evidence_text: str) -> list[str]:
    claim_tokens = _tokenize(claim)
    evidence_cf = _strip_html(evidence_text).casefold()
    candidates: list[str] = []

    for size in range(min(7, len(claim_tokens)), 1, -1):
        for start in range(0, len(claim_tokens) - size + 1):
            tokens = claim_tokens[start:start + size]
            if tokens[0].casefold() in _STOP_EDGE or tokens[-1].casefold() in _STOP_EDGE:
                continue
            phrase = _clean_concept(" ".join(tokens))
            if not _good_concept(phrase):
                continue
            if phrase.casefold() in evidence_cf:
                candidates.append(phrase)

    # Keep maximal matching phrases to avoid noisy ladders such as
    # "3 Đại tướng", "3 Đại tướng cùng", "3 Đại tướng cùng công".
    kept: list[str] = []
    for phrase in sorted(set(candidates), key=lambda value: (len(_tokenize(value)), len(value)), reverse=True):
        phrase_cf = phrase.casefold()
        if any(phrase_cf in existing.casefold() for existing in kept):
            continue
        kept.append(phrase)
    return kept


def _local_raw_concepts(claim: str, text_evidence: list[EvidenceItem]) -> list[dict]:
    """Local Extraction-then-Filter candidates; no LLM/API dependency."""
    concepts: dict[str, dict] = {}

    def add(value: str, evidence_id: str) -> None:
        cleaned = _clean_concept(value)
        if not _good_concept(cleaned):
            return
        key = cleaned.casefold()
        row = concepts.setdefault(key, {"concept": cleaned, "evidence_ids": set()})
        row["evidence_ids"].add(evidence_id)
        # Prefer a longer surface form when two candidates normalize closely.
        if len(cleaned) > len(row["concept"]):
            row["concept"] = cleaned

    for item in text_evidence[:12]:
        text = _strip_html(item.content or "")
        if not text:
            continue

        for phrase in _claim_guided_ngrams(claim, text):
            add(phrase, item.id)
        for phrase in _capitalized_sequences(text):
            add(phrase, item.id)

        for match in re.findall(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", text):
            add(match, item.id)
        for match in re.findall(r"\b(?:19|20)\d{2}\b", text):
            add(match, item.id)
        for match in re.findall(r"\b\d+(?:[.,]\d+)?\s*%\b", text):
            add(match, item.id)
        for match in re.findall(
            r"\b\d+\s+(?:Đại tướng|đại tướng|quân nhân|người|chức vụ|ngôi sao|tướng|năm|tháng)\b",
            text,
            flags=re.UNICODE,
        ):
            add(match, item.id)

    rows = [
        {"concept": row["concept"], "evidence_ids": sorted(row["evidence_ids"])}
        for row in concepts.values()
    ]
    rows.sort(
        key=lambda row: (len(row["evidence_ids"]), len(row["concept"])),
        reverse=True,
    )
    return rows[:100]




def _anchor_from_evidence(evidence: list[EvidenceItem]) -> date | None:
    for item in evidence:
        raw = str((item.metadata or {}).get("temporal_anchor") or "").strip()
        if not raw:
            continue
        try:
            return datetime.fromisoformat(raw).date()
        except ValueError:
            continue
    return None


def _concept_after_anchor(concept: str, temporal_anchor: date | None) -> bool:
    if temporal_anchor is None:
        return False
    for year in re.findall(r"\b(20\d{2})\b", concept or ""):
        if int(year) > temporal_anchor.year:
            return True
    for day, month, year in re.findall(
        r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b",
        concept or "",
    ):
        try:
            if date(int(year), int(month), int(day)) > temporal_anchor:
                return True
        except ValueError:
            pass
    return False


def _extract_raw_concepts_sync(claim: str, text_evidence: list[EvidenceItem]) -> list[dict]:
    if _client is None or not text_evidence:
        return []
    payload = [{"id": item.id, "content": item.content[:1500]} for item in text_evidence]
    prompt = f"""
Bạn là module Concept Extraction trong hệ thống evidence-based fact-checking.
CLAIM: {claim}
TEXT EVIDENCE: {json.dumps(payload, ensure_ascii=False)}
Trích xuất key concept factual quan trọng từ evidence: người, tổ chức, địa điểm,
sự kiện, số liệu, ngày tháng và quan hệ subject-predicate-object. Mỗi concept phải
có evidence_ids. Không thêm kiến thức ngoài evidence và không đưa ra verdict.
Trả duy nhất JSON: {{"concepts":[{{"concept":"...","evidence_ids":["text_..."]}}]}}
""".strip()
    response, _ = generate_content_with_retry(
        _client,
        model=GEMINI_ANALYST_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    data = _extract_json(_response_text(response))
    return [item for item in data.get("concepts", []) if isinstance(item, dict)]


def _merge_and_filter_concepts(
    claim: str,
    raw_concepts: list[dict],
    evidence_count: int,
    *,
    temporal_anchor: date | None = None,
) -> list[dict]:
    concepts = []
    for item in raw_concepts:
        text = _clean_concept(str(item.get("concept") or ""))
        if not _good_concept(text):
            continue
        if _concept_after_anchor(text, temporal_anchor):
            continue
        evidence_ids = {
            str(value).strip()
            for value in item.get("evidence_ids", [])
            if str(value).strip()
        }
        concepts.append({"text": text, "evidence_ids": evidence_ids})

    if not concepts:
        return []

    try:
        encoder = get_text_encoder()
        concept_texts = [item["text"] for item in concepts]
        concept_embeddings = encoder.encode(concept_texts)
        claim_embedding = encoder.encode([claim])[0]
    except Exception:
        return [
            {
                "concept": item["text"],
                "claim_similarity": None,
                "frequency_ratio": round(len(item["evidence_ids"]) / max(evidence_count, 1), 4),
                "evidence_ids": sorted(item["evidence_ids"]),
            }
            for item in concepts[:20]
        ]

    clusters: list[list[int]] = []
    used = set()
    for i in range(len(concepts)):
        if i in used:
            continue
        cluster = [i]
        used.add(i)
        for j in range(i + 1, len(concepts)):
            if j in used:
                continue
            similarity = torch.dot(concept_embeddings[i], concept_embeddings[j]).item()
            if similarity >= CONCEPT_CLUSTER_THRESHOLD:
                cluster.append(j)
                used.add(j)
        clusters.append(cluster)

    final = []
    for cluster in clusters:
        evidence_ids = set()
        for index in cluster:
            evidence_ids.update(concepts[index]["evidence_ids"])

        frequency_ratio = len(evidence_ids) / max(evidence_count, 1)
        best_index = None
        best_similarity = -1.0
        for index in cluster:
            similarity = torch.dot(concept_embeddings[index], claim_embedding).item()
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = index

        if best_index is None:
            continue

        # Stronger noise gate for deterministic Extraction-then-Filter.
        # A concept must be either clearly claim-relevant, or moderately relevant
        # and repeated across evidence. Very weak temporal/site boilerplate is
        # removed even when it appears in one selected passage.
        if best_similarity < 0.12:
            continue
        if (
            best_similarity < CONCEPT_CLAIM_THRESHOLD
            and frequency_ratio < max(0.40, CONCEPT_MIN_FREQ_RATIO)
        ):
            continue

        final.append(
            {
                "concept": concepts[best_index]["text"],
                "claim_similarity": round(float(best_similarity), 4),
                "frequency_ratio": round(float(frequency_ratio), 4),
                "evidence_ids": sorted(evidence_ids),
            }
        )

    final.sort(
        key=lambda item: (
            item["claim_similarity"] if item["claim_similarity"] is not None else -1,
            item["frequency_ratio"],
            len(item["concept"]),
        ),
        reverse=True,
    )
    return final[:12]


def _generate_reasoning_support_sync(claim: str, key_concepts: list[dict], evidence: list[EvidenceItem]) -> str:
    if _client is None or not key_concepts:
        return ""
    evidence_payload = [
        {
            "id": item.id,
            "modality": item.modality,
            "title": item.title,
            "content": item.content[:1200],
            "source_domain": item.source_domain,
        }
        for item in evidence
    ]
    prompt = f"""
Bạn là Concept-Centered Reasoning Support Generator.
CLAIM: {claim}
KEY CONCEPTS: {json.dumps(key_concepts, ensure_ascii=False)}
RETRIEVED EVIDENCE: {json.dumps(evidence_payload, ensure_ascii=False)}
Tạo reasoning-support ngắn, khách quan, liên kết key concepts với evidence.
Chỉ dùng evidence; nếu evidence mâu thuẫn, nêu mâu thuẫn. Không đưa verdict.
Trả duy nhất JSON: {{"reasoning_support":"..."}}
""".strip()
    response, _ = generate_content_with_retry(
        _client,
        model=GEMINI_ANALYST_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    data = _extract_json(_response_text(response))
    return str(data.get("reasoning_support") or "").strip()


def _deterministic_reasoning_support(key_concepts: list[dict], evidence: list[EvidenceItem]) -> str:
    concept_names = [str(item.get("concept") or "").strip() for item in key_concepts[:8]]
    concept_names = [value for value in concept_names if value]
    selected = [item for item in evidence if item.modality == "text" and item.content][:4]

    pieces = []
    if concept_names:
        pieces.append("Khái niệm trọng tâm: " + "; ".join(concept_names) + ".")
    if selected:
        pieces.append("Bằng chứng được chọn:")
    for item in selected:
        clean = _strip_html(item.content)[:380]
        pieces.append(f"[{item.id}] ({item.source_domain}) {clean}")
    return " ".join(pieces).strip()


def _unit_score(cosine_score: float) -> float:
    return max(0.0, min(1.0, (float(cosine_score) + 1.0) / 2.0))


def _semantic_consistency(reasoning_text: str, evidence: list[EvidenceItem]) -> dict:
    if not reasoning_text:
        return {"text": None, "image": None, "overall": None}

    text_items = [item for item in evidence if item.modality == "text" and item.content]
    image_items = [item for item in evidence if item.modality == "image" and item.media_url]
    text_score = None
    image_score = None

    if text_items:
        try:
            encoder = get_text_encoder()
            embeddings = encoder.encode([reasoning_text] + [item.content[:1200] for item in text_items])
            support = embeddings[0:1]
            evidence_embeddings = embeddings[1:]
            raw_scores = torch.mm(evidence_embeddings, support.T).squeeze(1)
            top_values = torch.topk(raw_scores, k=min(3, raw_scores.numel())).values
            text_score = round(_unit_score(float(top_values.mean().item())), 4)
        except Exception:
            text_score = None

    if image_items:
        try:
            encoder = get_image_encoder()
            support_embedding = encoder.encode_text(reasoning_text)
            raw_scores = []
            for item in image_items[:3]:
                image = load_remote_image(item.media_url)
                if image is None:
                    continue
                image_embedding = encoder.encode_image(image)
                raw_scores.append(torch.mm(image_embedding, support_embedding.T).item())
            if raw_scores:
                image_score = round(sum(_unit_score(value) for value in raw_scores) / len(raw_scores), 4)
        except Exception:
            image_score = None

    available = [value for value in (text_score, image_score) if value is not None]
    overall = round(sum(available) / len(available), 4) if available else None
    return {"text": text_score, "image": image_score, "overall": overall}


async def build_reasoning_support(claim: str, evidence: list[EvidenceItem]) -> dict:
    if not evidence:
        return {
            "status": "SKIPPED",
            "generator": "none",
            "key_concepts": [],
            "reasoning_support": "",
            "semantic_consistency": {"text": None, "image": None, "overall": None},
            "errors": [],
        }

    text_evidence = [item for item in evidence if item.modality == "text"]
    errors: list[str] = []
    generator = "local_concept_extraction"
    status = "SUCCESS"

    # Local extraction is the default research path so Phase 3 does not depend
    # on a cloud LLM. Gemini is an optional augmentation only.
    raw_concepts = _local_raw_concepts(claim, text_evidence) if text_evidence else []

    if REASONING_USE_GEMINI and _client is not None and text_evidence:
        try:
            lmm_concepts = await asyncio.to_thread(_extract_raw_concepts_sync, claim, text_evidence)
            if lmm_concepts:
                raw_concepts.extend(lmm_concepts)
                generator = "local_plus_conceptual_lmm"
        except Exception as exc:
            errors.append(f"Optional concept augmentation: {type(exc).__name__}: {exc}")
            status = "DEGRADED"

    key_concepts = _merge_and_filter_concepts(
        claim=claim,
        raw_concepts=raw_concepts,
        evidence_count=len(text_evidence),
        temporal_anchor=_anchor_from_evidence(text_evidence),
    )

    reasoning_text = _deterministic_reasoning_support(key_concepts, evidence)

    if REASONING_USE_GEMINI and _client is not None and key_concepts:
        try:
            lmm_reasoning = await asyncio.to_thread(
                _generate_reasoning_support_sync,
                claim,
                key_concepts,
                evidence,
            )
            if lmm_reasoning:
                reasoning_text = lmm_reasoning
                generator = "conceptual_lmm"
        except Exception as exc:
            errors.append(f"Optional reasoning augmentation: {type(exc).__name__}: {exc}")
            status = "DEGRADED"

    consistency = await asyncio.to_thread(_semantic_consistency, reasoning_text, evidence)

    return {
        "status": status,
        "generator": generator,
        "key_concepts": key_concepts,
        "reasoning_support": reasoning_text,
        "semantic_consistency": consistency,
        "errors": errors,
    }
