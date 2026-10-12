from __future__ import annotations

import json
from io import BytesIO

try:
    from google import genai
    from google.genai import types
except ImportError:  # Allows offline/unit-test imports.
    genai = None
    types = None

from config import (
    GEMINI_ANALYST_MODEL,
    GEMINI_API_KEY,
    VERIFICATION_BATCH_EVIDENCE_CHARS,
    VERIFICATION_BATCH_MAX_IMAGES_TOTAL,
    VERIFICATION_MAX_EVIDENCE_CHARS,
    VERIFICATION_MAX_IMAGES,
)
from core.gemini_utils import generate_content_with_retry
from fact_checking.evidence.evidence_schema import EvidenceItem
from fact_checking.evidence.image_retriever import load_remote_image
from fact_checking.schemas import CandidateClaim, MultimediaAnalysis


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


def _multimedia_context(claim: CandidateClaim, multimedia: MultimediaAnalysis) -> dict:
    return {
        "phase2_status": multimedia.status,
        "claim_source_modality": claim.source_modality,
        "claim_timestamp": claim.timestamp,
        "claim_type": claim.claim_type,
        "visual_observations": [
            item.model_dump() for item in multimedia.visual_observations[:20]
        ],
        "speech_segments": [
            item.model_dump() for item in multimedia.speech_segments[:30]
        ],
        "ocr_segments": [
            item.model_dump() for item in multimedia.ocr_segments[:30]
        ],
        "caption": multimedia.caption,
        "cross_modal_notes": multimedia.cross_modal_notes[:15],
    }


def _evidence_payload(evidence: list[EvidenceItem]) -> list[dict]:
    rows = []
    for item in evidence:
        rows.append(
            {
                "id": item.id,
                "modality": item.modality,
                "title": item.title,
                "source_url": item.source_url,
                "source_domain": item.source_domain,
                "provider": item.provider,
                "source_quality": item.source_quality,
                "content": item.content[:VERIFICATION_MAX_EVIDENCE_CHARS],
                "retrieval_score": round(float(item.retrieval_score), 4),
                "rerank_score": (
                    round(float(item.rerank_score), 4)
                    if item.rerank_score is not None
                    else None
                ),
                "relevance_score": round(float(item.relevance_score), 4),
            }
        )
    return rows


def _image_part(item: EvidenceItem):
    if item.modality != "image" or not item.media_url:
        return None
    image = load_remote_image(item.media_url)
    if image is None:
        return None
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")


def _build_prompt(
    claim: CandidateClaim,
    multimedia: MultimediaAnalysis,
    evidence: list[EvidenceItem],
    reasoning_support: dict,
) -> str:
    return f"""
# VAI TRÒ
Bạn là module Multimodal Claim Verification trong hệ thống kiểm chứng thông tin dựa trên bằng chứng.

# CLAIM
{claim.text}

# NGỮ CẢNH MULTIMEDIA GỐC TỪ TIKTOK
{json.dumps(_multimedia_context(claim, multimedia), ensure_ascii=False)}

# RETRIEVED EVIDENCE
{json.dumps(_evidence_payload(evidence), ensure_ascii=False)}

# CONCEPTUAL REASONING SUPPORT
Key concepts:
{json.dumps(reasoning_support.get('key_concepts') or [], ensure_ascii=False)}

Reasoning-support text:
{str(reasoning_support.get('reasoning_support') or '').strip() or '(không có)'}

Semantic-consistency diagnostic:
{json.dumps(reasoning_support.get('semantic_consistency') or {}, ensure_ascii=False)}

# NHÃN
Chọn đúng một nhãn:
- SUPPORTED: evidence đủ và trực tiếp hỗ trợ nội dung cốt lõi của claim.
- REFUTED: evidence đủ và trực tiếp mâu thuẫn/bác bỏ nội dung cốt lõi của claim.
- NEI: evidence không đủ, gián tiếp, xung đột, hoặc chưa xác minh được chi tiết cốt lõi.

# QUY TẮC BẮT BUỘC
- Chỉ dùng evidence được cung cấp. Không dùng kiến thức nền bên ngoài.
- Ngữ cảnh TikTok cho biết claim được thể hiện như thế nào; nó không tự động chứng minh claim đúng.
- Không coi comment/social reaction là authoritative evidence.
- Chỉ dẫn evidence bằng ID có trong RETRIEVED EVIDENCE.
- SUPPORTED/REFUTED phải có ít nhất một evidence_id hợp lệ và assessment stance tương ứng.
- relevance_score chỉ là điểm liên quan retrieval, không phải xác suất claim đúng.
- Nếu text/image evidence mâu thuẫn, nêu rõ trong uncertainty.
- Không viết chain-of-thought nội bộ; chỉ viết rationale ngắn, kiểm tra được và bám evidence.

# OUTPUT
Trả duy nhất JSON hợp lệ:
{{
  "verdict": "SUPPORTED | REFUTED | NEI",
  "confidence": 0.0,
  "evidence_ids": ["text_...", "image_..."],
  "evidence_assessments": [
    {{
      "evidence_id": "text_...",
      "stance": "support | refute | neutral",
      "confidence": 0.0,
      "rationale": "giải thích ngắn dựa trên evidence"
    }}
  ],
  "rationale": "giải thích ngắn cho verdict, nêu evidence ID",
  "uncertainty": "điểm chưa đủ hoặc mâu thuẫn",
  "limitations": ["..."],
  "multimodal_used": true
}}
""".strip()


def verify_with_vlm_sync(
    claim: CandidateClaim,
    multimedia: MultimediaAnalysis,
    evidence: list[EvidenceItem],
    reasoning_support: dict,
) -> dict:
    if _client is None or types is None:
        raise RuntimeError("Gemini client is unavailable")

    contents: list = [
        _build_prompt(
            claim=claim,
            multimedia=multimedia,
            evidence=evidence,
            reasoning_support=reasoning_support,
        )
    ]

    image_count = 0
    for item in evidence:
        if image_count >= VERIFICATION_MAX_IMAGES:
            break
        if item.modality != "image" or not item.media_url:
            continue
        try:
            image_part = _image_part(item)
        except Exception:
            image_part = None
        if image_part is None:
            continue

        contents.append(
            f"IMAGE_EVIDENCE_ID={item.id}; TITLE={item.title}; SOURCE={item.source_domain}"
        )
        contents.append(image_part)
        image_count += 1

    response, model_used = generate_content_with_retry(
        _client,
        model=GEMINI_ANALYST_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    data = _extract_json(_response_text(response))
    if not data:
        raise RuntimeError("Verifier returned invalid JSON")

    data["_attached_image_count"] = image_count
    data["_model"] = str(model_used or GEMINI_ANALYST_MODEL)
    return data



def _batch_claim_payload(claim, evidence: list[EvidenceItem], reasoning_support: dict) -> dict:
    return {
        "claim_id": claim.id,
        "claim": claim.text,
        "claim_type": claim.claim_type,
        "source_modality": claim.source_modality,
        "evidence": [
            {
                "id": item.id,
                "modality": item.modality,
                "title": item.title,
                "source_domain": item.source_domain,
                "source_quality": item.source_quality,
                "content": item.content[:VERIFICATION_BATCH_EVIDENCE_CHARS],
                "relevance_score": round(float(item.relevance_score), 4),
                "temporal_relation": item.metadata.get("temporal_relation"),
                "temporal_score": item.metadata.get("temporal_score"),
            }
            for item in evidence
        ],
        "key_concepts": reasoning_support.get("key_concepts") or [],
        "reasoning_support": str(reasoning_support.get("reasoning_support") or "")[:2500],
    }


def verify_batch_with_vlm_sync(
    jobs: list[tuple[CandidateClaim, list[EvidenceItem], dict]],
    multimedia: MultimediaAnalysis,
) -> dict[str, dict]:
    """Verify multiple claims in one Gemini request to reduce quota pressure."""
    if _client is None or types is None:
        raise RuntimeError("Gemini client is unavailable")
    if not jobs:
        return {}

    payload = [
        _batch_claim_payload(claim, evidence, reasoning_support)
        for claim, evidence, reasoning_support in jobs
    ]

    multimedia_summary = {
        "phase2_status": multimedia.status,
        "caption": multimedia.caption,
        "upload_date": (multimedia.metadata or {}).get("upload_date"),
        "cross_modal_notes": multimedia.cross_modal_notes[:10],
    }

    prompt = f"""
# VAI TRÒ
Bạn là module batch Multimodal Claim Verification trong hệ thống fact-checking dựa trên evidence.

# NGỮ CẢNH TIKTOK
{json.dumps(multimedia_summary, ensure_ascii=False)}

# CLAIM JOBS
{json.dumps(payload, ensure_ascii=False)}

# NHÃN
SUPPORTED / REFUTED / NEI.

# QUY TẮC
- Xử lý từng claim_id độc lập.
- Chỉ dùng evidence nằm trong job tương ứng; không dùng kiến thức nền ngoài evidence.
- TikTok context mô tả claim, không tự chứng minh claim đúng.
- SUPPORTED/REFUTED phải cite evidence ID hợp lệ của chính claim đó và stance phải khớp.
- Chú ý temporal_relation/temporal_score với claim phụ thuộc thời điểm.
- Nếu evidence xung đột hoặc không đủ, chọn NEI.
- Không viết chain-of-thought; rationale ngắn và kiểm tra được.

# OUTPUT
Trả duy nhất JSON:
{{
  "results": [
    {{
      "claim_id": "claim_01",
      "verdict": "SUPPORTED | REFUTED | NEI",
      "confidence": 0.0,
      "evidence_ids": ["text_...", "image_..."],
      "evidence_assessments": [
        {{"evidence_id":"text_...","stance":"support | refute | neutral","confidence":0.0,"rationale":"..."}}
      ],
      "rationale": "...",
      "uncertainty": "...",
      "limitations": ["..."],
      "multimodal_used": true
    }}
  ]
}}
""".strip()

    contents: list = [prompt]
    attached_counts: dict[str, int] = {claim.id: 0 for claim, _, _ in jobs}
    total_images = 0
    for claim, evidence, _ in jobs:
        for item in evidence:
            if total_images >= VERIFICATION_BATCH_MAX_IMAGES_TOTAL:
                break
            if item.modality != "image" or not item.media_url:
                continue
            try:
                image_part = _image_part(item)
            except Exception:
                image_part = None
            if image_part is None:
                continue
            contents.append(
                f"CLAIM_ID={claim.id}; IMAGE_EVIDENCE_ID={item.id}; "
                f"TITLE={item.title}; SOURCE={item.source_domain}"
            )
            contents.append(image_part)
            attached_counts[claim.id] += 1
            total_images += 1

    response, model_used = generate_content_with_retry(
        _client,
        model=GEMINI_ANALYST_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    data = _extract_json(_response_text(response))
    if not data or not isinstance(data.get("results"), list):
        raise RuntimeError("Batch verifier returned invalid JSON")

    out: dict[str, dict] = {}
    for row in data["results"]:
        if not isinstance(row, dict):
            continue
        claim_id = str(row.get("claim_id") or "").strip()
        if claim_id not in attached_counts:
            continue
        normalized = dict(row)
        normalized["_attached_image_count"] = attached_counts[claim_id]
        normalized["_model"] = str(model_used or GEMINI_ANALYST_MODEL)
        out[claim_id] = normalized
    return out
