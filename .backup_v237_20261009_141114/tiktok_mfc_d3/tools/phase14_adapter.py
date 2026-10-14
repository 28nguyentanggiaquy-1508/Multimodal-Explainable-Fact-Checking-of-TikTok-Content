from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .common import append_jsonl, iter_jsonl
from .contract_guard import normalize_system_response
from .id_generator import IdGenerator

CLAIM_TYPE_MAP = {
    "statistical": "NUMERICAL",
    "numerical": "NUMERICAL",
    "event_place_person": "EVENT",
    "event": "EVENT",
    "entity": "ENTITY",
    "location": "LOCATION",
    "causal": "CAUSAL",
    "authenticity": "AUTHENTICITY",
    "out_of_context": "OUT_OF_CONTEXT",
    "other_factual": "OTHER_FACTUAL",
}

MODALITY_MAP = {
    "speech": ["SPEECH"],
    "audio": ["SPEECH"],
    "ocr": ["OCR"],
    "caption": ["CAPTION"],
    "visual": ["VISUAL"],
    "cross_modal": ["SPEECH", "VISUAL"],
    "multimodal": ["SPEECH", "VISUAL"],
}

STANCE_TO_ROLE = {
    "support": "SUPPORT",
    "supported": "SUPPORT",
    "refute": "REFUTE",
    "refuted": "REFUTE",
    "neutral": "CONTEXT",
    "context": "CONTEXT",
    "insufficient": "INSUFFICIENT",
}


def _parse_timestamp(value: Any) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    if isinstance(value, (int, float)):
        return float(value), None
    if not isinstance(value, str):
        return None, None
    m = re.fullmatch(
        r"\s*(\d{1,2}):(\d{2})(?:\.(\d+))?\s*-\s*(\d{1,2}):(\d{2})(?:\.(\d+))?\s*",
        value,
    )
    if not m:
        return None, None
    sh, sm, sf, eh, em, ef = m.groups()
    start = int(sh) * 60 + int(sm) + (float(f"0.{sf}") if sf else 0.0)
    end = int(eh) * 60 + int(em) + (float(f"0.{ef}") if ef else 0.0)
    return start, end


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        p = urlsplit(str(url).strip())
        return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path.rstrip("/"), "", ""))
    except Exception:
        return str(url).strip()


def _tiktok_video_id(url: str | None) -> str | None:
    if not url:
        return None
    m = re.search(r"/video/(\d+)", str(url))
    return m.group(1) if m else None


def _source_metadata(response: dict[str, Any]) -> dict[str, Any]:
    mm = response.get("multimedia") or response.get("multimodal") or response.get("context") or {}
    metadata = mm.get("metadata") or response.get("metadata") or {}
    if not metadata and isinstance(response.get("analysis"), dict):
        metadata = response["analysis"].get("metadata") or {}
    return metadata


def _find_existing_video(dataset_root: Path, platform_video_id: str | None, canonical_url: str | None) -> dict[str, Any] | None:
    for record in iter_jsonl(dataset_root / "records" / "videos.jsonl"):
        src = record.get("source") or {}
        existing_id = src.get("platform_video_id")
        existing_url = _canonical_url(src.get("original_url"))
        if platform_video_id and existing_id and str(existing_id) == str(platform_video_id):
            return record
        if canonical_url and existing_url and canonical_url == existing_url:
            return record
    return None




def _normalize_claim_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _existing_claim_records(dataset_root: Path, video_id: str) -> list[dict[str, Any]]:
    return [
        record
        for record in iter_jsonl(dataset_root / "records" / "claims.jsonl")
        if record.get("video_id") == video_id
    ]


def _next_claim_index(records: list[dict[str, Any]]) -> int:
    max_index = 0
    for record in records:
        claim_id = str(record.get("claim_id") or "")
        m = re.search(r"_(\d{2})$", claim_id)
        if m:
            max_index = max(max_index, int(m.group(1)))
    return max_index + 1


def _assessment_map(item: dict[str, Any]) -> dict[str, dict[str, Any]]:
    verification = item.get("verification") or {}
    result: dict[str, dict[str, Any]] = {}
    for assessment in verification.get("evidence_assessments") or []:
        evidence_id = str(assessment.get("evidence_id") or "")
        if evidence_id:
            result[evidence_id] = assessment
    return result


def _build_multimedia_record(response: dict[str, Any], video_id: str) -> dict[str, Any]:
    mm = response.get("multimedia") or response.get("multimodal") or {}
    return {
        "video_id": video_id,
        "status": mm.get("status") or response.get("status"),
        "content_type": mm.get("content_type"),
        "available_modalities": mm.get("available_modalities") or [],
        "visual_observations": mm.get("visual_observations") or [],
        "speech_segments": mm.get("speech_segments") or [],
        "speech_transcript": mm.get("speech_transcript") or "",
        "ocr_segments": mm.get("ocr_segments") or [],
        "ocr_text": mm.get("ocr_text") or "",
        "caption": mm.get("caption") or (_source_metadata(response).get("caption") or ""),
        "cross_modal_notes": mm.get("cross_modal_notes") or [],
        "claim_extraction": {
            "status": response.get("claim_extraction_status")
            or (((response.get("phase_status") or {}).get("phase_2") or {}).get("status")),
            "completed": response.get("claim_extraction_completed"),
            "no_claim_detected": response.get("no_claim_detected"),
            "candidate_claim_count": ((((response.get("phase_status") or {}).get("phase_2") or {}).get("details") or {}).get("candidate_claim_count")),
            "selected_claim_count": ((((response.get("phase_status") or {}).get("phase_2") or {}).get("details") or {}).get("selected_claim_count")),
            "skip_reason": response.get("skip_reason"),
        },
        "system": {
            "trace_id": response.get("trace_id"),
            "pipeline_version": response.get("pipeline_version"),
            "source": mm.get("source"),
            "model": mm.get("model"),
            "error": mm.get("error"),
            "degraded_reason": mm.get("degraded_reason"),
            "phase_status": response.get("phase_status") or {},
        },
    }


def import_phase14_response(
    response: dict[str, Any],
    dataset_root: Path,
    *,
    collection_context: dict[str, Any] | None = None,
    skip_duplicate_source: bool = True,
) -> dict[str, Any]:
    """Import a Phase 1-4 backend response as SYSTEM/CANDIDATE data.

    This function NEVER promotes backend output to Gold. Human annotation remains
    authoritative for gold claims, gold evidence, verdicts and explanations.
    """
    response = normalize_system_response(response)
    collection_context = collection_context or {}
    ids = IdGenerator(dataset_root / "state" / "id_counters.json")
    metadata = _source_metadata(response)

    original_url = metadata.get("webpage_url") or metadata.get("url") or response.get("video_url")
    canonical_url = _canonical_url(original_url)
    platform_video_id = metadata.get("id") or metadata.get("video_id") or _tiktok_video_id(original_url)

    existing = _find_existing_video(dataset_root, platform_video_id, canonical_url)
    source_reused = bool(existing and skip_duplicate_source)

    if source_reused:
        video_id = str(existing.get("video_id"))
    else:
        video_id = ids.video_id()
        video_record = {
            "video_id": video_id,
            "platform": "tiktok",
            "source": {
                "platform_video_id": str(platform_video_id) if platform_video_id else None,
                "original_url": original_url,
                "canonical_url": canonical_url,
                "published_at": metadata.get("upload_date"),
                "collected_at": _utc_now(),
                "language": collection_context.get("language") or "vi",
            },
            "media": {
                "video_path": collection_context.get("video_path"),
                "sha256": collection_context.get("sha256"),
                "duration_seconds": metadata.get("duration"),
            },
            "metadata": {
                "caption": metadata.get("caption") or "",
                "domain": collection_context.get("target_domain"),
                "topics": collection_context.get("topics") or [],
                "batch_id": collection_context.get("batch_id"),
                "discovery_method": collection_context.get("discovery_method"),
                "discovery_query": collection_context.get("discovery_query"),
                "backend_trace_id": response.get("trace_id"),
                "backend_pipeline_version": response.get("pipeline_version"),
                "backend_status": response.get("status"),
            },
            # Source registration is independent from successful claim extraction.
            # A video with a technical Phase-2 failure must not be marked as fully preprocessed.
            "status": "COLLECTED",
        }
        append_jsonl(dataset_root / "records" / "videos.jsonl", video_record)

    # Keep run-level multimedia/system output even when the source already exists.
    # This allows a later successful run to enrich a source that previously failed.
    append_jsonl(dataset_root / "records" / "multimedia.jsonl", _build_multimedia_record(response, video_id))

    existing_claim_records = _existing_claim_records(dataset_root, video_id)
    existing_claim_texts = {
        _normalize_claim_text(record.get("normalized_text") or record.get("original_text"))
        for record in existing_claim_records
    }
    next_claim_index = _next_claim_index(existing_claim_records)

    imported_claims: list[str] = []
    imported_evidence: list[str] = []
    contract_violations: list[dict[str, Any]] = []

    skipped_duplicate_claims: list[str] = []
    for item in response.get("claims") or []:
        src = item.get("claim") or {}
        claim_text_key = _normalize_claim_text(src.get("text"))
        if claim_text_key and claim_text_key in existing_claim_texts:
            skipped_duplicate_claims.append(str(src.get("id") or src.get("text") or ""))
            continue

        index = next_claim_index
        next_claim_index += 1
        claim_id = ids.claim_id(video_id, index)
        if claim_text_key:
            existing_claim_texts.add(claim_text_key)
        source_modality = str(src.get("source_modality") or "").lower()
        modalities = MODALITY_MAP.get(source_modality, ["CAPTION"] if source_modality == "caption" else [])
        start_time, end_time = _parse_timestamp(src.get("timestamp"))

        claim_record = {
            "claim_id": claim_id,
            "video_id": video_id,
            "original_text": src.get("text") or "",
            "normalized_text": src.get("text") or "",
            "source": {
                "modalities": modalities,
                "primary_modality": modalities[0] if modalities else None,
                "start_time": start_time,
                "end_time": end_time,
                "frame_ids": [],
                "transcript_span_ids": [],
            },
            "annotation": {
                "checkable": src.get("checkable"),
                "checkworthy": src.get("check_worthy"),
                "claim_type": CLAIM_TYPE_MAP.get(str(src.get("claim_type") or "").lower(), "OTHER_FACTUAL"),
                "requires_multimodal_context": source_modality in {"cross_modal", "multimodal"},
                "claim_origin": "REAL_TIKTOK",
                "annotation_status": "CANDIDATE_SYSTEM_OUTPUT",
            },
            "system": {
                "backend_claim_id": src.get("id"),
                "rationale": src.get("rationale"),
                "source_modality_raw": src.get("source_modality"),
                "backend_trace_id": response.get("trace_id"),
            },
        }
        append_jsonl(dataset_root / "records" / "claims.jsonl", claim_record)
        imported_claims.append(claim_id)

        # Import candidate evidence from current backend without promoting it to Gold.
        assessment_by_backend_id = _assessment_map(item)
        backend_to_dataset_evidence: dict[str, str] = {}
        document_ids: dict[str, str] = {}

        for ev in item.get("evidence") or []:
            modality = str(ev.get("modality") or "text").lower()
            if modality not in {"text", "image"}:
                continue
            evidence_id = ids.evidence_id("TEXT" if modality == "text" else "IMAGE")
            backend_id = str(ev.get("id") or "")
            if backend_id:
                backend_to_dataset_evidence[backend_id] = evidence_id
            imported_evidence.append(evidence_id)

            source_url = ev.get("source_url")
            if source_url and source_url not in document_ids:
                document_ids[source_url] = ids.document_id()
            assessment = assessment_by_backend_id.get(backend_id) or {}

            evidence_record = {
                "evidence_id": evidence_id,
                "claim_id": claim_id,
                "type": modality,
                "source": {
                    "source_type": ev.get("source_quality") or "unclassified",
                    "publisher": ev.get("source_domain") or "",
                    "title": ev.get("title") or "",
                    "url": source_url,
                    "published_at": ev.get("published_at") or (ev.get("metadata") or {}).get("source_date"),
                    "retrieved_at": ev.get("retrieved_at"),
                    "language": collection_context.get("evidence_language") or "vi",
                },
                "content": {
                    "document_id": document_ids.get(source_url) if source_url else None,
                    "text": ev.get("content") or "",
                    "image_path": None,
                    "media_url": ev.get("media_url"),
                    "caption": ev.get("content") if modality == "image" else None,
                },
                "location": {
                    "paragraph_index": (ev.get("metadata") or {}).get("paragraph_index"),
                    "sentence_start": None,
                    "sentence_end": None,
                },
                "annotation": {
                    "is_relevant": None,
                    "is_gold": False,
                    "evidence_role": None,
                },
                "system": {
                    "backend_evidence_id": backend_id or None,
                    "retrieval_score": ev.get("retrieval_score"),
                    "rerank_score": ev.get("rerank_score"),
                    "relevance_score": ev.get("relevance_score"),
                    "provider": ev.get("provider"),
                    "source_quality": ev.get("source_quality"),
                    "retrieval_metadata": ev.get("metadata") or {},
                    "predicted_stance": assessment.get("stance"),
                    "predicted_role": STANCE_TO_ROLE.get(str(assessment.get("stance") or "").lower()),
                    "stance_confidence": assessment.get("confidence"),
                    "stance_rationale": assessment.get("rationale"),
                },
            }
            append_jsonl(dataset_root / "records" / "evidence.jsonl", evidence_record)

        verification = item.get("verification") or {}
        violations = verification.get("contract_violations") or []
        contract_violations.extend({"claim_id": claim_id, "message": v} for v in violations)
        backend_evidence_ids = verification.get("evidence_ids") or []
        mapped_evidence_ids = [
            backend_to_dataset_evidence[eid]
            for eid in backend_evidence_ids
            if eid in backend_to_dataset_evidence
        ]
        system_verification_record = {
            "verification_id": ids.verification_id(video_id, index),
            "claim_id": claim_id,
            "gold_verdict": None,
            "gold_evidence_ids": [],
            "evidence_coverage": None,
            "evidence_conflict": False,
            "verification_time": None,
            "annotation_status": "PENDING",
            "needs_adjudication": False,
            "system": {
                "predicted_verdict": verification.get("verdict"),
                "raw_verdict": verification.get("raw_verdict"),
                "confidence": verification.get("confidence"),
                "system_status": verification.get("system_status") or verification.get("status"),
                "evidence_ids": mapped_evidence_ids,
                "backend_evidence_ids": backend_evidence_ids,
                "evidence_assessments": verification.get("evidence_assessments") or [],
                "rationale": verification.get("rationale"),
                "uncertainty": verification.get("uncertainty"),
                "limitations": verification.get("limitations") or [],
                "multimodal_used": verification.get("multimodal_used"),
                "model": verification.get("model"),
                "contract_violations": violations,
            },
        }
        append_jsonl(dataset_root / "records" / "verifications.jsonl", system_verification_record)

        system_reasoning = item.get("reasoning_support") or {}
        explanation = item.get("explanation")
        explanation_record = {
            "explanation_id": ids.explanation_id(video_id, index),
            "claim_id": claim_id,
            "gold": {
                "verdict": None,
                "text": "",
                "evidence_ids": [],
                "evidence_attribution": [],
            },
            "quality": {
                "faithful": None,
                "verdict_consistent": None,
                "evidence_complete": None,
                "contains_unsupported_claim": None,
                "review_status": "PENDING",
            },
            "system": {
                "key_concepts": system_reasoning.get("key_concepts") or [],
                "reasoning_support": system_reasoning.get("reasoning_support") or "",
                "semantic_consistency": system_reasoning.get("semantic_consistency") or {},
                "generated_explanation": explanation,
            },
        }
        append_jsonl(dataset_root / "records" / "explanations.jsonl", explanation_record)

    return {
        "video_id": video_id,
        "duplicate_source": source_reused,
        "source_reused": source_reused,
        "imported_claim_ids": imported_claims,
        "imported_evidence_ids": imported_evidence,
        "skipped_duplicate_claims": skipped_duplicate_claims,
        "claim_extraction_status": response.get("claim_extraction_status"),
        "claim_extraction_completed": response.get("claim_extraction_completed"),
        "no_claim_detected": response.get("no_claim_detected"),
        "contract_violations": contract_violations,
        "message": (
            "Existing source reused; new system outputs were merged without duplicating the video."
            if source_reused
            else "New TikTok source registered and system outputs imported."
        ),
    }
