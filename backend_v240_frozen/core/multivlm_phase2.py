from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from core.gemini_vision import (
    _build_fact_check_multimedia_prompt,
    _extract_tiktok_post_metadata,
    _normalize_fact_check_payload,
    analyze_tiktok_multimedia as _legacy_gemini_analyze,
    get_or_download_video,
)
from core.vlm import cache, circuit
from core.vlm.local_stt import transcribe_local
from core.vlm.media import sample_video_frames
from core.vlm.providers import local_provider, openai_provider
from core.vlm.router import csv_order


def _provider_order() -> list[str]:
    return csv_order(os.getenv("VLM_PROVIDER_ORDER"), "local,openai,gemini")


def _openai_models() -> list[str]:
    raw = os.getenv("OPENAI_VLM_MODELS", os.getenv("OPENAI_VLM_MODEL", "gpt-6-luna"))
    models = [x.strip() for x in raw.split(",") if x.strip()]
    return models or ["gpt-6-luna"]


def _gemini_models() -> list[str]:
    models = [
        os.getenv("GEMINI_VISION_MODEL", "gemini-3.8-flash").strip(),
        os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.7-flash").strip(),
    ]
    return list(dict.fromkeys(model for model in models if model))


def _format_timestamp(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _safe_tiktok_metadata(video_url: str) -> tuple[dict[str, Any], str | None]:
    try:
        raw = _extract_tiktok_post_metadata(video_url)
    except Exception as exc:
        return {
            "caption": "",
            "uploader": "",
            "uploader_id": "",
            "upload_date": "",
            "duration": None,
            "webpage_url": video_url,
        }, f"{type(exc).__name__}: {exc}"
    if not isinstance(raw, dict):
        return {
            "caption": str(raw or "").strip(),
            "uploader": "",
            "uploader_id": "",
            "upload_date": "",
            "duration": None,
            "webpage_url": video_url,
        }, f"metadata_type={type(raw).__name__}"
    metadata = dict(raw)
    metadata["caption"] = str(metadata.get("caption") or "").strip()
    metadata["webpage_url"] = str(metadata.get("webpage_url") or video_url).strip()
    return metadata, None


def _stage_error(stage: str, exc: Exception) -> RuntimeError:
    return RuntimeError(f"{stage}: {type(exc).__name__}: {exc}")


def _cache_key(source_hash: str, provider: str, model: str) -> str:
    return cache.build_key(
        task="phase2_multimedia",
        provider=provider,
        model=model,
        prompt_version=os.getenv("PHASE2_PROMPT_VERSION", "v2.4.0-sampled-frames-local-stt"),
        payload={
            "source_sha256": source_hash,
            "frame_count": int(os.getenv("OPENAI_PHASE2_FRAME_COUNT", "10")),
            "local_stt_enabled": os.getenv("LOCAL_STT_ENABLED", "true"),
            "local_stt_model": os.getenv("FASTER_WHISPER_MODEL", "medium"),
        },
    )


def _load_phase2_cache(source_hash: str, provider: str, model: str) -> dict[str, Any] | None:
    key = _cache_key(source_hash, provider, model)
    result = cache.get_json(task="phase2_multimedia", key=key)
    if result:
        result = dict(result)
        metadata = dict(result.get("metadata") or {})
        metadata["phase2_cache_hit"] = True
        metadata["phase2_cache_provider"] = provider
        metadata["phase2_cache_model"] = model
        result["metadata"] = metadata
        return result
    return None


def _store_phase2_cache(source_hash: str, provider: str, model: str, result: dict[str, Any]) -> None:
    key = _cache_key(source_hash, provider, model)
    cache.set_json(
        task="phase2_multimedia",
        key=key,
        value=result,
        metadata={"provider": provider, "model": model, "source_sha256": source_hash},
    )


def _prepare_local_inputs(
    *,
    video_url: str,
    video_path: Path,
    metadata: dict[str, Any],
    metadata_error: str | None,
) -> dict[str, Any]:
    frame_count = int(os.getenv("OPENAI_PHASE2_FRAME_COUNT", "10"))
    try:
        sampled = sample_video_frames(video_path, count=frame_count)
    except Exception as exc:
        raise _stage_error("phase2_frame_sampling", exc) from exc
    frame_paths = [path for path, _ in sampled]

    transcript = ""
    transcript_segments: list[dict[str, Any]] = []
    transcription_error = None
    transcription_provider = "unavailable"
    transcription_model = ""

    local_stt_enabled = os.getenv("LOCAL_STT_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    if local_stt_enabled:
        try:
            stt = transcribe_local(video_path)
            transcript = str(stt.get("transcript") or "").strip()
            transcript_segments = [
                item
                for item in (stt.get("segments") or [])
                if isinstance(item, dict) and str(item.get("text") or "").strip()
            ]
            if transcript:
                transcription_provider = str(stt.get("provider") or "faster-whisper")
                transcription_model = str(stt.get("model") or "")
        except Exception as exc:
            transcription_error = f"local_stt: {type(exc).__name__}: {exc}"

    timestamp_legend = "\n".join(
        f"- image #{idx + 1}: ~{_format_timestamp(ts)}"
        for idx, (_, ts) in enumerate(sampled)
    )
    base_prompt = _build_fact_check_multimedia_prompt(metadata)
    prompt = f"""
{base_prompt}

# MULTI-VLM ADAPTER CONTEXT
You are not receiving the raw video file. You receive uniformly sampled frames,
plus the TikTok caption and an automatic speech transcript when available.
Use only these supplied observations. Do not invent missing speech, OCR, dates,
names, or events.

FRAME TIMESTAMP LEGEND:
{timestamp_legend}

AUTOMATIC SPEECH TRANSCRIPT:
{transcript or '(transcript unavailable)'}

TRANSCRIPTION ERROR (engineering metadata; do not treat as content):
{transcription_error or '(none)'}

METADATA EXTRACTION ERROR (engineering metadata; do not treat as content):
{metadata_error or '(none)'}

Return ONLY one JSON object matching the schema requested above.
""".strip()
    return {
        "prompt": prompt,
        "frame_paths": frame_paths,
        "transcript": transcript,
        "transcript_segments": transcript_segments,
        "transcription_error": transcription_error,
        "transcription_provider": transcription_provider,
        "transcription_model": transcription_model,
        "sampled_count": len(sampled),
    }


def _normalize_provider_result(
    *,
    data: dict[str, Any],
    metadata: dict[str, Any],
    provider: str,
    model: str,
    prepared: dict[str, Any] | None,
    metadata_error: str | None,
) -> dict[str, Any]:
    result = _normalize_fact_check_payload(data, metadata)
    if prepared:
        transcript = str(prepared.get("transcript") or "")
        if transcript:
            result["speech_transcript"] = transcript
            result["speech_segments"] = prepared.get("transcript_segments") or [{"timestamp": None, "text": transcript}]
    result["source"] = f"{provider}_multimodal"
    result["provider"] = provider
    result["model"] = model
    result["error"] = None
    result.setdefault("metadata", metadata)
    result["metadata"]["phase2_frame_count"] = prepared.get("sampled_count", 0) if prepared else None
    if prepared:
        result["metadata"]["transcription_provider"] = prepared.get("transcription_provider")
        result["metadata"]["transcription_model"] = prepared.get("transcription_model") or None
        result["metadata"]["transcription_segment_count"] = len(prepared.get("transcript_segments") or [])
        if prepared.get("transcription_error"):
            result["metadata"]["transcription_error"] = prepared["transcription_error"]
    if metadata_error:
        result["metadata"]["metadata_extraction_error"] = metadata_error
    return result


def _remember_gemini_quota(error_text: str) -> None:
    if "429" not in error_text and "RESOURCE_EXHAUSTED" not in error_text.upper():
        return
    retry = circuit.parse_retry_after_seconds(error_text, 300)
    for model in _gemini_models():
        circuit.open_circuit("gemini", model, seconds=retry, reason=error_text)


def analyze_tiktok_multimedia(video_url: str) -> dict:
    """Phase 2 cache-first, local-first-capable provider router.

    Cached successful outputs are reused before any cloud request. When cache misses,
    local STT/frame extraction runs once and can feed either a local VLM or OpenAI.
    Legacy Gemini remains an independent cloud fallback for Phase 2.
    """
    metadata, metadata_error = _safe_tiktok_metadata(video_url)
    try:
        video_path = Path(get_or_download_video(video_url))
    except Exception as exc:
        raise _stage_error("phase2_video_acquisition", exc) from exc

    try:
        source_hash = cache.file_sha256(video_path)
    except Exception:
        source_hash = str(metadata.get("id") or metadata.get("video_id") or video_url)

    # Cache lookup happens before frame sampling/STT/provider calls.
    for provider in _provider_order():
        if provider == "local" and local_provider.is_configured():
            model = local_provider.configured_model()
            cached = _load_phase2_cache(source_hash, provider, model)
            if cached:
                print(f"[MultiVLM Phase2] CACHE HIT provider={provider} model={model}")
                return cached
        elif provider == "openai" and openai_provider.is_configured():
            for model in _openai_models():
                cached = _load_phase2_cache(source_hash, provider, model)
                if cached:
                    print(f"[MultiVLM Phase2] CACHE HIT provider={provider} model={model}")
                    return cached
        elif provider == "gemini":
            for model in _gemini_models():
                cached = _load_phase2_cache(source_hash, provider, model)
                if cached:
                    print(f"[MultiVLM Phase2] CACHE HIT provider={provider} model={model}")
                    return cached

    prepared: dict[str, Any] | None = None
    failures: list[str] = []

    for provider in _provider_order():
        if provider in {"local", "openai"} and prepared is None:
            try:
                prepared = _prepare_local_inputs(
                    video_url=video_url,
                    video_path=video_path,
                    metadata=metadata,
                    metadata_error=metadata_error,
                )
            except Exception as exc:
                failures.append(f"local_preprocessing: {type(exc).__name__}: {exc}")
                prepared = None

        if provider == "local":
            if not local_provider.is_configured():
                failures.append("local: not configured")
                continue
            if prepared is None:
                failures.append("local: preprocessing unavailable")
                continue
            model = local_provider.configured_model()
            if circuit.is_open("local", model):
                failures.append(f"local: circuit_open remaining~{circuit.remaining_seconds('local', model):.0f}s")
                continue
            try:
                response = local_provider.generate_json(
                    model=model,
                    prompt=prepared["prompt"],
                    image_paths=prepared["frame_paths"],
                    max_output_tokens=int(os.getenv("LOCAL_PHASE2_MAX_OUTPUT_TOKENS", "3000")),
                    timeout_seconds=float(os.getenv("LOCAL_VLM_TIMEOUT", "180")),
                )
                result = _normalize_provider_result(
                    data=response.data,
                    metadata=metadata,
                    provider="local",
                    model=response.model,
                    prepared=prepared,
                    metadata_error=metadata_error,
                )
                _store_phase2_cache(source_hash, "local", response.model, result)
                print(f"[MultiVLM Phase2] OK provider=local model={response.model} claims={len(result.get('candidate_claims') or [])}")
                return result
            except Exception as exc:
                failures.append(f"local: {type(exc).__name__}: {exc}")
                continue

        if provider == "openai":
            if not openai_provider.is_configured():
                failures.append("openai: not configured")
                continue
            if prepared is None:
                failures.append("openai: preprocessing unavailable")
                continue
            for model in _openai_models():
                if circuit.is_open("openai", model):
                    failures.append(f"openai/{model}: circuit_open remaining~{circuit.remaining_seconds('openai', model):.0f}s")
                    continue
                try:
                    response = openai_provider.generate_json(
                        model=model,
                        prompt=prepared["prompt"],
                        image_paths=prepared["frame_paths"],
                        max_output_tokens=int(os.getenv("OPENAI_PHASE2_MAX_OUTPUT_TOKENS", "4000")),
                        timeout_seconds=float(os.getenv("OPENAI_VLM_TIMEOUT", "120")),
                        image_detail=os.getenv("OPENAI_IMAGE_DETAIL", "low"),
                    )
                    result = _normalize_provider_result(
                        data=response.data,
                        metadata=metadata,
                        provider="openai",
                        model=response.model,
                        prepared=prepared,
                        metadata_error=metadata_error,
                    )
                    _store_phase2_cache(source_hash, "openai", response.model, result)
                    print(f"[MultiVLM Phase2] OK provider=openai model={response.model} claims={len(result.get('candidate_claims') or [])}")
                    return result
                except Exception as exc:
                    failures.append(f"openai/{model}: {type(exc).__name__}: {exc}")
                    continue

        if provider == "gemini":
            if all(circuit.is_open("gemini", model) for model in _gemini_models()):
                failures.append("gemini: persistent circuit open")
                continue
            try:
                result = _legacy_gemini_analyze(video_url)
                error = str(result.get("error") or "") if isinstance(result, dict) else ""
                if error:
                    _remember_gemini_quota(error)
                    if isinstance(result, dict) and (result.get("candidate_claims") or []):
                        result["provider"] = "gemini"
                        return result
                    failures.append(f"gemini: {error}")
                    continue
                if isinstance(result, dict):
                    model = str(result.get("model") or os.getenv("GEMINI_VISION_MODEL", "gemini-3.8-flash"))
                    result["provider"] = "gemini"
                    result.setdefault("source", "gemini_multimodal")
                    result.setdefault("metadata", metadata)
                    _store_phase2_cache(source_hash, "gemini", model, result)
                    return result
            except Exception as exc:
                text = f"{type(exc).__name__}: {exc}"
                _remember_gemini_quota(text)
                failures.append(f"gemini: {text}")
                continue

    result = _normalize_fact_check_payload({}, metadata)
    if metadata_error:
        result.setdefault("metadata", metadata)
        if isinstance(result.get("metadata"), dict):
            result["metadata"]["metadata_extraction_error"] = metadata_error
    result["source"] = "multivlm_unavailable"
    result["provider"] = None
    result["model"] = ""
    result["error"] = " | ".join(failures) or "No VLM provider available"
    return result
