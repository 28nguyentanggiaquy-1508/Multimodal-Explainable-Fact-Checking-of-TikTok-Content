from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any


class LocalSTTUnavailable(RuntimeError):
    """Local speech-to-text component cannot be used in this environment."""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _cache_dir() -> Path:
    # Keep the model cache inside the backend/project by default so it does not
    # depend on any previously broken global Hugging Face cache path.
    default = Path(__file__).resolve().parents[2] / ".cache" / "faster_whisper"
    path = Path(os.getenv("FASTER_WHISPER_CACHE_DIR", str(default))).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _settings() -> tuple[str, str, str, int, int]:
    return (
        os.getenv("FASTER_WHISPER_MODEL", "small").strip() or "small",
        os.getenv("FASTER_WHISPER_DEVICE", "cpu").strip() or "cpu",
        os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "int8").strip() or "int8",
        int(os.getenv("FASTER_WHISPER_CPU_THREADS", "0")),
        int(os.getenv("FASTER_WHISPER_NUM_WORKERS", "1")),
    )


@lru_cache(maxsize=2)
def _load_model(
    model_name: str,
    device: str,
    compute_type: str,
    cpu_threads: int,
    num_workers: int,
):
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise LocalSTTUnavailable(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from exc

    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
        download_root=str(_cache_dir()),
    )


def _timestamp(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def transcribe_local(path: str | Path) -> dict[str, Any]:
    """Transcribe media locally with faster-whisper.

    Returns a project-friendly structure with both full transcript and segment
    timestamps. The model is cached in-process and loaded once.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Media file not found: {p}")

    model_name, device, compute_type, cpu_threads, num_workers = _settings()
    model = _load_model(
        model_name,
        device,
        compute_type,
        cpu_threads,
        num_workers,
    )

    language = os.getenv("FASTER_WHISPER_LANGUAGE", "vi").strip() or None
    beam_size = max(1, int(os.getenv("FASTER_WHISPER_BEAM_SIZE", "5")))
    vad_filter = _env_bool("FASTER_WHISPER_VAD_FILTER", True)
    condition_on_previous_text = _env_bool(
        "FASTER_WHISPER_CONDITION_ON_PREVIOUS_TEXT",
        False,
    )

    segments_iter, info = model.transcribe(
        str(p),
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        condition_on_previous_text=condition_on_previous_text,
        word_timestamps=False,
    )

    segments: list[dict[str, Any]] = []
    transcript_parts: list[str] = []

    # faster-whisper returns a generator; inference happens while iterating.
    for segment in segments_iter:
        text = str(getattr(segment, "text", "") or "").strip()
        if not text:
            continue
        start = float(getattr(segment, "start", 0.0) or 0.0)
        end = float(getattr(segment, "end", start) or start)
        transcript_parts.append(text)
        segments.append(
            {
                "timestamp": f"{_timestamp(start)}-{_timestamp(end)}",
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
            }
        )

    detected_language = str(getattr(info, "language", "") or "")
    language_probability = getattr(info, "language_probability", None)
    duration = getattr(info, "duration", None)

    return {
        "transcript": " ".join(transcript_parts).strip(),
        "segments": segments,
        "provider": "faster-whisper",
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "language": detected_language or language,
        "language_probability": (
            float(language_probability)
            if language_probability is not None
            else None
        ),
        "duration": float(duration) if duration is not None else None,
    }
