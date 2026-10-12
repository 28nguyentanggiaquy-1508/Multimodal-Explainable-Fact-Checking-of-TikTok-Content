from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Iterable

import requests

from core.vlm import circuit
from core.vlm.exceptions import VLMResponseError, VLMUnavailableError

from .base import ProviderJSONResult

provider_name = "local"


def _enabled() -> bool:
    return os.getenv("LOCAL_VLM_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def _model() -> str:
    return os.getenv("LOCAL_VLM_MODEL", "qwen2.5vl:7b").strip()


def _base_url() -> str:
    return os.getenv("LOCAL_VLM_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def _style() -> str:
    return os.getenv("LOCAL_VLM_API_STYLE", "ollama").strip().lower()


def is_configured() -> bool:
    return _enabled() and bool(_model()) and bool(_base_url())


def configured_model() -> str:
    return _model()


def _json_from_text(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(raw[start : end + 1])
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    raise VLMResponseError("Local VLM did not return a valid JSON object")


def _image_b64_from_path(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _image_b64_from_data_url(value: str) -> str:
    text = str(value or "")
    if text.startswith("data:") and "," in text:
        return text.split(",", 1)[1]
    raise ValueError("local provider accepts image data URLs or local image paths only")


def generate_json(
    *,
    model: str | None = None,
    prompt: str,
    image_paths: Iterable[str | Path] | None = None,
    image_urls: Iterable[str] | None = None,
    max_output_tokens: int = 1200,
    timeout_seconds: float = 180.0,
    image_detail: str = "low",
) -> ProviderJSONResult:
    del image_detail
    if not is_configured():
        raise VLMUnavailableError("LOCAL_VLM_ENABLED is false or local VLM endpoint/model is not configured")
    model = (model or _model()).strip()
    if circuit.is_open(provider_name, model):
        remain = circuit.remaining_seconds(provider_name, model)
        raise VLMUnavailableError(f"Local VLM circuit open for model={model}; remaining~{remain:.0f}s")

    images: list[str] = []
    for path in image_paths or []:
        images.append(_image_b64_from_path(path))
    for value in image_urls or []:
        if str(value).startswith("data:"):
            images.append(_image_b64_from_data_url(str(value)))

    try:
        if _style() == "openai":
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for encoded in images:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
            body = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": int(max_output_tokens),
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
            key = os.getenv("LOCAL_VLM_API_KEY", "local").strip() or "local"
            response = requests.post(
                f"{_base_url()}/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=body,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            text = payload["choices"][0]["message"]["content"]
        else:
            body = {
                "model": model,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "num_predict": int(max_output_tokens)},
                "messages": [{"role": "user", "content": prompt, "images": images}],
            }
            response = requests.post(f"{_base_url()}/api/chat", json=body, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            text = ((payload.get("message") or {}).get("content") or "")
    except requests.RequestException as exc:
        circuit.open_circuit(provider_name, model, seconds=30, reason=str(exc))
        raise VLMUnavailableError(f"Local VLM request failed: {type(exc).__name__}: {exc}") from exc

    data = _json_from_text(text)
    circuit.close_circuit(provider_name, model)
    return ProviderJSONResult(data=data, provider=provider_name, model=model)
