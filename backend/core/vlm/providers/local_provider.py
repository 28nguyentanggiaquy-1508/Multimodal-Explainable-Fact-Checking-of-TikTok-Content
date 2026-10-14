from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Iterable, Literal

import requests

from core.vlm import circuit
from core.vlm.exceptions import VLMResponseError, VLMUnavailableError

from .base import ProviderJSONResult

provider_name = "local"
TaskName = Literal["phase2", "phase4"]


def _enabled() -> bool:
    return os.getenv("LOCAL_VLM_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def configured_model(task: TaskName | None = None) -> str:
    """Return the task-specific local model, preserving LOCAL_VLM_MODEL fallback."""
    fallback = (os.getenv("LOCAL_VLM_MODEL") or "qwen3-vl:4b-instruct-q4_K_M").strip()
    if task == "phase2":
        return (os.getenv("LOCAL_PHASE2_MODEL") or fallback).strip()
    if task == "phase4":
        return (os.getenv("LOCAL_PHASE4_MODEL") or fallback).strip()
    return fallback


def _base_url() -> str:
    return os.getenv("LOCAL_VLM_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def _style() -> str:
    return os.getenv("LOCAL_VLM_API_STYLE", "ollama").strip().lower()


def is_configured(task: TaskName | None = None) -> bool:
    return _enabled() and bool(configured_model(task)) and bool(_base_url())


def health_check(task: TaskName | None = None, *, timeout_seconds: float = 5.0) -> dict[str, Any]:
    """Check the local Ollama endpoint and whether the configured model is installed.

    This does not download or run a model. It is safe to use as a readiness check.
    """
    model = configured_model(task)
    result: dict[str, Any] = {
        "enabled": _enabled(),
        "base_url": _base_url(),
        "api_style": _style(),
        "model": model,
        "reachable": False,
        "model_installed": False,
        "installed_models": [],
    }
    if not _enabled():
        result["reason"] = "LOCAL_VLM_ENABLED=false"
        return result
    if _style() != "ollama":
        result["reason"] = "health_check currently supports LOCAL_VLM_API_STYLE=ollama"
        return result
    try:
        response = requests.get(f"{_base_url()}/api/tags", timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        names = [str(item.get("name") or item.get("model") or "") for item in payload.get("models", [])]
        names = [name for name in names if name]
        result["reachable"] = True
        result["installed_models"] = names
        # Ollama may return a canonical name with :latest. Allow exact/base matching.
        requested_base = model.split(":", 1)[0]
        result["model_installed"] = any(
            name == model or name.split(":", 1)[0] == requested_base for name in names
        )
        if not result["model_installed"]:
            result["reason"] = f"configured model is not installed: {model}"
        return result
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result


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


def _even_sample(values: list[str], limit: int) -> list[str]:
    if limit <= 0 or len(values) <= limit:
        return values
    if limit == 1:
        return [values[len(values) // 2]]
    # Preserve first/last frames while sampling the timeline evenly.
    indices = [round(i * (len(values) - 1) / (limit - 1)) for i in range(limit)]
    return [values[index] for index in indices]


def generate_json(
    *,
    model: str | None = None,
    prompt: str,
    image_paths: Iterable[str | Path] | None = None,
    image_urls: Iterable[str] | None = None,
    max_output_tokens: int = 1200,
    timeout_seconds: float = 180.0,
    image_detail: str = "low",
    response_schema: dict[str, Any] | None = None,
) -> ProviderJSONResult:
    del image_detail
    if not is_configured():
        raise VLMUnavailableError("LOCAL_VLM_ENABLED is false or local VLM endpoint/model is not configured")

    model = (model or configured_model()).strip()
    if circuit.is_open(provider_name, model):
        remain = circuit.remaining_seconds(provider_name, model)
        raise VLMUnavailableError(f"Local VLM circuit open for model={model}; remaining~{remain:.0f}s")

    images: list[str] = []
    for path in image_paths or []:
        images.append(_image_b64_from_path(path))
    for value in image_urls or []:
        if str(value).startswith("data:"):
            images.append(_image_b64_from_data_url(str(value)))

    max_images = max(1, int(os.getenv("LOCAL_VLM_MAX_IMAGES", "6")))
    images = _even_sample(images, max_images)
    json_attempts = max(1, int(os.getenv("LOCAL_VLM_JSON_RETRIES", "2")))

    try:
        if _style() == "openai":
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for encoded in images:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
            body: dict[str, Any] = {
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
            try:
                data = _json_from_text(text)
            except VLMResponseError as exc:
                preview = str(text or "")[:1000].replace("\n", "\\n")
                raise VLMResponseError(
                    f"Local OpenAI-compatible VLM did not return valid JSON; chars={len(str(text or ''))}; preview={preview!r}"
                ) from exc
        else:
            options: dict[str, Any] = {
                "temperature": float(os.getenv("LOCAL_VLM_TEMPERATURE", "0")),
                "num_predict": int(max_output_tokens),
                "num_ctx": int(os.getenv("LOCAL_VLM_NUM_CTX", "8192")),
            }
            num_gpu = os.getenv("LOCAL_VLM_NUM_GPU", "").strip()
            if num_gpu:
                options["num_gpu"] = int(num_gpu)

            current_prompt = prompt
            last_parse_error: Exception | None = None
            data: dict[str, Any] | None = None
            for json_attempt in range(1, json_attempts + 1):
                body = {
                    "model": model,
                    "stream": False,
                    # Ollama accepts a JSON Schema object here. This is more
                    # reliable than plain "json" mode for long Phase 4 payloads.
                    "format": response_schema if response_schema else "json",
                    "keep_alive": os.getenv("LOCAL_VLM_KEEP_ALIVE", "10m"),
                    "options": options,
                    "messages": [{"role": "user", "content": current_prompt, "images": images}],
                }
                response = requests.post(
                    f"{_base_url()}/api/chat",
                    json=body,
                    timeout=timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                text = ((payload.get("message") or {}).get("content") or "")
                try:
                    data = _json_from_text(text)
                    break
                except VLMResponseError as exc:
                    last_parse_error = exc
                    if json_attempt >= json_attempts:
                        done_reason = str(payload.get("done_reason") or payload.get("done") or "unknown")
                        preview = str(text or "")[:1200].replace("\n", "\\n")
                        raise VLMResponseError(
                            "Local VLM did not return a valid JSON object; "
                            f"attempts={json_attempts}; done_reason={done_reason}; "
                            f"chars={len(str(text or ''))}; preview={preview!r}"
                        ) from exc
                    # Retry locally only; do not mutate evidence or invent a verdict.
                    current_prompt = (
                        prompt
                        + "\n\n# JSON RETRY\n"
                        + "Your previous response was not valid JSON. Return ONLY one compact JSON object "
                        + "matching the requested schema. Keep rationales brief and close all arrays/objects."
                    )
            if data is None:
                raise VLMResponseError(f"Local VLM JSON parsing failed: {last_parse_error}")
    except requests.RequestException as exc:
        circuit.open_circuit(provider_name, model, seconds=30, reason=str(exc))
        raise VLMUnavailableError(f"Local VLM request failed: {type(exc).__name__}: {exc}") from exc
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VLMResponseError(f"Local VLM response envelope is invalid: {type(exc).__name__}: {exc}") from exc

    circuit.close_circuit(provider_name, model)
    return ProviderJSONResult(data=data, provider=provider_name, model=model)
