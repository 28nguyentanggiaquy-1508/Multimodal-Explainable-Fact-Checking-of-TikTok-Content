from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Iterable

import requests

from . import circuit, quota_manager
from .exceptions import VLMQuotaError, VLMResponseError, VLMUnavailableError


OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def _api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def is_configured() -> bool:
    return bool(_api_key())


def _headers() -> dict[str, str]:
    key = _api_key()
    if not key:
        raise VLMUnavailableError("OPENAI_API_KEY is not configured")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _raise_http_error(response: requests.Response, *, provider: str, model: str) -> None:
    if response.ok:
        return

    body = response.text[:8000]
    message = f"HTTP {response.status_code}: {body}"

    if response.status_code == 429:
        retry = response.headers.get("Retry-After")
        try:
            retry_seconds = float(retry) if retry else circuit.parse_retry_after_seconds(body, 300)
        except Exception:
            retry_seconds = 300.0
        circuit.open_circuit(provider, model, seconds=retry_seconds, reason=message)
        quota_manager.record_quota_error(provider, model, message, retry_seconds)
        raise VLMQuotaError(message, retry_after_seconds=retry_seconds)

    if response.status_code in {401, 403, 404}:
        raise VLMUnavailableError(message)

    if 500 <= response.status_code <= 599:
        raise VLMUnavailableError(message)

    raise VLMResponseError(message)


def _extract_output_text(payload: Any) -> str:
    """Extract textual model output from Responses API defensively.

    Some proxies/gateways or tests may return a JSON string instead of the
    normal Response object. This function must never assume payload is a dict.
    """
    if payload is None:
        return ""

    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8", errors="replace")
        except Exception:
            return ""

    if isinstance(payload, str):
        text = payload.strip()
        # If the HTTP body is a JSON-encoded string/object, unwrap once.
        try:
            decoded = json.loads(text)
            if decoded is not payload:
                if isinstance(decoded, str):
                    return decoded.strip()
                if isinstance(decoded, (dict, list)):
                    return _extract_output_text(decoded)
        except Exception:
            pass
        return text

    if isinstance(payload, list):
        parts = [_extract_output_text(x) for x in payload]
        return "\n".join(x for x in parts if x).strip()

    if not isinstance(payload, dict):
        return str(payload).strip()

    # Some SDK-like gateways may expose output_text directly.
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    # If a gateway directly returns the model JSON object instead of a
    # Responses envelope, serialize it back to text for the JSON parser.
    direct_json_markers = {
        "candidate_claims", "content_type", "visual_observations",
        "verdict", "evidence_ids", "evidence_assessments",
    }
    if direct_json_markers.intersection(payload.keys()):
        return json.dumps(payload, ensure_ascii=False)

    texts: list[str] = []
    for item in payload.get("output") or []:
        if isinstance(item, str):
            if item.strip():
                texts.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        content_value = item.get("content") or []
        if isinstance(content_value, str):
            if content_value.strip():
                texts.append(content_value.strip())
            continue
        for content in content_value:
            if isinstance(content, str):
                if content.strip():
                    texts.append(content.strip())
                continue
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"}:
                value = content.get("text")
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
    return "\n".join(texts).strip()


def _json_from_text(text: str) -> dict[str, Any]:
    if not text:
        raise VLMResponseError("OpenAI response contained no text")
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    raise VLMResponseError("OpenAI response did not contain a valid JSON object")


def file_to_data_url(path: str | Path) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
    encoded = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def generate_json(
    *,
    model: str,
    prompt: str,
    image_paths: Iterable[str | Path] | None = None,
    image_urls: Iterable[str] | None = None,
    max_output_tokens: int = 3500,
    timeout_seconds: float = 120.0,
    image_detail: str = "low",
) -> dict[str, Any]:
    if circuit.is_open("openai", model):
        remain = circuit.remaining_seconds("openai", model)
        raise VLMQuotaError(
            f"OpenAI circuit is open for model={model}; remaining~{remain:.0f}s",
            retry_after_seconds=remain,
        )

    # Normalize iterables once because generators would otherwise be consumed by counting.
    image_paths = list(image_paths or [])
    image_urls = list(image_urls or [])
    image_count = len(image_paths) + len(image_urls)
    estimated_input = quota_manager.estimate_input_tokens(prompt, image_count=image_count)
    decision = quota_manager.can_request(
        "openai",
        model,
        estimated_input_tokens=estimated_input,
        max_output_tokens=int(max_output_tokens),
    )
    if not decision.allowed:
        raise VLMQuotaError(
            "OpenAI quota preflight blocked request: "
            f"reason={decision.reason} estimated_request_tokens={decision.estimated_request_tokens} "
            f"remaining_tokens={decision.remaining_tokens}",
            retry_after_seconds=decision.retry_after_seconds,
        )

    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]

    for path in image_paths or []:
        content.append(
            {
                "type": "input_image",
                "image_url": file_to_data_url(path),
                "detail": image_detail,
            }
        )

    for url in image_urls or []:
        if str(url).strip():
            content.append(
                {
                    "type": "input_image",
                    "image_url": str(url).strip(),
                    "detail": image_detail,
                }
            )

    reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "low").strip().lower() or "low"

    # Responses API JSON mode: force syntactically valid JSON.
    # This generic adapter is shared by Phase 2 and Phase 4, so JSON mode is
    # used here; phase-specific strict JSON Schema can be added later.
    body = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": int(max_output_tokens),
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "format": {
                "type": "json_object"
            }
        },
    }

    response = requests.post(
        f"{OPENAI_BASE_URL}/responses",
        headers=_headers(),
        json=body,
        timeout=float(timeout_seconds),
    )
    quota_manager.record_response_headers("openai", model, response.headers)
    _raise_http_error(response, provider="openai", model=model)

    try:
        payload: Any = response.json()
    except Exception:
        payload = response.text
    if isinstance(payload, dict):
        status = str(payload.get("status") or "").strip().lower()
        if status == "incomplete":
            details = payload.get("incomplete_details")
            raise VLMResponseError(
                f"OpenAI response incomplete: {details!r}"
            )
        if status == "failed":
            error = payload.get("error")
            raise VLMResponseError(
                f"OpenAI response failed: {error!r}"
            )

    text = _extract_output_text(payload)
    try:
        data = _json_from_text(text)
    except VLMResponseError as exc:
        preview = text[:2000].replace("\n", "\\n")
        raise VLMResponseError(
            f"{exc}; output_preview={preview!r}"
        ) from exc

    circuit.close_circuit("openai", model)
    quota_manager.record_success("openai", model)
    return data


def transcribe_media(
    path: str | Path,
    *,
    model: str | None = None,
    timeout_seconds: float = 120.0,
) -> str:
    """Transcribe audio contained in an MP4/MP3/etc. using OpenAI Audio API."""
    model = (model or os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")).strip()
    if not model:
        return ""

    if circuit.is_open("openai-transcribe", model):
        remain = circuit.remaining_seconds("openai-transcribe", model)
        raise VLMQuotaError(
            f"OpenAI transcription circuit open; remaining~{remain:.0f}s",
            retry_after_seconds=remain,
        )

    key = _api_key()
    if not key:
        raise VLMUnavailableError("OPENAI_API_KEY is not configured")

    p = Path(path)
    headers = {"Authorization": f"Bearer {key}"}
    with p.open("rb") as fh:
        response = requests.post(
            f"{OPENAI_BASE_URL}/audio/transcriptions",
            headers=headers,
            files={"file": (p.name, fh, mimetypes.guess_type(str(p))[0] or "video/mp4")},
            data={"model": model},
            timeout=float(timeout_seconds),
        )

    if not response.ok:
        body = response.text[:8000]
        if response.status_code == 429:
            retry = response.headers.get("Retry-After")
            try:
                retry_seconds = float(retry) if retry else circuit.parse_retry_after_seconds(body, 300)
            except Exception:
                retry_seconds = 300.0
            circuit.open_circuit(
                "openai-transcribe",
                model,
                seconds=retry_seconds,
                reason=body,
            )
            quota_manager.record_quota_error("openai-transcribe", model, body, retry_seconds)
            raise VLMQuotaError(body, retry_after_seconds=retry_seconds)
        raise VLMUnavailableError(f"OpenAI transcription failed: HTTP {response.status_code}: {body}")

    try:
        data: Any = response.json()
    except Exception:
        data = response.text

    if isinstance(data, dict):
        text = str(data.get("text") or "").strip()
    elif isinstance(data, str):
        raw = data.strip()
        try:
            decoded = json.loads(raw)
        except Exception:
            decoded = None
        if isinstance(decoded, dict):
            text = str(decoded.get("text") or "").strip()
        elif isinstance(decoded, str):
            text = decoded.strip()
        else:
            text = raw
    else:
        text = ""

    circuit.close_circuit("openai-transcribe", model)
    return text
