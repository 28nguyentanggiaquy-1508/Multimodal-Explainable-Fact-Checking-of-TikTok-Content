from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - lightweight/offline test environments
    genai = None
    types = None

from core.vlm import circuit, quota_manager
from core.vlm.exceptions import VLMQuotaError, VLMResponseError, VLMUnavailableError

from .base import ProviderJSONResult

provider_name = "gemini"


class _EvidenceAssessmentPayload(BaseModel):
    evidence_id: str
    stance: Literal["support", "refute", "neutral", "context"] = "neutral"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""


class _VerificationPayload(BaseModel):
    verdict: Literal["SUPPORTED", "REFUTED", "NEI"]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_coverage: Literal["COMPLETE", "PARTIAL", "INSUFFICIENT"]
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_assessments: list[_EvidenceAssessmentPayload] = Field(default_factory=list)
    rationale: str = ""
    uncertainty: str = ""
    limitations: list[str] = Field(default_factory=list)


def _api_key() -> str:
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def is_configured() -> bool:
    return bool(_api_key()) and genai is not None and types is not None


def _client():
    if not is_configured():
        raise VLMUnavailableError("Gemini client is not configured")
    return genai.Client(api_key=_api_key())


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

    raise VLMResponseError("Gemini response did not contain valid JSON")


def _parsed_response(response: Any) -> dict[str, Any] | None:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, BaseModel):
        return parsed.model_dump(mode="json")

    model_dump = getattr(parsed, "model_dump", None)
    if callable(model_dump):
        try:
            value = model_dump(mode="json")
            return value if isinstance(value, dict) else None
        except Exception:
            return None
    return None


def _finish_reason(response: Any) -> str:
    values: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        reason = getattr(candidate, "finish_reason", None)
        if reason is not None:
            values.append(str(reason))
    return ",".join(values) or "unknown"


def _response_text(response: Any) -> str:
    direct = getattr(response, "text", None)
    if direct:
        return str(direct)

    chunks: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                chunks.append(str(text))
    return "".join(chunks)


def _part_from_data_url(data_url: str):
    if types is None:
        raise VLMUnavailableError("google-genai is not installed")

    prefix, encoded = str(data_url).split(",", 1)
    mime_type = "image/jpeg"
    if prefix.startswith("data:") and ";" in prefix:
        mime_type = prefix[5:].split(";", 1)[0] or mime_type

    return types.Part.from_bytes(
        data=base64.b64decode(encoded),
        mime_type=mime_type,
    )


def generate_json(
    *,
    model: str,
    prompt: str,
    image_urls: Iterable[str] | None = None,
    max_output_tokens: int = 1200,
    timeout_seconds: float = 120.0,
) -> ProviderJSONResult:
    del timeout_seconds  # google-genai timeout is client-side library controlled here.

    if not is_configured():
        raise VLMUnavailableError(
            "GEMINI_API_KEY is not configured or google-genai is unavailable"
        )

    if circuit.is_open(provider_name, model):
        remain = circuit.remaining_seconds(provider_name, model)
        raise VLMQuotaError(
            f"Gemini circuit is open for model={model}; remaining~{remain:.0f}s",
            retry_after_seconds=remain,
        )

    image_urls = list(image_urls or [])

    # Resolve Phase 4 runtime configuration ONCE.
    # Dùng cùng một giá trị cho quota preflight và request thật.
    effective_max_output_tokens = int(
        os.getenv(
            "GEMINI_PHASE4_MAX_OUTPUT_TOKENS",
            str(max_output_tokens),
        )
    )

    thinking_level = (
        os.getenv("GEMINI_PHASE4_THINKING_LEVEL", "low").strip().lower()
    )

    # Defensive validation
    allowed_thinking_levels = {"minimal", "low", "medium", "high"}
    if thinking_level not in allowed_thinking_levels:
        thinking_level = "low"

    # -----------------------------
    # QUOTA PREFLIGHT
    # -----------------------------
    decision = quota_manager.can_request(
        provider_name,
        model,
        estimated_input_tokens=quota_manager.estimate_input_tokens(
            prompt,
            image_count=len(image_urls),
        ),
        max_output_tokens=effective_max_output_tokens,
    )

    if not decision.allowed:
        raise VLMQuotaError(
            f"Gemini quota preflight blocked request: reason={decision.reason}",
            retry_after_seconds=decision.retry_after_seconds,
        )

    # -----------------------------
    # BUILD CONTENTS
    # -----------------------------
    contents: list[Any] = [prompt]
    for value in image_urls:
        text = str(value or "")
        if text.startswith("data:") and "," in text:
            contents.append(_part_from_data_url(text))

    response = None
    max_retries = max(1, int(os.getenv("GEMINI_MAX_RETRIES", "3")))
    base_delay = max(0.0, float(os.getenv("GEMINI_RETRY_BASE_SECONDS", "2")))
    last_error: Exception | None = None

    # -----------------------------
    # GEMINI GENERATION CONFIG
    # -----------------------------
    generation_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=_VerificationPayload,
        max_output_tokens=effective_max_output_tokens,
        # QUAN TRỌNG:
        # thinking_level phải nằm trong thinking_config,
        # KHÔNG truyền trực tiếp vào generate_content().
        thinking_config=types.ThinkingConfig(
            thinking_level=thinking_level,
        ),
        # Phase 4 không dùng function/tool calling.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True,
        ),
    )

    # -----------------------------
    # REQUEST
    # -----------------------------
    client = _client()
    try:
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generation_config,
                )
                break

            except Exception as exc:
                last_error = exc
                message = f"{type(exc).__name__}: {exc}"

                # Provider quota / rate limit
                if "429" in message or "RESOURCE_EXHAUSTED" in message.upper():
                    retry = circuit.parse_retry_after_seconds(message, 300)

                    circuit.open_circuit(
                        provider_name,
                        model,
                        seconds=retry,
                        reason=message,
                    )
                    quota_manager.record_quota_error(
                        provider_name,
                        model,
                        message,
                        retry,
                    )
                    raise VLMQuotaError(
                        message,
                        retry_after_seconds=retry,
                    ) from exc

                # Transient provider/network errors
                retryable = any(
                    marker in message.lower()
                    for marker in (
                        "timeout",
                        "timed out",
                        "503",
                        "502",
                        "500",
                        "unavailable",
                        "connection",
                    )
                )
                if not retryable or attempt >= max_retries:
                    raise VLMUnavailableError(message) from exc

                time.sleep(base_delay * (2 ** (attempt - 1)))

        if response is None:
            raise VLMUnavailableError(f"Gemini request failed: {last_error}")

        # -----------------------------
        # STRUCTURED RESPONSE
        # -----------------------------
        data = _parsed_response(response)
        if data is None:
            raw_text = _response_text(response)
            try:
                data = _json_from_text(raw_text)
            except VLMResponseError as exc:
                finish_reason = _finish_reason(response)
                preview = raw_text[:1000].replace("\n", "\\n")
                raise VLMResponseError(
                    "Gemini response did not contain valid JSON; "
                    f"finish_reason={finish_reason}; "
                    f"chars={len(raw_text)}; "
                    f"preview={preview!r}"
                ) from exc

    finally:
        # Chỉ đóng client sau khi toàn bộ request/response kết thúc.
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    circuit.close_circuit(provider_name, model)
    quota_manager.record_success(provider_name, model)
    return ProviderJSONResult(data=data, provider=provider_name, model=model)
