from __future__ import annotations

import random
import re
import threading
import time
from typing import Any

from config import (
    GEMINI_CIRCUIT_DEFAULT_SECONDS,
    GEMINI_FALLBACK_MODEL,
    GEMINI_MAX_RETRIES,
    GEMINI_RETRY_BASE_SECONDS,
)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class GeminiCircuitOpenError(RuntimeError):
    """Raised locally when a model has an active hard-quota circuit."""


_CIRCUIT_LOCK = threading.Lock()
_CIRCUIT_OPEN_UNTIL: dict[str, float] = {}


def _get_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass

    text = str(exc)
    for code in RETRYABLE_STATUS_CODES:
        if str(code) in text:
            return code
    return None


def _is_hard_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    hard_quota_markers = (
        "exceeded your current quota",
        "check your plan and billing details",
        "billing details",
        "quota has been exhausted",
        "quotafailure",
    )
    return (
        ("resource_exhausted" in text or "429" in text)
        and any(marker in text for marker in hard_quota_markers)
    )


def _retry_delay_seconds(exc: Exception) -> int:
    text = str(exc)
    matches = [
        re.search(r"retryDelay['\"\s:]+(\d+)s", text, flags=re.I),
        re.search(r"retry\s+in\s+(\d+)h(?:(\d+)m)?(?:(\d+)(?:\.\d+)?s)?", text, flags=re.I),
        re.search(r"retry\s+in\s+(\d+)m(?:(\d+)(?:\.\d+)?s)?", text, flags=re.I),
        re.search(r"retry\s+in\s+(\d+)(?:\.\d+)?s", text, flags=re.I),
    ]

    if matches[0]:
        return max(1, int(matches[0].group(1)))
    if matches[1]:
        hours = int(matches[1].group(1) or 0)
        minutes = int(matches[1].group(2) or 0)
        seconds = int(matches[1].group(3) or 0)
        return max(1, hours * 3600 + minutes * 60 + seconds)
    if matches[2]:
        minutes = int(matches[2].group(1) or 0)
        seconds = int(matches[2].group(2) or 0)
        return max(1, minutes * 60 + seconds)
    if matches[3]:
        return max(1, int(matches[3].group(1)))
    return max(1, int(GEMINI_CIRCUIT_DEFAULT_SECONDS))


def _open_circuit(model: str, seconds: int) -> None:
    with _CIRCUIT_LOCK:
        _CIRCUIT_OPEN_UNTIL[model] = time.monotonic() + max(1, seconds)


def _circuit_remaining(model: str) -> int:
    with _CIRCUIT_LOCK:
        until = _CIRCUIT_OPEN_UNTIL.get(model, 0.0)
        remaining = until - time.monotonic()
        if remaining <= 0:
            _CIRCUIT_OPEN_UNTIL.pop(model, None)
            return 0
        return max(1, int(round(remaining)))


def reset_gemini_circuits() -> None:
    """Primarily useful for tests or after deliberately changing credentials."""
    with _CIRCUIT_LOCK:
        _CIRCUIT_OPEN_UNTIL.clear()


def _is_retryable(exc: Exception) -> bool:
    if _is_hard_quota_error(exc):
        return False

    status_code = _get_status_code(exc)
    if status_code in RETRYABLE_STATUS_CODES:
        return True

    # httpx/httpcore timeout subclasses can have messages such as
    # "The read operation timed out", which do not contain "timeout".
    exc_name = type(exc).__name__.lower()
    retryable_exception_names = (
        "timeout",
        "readtimeout",
        "connecttimeout",
        "pooltimeout",
        "networkerror",
        "connectionerror",
    )
    if any(value in exc_name for value in retryable_exception_names):
        return True

    text = str(exc).lower()
    retry_keywords = (
        "unavailable",
        "high demand",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "read operation timed out",
        "read timeout",
        "deadline exceeded",
        "server error",
        "internal error",
        "rate limit",
        "connection reset",
        "connection aborted",
    )
    return any(keyword in text for keyword in retry_keywords)


def generate_content_with_retry(
    client,
    *,
    model: str,
    contents: Any,
    config=None,
):
    if client is None:
        raise RuntimeError(
            "Gemini client chưa được khởi tạo. Kiểm tra GEMINI_API_KEY."
        )

    primary_model = str(model or "").strip()
    if not primary_model:
        raise ValueError("Gemini model không được để trống.")

    models_to_try = [primary_model]
    fallback_model = str(GEMINI_FALLBACK_MODEL or "").strip()
    if fallback_model and fallback_model != primary_model:
        models_to_try.append(fallback_model)

    last_error: Exception | None = None
    circuit_errors: list[str] = []

    for current_model in models_to_try:
        remaining = _circuit_remaining(current_model)
        if remaining > 0:
            circuit_errors.append(
                f"{current_model} circuit open for ~{remaining}s"
            )
            print(
                f"[Gemini] Circuit open model={current_model}; "
                f"skip call remaining~{remaining}s"
            )
            continue

        print(f"[Gemini] Trying model={current_model}")

        for attempt in range(1, GEMINI_MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=config,
                )

                if response is None:
                    raise RuntimeError("Gemini returned None response.")

                print(
                    f"[Gemini] Success model={current_model} attempt={attempt}"
                )
                return response, current_model

            except Exception as exc:
                last_error = exc

                if _is_hard_quota_error(exc):
                    delay = _retry_delay_seconds(exc)
                    _open_circuit(current_model, delay)
                    print(
                        "[Gemini] Hard quota exhausted - opening circuit; "
                        f"model={current_model} cooldown~{delay}s"
                    )
                    # Hard quota is not retried for this model. A distinct fallback
                    # model may still be attempted if its own circuit is closed.
                    break

                if not _is_retryable(exc):
                    print(
                        "[Gemini] Non-retryable error: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    raise

                if attempt >= GEMINI_MAX_RETRIES:
                    print(
                        f"[Gemini] Exhausted retries model={current_model}"
                    )
                    break

                wait_seconds = (
                    GEMINI_RETRY_BASE_SECONDS
                    * (2 ** (attempt - 1))
                    + random.uniform(0.0, 0.5)
                )

                print(
                    "[Gemini Retry] "
                    f"model={current_model} "
                    f"attempt={attempt}/{GEMINI_MAX_RETRIES} "
                    f"error={type(exc).__name__} "
                    f"wait={wait_seconds:.1f}s"
                )
                time.sleep(wait_seconds)

    if last_error is not None:
        raise last_error

    if circuit_errors:
        raise GeminiCircuitOpenError(
            "Gemini circuit open: " + "; ".join(circuit_errors)
        )

    raise RuntimeError("Gemini request failed without exception.")
