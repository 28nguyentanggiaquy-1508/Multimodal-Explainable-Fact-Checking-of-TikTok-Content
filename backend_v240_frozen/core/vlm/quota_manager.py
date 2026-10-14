from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from . import circuit

_LOCK = threading.Lock()


@dataclass
class QuotaDecision:
    allowed: bool
    reason: str
    estimated_input_tokens: int
    max_output_tokens: int
    estimated_request_tokens: int
    remaining_tokens: int | None = None
    retry_after_seconds: float = 0.0


def _state_path() -> Path:
    configured = os.getenv("VLM_QUOTA_STATE_FILE", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        return path.resolve()
    return (Path(__file__).resolve().parents[2] / "runtime" / "vlm_quota_state.json").resolve()


def _read_state() -> dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_state(state: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _key(provider: str, model: str) -> str:
    return f"{provider.strip().lower()}::{model.strip()}"


def _parse_duration(value: str | None) -> float | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    total = 0.0
    matched = False
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", text):
        matched = True
        number = float(amount)
        if unit == "ms":
            total += number / 1000.0
        elif unit == "s":
            total += number
        elif unit == "m":
            total += number * 60
        elif unit == "h":
            total += number * 3600
    return total if matched else None


def estimate_input_tokens(prompt: str, image_count: int = 0) -> int:
    chars_per_token = max(1.0, float(os.getenv("VLM_TOKEN_ESTIMATE_CHARS_PER_TOKEN", "2.5")))
    image_tokens = max(0, int(os.getenv("VLM_TOKEN_ESTIMATE_LOW_DETAIL_IMAGE", "900")))
    return max(1, int(len(prompt or "") / chars_per_token) + max(0, image_count) * image_tokens)


def can_request(
    provider: str,
    model: str,
    *,
    estimated_input_tokens: int,
    max_output_tokens: int,
) -> QuotaDecision:
    estimated_input_tokens = max(0, int(estimated_input_tokens))
    max_output_tokens = max(0, int(max_output_tokens))
    total = estimated_input_tokens + max_output_tokens

    if circuit.is_open(provider, model):
        remain = circuit.remaining_seconds(provider, model)
        return QuotaDecision(
            False,
            "circuit_open",
            estimated_input_tokens,
            max_output_tokens,
            total,
            retry_after_seconds=remain,
        )

    reserve = max(0, int(os.getenv("VLM_QUOTA_TOKEN_RESERVE", "1000")))
    freshness = max(1.0, float(os.getenv("VLM_QUOTA_OBSERVATION_TTL_SECONDS", "120")))
    now = time.time()
    with _LOCK:
        item = _read_state().get(_key(provider, model), {})
    if not isinstance(item, dict):
        item = {}
    blocked_until = float(item.get("blocked_until", 0) or 0)
    if blocked_until > now:
        return QuotaDecision(
            False,
            "quota_state_blocked",
            estimated_input_tokens,
            max_output_tokens,
            total,
            retry_after_seconds=max(0.0, blocked_until - now),
        )

    observed_at = float(item.get("observed_at", 0) or 0)
    reset_at = float(item.get("reset_at", 0) or 0)
    remaining = item.get("remaining_tokens")
    try:
        remaining_int = int(remaining) if remaining is not None else None
    except (TypeError, ValueError):
        remaining_int = None

    observation_fresh = observed_at > 0 and now - observed_at <= freshness
    reset_active = reset_at > now
    if remaining_int is not None and (observation_fresh or reset_active):
        if total + reserve > remaining_int:
            retry_after = max(0.0, reset_at - now) if reset_active else 0.0
            return QuotaDecision(
                False,
                "known_remaining_tokens_insufficient",
                estimated_input_tokens,
                max_output_tokens,
                total,
                remaining_tokens=remaining_int,
                retry_after_seconds=retry_after,
            )

    return QuotaDecision(
        True,
        "allowed",
        estimated_input_tokens,
        max_output_tokens,
        total,
        remaining_tokens=remaining_int if (observation_fresh or reset_active) else None,
    )


def record_response_headers(provider: str, model: str, headers: Mapping[str, Any]) -> None:
    normalized = {str(k).lower(): str(v) for k, v in headers.items()}
    remaining = normalized.get("x-ratelimit-remaining-tokens")
    limit = normalized.get("x-ratelimit-limit-tokens")
    reset = normalized.get("x-ratelimit-reset-tokens")
    now = time.time()
    item: dict[str, Any] = {"provider": provider, "model": model, "observed_at": now}
    try:
        if remaining is not None:
            item["remaining_tokens"] = int(float(remaining))
    except ValueError:
        pass
    try:
        if limit is not None:
            item["limit_tokens"] = int(float(limit))
    except ValueError:
        pass
    reset_seconds = _parse_duration(reset)
    if reset_seconds is not None:
        item["reset_at"] = now + reset_seconds
    if len(item) <= 3:
        return
    with _LOCK:
        state = _read_state()
        existing = state.get(_key(provider, model), {})
        if isinstance(existing, dict):
            existing.update(item)
            item = existing
        state[_key(provider, model)] = item
        _write_state(state)


def record_quota_error(provider: str, model: str, message: str, retry_after_seconds: float) -> None:
    text = str(message or "")
    limit = used = requested = None
    match = re.search(
        r"Limit\s+(\d+).*?Used\s+(\d+).*?Requested\s+(\d+)",
        text,
        flags=re.I | re.S,
    )
    if match:
        limit, used, requested = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    now = time.time()
    item: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "observed_at": now,
        "blocked_until": now + max(0.0, float(retry_after_seconds)),
        "last_quota_error": text[:4000],
    }
    if limit is not None:
        item["limit_tokens"] = limit
        item["used_tokens"] = used
        item["requested_tokens"] = requested
        item["remaining_tokens"] = max(0, limit - used)
    with _LOCK:
        state = _read_state()
        state[_key(provider, model)] = item
        _write_state(state)


def record_success(provider: str, model: str) -> None:
    with _LOCK:
        state = _read_state()
        item = state.get(_key(provider, model), {})
        if not isinstance(item, dict):
            item = {}
        item.update({"provider": provider, "model": model, "last_success_at": time.time()})
        item.pop("blocked_until", None)
        state[_key(provider, model)] = item
        _write_state(state)


def snapshot(provider: str, model: str) -> dict[str, Any]:
    with _LOCK:
        item = _read_state().get(_key(provider, model), {})
    return dict(item) if isinstance(item, dict) else {}


def decision_dict(decision: QuotaDecision) -> dict[str, Any]:
    return asdict(decision)
