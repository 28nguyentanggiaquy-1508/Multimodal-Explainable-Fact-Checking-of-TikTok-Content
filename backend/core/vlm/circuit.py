from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any


_LOCK = threading.Lock()


def _default_state_path() -> Path:
    backend_dir = Path(__file__).resolve().parents[2]
    runtime = backend_dir / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    return runtime / "vlm_circuit_state.json"


def _state_path() -> Path:
    configured = os.getenv("VLM_CIRCUIT_STATE_FILE", "").strip()
    return Path(configured) if configured else _default_state_path()


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


def key(provider: str, model: str) -> str:
    return f"{provider.strip().lower()}::{model.strip()}"


def remaining_seconds(provider: str, model: str) -> float:
    with _LOCK:
        state = _read_state()
        item = state.get(key(provider, model), {})
        # Defensive compatibility: older/hand-edited state files may contain
        # strings or other non-dict values. A malformed entry must never crash
        # the whole VLM router. Treat it as an expired/closed circuit.
        if not isinstance(item, dict):
            return 0.0
        try:
            blocked_until = float(item.get("blocked_until", 0) or 0)
        except Exception:
            return 0.0
        return max(0.0, blocked_until - time.time())


def is_open(provider: str, model: str) -> bool:
    return remaining_seconds(provider, model) > 0


def open_circuit(
    provider: str,
    model: str,
    *,
    seconds: float,
    reason: str = "",
) -> None:
    seconds = max(1.0, float(seconds))
    now = time.time()
    with _LOCK:
        state = _read_state()
        state[key(provider, model)] = {
            "provider": provider,
            "model": model,
            "blocked_until": now + seconds,
            "opened_at": now,
            "reason": reason[:2000],
        }
        _write_state(state)


def close_circuit(provider: str, model: str) -> None:
    with _LOCK:
        state = _read_state()
        if state.pop(key(provider, model), None) is not None:
            _write_state(state)


def parse_retry_after_seconds(message: str, default: float = 300.0) -> float:
    """Parse common provider retry hints, including Gemini retryDelay strings."""
    text = str(message or "")

    # e.g. retryDelay': '74792s' or "retryDelay": "74792s"
    m = re.search(r"retryDelay[^0-9]{0,20}(\d+(?:\.\d+)?)s", text, flags=re.I)
    if m:
        return max(1.0, float(m.group(1)))

    # e.g. retry in 20h46m32.706s
    m = re.search(
        r"retry\s+in\s+(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?",
        text,
        flags=re.I,
    )
    if m and any(part is not None for part in m.groups()):
        h = float(m.group(1) or 0)
        minutes = float(m.group(2) or 0)
        sec = float(m.group(3) or 0)
        total = h * 3600 + minutes * 60 + sec
        if total > 0:
            return total

    return max(1.0, float(default))
