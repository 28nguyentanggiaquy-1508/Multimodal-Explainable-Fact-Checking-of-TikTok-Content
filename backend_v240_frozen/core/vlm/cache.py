from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    return _bool_env("VLM_CACHE_ENABLED", True)


def _root() -> Path:
    configured = os.getenv("VLM_CACHE_DIR", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            backend_dir = Path(__file__).resolve().parents[2]
            path = backend_dir / path
        return path.resolve()
    return (Path(__file__).resolve().parents[2] / "runtime" / "vlm_cache").resolve()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def build_key(*, task: str, provider: str, model: str, prompt_version: str, payload: Any) -> str:
    material = stable_json(
        {
            "task": str(task),
            "provider": str(provider).lower(),
            "model": str(model),
            "prompt_version": str(prompt_version),
            "payload": payload,
        }
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _path(task: str, key: str) -> Path:
    safe_task = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in task)[:80] or "task"
    return _root() / safe_task / f"{key}.json"


def get_json(*, task: str, key: str, ttl_seconds: float | None = None) -> dict[str, Any] | None:
    if not enabled():
        return None
    path = _path(task, key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    created_at = float(payload.get("created_at", 0) or 0)
    if ttl_seconds is not None and ttl_seconds > 0 and created_at > 0:
        if time.time() - created_at > ttl_seconds:
            return None
    value = payload.get("value")
    return value if isinstance(value, dict) else None


def set_json(*, task: str, key: str, value: dict[str, Any], metadata: dict[str, Any] | None = None) -> Path | None:
    if not enabled():
        return None
    path = _path(task, key)
    payload = {
        "created_at": time.time(),
        "task": task,
        "key": key,
        "metadata": metadata or {},
        "value": value,
    }
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
    return path
