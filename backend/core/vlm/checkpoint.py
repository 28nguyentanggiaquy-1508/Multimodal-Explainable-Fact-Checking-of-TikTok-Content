from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _root() -> Path:
    configured = os.getenv("VLM_CHECKPOINT_DIR", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        return path.resolve()
    return (Path(__file__).resolve().parents[2] / "runtime" / "checkpoints").resolve()


def save_checkpoint(*, trace_id: str, phase: str, payload: dict[str, Any]) -> Path:
    root = _root() / str(trace_id)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{phase}.json"
    envelope = {
        "trace_id": trace_id,
        "phase": phase,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path
