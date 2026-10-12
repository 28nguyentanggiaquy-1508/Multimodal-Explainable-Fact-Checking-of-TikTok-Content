from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _tooling_root() -> Path:
    configured = os.getenv("TIKTOK_MFC_ROOT", "").strip()
    if configured:
        p = Path(configured).expanduser()
        if not p.is_absolute():
            # Resolve relative paths from project root, not current terminal directory.
            project_root = Path(__file__).resolve().parents[2]
            p = project_root / p
        return p.resolve()
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / "tiktok_mfc_d3").resolve()


def _as_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="json")
        if isinstance(value, dict):
            return value
    dict_method = getattr(response, "dict", None)
    if callable(dict_method):
        value = dict_method()
        if isinstance(value, dict):
            return value
    raise TypeError("Fact-check response must be a dict or Pydantic model.")


def _safe_name(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum() or ch in {"-", "_"})[:80] or "run"


@contextmanager
def _file_lock(path: Path, timeout_seconds: float = 20.0):
    """Small cross-platform inter-process lock using atomic file creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    stale_after = max(timeout_seconds * 3, 60.0)
    fd: int | None = None
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} time={time.time()}".encode("utf-8"))
            break
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
                if age > stale_after:
                    path.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.monotonic() - start >= timeout_seconds:
                raise TimeoutError(f"Could not acquire dataset lock: {path}")
            time.sleep(0.1)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        path.unlink(missing_ok=True)


def _write_raw_inbox(tooling_root: Path, payload: dict[str, Any]) -> Path:
    inbox = tooling_root / "inbox" / "phase14"
    inbox.mkdir(parents=True, exist_ok=True)
    trace_id = _safe_name(str(payload.get("trace_id") or uuid4()))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = inbox / f"{stamp}_{trace_id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def sync_phase14_response(
    response: Any,
    *,
    batch_id: str | None = None,
    target_domain: str | None = None,
    discovery_method: str | None = "BACKEND_FACT_CHECK",
    discovery_query: str | None = None,
    topics: list[str] | None = None,
) -> dict[str, Any]:
    """Persist a Phase 1-4 result into the sibling TikTok-MFC tooling.

    The raw response is always written to inbox when integration is enabled.
    Auto-ingestion converts backend output into SYSTEM/CANDIDATE records only;
    it never creates Gold labels or Gold evidence.
    """
    if not _bool_env("TIKTOK_MFC_ENABLED", True):
        return {"status": "DISABLED"}

    payload = _as_dict(response)

    # Normalize semantic status before dataset ingestion. This prevents a
    # technical Phase-2 failure from being interpreted as a true no-claim sample.
    try:
        from .response_guard import normalize_factcheck_response_semantics
        normalized_payload = normalize_factcheck_response_semantics(payload)
    except Exception:
        normalized_payload = payload
    tooling_root = _tooling_root()
    dataset_root = tooling_root / "dataset"
    required_state = dataset_root / "state" / "id_counters.json"
    if not required_state.exists():
        return {
            "status": "ERROR",
            "message": f"TikTok-MFC tooling not found or incomplete: {tooling_root}",
        }

    try:
        # Preserve the original backend response for audit.
        raw_path = _write_raw_inbox(tooling_root, payload)
    except Exception as exc:
        return {"status": "ERROR", "stage": "RAW_INBOX", "message": str(exc)}

    if not _bool_env("TIKTOK_MFC_AUTO_INGEST", True):
        return {
            "status": "QUEUED",
            "raw_response": str(raw_path),
            "auto_ingest": False,
        }

    context = {
        "batch_id": batch_id or os.getenv("TIKTOK_MFC_BATCH_ID") or None,
        "target_domain": target_domain or os.getenv("TIKTOK_MFC_TARGET_DOMAIN") or None,
        "discovery_method": discovery_method,
        "discovery_query": discovery_query,
        "topics": topics or [],
        "language": "vi",
        "evidence_language": "vi",
    }

    try:
        root_str = str(tooling_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        from tools.manifest import refresh_manifest
        from tools.phase14_adapter import import_phase14_response
        from tools.qc import run_qc
        from tools.validate import validate_dataset

        lock_path = tooling_root / ".locks" / "ingest.lock"
        with _file_lock(lock_path):
            imported = import_phase14_response(
                normalized_payload,
                dataset_root,
                collection_context=context,
                skip_duplicate_source=_bool_env("TIKTOK_MFC_SKIP_DUPLICATE_SOURCE", True),
            )
            manifest = refresh_manifest(dataset_root)
            schema = validate_dataset(dataset_root)
            qc = run_qc(dataset_root)

        return {
            "status": "SYNCED" if schema.get("valid") and qc.get("summary", {}).get("errors", 0) == 0 else "SYNCED_WITH_QC_ISSUES",
            "raw_response": str(raw_path),
            "tooling_root": str(tooling_root),
            "import": imported,
            "manifest_statistics": manifest.get("statistics", {}),
            "schema_valid": schema.get("valid"),
            "schema_errors": schema.get("errors", []),
            "qc_summary": qc.get("summary", {}),
            "semantic_guard": {
                "claim_extraction_status": normalized_payload.get("claim_extraction_status"),
                "claim_extraction_completed": normalized_payload.get("claim_extraction_completed"),
                "no_claim_detected": normalized_payload.get("no_claim_detected"),
            },
        }
    except Exception as exc:
        return {
            "status": "ERROR",
            "stage": "AUTO_INGEST",
            "raw_response": str(raw_path),
            "message": f"{type(exc).__name__}: {exc}",
        }


async def async_sync_phase14_response(response: Any, **kwargs: Any) -> dict[str, Any]:
    """Non-blocking FastAPI helper; file/QC work runs in a worker thread."""
    return await asyncio.to_thread(sync_phase14_response, response, **kwargs)
