from __future__ import annotations

"""Resume Phase 4 from a saved v2.3.7/v2.4 pipeline response.

v2.4 uses the same provider-agnostic quota-safe verifier as the main pipeline, so
this tool no longer assumes OpenAI is the only valid recovery provider.
It never replays canonical dataset sync.
"""

import argparse
import asyncio
import json
import os
import hashlib
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_VERSION = "2.0.1-v242-windows-path-hotfix"
ALLOWED_VERDICTS = {"SUPPORTED", "REFUTED", "NEI"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("Input JSON root must be an object")
    # Accept runtime checkpoint envelopes produced by core.vlm.checkpoint.
    if isinstance(data.get("payload"), dict) and data.get("trace_id"):
        data = dict(data["payload"])
    return data


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _validate(data: dict[str, Any], allow_version_mismatch: bool) -> None:
    if not isinstance(data.get("claims"), list) or not data["claims"]:
        raise RuntimeError("Input contains no claims[]")
    if not isinstance(data.get("multimedia"), dict):
        raise RuntimeError("Input is missing multimedia")
    version = str(data.get("pipeline_version") or "")
    supported = version.startswith("v2.3.7") or version.startswith("v2.4")
    if version and not supported and not allow_version_mismatch:
        raise RuntimeError(
            f"Unsupported pipeline_version={version!r}. Use --allow-version-mismatch only if intentional."
        )


def _claim_id(row: dict[str, Any]) -> str:
    return str((row.get("claim") or {}).get("id") or "").strip()


def _status(row: dict[str, Any]) -> str:
    verification = row.get("verification")
    return str((verification or {}).get("status") or "MISSING").upper()


def _build_explanation(verification: dict[str, Any]) -> dict[str, Any] | None:
    if str(verification.get("status") or "").upper() != "SUCCESS":
        return None
    verdict = str(verification.get("verdict") or "").upper()
    rationale = str(verification.get("rationale") or "").strip()
    if verdict not in ALLOWED_VERDICTS or not rationale:
        return None
    return {
        "status": "SUCCESS",
        "generator": "phase4_verification_rationale_projection",
        "origin": "SYSTEM",
        "is_gold": False,
        "verdict": verdict,
        "text": rationale,
        "evidence_ids": list(verification.get("evidence_ids") or []),
        "evidence_coverage": verification.get("evidence_coverage"),
        "multimodal_used": bool(verification.get("multimodal_used")),
        "provider": verification.get("provider"),
        "model": verification.get("model"),
    }


def _aggregate(statuses: list[str]) -> str:
    if not statuses:
        return "SKIPPED"
    if all(s == "SUCCESS" for s in statuses):
        return "SUCCESS"
    if all(s in {"UNAVAILABLE", "FAILED"} for s in statuses):
        return "UNAVAILABLE" if "UNAVAILABLE" in statuses else "FAILED"
    return "DEGRADED"


def _refresh(data: dict[str, Any], resume_id: str, duration_ms: int) -> None:
    claims = data.get("claims") or []
    statuses = [_status(row) for row in claims]
    phase_status = data.setdefault("phase_status", {})
    phase_status["phase_4"] = {
        "status": _aggregate(statuses),
        "message": "Evidence-grounded verification updated via provider-agnostic Phase 4 resume.",
        "duration_ms": duration_ms,
        "details": {
            "claim_statuses": statuses,
            "batch_candidate_count": len(claims),
            "resumed": True,
            "resume_id": resume_id,
        },
    }
    data["verdict_ready"] = bool(claims) and all(
        isinstance(row.get("verification"), dict)
        and str(row["verification"].get("status") or "").upper() == "SUCCESS"
        and str(row["verification"].get("verdict") or "").upper() in ALLOWED_VERDICTS
        for row in claims
    )
    data["explanation_ready"] = bool(claims) and all(
        isinstance(row.get("explanation"), dict)
        and row["explanation"].get("status") == "SUCCESS"
        and bool(str(row["explanation"].get("text") or "").strip())
        for row in claims
    )
    data["status"] = _aggregate(
        [str((report or {}).get("status") or "FAILED") for report in phase_status.values() if isinstance(report, dict)]
    )
    data["skip_reason"] = None if data["verdict_ready"] else (
        "Ít nhất một claim chưa hoàn tất verification do provider/dependency chưa khả dụng."
    )


def _output_path(input_path: Path) -> Path:
    """Build a short checkpoint name safe for common Windows MAX_PATH setups.

    Older versions appended a new suffix to the entire input stem on every resume.
    Repeated resumes could push the temporary ``.json.tmp`` path past ~260 chars
    on Windows.  Use a stable short digest instead; provenance is still preserved
    inside resume_metadata.source_file.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha1(str(input_path).encode("utf-8")).hexdigest()[:10]
    return input_path.with_name(f"phase4_v240_{stamp}_{digest}.json")


def _configure_backend() -> None:
    backend = Path(__file__).resolve().parents[1]
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))


async def _run(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    data = _load_json(input_path)
    _validate(data, args.allow_version_mismatch)

    requested = set(args.claim_id or [])
    selected = []
    for row in data["claims"]:
        cid = _claim_id(row)
        if requested and cid not in requested:
            continue
        if not args.retry_success and _status(row) == "SUCCESS":
            continue
        selected.append(row)
        if args.max_claims is not None and len(selected) >= args.max_claims:
            break

    print(f"[Phase4Resume/v240] input={input_path}")
    print(f"[Phase4Resume/v240] pipeline={data.get('pipeline_version')} total={len(data['claims'])} retry={len(selected)}")
    for row in selected:
        print(f"  - {_claim_id(row)} status={_status(row)} evidence={len(row.get('evidence') or [])}")
    if args.inspect_only:
        print("[Phase4Resume/v240] inspect-only: no API call, no file mutation.")
        return 0
    if not selected:
        print("[Phase4Resume/v240] nothing to retry")
        return 0

    _configure_backend()
    if args.provider_order:
        os.environ["VLM_VERIFICATION_PROVIDER_ORDER"] = args.provider_order
    from fact_checking.verification.verification_service_multivlm import verify_claim

    output = Path(args.output).expanduser().resolve() if args.output else _output_path(input_path)
    resume_id = str(uuid.uuid4())
    started = time.perf_counter()
    data["resume_metadata"] = {
        "tool": "phase4_resume",
        "tool_version": TOOL_VERSION,
        "resume_id": resume_id,
        "source_file": str(input_path),
        "source_trace_id": data.get("trace_id"),
        "started_at": _utc_now(),
        "provider_order": os.getenv("VLM_VERIFICATION_PROVIDER_ORDER", "local,openai,gemini"),
        "dataset_sync_replayed": False,
        "retried_claim_ids": [],
        "completed_claim_ids": [],
    }
    _refresh(data, resume_id, 0)
    _atomic_write(output, data)

    for row in selected:
        cid = _claim_id(row)
        print(f"\n[Phase4Resume/v240] ===== {cid} =====")
        result = await verify_claim(
            claim=row["claim"],
            multimedia=data["multimedia"],
            evidence=list(row.get("evidence") or []),
            reasoning_support=row.get("reasoning_support") or {},
        )
        verification = result.model_dump()
        row["verification"] = verification
        row["explanation"] = _build_explanation(verification)
        data["resume_metadata"]["retried_claim_ids"].append(cid)
        if verification.get("status") == "SUCCESS":
            data["resume_metadata"]["completed_claim_ids"].append(cid)
        elapsed = int(round((time.perf_counter() - started) * 1000))
        _refresh(data, resume_id, elapsed)
        _atomic_write(output, data)
        print(
            f"[Phase4Resume/v240] {cid} status={verification.get('status')} "
            f"verdict={verification.get('verdict')} provider={verification.get('provider')} model={verification.get('model')}"
        )
        if verification.get("status") != "SUCCESS" and not args.continue_on_error:
            print("[Phase4Resume/v240] STOP: claim not SUCCESS; checkpoint preserved.")
            break
        if args.delay_seconds > 0:
            await asyncio.sleep(args.delay_seconds)

    elapsed = int(round((time.perf_counter() - started) * 1000))
    _refresh(data, resume_id, elapsed)
    data["resume_metadata"]["updated_at"] = _utc_now()
    _atomic_write(output, data)
    print(f"\n[Phase4Resume/v240] verdict_ready={data.get('verdict_ready')} output={output}")
    print("[Phase4Resume/v240] dataset_sync_replayed=False")
    return 0 if data.get("verdict_ready") else 2


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Resume Phase 4 using the v2.4 provider-agnostic quota-safe verifier.")
    p.add_argument("--input", required=True)
    p.add_argument("--output")
    p.add_argument("--claim-id", action="append")
    p.add_argument("--max-claims", type=int)
    p.add_argument("--retry-success", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--delay-seconds", type=float, default=1.0)
    p.add_argument("--provider-order", default=None, help="Example: local,openai,gemini")
    p.add_argument("--inspect-only", action="store_true")
    p.add_argument("--allow-version-mismatch", action="store_true")
    return p


def main() -> int:
    args = _parser().parse_args()
    if args.max_claims is not None and args.max_claims <= 0:
        raise SystemExit("--max-claims must be >= 1")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
