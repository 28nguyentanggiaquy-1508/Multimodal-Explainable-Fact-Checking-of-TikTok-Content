from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.full_pipeline_v081 import FULL_PIPELINE_VERSION, evaluate_full_pipeline, save_evaluation_bundle
from evaluation.temporary_gold_v071 import load_temporary_gold, read_jsonl, sha256_file, write_jsonl


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _video_id_from_record(record) -> str:
    return record.video_id


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {str(x.get("video_id") or ""): x for x in rows if str(x.get("video_id") or "")}


def _response_retryable(row: dict[str, Any]) -> bool:
    response = row.get("response") if isinstance(row.get("response"), dict) else {}
    claims = [x for x in (response.get("claims") or []) if isinstance(x, dict)]
    if not response:
        return True
    for item in claims:
        verification = item.get("verification") if isinstance(item.get("verification"), dict) else {}
        if str(verification.get("status") or "").upper() in {"FAILED", "UNAVAILABLE"}:
            return True
        if str(item.get("runtime_limit_reason") or "").upper() in {
            "TECHNICAL_STATUS_CANNOT_HAVE_VERDICT",
            "RETRIEVAL_TECHNICAL_FAILURE",
        }:
            return True
    phases = response.get("phase_status") if isinstance(response.get("phase_status"), dict) else {}
    return any(str((phases.get(k) or {}).get("status") or "").upper() in {"FAILED", "UNAVAILABLE"} for k in ("phase_2", "phase_3", "phase_4"))


def _health(base_url: str, timeout: float) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/fact-check/health"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError("Health endpoint did not return JSON object")
    return data


def _post_video(base_url: str, video_url: str, timeout: float) -> tuple[int, dict[str, Any]]:
    url = base_url.rstrip("/") + "/fact-check/fact-check"
    r = requests.post(url, json={"url": video_url, "sync_dataset": False}, timeout=timeout)
    status = r.status_code
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError("Fact-check endpoint did not return JSON object")
    return status, data


def _save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda x: str(x.get("video_id") or ""))
    write_jsonl(path, rows)


def cmd_run(args) -> int:
    dataset = load_temporary_gold(args.dataset)
    source_before = dataset.source_sha256
    gold_by_video: dict[str, list[Any]] = {}
    for rec in dataset.usable:
        gold_by_video.setdefault(rec.video_id, []).append(rec)
    selected_video_ids = {str(x).strip() for x in (args.video_id or []) if str(x).strip()}
    if selected_video_ids:
        gold_by_video = {k: v for k, v in gold_by_video.items() if k in selected_video_ids}
        if not gold_by_video:
            raise SystemExit("No requested --video-id exists in the Temporary Gold dataset")

    health = _health(args.base_url, args.health_timeout)
    existing = _load_existing(args.runs_output)
    run_rows = list(existing.values())
    attempted = skipped_resume = failures = 0

    for idx, (video_id, records) in enumerate(sorted(gold_by_video.items()), 1):
        previous = existing.get(video_id)
        if previous and not args.force:
            if not args.retry_technical or not _response_retryable(previous):
                skipped_resume += 1
                print(f"[v0.8.1] {idx}/{len(gold_by_video)} SKIP {video_id} (resume)", file=sys.stderr)
                continue
        attempted += 1
        video_url = records[0].video_url
        print(f"[v0.8.1] {idx}/{len(gold_by_video)} RUN {video_id}", file=sys.stderr)
        started = time.perf_counter()
        try:
            http_status, response = _post_video(args.base_url, video_url, args.request_timeout)
            row = {
                "benchmark_version": FULL_PIPELINE_VERSION,
                "video_id": video_id,
                "video_url": video_url,
                "gold_claim_count_for_video": len(records),
                "run_status": "SUCCESS",
                "http_status": http_status,
                "started_at": _utc_now(),
                "duration_seconds": round(time.perf_counter() - started, 3),
                "response": response,
            }
        except Exception as exc:
            failures += 1
            row = {
                "benchmark_version": FULL_PIPELINE_VERSION,
                "video_id": video_id,
                "video_url": video_url,
                "gold_claim_count_for_video": len(records),
                "run_status": "FAILED",
                "http_status": None,
                "started_at": _utc_now(),
                "duration_seconds": round(time.perf_counter() - started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc)[:800],
                "response": None,
            }
        existing[video_id] = row
        _save_rows(args.runs_output, list(existing.values()))
        if args.min_request_interval_seconds > 0:
            time.sleep(args.min_request_interval_seconds)

    run_rows = list(existing.values())
    predictions, report = evaluate_full_pipeline(
        dataset.usable,
        run_rows,
        min_alignment_score=args.min_alignment_score,
        min_text_similarity=args.min_text_similarity,
    )
    report["execution"] = {
        "base_url": args.base_url,
        "health": health,
        "attempted_this_run": attempted,
        "resume_skipped_this_run": skipped_resume,
        "http_failures_this_run": failures,
        "run_rows_total": len(run_rows),
        "runs_output": str(args.runs_output),
        "predictions_output": str(args.predictions_output),
        "report_output": str(args.report_output),
        "sync_dataset": False,
        "resume_enabled": not args.force,
        "retry_technical": bool(args.retry_technical),
    }
    report["dataset"] = dataset.metadata
    report["dataset"]["source_sha256"] = dataset.source_sha256
    report["source_unchanged"] = sha256_file(args.dataset.resolve()) == source_before
    save_evaluation_bundle(
        predictions_path=args.predictions_output,
        predictions=predictions,
        report_path=args.report_output,
        report=report,
    )
    _print({
        "benchmark_version": FULL_PIPELINE_VERSION,
        "mode": report["mode"],
        "video_count": len(gold_by_video),
        "gold_claim_count": len(dataset.usable),
        "matched_gold_claim_count": report["alignment"]["matched_gold_claim_count"],
        "gold_claim_recall": report["alignment"]["gold_claim_recall"],
        "prediction_count": report["full_pipeline_metrics"]["prediction_count"],
        "coverage": report["full_pipeline_metrics"]["coverage"],
        "overall_accuracy": report["full_pipeline_metrics"]["overall_accuracy"],
        "safety_violation_count": report["safety"]["violation_count"],
        "root_cause_distribution": report["root_cause_distribution"],
        "source_unchanged": report["source_unchanged"],
        "runs_output": str(args.runs_output),
        "predictions_output": str(args.predictions_output),
        "report_output": str(args.report_output),
    })
    return 0


def cmd_evaluate(args) -> int:
    dataset = load_temporary_gold(args.dataset)
    source_before = dataset.source_sha256
    run_rows = read_jsonl(args.runs)
    predictions, report = evaluate_full_pipeline(
        dataset.usable,
        run_rows,
        min_alignment_score=args.min_alignment_score,
        min_text_similarity=args.min_text_similarity,
    )
    report["dataset"] = dataset.metadata
    report["source_unchanged"] = sha256_file(args.dataset.resolve()) == source_before
    save_evaluation_bundle(
        predictions_path=args.predictions_output,
        predictions=predictions,
        report_path=args.report_output,
        report=report,
    )
    _print(report)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full Pipeline Temporary-Gold endpoint evaluator v0.8.1")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("run", help="Run real /fact-check/fact-check per Temporary-Gold video and evaluate")
    a.add_argument("--dataset", type=Path, required=True)
    a.add_argument("--base-url", default="http://127.0.0.1:8000")
    a.add_argument("--runs-output", type=Path, required=True)
    a.add_argument("--predictions-output", type=Path, required=True)
    a.add_argument("--report-output", type=Path, required=True)
    a.add_argument("--request-timeout", type=float, default=1800.0)
    a.add_argument("--health-timeout", type=float, default=15.0)
    a.add_argument("--min-request-interval-seconds", type=float, default=1.0)
    a.add_argument("--retry-technical", action="store_true")
    a.add_argument("--video-id", action="append", default=[], help="Run/re-run only selected video_id; may be repeated")
    a.add_argument("--force", action="store_true")
    a.add_argument("--min-alignment-score", type=float, default=0.68)
    a.add_argument("--min-text-similarity", type=float, default=0.62)
    a.set_defaults(func=cmd_run)

    a = sub.add_parser("evaluate", help="Evaluate an already captured v0.8.0 runs JSONL without network calls")
    a.add_argument("--dataset", type=Path, required=True)
    a.add_argument("--runs", type=Path, required=True)
    a.add_argument("--predictions-output", type=Path, required=True)
    a.add_argument("--report-output", type=Path, required=True)
    a.add_argument("--min-alignment-score", type=float, default=0.68)
    a.add_argument("--min-text-similarity", type=float, default=0.62)
    a.set_defaults(func=cmd_evaluate)
    return p


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
