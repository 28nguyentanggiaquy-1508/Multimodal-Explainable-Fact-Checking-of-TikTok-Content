from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _claim_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in payload.get("claims") or []:
        claim = row.get("claim") or {}
        cid = str(claim.get("id") or "")
        if cid:
            result[cid] = row
    return result


def _verification(row: dict[str, Any] | None) -> dict[str, Any]:
    return dict((row or {}).get("verification") or {})


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Gemini baseline against local-Qwen Phase 4 outputs.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--qwen", required=True, action="append", type=Path, help="Repeat for each Qwen result JSON")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = _claim_map(_load(args.baseline))
    qwen_rows: dict[str, dict[str, Any]] = {}
    for path in args.qwen:
        qwen_rows.update(_claim_map(_load(path)))

    rows: list[dict[str, Any]] = []
    for cid in sorted(set(baseline) & set(qwen_rows)):
        b = _verification(baseline[cid])
        q = _verification(qwen_rows[cid])
        guard = ((q.get("operational_metadata") or {}).get("local_component_guard") or {})
        item = {
            "claim_id": cid,
            "baseline_verdict": b.get("verdict"),
            "qwen_verdict": q.get("verdict"),
            "verdict_agreement": b.get("verdict") == q.get("verdict"),
            "baseline_coverage": b.get("evidence_coverage"),
            "qwen_coverage": q.get("evidence_coverage"),
            "coverage_agreement": b.get("evidence_coverage") == q.get("evidence_coverage"),
            "baseline_confidence": b.get("confidence"),
            "qwen_confidence": q.get("confidence"),
            "qwen_provider": q.get("provider"),
            "qwen_model": q.get("model"),
            "component_count": guard.get("component_count"),
            "resolved_components": guard.get("resolved_components"),
            "guard_mode": guard.get("mode"),
        }
        rows.append(item)

    print("claim | gemini verdict/coverage | qwen verdict/coverage | verdict | coverage")
    print("-" * 96)
    for row in rows:
        print(
            f"{row['claim_id']:8} | "
            f"{str(row['baseline_verdict']):9}/{str(row['baseline_coverage']):12} | "
            f"{str(row['qwen_verdict']):9}/{str(row['qwen_coverage']):12} | "
            f"{'PASS' if row['verdict_agreement'] else 'DIFF':7} | "
            f"{'PASS' if row['coverage_agreement'] else 'DIFF'}"
        )

    summary = {
        "compared": len(rows),
        "verdict_agreement": sum(1 for row in rows if row["verdict_agreement"]),
        "coverage_agreement": sum(1 for row in rows if row["coverage_agreement"]),
        "rows": rows,
    }
    print("\nsummary=", json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
