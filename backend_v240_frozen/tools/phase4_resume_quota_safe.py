from __future__ import annotations

"""Compatibility wrapper.

In v2.4 quota-safe compaction is part of the core verifier, so the normal
phase4_resume.py already performs the quota-safe path. This filename is kept so
old commands/scripts do not break.
"""

import json
import sys
from pathlib import Path


def _preview(input_path: str) -> int:
    backend = Path(__file__).resolve().parents[1]
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from fact_checking.verification.quota_safe import prepare_verification_inputs

    path = Path(input_path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data.get("payload"), dict):
        data = data["payload"]
    multimedia = data.get("multimedia") or {}
    print("[Phase4QS/v240] preview only; quota-safe compaction is now built into core Phase 4.")
    for row in data.get("claims") or []:
        if str((row.get("verification") or {}).get("status") or "").upper() == "SUCCESS":
            continue
        refined, _, _, stats = prepare_verification_inputs(
            claim=row.get("claim") or {},
            multimedia=multimedia,
            evidence=row.get("evidence") or [],
            reasoning_support=row.get("reasoning_support") or {},
        )
        cid = str((row.get("claim") or {}).get("id") or "")
        print(
            f"[Phase4QS/v240] {cid}: evidence {stats['evidence_before']}->{stats['evidence_after']} "
            f"ids={stats['selected_evidence_ids']}"
        )
    return 0


def main() -> int:
    if "--preview-quota-safe" in sys.argv:
        try:
            index = sys.argv.index("--input")
            input_path = sys.argv[index + 1]
        except Exception:
            print("--preview-quota-safe requires --input <json>", file=sys.stderr)
            return 2
        return _preview(input_path)

    print("[Phase4QS/v240] quota-safe behavior moved into core; delegating to phase4_resume.py")
    from phase4_resume import main as resume_main
    return resume_main()


if __name__ == "__main__":
    raise SystemExit(main())
