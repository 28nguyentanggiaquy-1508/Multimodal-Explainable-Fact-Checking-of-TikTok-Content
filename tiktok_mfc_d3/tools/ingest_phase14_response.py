from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import refresh_manifest
from .phase14_adapter import import_phase14_response
from .qc import run_qc


def main() -> None:
    parser = argparse.ArgumentParser(description="Import current Phase 1-4 backend JSON into TikTok-MFC candidate records.")
    parser.add_argument("response_json", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    args = parser.parse_args()

    with args.response_json.open("r", encoding="utf-8") as f:
        response = json.load(f)

    result = import_phase14_response(response, args.dataset_root)
    manifest = refresh_manifest(args.dataset_root)
    qc = run_qc(args.dataset_root)

    print(json.dumps({"import": result, "manifest": manifest, "qc": qc}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
