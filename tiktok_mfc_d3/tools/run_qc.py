from __future__ import annotations

import argparse
import json
from pathlib import Path
from .qc import run_qc
from .validate import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    result = {
        "schema": validate_dataset(args.dataset_root),
        "semantic_qc": run_qc(args.dataset_root),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["schema"]["valid"] and result["semantic_qc"]["summary"]["errors"] == 0 else 1)


if __name__ == "__main__":
    main()
