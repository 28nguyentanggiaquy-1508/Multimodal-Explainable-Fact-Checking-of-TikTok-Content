from pathlib import Path
import json
import tempfile
import unittest

from tools.contract_guard import normalize_system_response
from tools.phase14_adapter import _parse_timestamp, import_phase14_response
from tools.validate import validate_dataset
from tools.qc import run_qc


class ToolingTests(unittest.TestCase):
    def test_failure_never_becomes_nei(self):
        payload = {
            "claims": [{
                "evidence": [],
                "verification": {
                    "verdict": "NEI",
                    "evidence_ids": [],
                    "rationale": "Không có bằng chứng truy xuất đủ để xác minh claim.",
                }
            }]
        }
        normalized = normalize_system_response(payload)
        v = normalized["claims"][0]["verification"]
        self.assertIsNone(v["verdict"])
        self.assertEqual(v["system_status"], "INSUFFICIENT_COLLECTION")
        self.assertEqual(v["raw_verdict"], "NEI")
        self.assertIsNone(v["confidence"])


    def test_single_timestamp_is_preserved(self):
        self.assertEqual(_parse_timestamp("00:07"), (7.0, None))
        self.assertEqual(_parse_timestamp("01:02.5"), (62.5, None))
        self.assertEqual(_parse_timestamp("00:07-00:12"), (7.0, 12.0))

    def test_adapter_and_schema(self):
        src_root = Path(__file__).resolve().parents[1]
        payload = json.loads((src_root / "examples" / "phase14_response_minimal.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "dataset"
            import shutil
            shutil.copytree(src_root / "dataset", dst)
            result = import_phase14_response(payload, dst)
            self.assertEqual(len(result["imported_claim_ids"]), 1)
            self.assertTrue(result["contract_violations"])
            schema_result = validate_dataset(dst)
            self.assertTrue(schema_result["valid"], schema_result["errors"])
            qc_result = run_qc(dst)
            self.assertEqual(qc_result["summary"]["errors"], 0)
            self.assertEqual(qc_result["summary"]["warnings"], 1)


if __name__ == "__main__":
    unittest.main()

