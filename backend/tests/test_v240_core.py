from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
import sys
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.vlm import cache, circuit, quota_manager
from fact_checking.verification.quota_safe import prepare_verification_inputs


def test_cache_roundtrip(tmp: Path) -> None:
    os.environ["VLM_CACHE_DIR"] = str(tmp / "cache")
    key = cache.build_key(task="x", provider="p", model="m", prompt_version="1", payload={"a": 1})
    assert cache.get_json(task="x", key=key) is None
    cache.set_json(task="x", key=key, value={"ok": True})
    assert cache.get_json(task="x", key=key) == {"ok": True}


def test_quota_preflight_blocks_known_remaining(tmp: Path) -> None:
    os.environ["VLM_QUOTA_STATE_FILE"] = str(tmp / "quota.json")
    os.environ["VLM_CIRCUIT_STATE_FILE"] = str(tmp / "circuit.json")
    os.environ["VLM_QUOTA_TOKEN_RESERVE"] = "1000"
    quota_manager.record_quota_error(
        "openai",
        "demo",
        "Rate limit tokens per min: Limit 100000, Used 98000, Requested 5000",
        30,
    )
    decision = quota_manager.can_request(
        "openai", "demo", estimated_input_tokens=3000, max_output_tokens=1000
    )
    assert decision.allowed is False
    assert decision.reason in {"circuit_open", "quota_state_blocked"}


def test_quota_safe_compaction() -> None:
    os.environ["VERIFICATION_REFINED_MAX_EVIDENCE"] = "5"
    os.environ["VERIFICATION_REFINED_MAX_TEXT"] = "4"
    os.environ["VERIFICATION_MAX_IMAGES"] = "1"
    evidence = []
    for i in range(7):
        evidence.append({
            "id": f"text_{i}", "modality": "text", "source_url": f"https://s{i}.vn/a",
            "source_domain": f"s{i}.vn", "content": "x" * 1000,
            "relevance_score": 1 - i * 0.05,
        })
    evidence.append({
        "id": "image_1", "modality": "image", "source_url": "https://img.vn/a",
        "source_domain": "img.vn", "content": "image", "media_url": "https://img.vn/a.jpg",
        "relevance_score": 0.8,
    })
    claim = {"id": "claim_01", "text": "demo", "timestamp": "00:20-00:30"}
    multimedia = {
        "caption": "cap",
        "speech_transcript": "full transcript" * 100,
        "speech_segments": [
            {"timestamp": "00:19-00:25", "text": "near claim"},
            {"timestamp": "01:00-01:10", "text": "far"},
        ],
        "ocr_segments": [],
        "ocr_text": "ocr",
        "visual_observations": [],
    }
    refined, compact_mm, _, stats = prepare_verification_inputs(
        claim=claim, multimedia=multimedia, evidence=evidence, reasoning_support={"reasoning_support": "r" * 5000}
    )
    assert len(refined) == 5
    assert sum(1 for item in refined if item["modality"] == "image") <= 1
    assert compact_mm["speech_transcript"] == "near claim"
    assert stats["evidence_before"] == 8 and stats["evidence_after"] == 5


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_cache_roundtrip(tmp)
        test_quota_preflight_blocks_known_remaining(tmp)
    test_quota_safe_compaction()
    print("v2.4 core offline tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
