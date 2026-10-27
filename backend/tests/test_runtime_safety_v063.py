from __future__ import annotations

from copy import deepcopy

from fact_checking.runtime_safety_v050 import enforce_runtime_verdict_policy


EXTERNAL_EVIDENCE = {
    "id": "text_01",
    "modality": "text",
    "source_url": "https://example.com/fact",
    "source_domain": "example.com",
    "content": "Independent external evidence.",
}


def _response(*, verification: dict, retrieval: dict | None = None, evidence: list[dict] | None = None) -> dict:
    return {
        "status": "SUCCESS",
        "phase_status": {
            "phase_4": {"status": "SUCCESS", "details": {}}
        },
        "claims": [
            {
                "claim": {"id": "claim_01", "text": "Demo claim"},
                "retrieval": retrieval or {"status": "SUCCESS"},
                "evidence": evidence if evidence is not None else [deepcopy(EXTERNAL_EVIDENCE)],
                "verification": verification,
                "explanation": {
                    "status": "SUCCESS",
                    "text": "System explanation",
                },
            }
        ],
    }


def test_supported_with_valid_external_citation_is_display_eligible_not_gold() -> None:
    result = enforce_runtime_verdict_policy(
        _response(
            verification={
                "status": "SUCCESS",
                "verdict": "SUPPORTED",
                "confidence": 0.9,
                "evidence_ids": ["text_01"],
                "evidence_assessments": [],
            }
        )
    )
    row = result["claims"][0]
    assert row["verification"]["verdict"] == "SUPPORTED"
    assert row["runtime_decision"] == "VERDICT_ELIGIBLE_FOR_DISPLAY_NOT_GOLD"
    assert result["runtime_policy"]["safe_to_freeze_gold"] is False


def test_tiktok_citation_cannot_support_decisive_verdict() -> None:
    result = enforce_runtime_verdict_policy(
        _response(
            verification={
                "status": "SUCCESS",
                "verdict": "SUPPORTED",
                "confidence": 0.9,
                "evidence_ids": ["tik_01"],
                "evidence_assessments": [],
            },
            evidence=[{
                "id": "tik_01",
                "modality": "text",
                "source_url": "https://www.tiktok.com/@demo/video/123",
                "content": "TikTok content is source context, not truth evidence.",
            }],
        )
    )
    row = result["claims"][0]
    assert row["verification"]["status"] == "UNAVAILABLE"
    assert row["verification"]["verdict"] is None
    assert row["runtime_decision"] == "WITHHELD"
    assert row["runtime_limit_reason"] == "CITATION_NOT_IN_RETRIEVED_EXTERNAL_EVIDENCE"
    assert result["verdict_ready"] is False


def test_technical_failure_never_keeps_semantic_verdict() -> None:
    result = enforce_runtime_verdict_policy(
        _response(
            verification={
                "status": "FAILED",
                "verdict": "NEI",
                "confidence": 0.2,
                "evidence_ids": [],
                "evidence_assessments": [],
            },
            evidence=[],
        )
    )
    row = result["claims"][0]
    assert row["verification"]["status"] == "UNAVAILABLE"
    assert row["verification"]["verdict"] is None
    assert row["runtime_limit_reason"] == "TECHNICAL_STATUS_CANNOT_HAVE_VERDICT"


def test_nei_without_explicit_collection_certificate_is_withheld() -> None:
    result = enforce_runtime_verdict_policy(
        _response(
            verification={
                "status": "SUCCESS",
                "verdict": "NEI",
                "confidence": 0.2,
                "evidence_ids": [],
                "evidence_assessments": [],
            },
            retrieval={
                "status": "SUCCESS",
                "text": {"status": "SUCCESS", "errors": []},
                "image": {"status": "SUCCESS", "errors": []},
                "errors": [],
            },
            evidence=[],
        )
    )
    row = result["claims"][0]
    assert row["verification"]["status"] == "UNAVAILABLE"
    assert row["verification"]["verdict"] is None
    assert row["runtime_limit_reason"] == "EVIDENCE_COLLECTION_COMPLETENESS_NOT_CERTIFIED"


def test_nei_with_explicit_collection_certificate_remains_nei() -> None:
    result = enforce_runtime_verdict_policy(
        _response(
            verification={
                "status": "SUCCESS",
                "verdict": "NEI",
                "confidence": 0.2,
                "evidence_ids": [],
                "evidence_assessments": [],
            },
            retrieval={
                "status": "SUCCESS",
                "collection_complete": True,
                "text": {"status": "SUCCESS", "errors": []},
                "image": {"status": "SUCCESS", "errors": []},
                "errors": [],
            },
            evidence=[],
        )
    )
    row = result["claims"][0]
    assert row["verification"]["status"] == "SUCCESS"
    assert row["verification"]["verdict"] == "NEI"
    assert row["runtime_decision"] == "VERDICT_ELIGIBLE_FOR_DISPLAY_NOT_GOLD"


def test_guard_does_not_mutate_input_response() -> None:
    original = _response(
        verification={
            "status": "SUCCESS",
            "verdict": "SUPPORTED",
            "confidence": 0.9,
            "evidence_ids": ["missing"],
            "evidence_assessments": [],
        }
    )
    snapshot = deepcopy(original)
    _ = enforce_runtime_verdict_policy(original)
    assert original == snapshot
