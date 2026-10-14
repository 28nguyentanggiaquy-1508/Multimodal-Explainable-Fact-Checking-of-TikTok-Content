from __future__ import annotations

from fact_checking.verification.local_component_guard import (
    apply_local_component_guard,
    decompose_claim,
)


def _base_result() -> dict:
    return {
        "status": "SUCCESS",
        "verdict": "SUPPORTED",
        "confidence": 1.0,
        "evidence_ids": ["e1", "e2", "e3", "e4"],
        "evidence_assessments": [],
        "evidence_coverage": "COMPLETE",
        "rationale": "raw model rationale",
        "uncertainty": "",
        "limitations": [],
        "multimodal_used": False,
        "model": "qwen3-vl:8b-instruct-q4_K_M",
        "provider": "local",
    }


def test_decompose_claim03_like_compound_claim() -> None:
    claim = (
        "Đại tướng Võ Nguyên Giáp là Tổng Tư lệnh duy nhất của Quân đội nhân dân Việt Nam, "
        "được Chủ tịch Hồ Chí Minh phong thẳng quân hàm Đại tướng năm 37 tuổi mà không qua các cấp trung gian."
    )
    parts = decompose_claim(claim)
    assert len(parts) == 2
    assert "Tổng Tư lệnh duy nhất" in parts[0].text
    assert "phong thẳng" in parts[1].text


def test_decompose_claim04_like_compound_claim() -> None:
    claim = (
        "Đại tướng Nguyễn Chí Thanh là người thứ hai được Chủ tịch Hồ Chí Minh phong thẳng quân hàm Đại tướng "
        "(thụ phong năm 1959), tên thật là Nguyễn Vịnh và là thân sinh của Thượng tướng Nguyễn Chí Vịnh."
    )
    parts = decompose_claim(claim)
    assert len(parts) == 3
    assert "1959" in parts[0].text
    assert "Nguyễn Vịnh" in parts[1].text
    assert "thân sinh" in parts[2].text


def test_all_components_supported_remains_supported_complete() -> None:
    components = decompose_claim("A là B và được C xác nhận")
    raw = {
        "verdict": "SUPPORTED",
        "confidence": 1.0,
        "evidence_coverage": "COMPLETE",
        "component_assessments": [
            {"component_id": "C1", "status": "SUPPORTED", "evidence_ids": ["e1"], "confidence": 0.98},
            {"component_id": "C2", "status": "SUPPORTED", "evidence_ids": ["e2"], "confidence": 0.97},
        ],
    }
    result = apply_local_component_guard(
        raw_data=raw,
        base_result=_base_result(),
        components=components,
        candidate_ids=["e1", "e2"],
        attached_image_ids=set(),
    )
    assert result["verdict"] == "SUPPORTED"
    assert result["evidence_coverage"] == "COMPLETE"
    assert result["confidence"] <= 0.95


def test_missing_component_downgrades_supported_to_nei_partial() -> None:
    claim = "A là B, giữ chức C và là cha của D"
    components = decompose_claim(claim)
    assert len(components) >= 3
    raw = {
        "verdict": "SUPPORTED",
        "confidence": 1.0,
        "evidence_coverage": "COMPLETE",
        "component_assessments": [
            {"component_id": components[0].component_id, "status": "SUPPORTED", "evidence_ids": ["e1"], "confidence": 1.0},
            {"component_id": components[1].component_id, "status": "SUPPORTED", "evidence_ids": ["e2"], "confidence": 1.0},
            # final component intentionally omitted
        ],
    }
    result = apply_local_component_guard(
        raw_data=raw,
        base_result=_base_result(),
        components=components,
        candidate_ids=["e1", "e2", "e3"],
        attached_image_ids=set(),
    )
    assert result["verdict"] == "NEI"
    assert result["evidence_coverage"] == "PARTIAL"
    assert result["confidence"] <= 0.85
    assert "insufficient external evidence" in " ".join(result["limitations"])


def test_supported_without_valid_evidence_id_becomes_insufficient() -> None:
    components = decompose_claim("A là B")
    raw = {
        "verdict": "SUPPORTED",
        "confidence": 1.0,
        "evidence_coverage": "COMPLETE",
        "component_assessments": [
            {"component_id": "C1", "status": "SUPPORTED", "evidence_ids": ["not-allowed"], "confidence": 1.0},
        ],
    }
    result = apply_local_component_guard(
        raw_data=raw,
        base_result=_base_result(),
        components=components,
        candidate_ids=["e1"],
        attached_image_ids=set(),
    )
    assert result["verdict"] == "NEI"
    assert result["evidence_coverage"] == "INSUFFICIENT"
    assert result["evidence_ids"] == []


def test_refuted_component_can_refute_conjunction_without_false_supported_upgrade() -> None:
    components = decompose_claim("A là B và được C xác nhận")
    raw = {
        "verdict": "SUPPORTED",  # raw local model can be wrong; guard must override it
        "confidence": 1.0,
        "evidence_coverage": "COMPLETE",
        "component_assessments": [
            {"component_id": "C1", "status": "REFUTED", "evidence_ids": ["e1"], "confidence": 0.9},
            {"component_id": "C2", "status": "SUPPORTED", "evidence_ids": ["e2"], "confidence": 0.95},
        ],
    }
    result = apply_local_component_guard(
        raw_data=raw,
        base_result=_base_result(),
        components=components,
        candidate_ids=["e1", "e2"],
        attached_image_ids=set(),
    )
    assert result["verdict"] == "REFUTED"
    assert result["evidence_coverage"] == "COMPLETE"
    assert result["confidence"] <= 0.9
