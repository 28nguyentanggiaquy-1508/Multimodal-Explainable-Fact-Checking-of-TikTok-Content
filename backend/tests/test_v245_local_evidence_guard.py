from __future__ import annotations

from fact_checking.verification.local_component_guard import apply_local_component_guard, decompose_claim


def _base_result() -> dict:
    return {
        "status": "SUCCESS",
        "verdict": "SUPPORTED",
        "confidence": 1.0,
        "evidence_ids": [],
        "evidence_assessments": [],
        "evidence_coverage": "COMPLETE",
        "rationale": "raw",
        "uncertainty": "",
        "limitations": [],
        "multimodal_used": False,
        "model": "qwen3-vl:8b-instruct-q4_K_M",
        "provider": "local",
    }


def _run(claim: str, evidence: list[dict], assessments: list[dict]):
    components = decompose_claim(claim)
    raw = {
        "verdict": "SUPPORTED",
        "confidence": 1.0,
        "evidence_coverage": "COMPLETE",
        "component_assessments": assessments,
    }
    return apply_local_component_guard(
        raw_data=raw,
        base_result=_base_result(),
        components=components,
        candidate_ids=[item["id"] for item in evidence],
        attached_image_ids=set(),
        evidence=evidence,
    )


def test_unique_qualifier_without_direct_unique_evidence_is_downgraded() -> None:
    claim = "Võ Nguyên Giáp là Tổng Tư lệnh duy nhất của Quân đội nhân dân Việt Nam"
    result = _run(
        claim,
        [{"id": "e1", "content": "Võ Nguyên Giáp từng giữ chức Tổng Tư lệnh Quân đội nhân dân Việt Nam."}],
        [{"component_id": "C1", "status": "SUPPORTED", "evidence_ids": ["e1"], "confidence": 1.0}],
    )
    assert result["verdict"] == "NEI"
    assert result["evidence_coverage"] == "INSUFFICIENT"
    assert result["_guard_metadata"]["qualifier_checks"][0]["kind"] == "unique"
    assert result["_guard_metadata"]["qualifier_checks"][0]["passed"] is False


def test_unique_qualifier_with_direct_unique_evidence_can_pass() -> None:
    claim = "Võ Nguyên Giáp là Tổng Tư lệnh duy nhất của Quân đội nhân dân Việt Nam"
    result = _run(
        claim,
        [{"id": "e1", "content": "Võ Nguyên Giáp là Tổng Tư lệnh duy nhất của Quân đội nhân dân Việt Nam."}],
        [{"component_id": "C1", "status": "SUPPORTED", "evidence_ids": ["e1"], "confidence": 0.9}],
    )
    assert result["verdict"] == "SUPPORTED"
    assert result["evidence_coverage"] == "COMPLETE"


def test_kinship_component_requires_direct_relation_evidence() -> None:
    claim = "Nguyễn Chí Thanh là thân sinh của Nguyễn Chí Vịnh"
    result = _run(
        claim,
        [{"id": "e1", "content": "Nguyễn Chí Thanh tên thật là Nguyễn Vịnh và được phong Đại tướng năm 1959."}],
        [{"component_id": "C1", "status": "SUPPORTED", "evidence_ids": ["e1"], "confidence": 1.0}],
    )
    assert result["verdict"] == "NEI"
    check = result["_guard_metadata"]["qualifier_checks"][0]
    assert check["kind"] == "kinship_parent"
    assert check["passed"] is False


def test_kinship_component_passes_with_direct_relation_evidence() -> None:
    claim = "Nguyễn Chí Thanh là thân sinh của Nguyễn Chí Vịnh"
    result = _run(
        claim,
        [{"id": "e1", "content": "Đại tướng Nguyễn Chí Thanh là thân sinh của Thượng tướng Nguyễn Chí Vịnh."}],
        [{"component_id": "C1", "status": "SUPPORTED", "evidence_ids": ["e1"], "confidence": 0.92}],
    )
    assert result["verdict"] == "SUPPORTED"


def test_exact_year_is_required_when_material() -> None:
    claim = "Nguyễn Chí Thanh được phong Đại tướng năm 1959"
    result = _run(
        claim,
        [{"id": "e1", "content": "Nguyễn Chí Thanh được phong quân hàm Đại tướng."}],
        [{"component_id": "C1", "status": "SUPPORTED", "evidence_ids": ["e1"], "confidence": 0.95}],
    )
    assert result["verdict"] == "NEI"
    assert any(c["kind"] == "year" and not c["passed"] for c in result["_guard_metadata"]["qualifier_checks"])
