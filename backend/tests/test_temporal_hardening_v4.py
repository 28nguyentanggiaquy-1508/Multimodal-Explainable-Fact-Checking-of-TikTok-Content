from datetime import date

from fact_checking.evidence.evidence_ranker import _diverse_select
from fact_checking.evidence.evidence_schema import EvidenceItem
from fact_checking.evidence.text_retriever import _claim_directness, _temporal_features
from fact_checking.evidence.image_retriever import _image_temporal_features
from fact_checking.reasoning.reasoning_support import (
    _concept_after_anchor,
    _merge_and_filter_concepts,
)


def _item(eid: str, score: float, eligible: bool):
    return EvidenceItem(
        id=eid,
        modality="text",
        source_url=f"https://example.com/{eid}",
        source_domain="example.com",
        content="evidence",
        retrieval_score=score,
        relevance_score=score,
        metadata={"temporal_eligible": eligible},
    )


def test_current_state_future_page_is_strongly_penalized():
    score, relation, parsed = _temporal_features(
        page_age="2025-08-24T08:29:53",
        content="Hiện có 3 Đại tướng cùng công tác.",
        temporal_anchor=date(2025, 7, 19),
        time_sensitive=True,
        temporal_mode="current_state",
    )
    assert parsed == "2025-08-24"
    assert relation == "after_anchor_180d"
    assert score <= 0.12


def test_directness_prefers_exact_current_state_count():
    claim = "Việt Nam hiện có 3 Đại tướng cùng công tác trong Quân đội"
    direct = _claim_directness(
        claim,
        "Việt Nam hiện có 3 Đại tướng cùng công tác trong Quân đội.",
        temporal_mode="current_state",
    )
    historical = _claim_directness(
        claim,
        "Kể từ khi thành lập Quân đội Nhân dân Việt Nam đã có 18 đại tướng.",
        temporal_mode="current_state",
    )
    assert direct > historical


def test_diverse_select_prefers_temporally_eligible():
    selected = _diverse_select(
        [
            _item("future", 0.99, False),
            _item("past1", 0.80, True),
            _item("past2", 0.79, True),
        ],
        2,
    )
    assert {x.id for x in selected} == {"past1", "past2"}


def test_image_future_year_is_ineligible():
    score, relation, eligible = _image_temporal_features(
        title="Hội nghị 6 tháng đầu năm 2026",
        source_url="https://example.com/article",
        image_url="https://cdn.example.com/2026/photo.jpg",
        temporal_anchor=date(2025, 7, 19),
        temporal_mode="current_state",
    )
    assert score == 0.10
    assert relation == "future_year_detected"
    assert eligible is False


def test_future_date_concept_is_filtered_by_anchor():
    assert _concept_after_anchor("01/9/2026", date(2025, 7, 19)) is True
    assert _concept_after_anchor("14/7/2025", date(2025, 7, 19)) is False


def test_low_similarity_single_source_concept_removed(monkeypatch):
    # The filtering function uses embeddings, so this behavior is covered mainly
    # by integration tests. This unit test protects the explicit temporal gate.
    raw = [{"concept": "01/9/2026", "evidence_ids": ["text_01"]}]
    out = _merge_and_filter_concepts(
        "Việt Nam hiện có 3 Đại tướng cùng công tác trong Quân đội",
        raw,
        1,
        temporal_anchor=date(2025, 7, 19),
    )
    assert out == []
