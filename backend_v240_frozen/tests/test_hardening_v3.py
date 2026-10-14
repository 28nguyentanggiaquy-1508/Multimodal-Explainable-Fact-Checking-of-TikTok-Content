from datetime import date

from fact_checking.evidence.evidence_ranker import _diverse_select
from fact_checking.evidence.evidence_schema import EvidenceItem
from fact_checking.evidence.text_retriever import _temporal_features
from fact_checking.reasoning.reasoning_support import _local_raw_concepts


def _evidence(eid, domain, url, score):
    return EvidenceItem(
        id=eid, modality="text", source_url=url, source_domain=domain,
        content="Nội dung evidence", retrieval_score=score, relevance_score=score,
    )


def test_temporal_current_state_after_anchor_is_penalized():
    score, relation, source_date = _temporal_features(
        page_age="2026-07-06T13:40:19",
        content="Hiện có 3 người đang công tác.",
        temporal_anchor=date(2025, 7, 19),
        time_sensitive=True,
    )
    assert relation in {"after_anchor_365d", "far_after_anchor"}
    assert score < 0.5
    assert source_date == "2026-07-06"


def test_diverse_select_avoids_one_url_dominating():
    items = [
        _evidence("a", "wiki.org", "https://wiki.org/x", 0.99),
        _evidence("b", "wiki.org", "https://wiki.org/x", 0.98),
        _evidence("c", "news.vn", "https://news.vn/y", 0.90),
        _evidence("d", "gov.vn", "https://gov.vn/z", 0.89),
    ]
    selected = _diverse_select(items, 3)
    assert len({item.source_url for item in selected}) == 3


def test_local_concepts_do_not_emit_broken_stopword_fragments():
    evidence = [
        EvidenceItem(
            id="text_01", modality="text", source_url="https://example.com",
            source_domain="example.com",
            content=("Hiện có 3 Đại tướng cùng công tác trong Quân đội nhân dân Việt Nam "
                     "là Phan Văn Giang, Nguyễn Tân Cương và Trịnh Văn Quyết."),
            retrieval_score=0.9, relevance_score=0.9,
        )
    ]
    rows = _local_raw_concepts(
        "Việt Nam hiện có 3 Đại tướng cùng công tác trong Quân đội",
        evidence,
    )
    concepts = {row["concept"] for row in rows}
    assert "Phan Văn Giang" in concepts
    assert "đội Có" not in concepts
