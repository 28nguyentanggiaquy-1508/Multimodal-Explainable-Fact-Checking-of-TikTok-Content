from fact_checking.evidence.evidence_schema import EvidenceItem
from fact_checking.verification.verification_service import _normalize_result


def test_decisive_verdict_without_valid_citation_is_downgraded():
    evidence = [
        EvidenceItem(
            id="text_01",
            modality="text",
            source_url="https://example.com",
            source_domain="example.com",
            content="Evidence",
            retrieval_score=0.8,
            relevance_score=0.8,
        )
    ]

    result = _normalize_result(
        {
            "verdict": "SUPPORTED",
            "confidence": 0.95,
            "evidence_ids": [],
            "evidence_assessments": [],
        },
        evidence,
    )

    assert result.status == "SUCCESS"
    assert result.verdict == "NEI"
    assert result.confidence <= 0.35
