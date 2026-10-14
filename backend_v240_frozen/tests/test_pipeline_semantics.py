import asyncio

from fact_checking.evidence.evidence_schema import (
    EvidenceItem,
    EvidenceRetrievalResult,
    RetrievalChannelResult,
)
from fact_checking.schemas import CandidateClaim, MultimediaAnalysis
from fact_checking.verification.verification_schema import VerificationResult
import fact_checking.pipeline as pipeline

URL = "https://www.tiktok.com/@demo/video/1234567890"
CLAIM = CandidateClaim(id="claim_01", text="Một factual claim có thể kiểm chứng.", source_modality="caption")


def _multimedia():
    return MultimediaAnalysis(
        video_url=URL, status="DEGRADED", available_modalities=["caption"],
        caption=CLAIM.text, candidate_claims=[CLAIM], source="caption_fallback",
    )


def test_provider_failure_is_not_nei(monkeypatch):
    async def fake_multimedia(_): return _multimedia()
    async def fake_claims(_): return [CLAIM]
    async def fake_retrieval(**_):
        return EvidenceRetrievalResult(
            status="UNAVAILABLE", evidence=[],
            text=RetrievalChannelResult(modality="text", status="UNAVAILABLE"),
            image=RetrievalChannelResult(modality="image", status="UNAVAILABLE"),
            errors=["provider unavailable"],
        )
    monkeypatch.setattr(pipeline, "analyze_multimedia", fake_multimedia)
    monkeypatch.setattr(pipeline, "extract_claims", fake_claims)
    monkeypatch.setattr(pipeline, "retrieve_evidence", fake_retrieval)
    result = asyncio.run(pipeline.run_fact_check_pipeline(URL))
    assert result["claims"][0]["verification"]["status"] == "UNAVAILABLE"
    assert result["claims"][0]["verification"]["verdict"] is None
    assert result["verdict_ready"] is False


def test_completed_retrieval_without_evidence_is_semantic_nei(monkeypatch):
    async def fake_multimedia(_): return _multimedia()
    async def fake_claims(_): return [CLAIM]
    async def fake_retrieval(**_):
        return EvidenceRetrievalResult(
            status="SUCCESS", evidence=[],
            text=RetrievalChannelResult(modality="text", status="SUCCESS"),
            image=RetrievalChannelResult(modality="image", status="SUCCESS"),
        )
    monkeypatch.setattr(pipeline, "analyze_multimedia", fake_multimedia)
    monkeypatch.setattr(pipeline, "extract_claims", fake_claims)
    monkeypatch.setattr(pipeline, "retrieve_evidence", fake_retrieval)
    result = asyncio.run(pipeline.run_fact_check_pipeline(URL))
    assert result["claims"][0]["verification"]["verdict"] == "NEI"
    assert result["verdict_ready"] is True
    assert result["contract_version"] == "D3.1-backend-contract-v1"
    assert result["claim_extraction_completed"] is True
    assert result["no_claim_detected"] is False


def test_evidence_backed_supported_verdict(monkeypatch):
    evidence = EvidenceItem(
        id="text_01_000", modality="text", source_url="https://example.com/source",
        source_domain="example.com", content="Evidence content", retrieval_score=0.9,
        relevance_score=0.9,
    )
    async def fake_multimedia(_): return _multimedia()
    async def fake_claims(_): return [CLAIM]
    async def fake_retrieval(**_):
        return EvidenceRetrievalResult(
            status="SUCCESS", evidence=[evidence],
            text=RetrievalChannelResult(modality="text", status="SUCCESS", evidence=[evidence]),
            image=RetrievalChannelResult(modality="image", status="SUCCESS"),
        )
    async def fake_reasoning(**_):
        return {"status":"SUCCESS","generator":"test","key_concepts":[],"reasoning_support":"",
                "semantic_consistency":{"text":0.9,"image":None,"overall":0.9},"errors":[]}
    async def fake_batch(jobs, multimedia):
        return {CLAIM.id: VerificationResult(status="SUCCESS", verdict="SUPPORTED", confidence=0.9,
                                              evidence_ids=["text_01_000"])}
    monkeypatch.setattr(pipeline, "analyze_multimedia", fake_multimedia)
    monkeypatch.setattr(pipeline, "extract_claims", fake_claims)
    monkeypatch.setattr(pipeline, "retrieve_evidence", fake_retrieval)
    monkeypatch.setattr(pipeline, "build_reasoning_support", fake_reasoning)
    monkeypatch.setattr(pipeline, "verify_claims_batch", fake_batch)
    result = asyncio.run(pipeline.run_fact_check_pipeline(URL))
    assert result["claims"][0]["verification"]["verdict"] == "SUPPORTED"
    assert result["verdict_ready"] is True



def test_successful_phase2_with_no_claim_is_true_no_claim(monkeypatch):
    multimedia = MultimediaAnalysis(
        video_url=URL,
        status="SUCCESS",
        available_modalities=["caption"],
        caption="Không có factual claim.",
        candidate_claims=[],
        source="test",
    )

    async def fake_multimedia(_): return multimedia
    async def fake_claims(_): return []
    monkeypatch.setattr(pipeline, "analyze_multimedia", fake_multimedia)
    monkeypatch.setattr(pipeline, "extract_claims", fake_claims)

    result = asyncio.run(pipeline.run_fact_check_pipeline(URL))
    assert result["claim_extraction_status"] == "SUCCESS"
    assert result["claim_extraction_completed"] is True
    assert result["no_claim_detected"] is True


def test_unavailable_phase2_is_not_true_no_claim(monkeypatch):
    multimedia = MultimediaAnalysis(
        video_url=URL,
        status="UNAVAILABLE",
        available_modalities=[],
        candidate_claims=[],
        source="multivlm_unavailable",
        error="provider unavailable",
    )

    async def fake_multimedia(_): return multimedia
    async def fake_claims(_): return []
    monkeypatch.setattr(pipeline, "analyze_multimedia", fake_multimedia)
    monkeypatch.setattr(pipeline, "extract_claims", fake_claims)

    result = asyncio.run(pipeline.run_fact_check_pipeline(URL))
    assert result["claim_extraction_completed"] is False
    assert result["no_claim_detected"] is None
