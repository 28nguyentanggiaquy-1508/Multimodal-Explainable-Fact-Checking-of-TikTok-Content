from types import SimpleNamespace

from fact_checking.evidence.evidence_schema import RetrievalChannelResult
from fact_checking.evidence.evidence_service import _collection_complete_certificate, _temporal_mode
from fact_checking.verification.multivlm_verifier import _multimedia_context
from fact_checking.verification.quota_safe import refine_evidence
from evaluation.full_pipeline_v081 import PENDING_GOLD_REAUDIT_IDS


def _channel(modality, status="SUCCESS", errors=None):
    return RetrievalChannelResult(modality=modality, status=status, errors=errors or [])


def test_collection_certificate_requires_both_channels_success_and_no_errors():
    text = _channel("text")
    image = _channel("image")
    assert _collection_complete_certificate(overall_status="SUCCESS", text_result=text, image_result=image, errors=[]) is True
    assert _collection_complete_certificate(overall_status="DEGRADED", text_result=text, image_result=image, errors=[]) is False
    assert _collection_complete_certificate(overall_status="SUCCESS", text_result=text, image_result=_channel("image", "FAILED"), errors=[]) is False
    assert _collection_complete_certificate(overall_status="SUCCESS", text_result=text, image_result=image, errors=["x"]) is False


def test_quota_refinement_reserves_direct_numeric_evidence_before_domain_diversity(monkeypatch):
    monkeypatch.setenv("VERIFICATION_REFINED_MAX_EVIDENCE", "5")
    monkeypatch.setenv("VERIFICATION_REFINED_MAX_TEXT", "4")
    monkeypatch.setenv("VERIFICATION_MAX_IMAGES", "1")
    claim = "Giá dầu mazut 180 CST 3.5S không cao hơn 23.373 đồng/kg."
    evidence = [
        {"id":"a","modality":"text","source_url":"https://a.vn/1","source_domain":"a.vn","content":"mazut 180CST 3.5S 17.496 đồng/kg","relevance_score":0.95},
        {"id":"b","modality":"text","source_url":"https://b.vn/1","source_domain":"b.vn","content":"mazut 180CST 3.5S 17.641 đồng/kg","relevance_score":0.90},
        {"id":"c","modality":"text","source_url":"https://same.vn/1","source_domain":"same.vn","content":"mazut 180CST 3.5S 22.135 đồng/kg","relevance_score":0.88},
        {"id":"direct","modality":"text","source_url":"https://same.vn/2","source_domain":"same.vn","content":"dầu mazut 180CST 3.5S không cao hơn 23.373 đồng/kg","relevance_score":0.80},
        {"id":"d","modality":"text","source_url":"https://d.vn/1","source_domain":"d.vn","content":"thông tin xăng dầu khác","relevance_score":0.79},
        {"id":"img","modality":"image","source_url":"https://img.vn/x","source_domain":"img.vn","content":"ảnh","relevance_score":0.70},
    ]
    selected = refine_evidence(evidence, claim_text=claim)
    ids = [x["id"] for x in selected]
    assert "direct" in ids
    assert len(selected) == 5


def test_relative_end_of_year_claim_is_temporally_sensitive():
    assert _temporal_mode("Giá card đồ họa sẽ tăng mạnh trong những tháng cuối năm.") == "relative_event"


def test_multimedia_context_includes_upload_date_anchor():
    mm = {"caption":"", "speech_transcript":"", "ocr_text":"", "visual_observations":[], "metadata":{"upload_date":"20260920"}}
    ctx = _multimedia_context(mm)
    assert "UPLOAD_DATE: 20260920" in ctx
    assert "relative time expressions" in ctx


def test_new_temporal_conflict_record_is_pending_gold_reaudit():
    assert "TK_7687526963484822792_AI_9a9af3394e3f" in PENDING_GOLD_REAUDIT_IDS
