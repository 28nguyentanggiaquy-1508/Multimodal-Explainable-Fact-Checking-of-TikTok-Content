import asyncio

import fact_checking.evidence.evidence_service as service
from fact_checking.evidence.evidence_schema import RetrievalChannelResult


def test_all_providers_unavailable_is_operational_unavailable(monkeypatch):
    def fake_text(_claim, _top_n=None, **_kwargs):
        return RetrievalChannelResult(modality="text", status="UNAVAILABLE")

    def fake_image(_claim, _top_n=None, **_kwargs):
        return RetrievalChannelResult(modality="image", status="UNAVAILABLE")

    monkeypatch.setattr(service, "retrieve_text_evidence", fake_text)
    monkeypatch.setattr(service, "retrieve_image_evidence", fake_image)

    result = asyncio.run(service.retrieve_evidence("claim"))
    assert result.status == "UNAVAILABLE"
    assert result.evidence == []
