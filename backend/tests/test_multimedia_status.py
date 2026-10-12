from fact_checking.multimedia.multimedia_service import _phase2_status


def test_caption_only_is_degraded():
    status, reason, modalities = _phase2_status(
        raw={"cross_modal_notes": []},
        visual_observations=[],
        speech_segments=[],
        speech_transcript="",
        ocr_segments=[],
        ocr_text="",
        caption="Một caption factual",
        candidate_claims=[object()],
    )
    assert status == "DEGRADED"
    assert "caption" in modalities
    assert reason


def test_primary_multimedia_is_success():
    status, reason, modalities = _phase2_status(
        raw={"cross_modal_notes": []},
        visual_observations=[object()],
        speech_segments=[],
        speech_transcript="",
        ocr_segments=[],
        ocr_text="",
        caption=None,
        candidate_claims=[],
    )
    assert status == "SUCCESS"
    assert "visual" in modalities
    assert reason is None
