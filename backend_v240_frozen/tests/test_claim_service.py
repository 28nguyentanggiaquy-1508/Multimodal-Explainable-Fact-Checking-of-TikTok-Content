import asyncio

from fact_checking.claims.claim_service import extract_claims
from fact_checking.schemas import CandidateClaim, MultimediaAnalysis


def test_claims_are_deduplicated_and_renumbered():
    multimedia = MultimediaAnalysis(
        video_url="https://www.tiktok.com/@demo/video/1234567890",
        candidate_claims=[
            CandidateClaim(
                id="x",
                text="Việt Nam hiện có 3 Đại tướng.",
                source_modality="caption",
            ),
            CandidateClaim(
                id="y",
                text="  Việt Nam hiện có 3 Đại tướng.  ",
                source_modality="speech",
            ),
        ],
    )

    claims = asyncio.run(extract_claims(multimedia))
    assert len(claims) == 1
    assert claims[0].id == "claim_01"



def test_research_hard_cap_never_selects_more_than_five_claims():
    multimedia = MultimediaAnalysis(
        video_url="https://www.tiktok.com/@demo/video/1234567890",
        candidate_claims=[
            CandidateClaim(
                id=f"raw_{i}",
                text=f"Factual claim số {i}.",
                source_modality="caption",
            )
            for i in range(1, 9)
        ],
    )

    claims = asyncio.run(extract_claims(multimedia))
    assert len(claims) == 5
    assert [claim.id for claim in claims] == [f"claim_{i:02d}" for i in range(1, 6)]
