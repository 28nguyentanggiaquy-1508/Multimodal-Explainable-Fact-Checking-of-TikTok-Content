from __future__ import annotations

import re

from config import MAX_CHECK_WORTHY_CLAIMS
from fact_checking.schemas import CandidateClaim, MultimediaAnalysis


# Research/prototype safety cap.
# Prevents an accidental .env override (for example 8 or 10 claims) from
# multiplying Phase 3 retrieval and Phase 4 VLM cost/TPM usage.
RESEARCH_HARD_MAX_CLAIMS = 5


def _claim_key(text: str) -> str:
    """Create a stable key for exact/format-level claim de-duplication."""
    normalized = re.sub(r"\s+", " ", text or "").strip().casefold()
    normalized = re.sub(r"[^\w\s]", "", normalized, flags=re.UNICODE)
    return normalized


def _effective_claim_limit() -> int:
    """
    Honor a lower configured limit, but never exceed the research hard cap.

    Examples:
      MAX_CHECK_WORTHY_CLAIMS=3 -> 3
      MAX_CHECK_WORTHY_CLAIMS=5 -> 5
      MAX_CHECK_WORTHY_CLAIMS=8 -> 5
    """
    try:
        configured = int(MAX_CHECK_WORTHY_CLAIMS)
    except (TypeError, ValueError):
        configured = RESEARCH_HARD_MAX_CLAIMS

    configured = max(1, configured)
    return min(configured, RESEARCH_HARD_MAX_CLAIMS)


async def extract_claims(
    multimedia: MultimediaAnalysis,
) -> list[CandidateClaim]:
    """
    Stage 1B: validate, de-duplicate, and deterministically cap claims.

    Selection policy:
    - keep only checkable + check-worthy claims;
    - remove exact/format-level duplicates;
    - preserve the VLM's original candidate order;
    - allow at most 5 claims in research mode.

    This is intentionally deterministic and does not invent a learned
    check-worthiness score that is not part of the current project design.
    """
    valid_claims: list[CandidateClaim] = []
    seen: set[str] = set()
    limit = _effective_claim_limit()

    if int(MAX_CHECK_WORTHY_CLAIMS) > RESEARCH_HARD_MAX_CLAIMS:
        print(
            "[ClaimSelection] configured MAX_CHECK_WORTHY_CLAIMS="
            f"{MAX_CHECK_WORTHY_CLAIMS} exceeds research hard cap="
            f"{RESEARCH_HARD_MAX_CLAIMS}; using {limit}."
        )

    for claim in multimedia.candidate_claims:
        text = re.sub(r"\s+", " ", claim.text or "").strip()

        if not text:
            continue

        if not claim.checkable or not claim.check_worthy:
            continue

        key = _claim_key(text)
        if not key or key in seen:
            continue

        seen.add(key)

        valid_claims.append(
            claim.model_copy(
                update={
                    "text": text,
                    "id": f"claim_{len(valid_claims) + 1:02d}",
                }
            )
        )

        if len(valid_claims) >= limit:
            break

    print(
        "[ClaimSelection] "
        f"candidates={len(multimedia.candidate_claims)} "
        f"selected={len(valid_claims)} limit={limit}"
    )

    return valid_claims
