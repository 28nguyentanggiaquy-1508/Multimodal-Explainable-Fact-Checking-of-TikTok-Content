async def build_explanation(
    claim: dict,
    multimedia: dict,
    evidence: list[dict],
    verification: dict,
) -> dict:

    return {
        "verdict": verification.get("verdict", "NEI"),
        "summary": verification.get("reasoning", ""),
        "supporting_evidence": evidence,
    }