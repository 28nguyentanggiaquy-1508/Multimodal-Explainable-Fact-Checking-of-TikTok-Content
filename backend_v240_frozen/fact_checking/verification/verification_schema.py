from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from fact_checking.schemas import StageStatus


Verdict = Literal["SUPPORTED", "REFUTED", "NEI"]
EvidenceStance = Literal["support", "refute", "neutral", "context"]
EvidenceCoverage = Literal["COMPLETE", "PARTIAL", "INSUFFICIENT"]


class EvidenceAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    evidence_id: str
    stance: EvidenceStance = "neutral"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Operational status is intentionally separate from the semantic verdict.
    # Provider/model failure must never be represented as NEI.
    status: StageStatus = "SUCCESS"
    verdict: Optional[Verdict] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    evidence_ids: list[str] = Field(default_factory=list)
    evidence_assessments: list[EvidenceAssessment] = Field(default_factory=list)
    evidence_coverage: Optional[EvidenceCoverage] = None

    rationale: str = ""
    uncertainty: str = ""
    limitations: list[str] = Field(default_factory=list)

    multimodal_used: bool = False
    model: str = ""
    provider: Optional[str] = None
    contract_violations: list[str] = Field(default_factory=list)

