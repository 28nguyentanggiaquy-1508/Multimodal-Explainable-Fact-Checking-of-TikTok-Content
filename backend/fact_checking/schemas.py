from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


CONTRACT_VERSION = "D3.1-backend-contract-v1"

DataOrigin = Literal["SYSTEM", "CANDIDATE", "GOLD"]

StageStatus = Literal[
    "SUCCESS",
    "DEGRADED",
    "UNAVAILABLE",
    "FAILED",
    "SKIPPED",
]

SourceModality = Literal[
    "speech",
    "ocr",
    "caption",
    "visual",
    "cross_modal",
]

ClaimType = Literal[
    "statistical",
    "event_place_person",
    "other_factual",
    "authenticity_context",
]

ContentType = Literal[
    "real_world",
    "fictional",
    "mixed",
    "uncertain",
]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")




class ContractProvenance(StrictBaseModel):
    """Run-level provenance shared by backend and dataset bridge.

    This model is intentionally model-agnostic so retrievers/verifiers can be
    replaced later without changing the D3.1 dataset contract.
    """

    contract_version: str = CONTRACT_VERSION
    origin: DataOrigin = "SYSTEM"
    pipeline_version: str = ""
    trace_id: str = ""
    provider: Optional[str] = None
    model: Optional[str] = None
    prompt_version: Optional[str] = None


class StageReport(StrictBaseModel):
    status: StageStatus
    message: str = ""
    duration_ms: Optional[int] = None
    details: dict = Field(default_factory=dict)


class VisualObservation(StrictBaseModel):
    timestamp: Optional[str] = None
    fact: str = Field(min_length=1)


class SpeechSegment(StrictBaseModel):
    timestamp: Optional[str] = None
    text: str = Field(min_length=1)


class OCRSegment(StrictBaseModel):
    timestamp: Optional[str] = None
    text: str = Field(min_length=1)


class CandidateClaim(StrictBaseModel):
    id: str = ""
    text: str = Field(min_length=1)
    source_modality: SourceModality
    timestamp: Optional[str] = None
    checkable: bool = True
    check_worthy: bool = True
    claim_type: ClaimType = "other_factual"
    rationale: str = ""


class MultimediaAnalysis(StrictBaseModel):
    video_url: str
    content_type: ContentType = "uncertain"

    # Phase-2 operational status. This prevents caption-only fallback from
    # being confused with successful full multimodal analysis.
    status: StageStatus = "SUCCESS"
    degraded_reason: Optional[str] = None
    available_modalities: list[SourceModality] = Field(default_factory=list)

    visual_observations: list[VisualObservation] = Field(default_factory=list)
    speech_segments: list[SpeechSegment] = Field(default_factory=list)
    speech_transcript: str = ""

    ocr_segments: list[OCRSegment] = Field(default_factory=list)
    ocr_text: str = ""

    caption: Optional[str] = None

    candidate_claims: list[CandidateClaim] = Field(default_factory=list)
    cross_modal_notes: list[str] = Field(default_factory=list)

    metadata: dict = Field(default_factory=dict)
    source: str = ""
    model: str = ""
    error: Optional[str] = None

