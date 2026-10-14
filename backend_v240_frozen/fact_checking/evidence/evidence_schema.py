from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from fact_checking.schemas import StageStatus


EvidenceModality = Literal["text", "image"]


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    modality: EvidenceModality

    title: str = ""
    source_url: str
    source_domain: str = ""

    content: str = ""
    media_url: Optional[str] = None

    retrieval_score: float = Field(default=0.0, ge=0.0, le=1.0)
    rerank_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Provenance fields used by UI/report/reproducibility.
    provider: str = ""
    published_at: Optional[str] = None
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source_quality: str = "unknown"

    metadata: dict = Field(default_factory=dict)


class ProviderAttempt(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str
    status: StageStatus
    result_count: int = 0
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class RetrievalChannelResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    modality: EvidenceModality
    status: StageStatus
    evidence: list[EvidenceItem] = Field(default_factory=list)
    provider_attempts: list[ProviderAttempt] = Field(default_factory=list)
    search_source_count: int = 0
    candidate_count: int = 0
    errors: list[str] = Field(default_factory=list)


class EvidenceRetrievalResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: StageStatus
    evidence: list[EvidenceItem] = Field(default_factory=list)
    text: RetrievalChannelResult
    image: RetrievalChannelResult
    reranker_status: StageStatus = "SKIPPED"
    errors: list[str] = Field(default_factory=list)

    @property
    def provider_available(self) -> bool:
        return any(
            channel.status in {"SUCCESS", "DEGRADED"}
            for channel in (self.text, self.image)
        )
