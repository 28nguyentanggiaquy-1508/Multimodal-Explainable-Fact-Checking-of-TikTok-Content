from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from dataset_integration.bridge import async_sync_phase14_response

router = APIRouter(prefix="/dataset", tags=["dataset"])


class DatasetSyncRequest(BaseModel):
    response: dict[str, Any]
    batch_id: str | None = None
    target_domain: str | None = None
    discovery_method: str | None = "MANUAL_API_SYNC"
    discovery_query: str | None = None
    topics: list[str] = Field(default_factory=list)


@router.post("/sync")
async def sync_dataset(req: DatasetSyncRequest):
    """Manual/debug path for importing an already-created Phase 1-4 response."""
    return await async_sync_phase14_response(
        req.response,
        batch_id=req.batch_id,
        target_domain=req.target_domain,
        discovery_method=req.discovery_method,
        discovery_query=req.discovery_query,
        topics=req.topics,
    )
