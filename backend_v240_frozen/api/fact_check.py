from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dataset_integration.bridge import async_sync_phase14_response
from dataset_integration.response_guard import normalize_factcheck_response_semantics
from fact_checking.pipeline import PIPELINE_VERSION, run_fact_check_pipeline
from fact_checking.url_utils import InvalidTikTokUrl, validate_tiktok_url

router = APIRouter(prefix="/fact-check", tags=["Fact Checking"])


class FactCheckRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2048)


@router.post("/fact-check")
async def fact_check(request: FactCheckRequest):
    try:
        video_url = validate_tiktok_url(request.url)
    except InvalidTikTokUrl as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = await run_fact_check_pipeline(video_url)
    # Normalize semantics before the dataset bridge receives the response.
    result = normalize_factcheck_response_semantics(result)
    dataset_sync = await async_sync_phase14_response(
        result,
        discovery_method="BACKEND_FACT_CHECK",
    )
    if isinstance(result, dict):
        result["dataset_sync"] = dataset_sync
    return result


@router.get("/health")
def fact_check_health():
    return {
        "status": "ok",
        "pipeline": "multimodal-evidence-based-fact-checking",
        "pipeline_version": PIPELINE_VERSION,
        "implemented_through": "phase_4",
        "execution_architecture": "cache-first/quota-aware/provider-agnostic/resumable",
    }
