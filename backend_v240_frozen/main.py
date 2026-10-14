from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.fact_check import router as fact_check_router
from config import CORS_ALLOW_ORIGINS
from core.gemini_vision import get_or_download_video
from core.model_manager import get_default_model_id, list_model_options
from fact_checking.url_utils import InvalidTikTokUrl, validate_tiktok_url
from services.ai_service import analyze_video

from api.dataset_sync import router as dataset_sync_router

app = FastAPI(
    title="TikTok Analyzer & Explainable Multimodal Fact-Checking API",
    version="2.4.0-hybrid-quota-independent",
)
app.include_router(fact_check_router)
app.include_router(dataset_sync_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class Request(BaseModel):
    url: str
    model_id: Optional[str] = None


@app.post("/analyze")
async def analyze(req: Request):
    # Legacy emotion endpoint is intentionally preserved.
    return await analyze_video(req.url, model_id=req.model_id)


@app.get("/models")
def models():
    return {
        "default_model_id": get_default_model_id(),
        "models": list_model_options(),
    }


@app.get("/video/{video_id}")
def video_proxy(video_id: str, url: str = Query(...)):
    if not video_id.isdigit():
        raise HTTPException(status_code=400, detail="Invalid video id")

    try:
        safe_url = validate_tiktok_url(url)
    except InvalidTikTokUrl as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        path = get_or_download_video(safe_url, video_id=video_id)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot fetch TikTok video: {type(exc).__name__}",
        ) from exc

    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.get("/")
def root():
    return {
        "message": "TikTok Analyzer API is running",
        "fact_check_pipeline": "v2.4.0-hybrid-quota-independent",
    }
