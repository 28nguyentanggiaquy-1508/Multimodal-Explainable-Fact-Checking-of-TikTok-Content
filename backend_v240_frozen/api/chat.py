from fastapi import APIRouter
from services.ai_service import analyze_comments

router = APIRouter()

@router.post("/chat")
async def chat(data: dict):
    comments = data.get("comments", [])
    result = analyze_comments(comments)
    return {"data": result}