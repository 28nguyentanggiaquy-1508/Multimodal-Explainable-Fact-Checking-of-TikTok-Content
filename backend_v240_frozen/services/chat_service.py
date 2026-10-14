from services.ai_service import analyze_video

def handle_message(message: str):
    try:
        if "tiktok.com" in message:
            return analyze_video(message)
        return "Hãy gửi link TikTok để phân tích."
    except Exception as e:
        return f"Lỗi: {str(e)}"