from __future__ import annotations

from urllib.parse import urlparse


class InvalidTikTokUrl(ValueError):
    pass


def validate_tiktok_url(url: str) -> str:
    value = (url or "").strip()

    if not value or len(value) > 2048:
        raise InvalidTikTokUrl("TikTok URL không hợp lệ.")

    try:
        parsed = urlparse(value)
    except Exception as exc:
        raise InvalidTikTokUrl(f"URL không hợp lệ: {exc}") from exc

    host = (parsed.hostname or "").lower().rstrip(".")

    if parsed.scheme not in {"http", "https"}:
        raise InvalidTikTokUrl("Chỉ chấp nhận URL dùng http/https.")

    if not (host == "tiktok.com" or host.endswith(".tiktok.com")):
        raise InvalidTikTokUrl("Chỉ chấp nhận URL thuộc tiktok.com.")

    return value
