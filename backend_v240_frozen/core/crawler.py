import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests


TIKTOK_COMMENT_API = "https://www.tiktok.com/api/comment/list/"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.tiktok.com/",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}


def extract_video_id(url: str) -> Optional[str]:
    """
    Lấy TikTok video ID từ URL dạng:
    https://www.tiktok.com/@username/video/123456789
    """

    if not url:
        return None

    match = re.search(r"/video/(\d+)", url)

    if not match:
        return None

    return match.group(1)


def _timestamp_to_iso(timestamp) -> Optional[str]:
    """
    Chuyển Unix timestamp TikTok thành ISO-8601 UTC.
    """

    try:
        timestamp = int(timestamp)

        if timestamp <= 0:
            return None

        return datetime.fromtimestamp(
            timestamp,
            tz=timezone.utc,
        ).isoformat()

    except (TypeError, ValueError, OSError):
        return None


def _parse_comment(comment: Dict) -> Optional[Dict]:
    """
    Chuẩn hóa một comment TikTok thành format dùng chung
    cho AI, database, ABSA và trend analysis.
    """

    if not isinstance(comment, dict):
        return None

    text = str(comment.get("text") or "").strip()

    if not text:
        return None

    user = comment.get("user") or {}

    if not isinstance(user, dict):
        user = {}

    create_time = comment.get("create_time")

    return {
        # TikTok comment ID
        "comment_id": str(
            comment.get("cid")
            or comment.get("comment_id")
            or ""
        ),

        # Nội dung comment
        "text": text,

        # Thời gian
        "created_at": (
            int(create_time)
            if str(create_time or "").isdigit()
            else None
        ),
        "created_at_iso": _timestamp_to_iso(create_time),

        # Engagement
        "like_count": int(
            comment.get("digg_count")
            or comment.get("like_count")
            or 0
        ),
        "reply_count": int(
            comment.get("reply_comment_total")
            or comment.get("reply_count")
            or 0
        ),

        # User metadata
        "author_id": str(
            user.get("uid")
            or user.get("id")
            or ""
        ),
        "author_name": str(
            user.get("nickname")
            or ""
        ),
        "author_unique_id": str(
            user.get("unique_id")
            or ""
        ),
    }


def get_all_comments(
    url: str,
    max_comments: int = 50,
) -> Tuple[List[Dict], Optional[str]]:
    """
    Crawl comments từ TikTok.

    Có hỗ trợ pagination bằng cursor.

    Returns:
        comments: List[Dict]
        video_id: str | None
    """

    video_id = extract_video_id(url)

    if not video_id:
        print("[Crawler] Không tìm thấy video ID")
        return [], None

    # Không cho giá trị bất hợp lý
    max_comments = max(1, min(int(max_comments), 5000))

    comments: List[Dict] = []

    # tránh duplicate comment
    seen_comment_ids = set()

    cursor = 0
    page_size = 20

    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    print(
        f"[Crawler] Start video={video_id}, "
        f"target={max_comments}"
    )

    while len(comments) < max_comments:

        remaining = max_comments - len(comments)
        count = min(page_size, remaining)

        params = {
            "aid": "1988",
            "aweme_id": video_id,
            "count": count,
            "cursor": cursor,
        }

        try:
            response = session.get(
                TIKTOK_COMMENT_API,
                params=params,
                timeout=15,
            )

            response.raise_for_status()

            data = response.json()

        except requests.RequestException as exc:
            print(
                f"[Crawler] Request error "
                f"cursor={cursor}: {exc}"
            )
            break

        except ValueError as exc:
            print(
                f"[Crawler] Invalid JSON "
                f"cursor={cursor}: {exc}"
            )
            break

        raw_comments = data.get("comments") or []

        if not raw_comments:
            print(
                f"[Crawler] Không còn comment "
                f"ở cursor={cursor}"
            )
            break

        added_this_page = 0

        for raw_comment in raw_comments:

            parsed = _parse_comment(raw_comment)

            if not parsed:
                continue

            comment_id = parsed.get("comment_id")

            # Nếu có ID thì chống duplicate
            if comment_id:
                if comment_id in seen_comment_ids:
                    continue

                seen_comment_ids.add(comment_id)

            comments.append(parsed)
            added_this_page += 1

            if len(comments) >= max_comments:
                break

        print(
            f"[Crawler] cursor={cursor} "
            f"received={len(raw_comments)} "
            f"added={added_this_page} "
            f"total={len(comments)}"
        )

        has_more = data.get("has_more")

        next_cursor = data.get("cursor")

        # TikTok có thể trả bool hoặc số
        if not has_more:
            break

        try:
            next_cursor = int(next_cursor)
        except (TypeError, ValueError):
            break

        # tránh infinite loop
        if next_cursor == cursor:
            print("[Crawler] Cursor không thay đổi, dừng crawl")
            break

        cursor = next_cursor

    print(
        f"[Crawler] Finished: "
        f"{len(comments)} comments"
    )

    return comments[:max_comments], video_id