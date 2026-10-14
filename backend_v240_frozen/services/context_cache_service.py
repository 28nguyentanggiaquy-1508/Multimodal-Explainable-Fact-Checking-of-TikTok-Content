import hashlib
import json
from datetime import datetime, timezone

from config import GEMINI_VISION_MODEL
from core.gemini_vision import analyze_tiktok_video_context
from database.db import get_connection


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _json_loads(value):
    if not value:
        return None

    try:
        return json.loads(value)
    except Exception:
        return None


def _fingerprint(context):
    raw = _json_dumps(context or {})

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]


def ensure_context_cache_table():
    conn = get_connection()

    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS video_context_cache (
                video_id TEXT PRIMARY KEY,
                video_url TEXT NOT NULL,
                context_json TEXT NOT NULL,
                model_name TEXT,
                context_source TEXT,
                fingerprint TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_video_context_cache_updated_at
            ON video_context_cache(updated_at DESC);
            """
        )

        conn.commit()

    finally:
        conn.close()


def get_cached_video_context(video_id):
    ensure_context_cache_table()

    conn = get_connection()

    try:
        row = conn.execute(
            """
            SELECT *
            FROM video_context_cache
            WHERE video_id = ?
            """,
            (str(video_id),),
        ).fetchone()

        if row is None:
            return None

        context = _json_loads(
            row["context_json"]
        )

        if not isinstance(context, dict):
            return None

        return {
            "video_id": row["video_id"],
            "video_url": row["video_url"],
            "context": context,
            "model_name": row["model_name"],
            "context_source": row["context_source"],
            "fingerprint":
                row["fingerprint"]
                or _fingerprint(context),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    finally:
        conn.close()


def save_video_context(
    *,
    video_id,
    video_url,
    context,
):
    ensure_context_cache_table()

    now = _utc_now_iso()

    source = str(
        (context or {}).get(
            "source",
            "",
        )
        or ""
    )

    fingerprint = _fingerprint(
        context
    )

    conn = get_connection()

    try:
        existing = conn.execute(
            """
            SELECT created_at
            FROM video_context_cache
            WHERE video_id = ?
            """,
            (str(video_id),),
        ).fetchone()

        created_at = (
            existing["created_at"]
            if existing
            else now
        )

        conn.execute(
            """
            INSERT INTO video_context_cache (
                video_id,
                video_url,
                context_json,
                model_name,
                context_source,
                fingerprint,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(video_id)
            DO UPDATE SET
                video_url = excluded.video_url,
                context_json = excluded.context_json,
                model_name = excluded.model_name,
                context_source = excluded.context_source,
                fingerprint = excluded.fingerprint,
                updated_at = excluded.updated_at
            """,
            (
                str(video_id),
                str(video_url),
                _json_dumps(context),
                GEMINI_VISION_MODEL,
                source,
                fingerprint,
                created_at,
                now,
            ),
        )

        conn.commit()

    finally:
        conn.close()

    return {
        "fingerprint": fingerprint,
        "created_at": created_at,
        "updated_at": now,
        "model_name": GEMINI_VISION_MODEL,
        "context_source": source,
    }


def delete_cached_video_context(video_id):
    ensure_context_cache_table()

    conn = get_connection()

    try:
        cursor = conn.execute(
            """
            DELETE FROM video_context_cache
            WHERE video_id = ?
            """,
            (str(video_id),),
        )

        conn.commit()

        return cursor.rowcount > 0

    finally:
        conn.close()


def _is_cache_compatible(cached):
    if not cached:
        return False

    cached_model = cached.get(
        "model_name"
    )

    if (
        cached_model
        and cached_model
        != GEMINI_VISION_MODEL
    ):
        return False

    context = (
        cached.get("context")
        or {}
    )

    # Chỉ freeze Gemini Vision context thành công.
    # Local fallback do lỗi API/network không được cache cố định.
    return (
        context.get("source")
        == "gemini_flash_vision"
    )


def get_stable_video_context(
    *,
    video_url,
    video_id,
    comments=None,
    refresh=False,
):
    """
    Context Stabilization V1.

    refresh=False:
        cache tốt đã có -> HIT.

    refresh=True:
        gọi Gemini Vision lại.
        - thành công -> REFRESHED + overwrite cache.
        - lỗi tạm thời -> dùng cache cũ nếu có.
    """
    cached = get_cached_video_context(
        video_id
    )

    compatible_cache = (
        cached
        if _is_cache_compatible(cached)
        else None
    )

    if (
        compatible_cache
        and not refresh
    ):
        return (
            compatible_cache["context"],
            {
                "status": "HIT",
                "hit": True,
                "refreshed": False,
                "saved": True,
                "fingerprint":
                    compatible_cache[
                        "fingerprint"
                    ],
                "model_name":
                    compatible_cache[
                        "model_name"
                    ],
                "cached_at":
                    compatible_cache[
                        "updated_at"
                    ],
                "context_source":
                    compatible_cache[
                        "context_source"
                    ],
                "refresh_failed": False,
            },
        )

    new_context = (
        analyze_tiktok_video_context(
            video_url,
            comments=comments or [],
        )
    )

    source = (
        (new_context or {})
        .get("source")
    )

    if (
        isinstance(new_context, dict)
        and source
        == "gemini_flash_vision"
    ):
        saved = save_video_context(
            video_id=video_id,
            video_url=video_url,
            context=new_context,
        )

        return (
            new_context,
            {
                "status":
                    "REFRESHED"
                    if refresh
                    else "MISS_SAVED",
                "hit": False,
                "refreshed": bool(refresh),
                "saved": True,
                "fingerprint":
                    saved["fingerprint"],
                "model_name":
                    saved["model_name"],
                "cached_at":
                    saved["updated_at"],
                "context_source":
                    saved["context_source"],
                "refresh_failed": False,
            },
        )

    if compatible_cache:
        return (
            compatible_cache["context"],
            {
                "status":
                    "REFRESH_FAILED_USING_CACHE",
                "hit": True,
                "refreshed": False,
                "saved": True,
                "fingerprint":
                    compatible_cache[
                        "fingerprint"
                    ],
                "model_name":
                    compatible_cache[
                        "model_name"
                    ],
                "cached_at":
                    compatible_cache[
                        "updated_at"
                    ],
                "context_source":
                    compatible_cache[
                        "context_source"
                    ],
                "refresh_failed": True,
                "refresh_error":
                    (new_context or {})
                    .get("error"),
            },
        )

    return (
        new_context,
        {
            "status":
                "FALLBACK_NOT_CACHED",
            "hit": False,
            "refreshed": False,
            "saved": False,
            "fingerprint":
                _fingerprint(
                    new_context
                ),
            "model_name":
                GEMINI_VISION_MODEL,
            "cached_at": None,
            "context_source":
                source
                or "fallback",
            "refresh_failed":
                bool(refresh),
        },
    )


def get_context_cache_info(video_id):
    cached = get_cached_video_context(
        video_id
    )

    if cached is None:
        return None

    return {
        "video_id":
            cached["video_id"],
        "video_url":
            cached["video_url"],
        "model_name":
            cached["model_name"],
        "context_source":
            cached["context_source"],
        "fingerprint":
            cached["fingerprint"],
        "created_at":
            cached["created_at"],
        "updated_at":
            cached["updated_at"],
        "context":
            cached["context"],
    }
