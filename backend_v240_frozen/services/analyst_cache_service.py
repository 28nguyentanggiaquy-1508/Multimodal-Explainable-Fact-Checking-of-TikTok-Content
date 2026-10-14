import hashlib
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import GEMINI_ANALYST_MODEL
from core.gemini_vision import (
    build_local_emotion_report,
    generate_emotion_analyst_report,
)


BACKEND_DIR = Path(__file__).resolve().parents[1]

DB_PATH = Path(
    os.getenv(
        "TIKTOK_ANALYZER_DB_PATH",
        str(BACKEND_DIR / "data" / "tiktok_analyzer.db"),
    )
)

CACHE_TTL_SECONDS = int(
    os.getenv(
        "GEMINI_ANALYST_CACHE_TTL_SECONDS",
        "21600",
    )
)

STALE_MAX_AGE_SECONDS = int(
    os.getenv(
        "GEMINI_ANALYST_STALE_MAX_AGE_SECONDS",
        "604800",
    )
)

QUOTA_COOLDOWN_SECONDS = int(
    os.getenv(
        "GEMINI_ANALYST_QUOTA_COOLDOWN_SECONDS",
        "1800",
    )
)

ERROR_COOLDOWN_SECONDS = int(
    os.getenv(
        "GEMINI_ANALYST_ERROR_COOLDOWN_SECONDS",
        "300",
    )
)

PROFILE_BUCKET_PERCENT = max(
    1,
    int(
        os.getenv(
            "GEMINI_ANALYST_PROFILE_BUCKET_PERCENT",
            "10",
        )
    ),
)

CACHE_SCHEMA_VERSION = "analyst-cache-v1"


def _now():
    return datetime.now(timezone.utc)


def _iso(value):
    if value is None:
        return None

    return (
        value
        .astimezone(timezone.utc)
        .isoformat()
    )


def _parse_time(value):
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(str(value))

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc)

    except Exception:
        return None


def _canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value):
    return hashlib.sha256(
        str(value).encode("utf-8")
    ).hexdigest()


def _connect():
    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
    )

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


def _ensure_tables():
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gemini_analyst_cache (
                cache_key TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                context_fingerprint TEXT,
                emotion_model_id TEXT,
                analyst_model TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                report_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_gemini_analyst_cache_video
            ON gemini_analyst_cache(
                video_id,
                analyst_model,
                updated_at
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gemini_analyst_state (
                state_key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def _bucket_percent(count, total):
    if total <= 0:
        return 0

    percent = (
        100.0
        * float(count)
        / float(total)
    )

    bucket = int(
        round(
            percent
            / PROFILE_BUCKET_PERCENT
        )
        * PROFILE_BUCKET_PERCENT
    )

    return max(
        0,
        min(100, bucket),
    )


def _build_profile(summary, emotions):
    summary = summary if isinstance(summary, dict) else {}
    emotions = emotions if isinstance(emotions, dict) else {}

    positive = int(summary.get("positive", 0) or 0)
    neutral = int(summary.get("neutral", 0) or 0)
    negative = int(summary.get("negative", 0) or 0)

    total = max(
        positive + neutral + negative,
        1,
    )

    ranked_emotions = sorted(
        (
            (str(name), int(value or 0))
            for name, value in emotions.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )

    return {
        "schema": CACHE_SCHEMA_VERSION,
        "sentiment_bucket_pct": {
            "positive": _bucket_percent(
                positive,
                total,
            ),
            "neutral": _bucket_percent(
                neutral,
                total,
            ),
            "negative": _bucket_percent(
                negative,
                total,
            ),
        },
        "top_emotions": [
            name
            for name, _
            in ranked_emotions[:2]
        ],
    }


def _fallback_context_fingerprint(video_context):
    safe_context = (
        video_context
        if isinstance(video_context, dict)
        else {"value": str(video_context or "")}
    )

    return _sha256(
        _canonical_json(safe_context)
    )[:16]


def _build_cache_key(
    video_id,
    context_fingerprint,
    emotion_model_id,
    profile,
):
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "video_id": str(video_id),
        "context_fingerprint": str(context_fingerprint),
        "emotion_model_id": str(emotion_model_id),
        "analyst_model": GEMINI_ANALYST_MODEL,
        "profile": profile,
    }

    return _sha256(
        _canonical_json(payload)
    )


def _row_age_seconds(row, now=None):
    now = now or _now()
    updated = _parse_time(row["updated_at"])

    if updated is None:
        return None

    return max(
        0.0,
        (now - updated).total_seconds(),
    )


def _get_exact(cache_key):
    _ensure_tables()

    with _connect() as conn:
        return conn.execute(
            """
            SELECT *
            FROM gemini_analyst_cache
            WHERE cache_key = ?
            LIMIT 1
            """,
            (cache_key,),
        ).fetchone()


def _get_latest_for_video(
    video_id,
    analyst_model,
    context_fingerprint=None,
    emotion_model_id=None,
):
    _ensure_tables()

    clauses = [
        "video_id = ?",
        "analyst_model = ?",
    ]

    params = [
        str(video_id),
        str(analyst_model),
    ]

    if context_fingerprint:
        clauses.append("context_fingerprint = ?")
        params.append(str(context_fingerprint))

    if emotion_model_id:
        clauses.append("emotion_model_id = ?")
        params.append(str(emotion_model_id))

    sql = f"""
        SELECT *
        FROM gemini_analyst_cache
        WHERE {" AND ".join(clauses)}
        ORDER BY updated_at DESC
        LIMIT 1
    """

    with _connect() as conn:
        return conn.execute(
            sql,
            tuple(params),
        ).fetchone()


def _touch(cache_key):
    with _connect() as conn:
        conn.execute(
            """
            UPDATE gemini_analyst_cache
            SET last_used_at = ?
            WHERE cache_key = ?
            """,
            (
                _iso(_now()),
                cache_key,
            ),
        )


def _save(
    *,
    cache_key,
    video_id,
    context_fingerprint,
    emotion_model_id,
    profile,
    report_text,
):
    now_text = _iso(_now())

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO gemini_analyst_cache (
                cache_key,
                video_id,
                context_fingerprint,
                emotion_model_id,
                analyst_model,
                profile_json,
                report_text,
                created_at,
                updated_at,
                last_used_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key)
            DO UPDATE SET
                report_text = excluded.report_text,
                profile_json = excluded.profile_json,
                updated_at = excluded.updated_at,
                last_used_at = excluded.last_used_at
            """,
            (
                cache_key,
                str(video_id),
                str(context_fingerprint),
                str(emotion_model_id),
                str(GEMINI_ANALYST_MODEL),
                _canonical_json(profile),
                str(report_text),
                now_text,
                now_text,
                now_text,
            ),
        )


def _state_key():
    return (
        "gemini_analyst_cooldown:"
        + str(GEMINI_ANALYST_MODEL)
    )


def _get_cooldown():
    _ensure_tables()

    with _connect() as conn:
        row = conn.execute(
            """
            SELECT value_json
            FROM gemini_analyst_state
            WHERE state_key = ?
            LIMIT 1
            """,
            (_state_key(),),
        ).fetchone()

    if row is None:
        return None

    try:
        data = json.loads(row["value_json"])
    except Exception:
        return None

    until = _parse_time(data.get("until"))

    if until is None or until <= _now():
        return None

    return {
        **data,
        "until": _iso(until),
    }


def _set_cooldown(seconds, reason, error=None):
    until = _now() + timedelta(
        seconds=max(1, int(seconds))
    )

    payload = {
        "until": _iso(until),
        "reason": str(reason),
        "error": str(error) if error else None,
    }

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO gemini_analyst_state (
                state_key,
                value_json,
                updated_at
            )
            VALUES (?, ?, ?)
            ON CONFLICT(state_key)
            DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = excluded.updated_at
            """,
            (
                _state_key(),
                _canonical_json(payload),
                _iso(_now()),
            ),
        )

    return payload


def _clear_cooldown():
    with _connect() as conn:
        conn.execute(
            """
            DELETE FROM gemini_analyst_state
            WHERE state_key = ?
            """,
            (_state_key(),),
        )


def _cached_response(
    row,
    *,
    status,
    cache_key,
    profile,
    context_fingerprint,
    emotion_model_id,
    request_attempted=False,
    cooldown=None,
):
    age = _row_age_seconds(row)

    _touch(row["cache_key"])

    return {
        "text": row["report_text"],
        "meta": {
            "status": status,
            "cache_hit": True,
            "source": "gemini_analyst_cache",
            "analyst_model": GEMINI_ANALYST_MODEL,
            "cache_key": cache_key[:16],
            "stored_cache_key": str(row["cache_key"])[:16],
            "context_fingerprint": context_fingerprint,
            "emotion_model_id": emotion_model_id,
            "profile": profile,
            "cache_age_seconds": (
                round(age, 1)
                if age is not None
                else None
            ),
            "request_attempted": request_attempted,
            "cooldown": cooldown,
        },
    }


def _local_response(
    *,
    summary,
    emotions,
    video_context,
    status,
    cache_key,
    profile,
    context_fingerprint,
    emotion_model_id,
    request_attempted,
    error=None,
    cooldown=None,
):
    return {
        "text": build_local_emotion_report(
            summary,
            emotions,
            video_context,
        ),
        "meta": {
            "status": status,
            "cache_hit": False,
            "source": "local_fallback",
            "analyst_model": GEMINI_ANALYST_MODEL,
            "cache_key": cache_key[:16],
            "context_fingerprint": context_fingerprint,
            "emotion_model_id": emotion_model_id,
            "profile": profile,
            "request_attempted": request_attempted,
            "error": error,
            "cooldown": cooldown,
        },
    }


def get_cached_analyst_report(
    *,
    video_id,
    summary,
    emotions,
    video_context=None,
    comments=None,
    emotion_model_id=None,
    context_fingerprint=None,
    refresh=False,
):
    """
    Quota-safe Gemini Analyst flow.

    - Same video/context/profile inside TTL: zero Gemini request.
    - Success: save only real Gemini output.
    - Quota/rate-limit: 30-min circuit breaker.
    - 5xx/timeout: short circuit breaker.
    - During cooldown: zero Gemini request.
    - On failure: stale successful Gemini cache first, then local fallback.
    """

    _ensure_tables()

    video_id = str(video_id or "unknown")
    emotion_model_id = str(emotion_model_id or "default")

    context_fingerprint = str(
        context_fingerprint
        or _fallback_context_fingerprint(
            video_context
        )
    )

    profile = _build_profile(
        summary,
        emotions,
    )

    cache_key = _build_cache_key(
        video_id,
        context_fingerprint,
        emotion_model_id,
        profile,
    )

    now = _now()

    exact = _get_exact(cache_key)

    if not refresh and exact is not None:
        age = _row_age_seconds(
            exact,
            now,
        )

        if (
            age is not None
            and age <= CACHE_TTL_SECONDS
        ):
            return _cached_response(
                exact,
                status="HIT",
                cache_key=cache_key,
                profile=profile,
                context_fingerprint=context_fingerprint,
                emotion_model_id=emotion_model_id,
                request_attempted=False,
            )

    cooldown = _get_cooldown()

    if cooldown is not None:
        stale = (
            exact
            or _get_latest_for_video(
                video_id,
                GEMINI_ANALYST_MODEL,
                context_fingerprint=context_fingerprint,
                emotion_model_id=emotion_model_id,
            )
            or _get_latest_for_video(
                video_id,
                GEMINI_ANALYST_MODEL,
            )
        )

        if stale is not None:
            age = _row_age_seconds(
                stale,
                now,
            )

            if (
                age is not None
                and age <= STALE_MAX_AGE_SECONDS
            ):
                return _cached_response(
                    stale,
                    status="COOLDOWN_STALE_CACHE",
                    cache_key=cache_key,
                    profile=profile,
                    context_fingerprint=context_fingerprint,
                    emotion_model_id=emotion_model_id,
                    request_attempted=False,
                    cooldown=cooldown,
                )

        return _local_response(
            summary=summary,
            emotions=emotions,
            video_context=video_context,
            status="COOLDOWN_LOCAL_FALLBACK",
            cache_key=cache_key,
            profile=profile,
            context_fingerprint=context_fingerprint,
            emotion_model_id=emotion_model_id,
            request_attempted=False,
            cooldown=cooldown,
        )

    generated = generate_emotion_analyst_report(
        summary=summary,
        emotions=emotions,
        video_context=video_context,
        comments=comments,
    )

    if generated.get("ok"):
        _save(
            cache_key=cache_key,
            video_id=video_id,
            context_fingerprint=context_fingerprint,
            emotion_model_id=emotion_model_id,
            profile=profile,
            report_text=generated["text"],
        )

        _clear_cooldown()

        return {
            "text": generated["text"],
            "meta": {
                "status": (
                    "REFRESHED_SAVED"
                    if refresh
                    else "MISS_SAVED"
                ),
                "cache_hit": False,
                "source": "gemini_analyst",
                "analyst_model": GEMINI_ANALYST_MODEL,
                "cache_key": cache_key[:16],
                "context_fingerprint": context_fingerprint,
                "emotion_model_id": emotion_model_id,
                "profile": profile,
                "request_attempted": True,
                "cooldown": None,
            },
        }

    cooldown = None

    if generated.get("quota_limited"):
        cooldown = _set_cooldown(
            QUOTA_COOLDOWN_SECONDS,
            reason="quota",
            error=generated.get("error"),
        )
        failure_prefix = "QUOTA"

    elif generated.get("retryable"):
        cooldown = _set_cooldown(
            ERROR_COOLDOWN_SECONDS,
            reason="transient_error",
            error=generated.get("error"),
        )
        failure_prefix = "ERROR"

    else:
        failure_prefix = (
            "NO_KEY"
            if not generated.get("request_attempted")
            else "ERROR"
        )

    stale = (
        exact
        or _get_latest_for_video(
            video_id,
            GEMINI_ANALYST_MODEL,
            context_fingerprint=context_fingerprint,
            emotion_model_id=emotion_model_id,
        )
        or _get_latest_for_video(
            video_id,
            GEMINI_ANALYST_MODEL,
        )
    )

    if stale is not None:
        age = _row_age_seconds(
            stale,
            now,
        )

        if (
            age is not None
            and age <= STALE_MAX_AGE_SECONDS
        ):
            return _cached_response(
                stale,
                status=f"{failure_prefix}_STALE_CACHE",
                cache_key=cache_key,
                profile=profile,
                context_fingerprint=context_fingerprint,
                emotion_model_id=emotion_model_id,
                request_attempted=bool(
                    generated.get("request_attempted")
                ),
                cooldown=cooldown,
            )

    return {
        "text": (
            generated.get("text")
            or build_local_emotion_report(
                summary,
                emotions,
                video_context,
            )
        ),
        "meta": {
            "status": f"{failure_prefix}_LOCAL_FALLBACK",
            "cache_hit": False,
            "source": "local_fallback",
            "analyst_model": GEMINI_ANALYST_MODEL,
            "cache_key": cache_key[:16],
            "context_fingerprint": context_fingerprint,
            "emotion_model_id": emotion_model_id,
            "profile": profile,
            "request_attempted": bool(
                generated.get("request_attempted")
            ),
            "error": generated.get("error"),
            "cooldown": cooldown,
        },
    }


def get_analyst_cache_status(video_id):
    """
    Diagnostic helper. Never calls Gemini.
    """
    _ensure_tables()

    row = _get_latest_for_video(
        video_id,
        GEMINI_ANALYST_MODEL,
    )

    cooldown = _get_cooldown()

    if row is None:
        return {
            "video_id": str(video_id),
            "cached": False,
            "cooldown": cooldown,
        }

    return {
        "video_id": str(video_id),
        "cached": True,
        "cache_age_seconds": _row_age_seconds(row),
        "updated_at": row["updated_at"],
        "last_used_at": row["last_used_at"],
        "analyst_model": row["analyst_model"],
        "emotion_model_id": row["emotion_model_id"],
        "context_fingerprint": row["context_fingerprint"],
        "cooldown": cooldown,
    }
