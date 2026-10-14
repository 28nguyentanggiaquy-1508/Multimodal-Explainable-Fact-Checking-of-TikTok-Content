import json
from datetime import datetime, timezone

from .db import (
    get_connection,
    init_db,
)


def _json_dumps(value):
    if value is None:
        value = {}

    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    )


def _json_loads(value, default):
    if not value:
        return default

    try:
        return json.loads(value)
    except Exception:
        return default


def _utc_now_iso():
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


def save_analysis(
    result,
    *,
    ate_model_version="ate_phobert_v1",
    asc_model_version="asc_phobert_v1",
):
    """
    Lưu một kết quả /analyze hoàn chỉnh.

    Return:
        analysis_id: int
    """
    init_db()

    video_id = str(
        result.get(
            "video_id",
            "",
        )
        or ""
    )

    video_url = str(
        result.get(
            "video_url",
            "",
        )
        or ""
    )

    if not video_id:
        raise ValueError(
            "Không thể lưu history: thiếu video_id"
        )

    if not video_url:
        raise ValueError(
            "Không thể lưu history: thiếu video_url"
        )

    comments = (
        result.get(
            "comments"
        )
        or []
    )

    model_id = result.get(
        "model_id"
    )

    created_at = (
        result.get(
            "created_at"
        )
        or _utc_now_iso()
    )

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO analyses (
                video_id,
                video_url,
                model_id,
                emotion_model_version,
                ate_model_version,
                asc_model_version,
                created_at,
                comment_count,
                context_json,
                summary_json,
                emotions_json,
                absa_summary_json,
                insight
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                video_id,
                video_url,
                model_id,
                model_id,
                ate_model_version,
                asc_model_version,
                created_at,
                len(comments),
                _json_dumps(
                    result.get(
                        "context"
                    )
                ),
                _json_dumps(
                    result.get(
                        "summary"
                    )
                ),
                _json_dumps(
                    result.get(
                        "emotions"
                    )
                ),
                _json_dumps(
                    result.get(
                        "absa_summary"
                    )
                ),
                str(
                    result.get(
                        "insight",
                        "",
                    )
                    or ""
                ),
            ),
        )

        analysis_id = int(
            cursor.lastrowid
        )

        for comment in comments:
            source_comment_id = (
                comment.get(
                    "comment_id"
                )
                or comment.get(
                    "id"
                )
            )

            metadata = {}

            for key in [
                "comment_id",
                "id",
                "create_time",
                "created_at",
                "like_count",
                "likes",
                "reply_count",
            ]:
                if key in comment:
                    metadata[
                        key
                    ] = comment[
                        key
                    ]

            cursor.execute(
                """
                INSERT INTO comments (
                    analysis_id,
                    source_comment_id,
                    text,
                    emotion_label,
                    emotion_confidence,
                    all_labels_json,
                    probabilities_json,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    (
                        str(
                            source_comment_id
                        )
                        if source_comment_id
                        is not None
                        else None
                    ),
                    str(
                        comment.get(
                            "text",
                            "",
                        )
                    ),
                    comment.get(
                        "label"
                    ),
                    (
                        float(
                            comment.get(
                                "confidence"
                            )
                        )
                        if comment.get(
                            "confidence"
                        )
                        is not None
                        else None
                    ),
                    _json_dumps(
                        comment.get(
                            "all_labels"
                        )
                        or []
                    ),
                    _json_dumps(
                        comment.get(
                            "probabilities"
                        )
                        or {}
                    ),
                    _json_dumps(
                        metadata
                    ),
                ),
            )

            comment_row_id = int(
                cursor.lastrowid
            )

            absa = (
                comment.get(
                    "absa"
                )
                or {}
            )

            aspects = (
                absa.get(
                    "aspects"
                )
                or []
            )

            for aspect in aspects:
                cursor.execute(
                    """
                    INSERT INTO aspect_results (
                        comment_row_id,
                        term,
                        start_pos,
                        end_pos,
                        sentiment,
                        confidence,
                        probabilities_json,
                        ate_model_version,
                        asc_model_version
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        comment_row_id,
                        str(
                            aspect.get(
                                "term",
                                "",
                            )
                        ),
                        aspect.get(
                            "start"
                        ),
                        aspect.get(
                            "end"
                        ),
                        aspect.get(
                            "sentiment"
                        ),
                        (
                            float(
                                aspect.get(
                                    "confidence"
                                )
                            )
                            if aspect.get(
                                "confidence"
                            )
                            is not None
                            else None
                        ),
                        _json_dumps(
                            aspect.get(
                                "probabilities"
                            )
                            or {}
                        ),
                        ate_model_version,
                        asc_model_version,
                    ),
                )

        conn.commit()

        return analysis_id

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


def list_analyses(
    limit=20,
    offset=0,
):
    init_db()

    limit = max(
        1,
        min(
            int(limit),
            100,
        ),
    )

    offset = max(
        int(offset),
        0,
    )

    conn = get_connection()

    try:
        total_row = conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM analyses
            """
        ).fetchone()

        rows = conn.execute(
            """
            SELECT
                id,
                video_id,
                video_url,
                model_id,
                emotion_model_version,
                ate_model_version,
                asc_model_version,
                created_at,
                comment_count,
                summary_json,
                absa_summary_json
            FROM analyses
            ORDER BY id DESC
            LIMIT ?
            OFFSET ?
            """,
            (
                limit,
                offset,
            ),
        ).fetchall()

        items = []

        for row in rows:
            items.append({
                "analysis_id":
                    row["id"],

                "video_id":
                    row["video_id"],

                "video_url":
                    row["video_url"],

                "model_id":
                    row["model_id"],

                "emotion_model_version":
                    row[
                        "emotion_model_version"
                    ],

                "ate_model_version":
                    row[
                        "ate_model_version"
                    ],

                "asc_model_version":
                    row[
                        "asc_model_version"
                    ],

                "created_at":
                    row["created_at"],

                "comment_count":
                    row[
                        "comment_count"
                    ],

                "summary":
                    _json_loads(
                        row[
                            "summary_json"
                        ],
                        {},
                    ),

                "absa_summary":
                    _json_loads(
                        row[
                            "absa_summary_json"
                        ],
                        {},
                    ),
            })

        return {
            "total":
                int(
                    total_row["total"]
                ),

            "limit":
                limit,

            "offset":
                offset,

            "items":
                items,
        }

    finally:
        conn.close()


def get_analysis(
    analysis_id,
):
    init_db()

    conn = get_connection()

    try:
        analysis = conn.execute(
            """
            SELECT *
            FROM analyses
            WHERE id = ?
            """,
            (
                int(
                    analysis_id
                ),
            ),
        ).fetchone()

        if analysis is None:
            return None

        comment_rows = conn.execute(
            """
            SELECT *
            FROM comments
            WHERE analysis_id = ?
            ORDER BY id ASC
            """,
            (
                int(
                    analysis_id
                ),
            ),
        ).fetchall()

        comments = []

        for comment_row in comment_rows:
            aspect_rows = conn.execute(
                """
                SELECT *
                FROM aspect_results
                WHERE comment_row_id = ?
                ORDER BY id ASC
                """,
                (
                    comment_row["id"],
                ),
            ).fetchall()

            aspects = []

            for aspect_row in aspect_rows:
                aspects.append({
                    "term":
                        aspect_row[
                            "term"
                        ],

                    "start":
                        aspect_row[
                            "start_pos"
                        ],

                    "end":
                        aspect_row[
                            "end_pos"
                        ],

                    "sentiment":
                        aspect_row[
                            "sentiment"
                        ],

                    "confidence":
                        aspect_row[
                            "confidence"
                        ],

                    "probabilities":
                        _json_loads(
                            aspect_row[
                                "probabilities_json"
                            ],
                            {},
                        ),

                    "ate_model_version":
                        aspect_row[
                            "ate_model_version"
                        ],

                    "asc_model_version":
                        aspect_row[
                            "asc_model_version"
                        ],
                })

            metadata = _json_loads(
                comment_row[
                    "metadata_json"
                ],
                {},
            )

            item = {
                "text":
                    comment_row[
                        "text"
                    ],

                "label":
                    comment_row[
                        "emotion_label"
                    ],

                "all_labels":
                    _json_loads(
                        comment_row[
                            "all_labels_json"
                        ],
                        [],
                    ),

                "confidence":
                    comment_row[
                        "emotion_confidence"
                    ],

                "probabilities":
                    _json_loads(
                        comment_row[
                            "probabilities_json"
                        ],
                        {},
                    ),

                "absa": {
                    "aspect_count":
                        len(
                            aspects
                        ),

                    "aspects":
                        aspects,
                },
            }

            item.update(
                metadata
            )

            comments.append(
                item
            )

        return {
            "analysis_id":
                analysis["id"],

            "video_id":
                analysis[
                    "video_id"
                ],

            "video_url":
                analysis[
                    "video_url"
                ],

            "model_id":
                analysis[
                    "model_id"
                ],

            "model_versions": {
                "emotion":
                    analysis[
                        "emotion_model_version"
                    ],

                "ate":
                    analysis[
                        "ate_model_version"
                    ],

                "asc":
                    analysis[
                        "asc_model_version"
                    ],
            },

            "created_at":
                analysis[
                    "created_at"
                ],

            "comment_count":
                analysis[
                    "comment_count"
                ],

            "context":
                _json_loads(
                    analysis[
                        "context_json"
                    ],
                    {},
                ),

            "summary":
                _json_loads(
                    analysis[
                        "summary_json"
                    ],
                    {},
                ),

            "emotions":
                _json_loads(
                    analysis[
                        "emotions_json"
                    ],
                    {},
                ),

            "absa_summary":
                _json_loads(
                    analysis[
                        "absa_summary_json"
                    ],
                    {},
                ),

            "insight":
                analysis[
                    "insight"
                ],

            "comments":
                comments,
        }

    finally:
        conn.close()


def get_video_history(
    video_id,
    limit=20,
):
    init_db()

    limit = max(
        1,
        min(
            int(limit),
            100,
        ),
    )

    conn = get_connection()

    try:
        rows = conn.execute(
            """
            SELECT
                id,
                video_id,
                video_url,
                model_id,
                ate_model_version,
                asc_model_version,
                created_at,
                comment_count,
                summary_json,
                absa_summary_json
            FROM analyses
            WHERE video_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (
                str(
                    video_id
                ),
                limit,
            ),
        ).fetchall()

        items = []

        for row in rows:
            items.append({
                "analysis_id":
                    row["id"],

                "video_id":
                    row["video_id"],

                "video_url":
                    row["video_url"],

                "model_id":
                    row["model_id"],

                "ate_model_version":
                    row[
                        "ate_model_version"
                    ],

                "asc_model_version":
                    row[
                        "asc_model_version"
                    ],

                "created_at":
                    row["created_at"],

                "comment_count":
                    row[
                        "comment_count"
                    ],

                "summary":
                    _json_loads(
                        row[
                            "summary_json"
                        ],
                        {},
                    ),

                "absa_summary":
                    _json_loads(
                        row[
                            "absa_summary_json"
                        ],
                        {},
                    ),
            })

        return {
            "video_id":
                str(
                    video_id
                ),

            "total":
                len(
                    items
                ),

            "items":
                items,
        }

    finally:
        conn.close()


def delete_analysis(
    analysis_id,
):
    init_db()

    conn = get_connection()

    try:
        cursor = conn.execute(
            """
            DELETE FROM analyses
            WHERE id = ?
            """,
            (
                int(
                    analysis_id
                ),
            ),
        )

        conn.commit()

        return (
            cursor.rowcount
            > 0
        )

    finally:
        conn.close()
