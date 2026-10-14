from services.history_service import (
    get_history_by_video,
)

from services.compare_service import (
    compare_analyses,
)


def _to_int(value):
    try:
        return int(value or 0)
    except Exception:
        return 0


def _average(values):
    clean = [
        float(value)
        for value in values
        if value is not None
    ]

    if not clean:
        return None

    return round(
        sum(clean)
        / len(clean),
        2,
    )


def _point_from_history_item(
    item,
):
    summary = (
        item.get(
            "summary"
        )
        or {}
    )

    absa = (
        item.get(
            "absa_summary"
        )
        or {}
    )

    sentiments = (
        absa.get(
            "sentiments"
        )
        or {}
    )

    return {
        "analysis_id":
            item.get(
                "analysis_id"
            ),

        "created_at":
            item.get(
                "created_at"
            ),

        "comment_count":
            _to_int(
                item.get(
                    "comment_count"
                )
            ),

        "model_id":
            item.get(
                "model_id"
            ),

        "ate_model_version":
            item.get(
                "ate_model_version"
            ),

        "asc_model_version":
            item.get(
                "asc_model_version"
            ),

        "emotion": {
            "positive":
                _to_int(
                    summary.get(
                        "positive"
                    )
                ),

            "neutral":
                _to_int(
                    summary.get(
                        "neutral"
                    )
                ),

            "negative":
                _to_int(
                    summary.get(
                        "negative"
                    )
                ),
        },

        "absa": {
            "comments_with_aspects":
                _to_int(
                    absa.get(
                        "comments_with_aspects"
                    )
                ),

            "total_aspects":
                _to_int(
                    absa.get(
                        "total_aspects"
                    )
                ),

            "positive":
                _to_int(
                    sentiments.get(
                        "POSITIVE"
                    )
                ),

            "neutral":
                _to_int(
                    sentiments.get(
                        "NEUTRAL"
                    )
                ),

            "negative":
                _to_int(
                    sentiments.get(
                        "NEGATIVE"
                    )
                ),

            "top_aspects":
                absa.get(
                    "top_aspects"
                )
                or [],
        },
    }


def _unique_non_empty(
    values,
):
    return sorted(
        {
            str(value)
            for value in values
            if value is not None
            and str(value).strip()
        }
    )


def get_video_trend(
    video_id,
    limit=20,
):
    """
    Trend V1:
    - chỉ đọc History đã lưu
    - không crawl lại TikTok
    - không chạy lại model
    - các điểm được sắp từ cũ -> mới

    transition[i] mô tả consistency giữa:
        points[i] -> points[i+1]
    """
    limit = max(
        2,
        min(
            int(limit),
            50,
        ),
    )

    history = (
        get_history_by_video(
            str(
                video_id
            ),
            limit=limit,
        )
    )

    items = (
        history.get(
            "items"
        )
        or []
    )

    if not items:
        raise LookupError(
            f"Không có history cho video_id={video_id}"
        )

    # Repository trả newest -> oldest.
    # Trend phải hiển thị oldest -> newest.
    items = sorted(
        items,
        key=lambda item: (
            str(
                item.get(
                    "created_at",
                    "",
                )
            ),
            _to_int(
                item.get(
                    "analysis_id"
                )
            ),
        ),
    )

    points = [
        _point_from_history_item(
            item
        )
        for item in items
    ]

    transitions = []

    for index in range(
        1,
        len(
            points
        ),
    ):
        previous = points[
            index - 1
        ]

        current = points[
            index
        ]

        comparison = (
            compare_analyses(
                previous[
                    "analysis_id"
                ],
                current[
                    "analysis_id"
                ],
            )
        )

        consistency = (
            comparison.get(
                "consistency"
            )
            or {}
        )

        comments = (
            consistency.get(
                "comments"
            )
            or {}
        )

        emotion = (
            consistency.get(
                "emotion"
            )
            or {}
        )

        absa = (
            consistency.get(
                "absa"
            )
            or {}
        )

        context = (
            consistency.get(
                "video_context"
            )
            or {}
        )

        transitions.append({
            "from_analysis_id":
                previous[
                    "analysis_id"
                ],

            "to_analysis_id":
                current[
                    "analysis_id"
                ],

            "from_created_at":
                previous[
                    "created_at"
                ],

            "to_created_at":
                current[
                    "created_at"
                ],

            "comment_overlap_percent":
                comments.get(
                    "overlap_percent"
                ),

            "comment_common":
                comments.get(
                    "common"
                ),

            "comment_only_previous":
                comments.get(
                    "only_a"
                ),

            "comment_only_current":
                comments.get(
                    "only_b"
                ),

            "emotion_agreement_percent":
                emotion.get(
                    "agreement_percent"
                ),

            "emotion_changed_label_count":
                emotion.get(
                    "changed_label"
                ),

            "absa_agreement_percent":
                absa.get(
                    "relevant_agreement_percent"
                ),

            "absa_pair_overlap_percent":
                absa.get(
                    "pair_overlap_percent"
                ),

            "context_changed":
                not bool(
                    context.get(
                        "same",
                        False,
                    )
                ),

            "context_changed_fields":
                context.get(
                    "changed_fields"
                )
                or [],
        })

    emotion_models = (
        _unique_non_empty(
            point.get(
                "model_id"
            )
            for point in points
        )
    )

    ate_models = (
        _unique_non_empty(
            point.get(
                "ate_model_version"
            )
            for point in points
        )
    )

    asc_models = (
        _unique_non_empty(
            point.get(
                "asc_model_version"
            )
            for point in points
        )
    )

    context_changed_count = sum(
        1
        for transition
        in transitions
        if transition[
            "context_changed"
        ]
    )

    return {
        "video_id":
            str(
                video_id
            ),

        "session_count":
            len(
                points
            ),

        "transition_count":
            len(
                transitions
            ),

        "model_consistency": {
            "same_emotion_model":
                len(
                    emotion_models
                )
                <= 1,

            "same_ate_model":
                len(
                    ate_models
                )
                <= 1,

            "same_asc_model":
                len(
                    asc_models
                )
                <= 1,

            "emotion_models":
                emotion_models,

            "ate_models":
                ate_models,

            "asc_models":
                asc_models,
        },

        "stability_summary": {
            "avg_comment_overlap_percent":
                _average(
                    transition.get(
                        "comment_overlap_percent"
                    )
                    for transition in transitions
                ),

            "avg_emotion_agreement_percent":
                _average(
                    transition.get(
                        "emotion_agreement_percent"
                    )
                    for transition in transitions
                ),

            "avg_absa_agreement_percent":
                _average(
                    transition.get(
                        "absa_agreement_percent"
                    )
                    for transition in transitions
                ),

            "avg_absa_pair_overlap_percent":
                _average(
                    transition.get(
                        "absa_pair_overlap_percent"
                    )
                    for transition in transitions
                ),

            "context_changed_count":
                context_changed_count,

            "context_stable_count":
                len(
                    transitions
                )
                - context_changed_count,
        },

        "points":
            points,

        "transitions":
            transitions,
    }
