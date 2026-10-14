import json
import re
from collections import Counter, defaultdict

from services.history_service import (
    get_history_detail,
)


EMOTION_KEYS = [
    "positive",
    "neutral",
    "negative",
]

ABSA_SENTIMENT_KEYS = [
    "POSITIVE",
    "NEUTRAL",
    "NEGATIVE",
]


def _to_int(value):
    try:
        return int(value or 0)
    except Exception:
        return 0


def _compare_counts(
    source_a,
    source_b,
    keys,
):
    result = {}

    source_a = source_a or {}
    source_b = source_b or {}

    for key in keys:
        value_a = _to_int(
            source_a.get(key)
        )

        value_b = _to_int(
            source_b.get(key)
        )

        result[key] = {
            "a": value_a,
            "b": value_b,
            # Delta = B - A
            "delta": (
                value_b
                - value_a
            ),
        }

    return result


def _session_summary(
    analysis,
):
    return {
        "analysis_id":
            analysis.get(
                "analysis_id"
            ),

        "video_id":
            analysis.get(
                "video_id"
            ),

        "video_url":
            analysis.get(
                "video_url"
            ),

        "created_at":
            analysis.get(
                "created_at"
            ),

        "comment_count":
            analysis.get(
                "comment_count",
                0,
            ),

        "model_id":
            analysis.get(
                "model_id"
            ),

        "model_versions":
            analysis.get(
                "model_versions"
            )
            or {},

        "summary":
            analysis.get(
                "summary"
            )
            or {},

        "absa_summary":
            analysis.get(
                "absa_summary"
            )
            or {},
    }


def _collect_aspects(
    analysis,
):
    """
    Compare V1 giữ term theo output model.
    Chỉ lower/strip để "Camera" và "camera"
    không bị tách thành hai mục.

    Chưa merge "mùa 1" với "mùa1" vì đó là
    normalization experiment riêng về sau.
    """
    counts = Counter()

    sentiments = defaultdict(
        lambda: {
            "POSITIVE": 0,
            "NEUTRAL": 0,
            "NEGATIVE": 0,
        }
    )

    display_names = {}

    for comment in (
        analysis.get(
            "comments"
        )
        or []
    ):
        aspects = (
            (
                comment.get(
                    "absa"
                )
                or {}
            )
            .get(
                "aspects"
            )
            or []
        )

        for aspect in aspects:
            term = str(
                aspect.get(
                    "term",
                    "",
                )
            ).strip()

            if not term:
                continue

            key = term.lower()

            counts[
                key
            ] += 1

            if key not in display_names:
                display_names[
                    key
                ] = term

            sentiment = (
                aspect.get(
                    "sentiment"
                )
            )

            if (
                sentiment
                in sentiments[
                    key
                ]
            ):
                sentiments[
                    key
                ][
                    sentiment
                ] += 1

    return {
        "counts":
            counts,

        "sentiments":
            sentiments,

        "display_names":
            display_names,
    }


def _compare_aspects(
    analysis_a,
    analysis_b,
    limit=15,
):
    collected_a = (
        _collect_aspects(
            analysis_a
        )
    )

    collected_b = (
        _collect_aspects(
            analysis_b
        )
    )

    keys = set(
        collected_a[
            "counts"
        ].keys()
    )

    keys.update(
        collected_b[
            "counts"
        ].keys()
    )

    rows = []

    for key in keys:
        count_a = (
            collected_a[
                "counts"
            ].get(
                key,
                0,
            )
        )

        count_b = (
            collected_b[
                "counts"
            ].get(
                key,
                0,
            )
        )

        term = (
            collected_b[
                "display_names"
            ].get(
                key
            )
            or collected_a[
                "display_names"
            ].get(
                key
            )
            or key
        )

        rows.append({
            "term":
                term,

            "count_a":
                count_a,

            "count_b":
                count_b,

            "delta":
                count_b
                - count_a,

            "sentiments_a":
                dict(
                    collected_a[
                        "sentiments"
                    ].get(
                        key,
                        {
                            "POSITIVE": 0,
                            "NEUTRAL": 0,
                            "NEGATIVE": 0,
                        },
                    )
                ),

            "sentiments_b":
                dict(
                    collected_b[
                        "sentiments"
                    ].get(
                        key,
                        {
                            "POSITIVE": 0,
                            "NEUTRAL": 0,
                            "NEGATIVE": 0,
                        },
                    )
                ),
        })

    rows.sort(
        key=lambda row: (
            -max(
                row[
                    "count_a"
                ],
                row[
                    "count_b"
                ],
            ),
            -(
                row[
                    "count_a"
                ]
                + row[
                    "count_b"
                ]
            ),
            row[
                "term"
            ].lower(),
        )
    )

    return rows[
        : max(
            1,
            min(
                int(
                    limit
                ),
                50,
            ),
        )
    ]



def _normalize_comment_text(value):
    """
    Fallback identity khi crawler metadata không có comment_id.
    Không dùng normalization mạnh để tránh merge nhầm 2 comment khác nhau.
    """
    text = str(
        value
        or ""
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return (
        text
        .strip()
        .casefold()
    )


def _comment_identity(comment):
    """
    Ưu tiên ID thật từ TikTok/crawler.
    Chỉ fallback về normalized text khi không có ID.
    """
    for key in [
        "comment_id",
        "id",
    ]:
        value = comment.get(
            key
        )

        if value is not None:
            value = str(
                value
            ).strip()

            if value:
                return (
                    "id",
                    value,
                )

    normalized_text = (
        _normalize_comment_text(
            comment.get(
                "text"
            )
        )
    )

    return (
        "text",
        normalized_text,
    )


def _group_comments_by_identity(
    analysis,
):
    groups = defaultdict(
        list
    )

    for comment in (
        analysis.get(
            "comments"
        )
        or []
    ):
        key = (
            _comment_identity(
                comment
            )
        )

        groups[
            key
        ].append(
            comment
        )

    return groups


def _aspect_signature(
    comment,
):
    """
    Exact ABSA pair set theo comment:
        (normalized aspect term, sentiment)

    Span không đưa vào signature vì cùng term vẫn có thể dịch vị trí
    nếu text preprocessing thay đổi nhẹ; mục tiêu V1.1 là kiểm tra
    semantic ABSA output consistency.
    """
    pairs = set()

    aspects = (
        (
            comment.get(
                "absa"
            )
            or {}
        )
        .get(
            "aspects"
        )
        or []
    )

    for aspect in aspects:
        term = (
            str(
                aspect.get(
                    "term",
                    "",
                )
            )
            .strip()
            .casefold()
        )

        sentiment = str(
            aspect.get(
                "sentiment",
                "",
            )
            or ""
        ).strip()

        if not term:
            continue

        pairs.add(
            (
                term,
                sentiment,
            )
        )

    return pairs


def _canonical_context_value(
    value,
):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _compare_context_consistency(
    analysis_a,
    analysis_b,
):
    context_a = (
        analysis_a.get(
            "context"
        )
        or {}
    )

    context_b = (
        analysis_b.get(
            "context"
        )
        or {}
    )

    fields = [
        "topic",
        "tone",
        "visual_summary",
        "audience_context",
        "keywords",
        "scores",
        "source",
    ]

    changed_fields = []

    for field in fields:
        value_a = (
            context_a.get(
                field
            )
        )

        value_b = (
            context_b.get(
                field
            )
        )

        if (
            _canonical_context_value(
                value_a
            )
            !=
            _canonical_context_value(
                value_b
            )
        ):
            changed_fields.append(
                field
            )

    return {
        "same":
            len(
                changed_fields
            )
            == 0,

        "changed_fields":
            changed_fields,

        "tone_a":
            context_a.get(
                "tone"
            ),

        "tone_b":
            context_b.get(
                "tone"
            ),

        "topic_a":
            context_a.get(
                "topic"
            ),

        "topic_b":
            context_b.get(
                "topic"
            ),
    }


def _compare_comment_consistency(
    analysis_a,
    analysis_b,
):
    """
    Đo:
    - mức overlap dữ liệu comment giữa 2 lần phân tích
    - emotion label agreement trên comment chung
    - ABSA agreement trên comment chung
    - context Gemini có thay đổi hay không

    Đây là diagnostic, KHÔNG phải metric đánh giá accuracy với ground truth.
    """
    groups_a = (
        _group_comments_by_identity(
            analysis_a
        )
    )

    groups_b = (
        _group_comments_by_identity(
            analysis_b
        )
    )

    keys = (
        set(
            groups_a.keys()
        )
        |
        set(
            groups_b.keys()
        )
    )

    count_a = sum(
        len(
            values
        )
        for values
        in groups_a.values()
    )

    count_b = sum(
        len(
            values
        )
        for values
        in groups_b.values()
    )

    common_count = 0
    only_a_count = 0
    only_b_count = 0

    matched_pairs = []

    matched_by_id = 0
    matched_by_text = 0

    for key in keys:
        comments_a = (
            groups_a.get(
                key,
                [],
            )
        )

        comments_b = (
            groups_b.get(
                key,
                [],
            )
        )

        common_here = min(
            len(
                comments_a
            ),
            len(
                comments_b
            ),
        )

        common_count += (
            common_here
        )

        only_a_count += max(
            len(
                comments_a
            )
            - len(
                comments_b
            ),
            0,
        )

        only_b_count += max(
            len(
                comments_b
            )
            - len(
                comments_a
            ),
            0,
        )

        if common_here:
            if key[0] == "id":
                matched_by_id += (
                    common_here
                )
            else:
                matched_by_text += (
                    common_here
                )

        for index in range(
            common_here
        ):
            matched_pairs.append(
                (
                    comments_a[
                        index
                    ],
                    comments_b[
                        index
                    ],
                    key,
                )
            )

    union_count = (
        common_count
        + only_a_count
        + only_b_count
    )

    overlap_percent = (
        round(
            (
                common_count
                / union_count
                * 100
            ),
            2,
        )
        if union_count
        else 100.0
    )

    coverage_a_percent = (
        round(
            (
                common_count
                / count_a
                * 100
            ),
            2,
        )
        if count_a
        else 100.0
    )

    coverage_b_percent = (
        round(
            (
                common_count
                / count_b
                * 100
            ),
            2,
        )
        if count_b
        else 100.0
    )

    # --------------------------------------------------------
    # Emotion consistency
    # --------------------------------------------------------

    same_emotion = 0
    changed_emotion = 0

    transition_counter = Counter()
    changed_emotion_examples = []

    # --------------------------------------------------------
    # ABSA consistency
    # --------------------------------------------------------

    absa_exact_all = 0

    absa_relevant_common = 0
    absa_exact_relevant = 0

    absa_pair_common = 0
    absa_pair_union = 0

    for (
        comment_a,
        comment_b,
        identity_key,
    ) in matched_pairs:
        label_a = (
            comment_a.get(
                "label"
            )
        )

        label_b = (
            comment_b.get(
                "label"
            )
        )

        if (
            label_a
            == label_b
        ):
            same_emotion += 1

        else:
            changed_emotion += 1

            transition_counter[
                (
                    str(
                        label_a
                    ),
                    str(
                        label_b
                    ),
                )
            ] += 1

            if (
                len(
                    changed_emotion_examples
                )
                < 10
            ):
                changed_emotion_examples.append({
                    "identity_type":
                        identity_key[
                            0
                        ],

                    "comment_id":
                        (
                            identity_key[
                                1
                            ]
                            if identity_key[
                                0
                            ]
                            == "id"
                            else None
                        ),

                    "text":
                        (
                            comment_a.get(
                                "text"
                            )
                            or comment_b.get(
                                "text"
                            )
                            or ""
                        ),

                    "label_a":
                        label_a,

                    "label_b":
                        label_b,

                    "confidence_a":
                        comment_a.get(
                            "confidence"
                        ),

                    "confidence_b":
                        comment_b.get(
                            "confidence"
                        ),
                })

        signature_a = (
            _aspect_signature(
                comment_a
            )
        )

        signature_b = (
            _aspect_signature(
                comment_b
            )
        )

        if (
            signature_a
            == signature_b
        ):
            absa_exact_all += 1

        if (
            signature_a
            or signature_b
        ):
            absa_relevant_common += 1

            if (
                signature_a
                == signature_b
            ):
                absa_exact_relevant += 1

        absa_pair_common += len(
            signature_a
            & signature_b
        )

        absa_pair_union += len(
            signature_a
            | signature_b
        )

    emotion_agreement_percent = (
        round(
            (
                same_emotion
                / common_count
                * 100
            ),
            2,
        )
        if common_count
        else 0.0
    )

    absa_all_agreement_percent = (
        round(
            (
                absa_exact_all
                / common_count
                * 100
            ),
            2,
        )
        if common_count
        else 0.0
    )

    absa_relevant_agreement_percent = (
        round(
            (
                absa_exact_relevant
                / absa_relevant_common
                * 100
            ),
            2,
        )
        if absa_relevant_common
        else 100.0
    )

    absa_pair_overlap_percent = (
        round(
            (
                absa_pair_common
                / absa_pair_union
                * 100
            ),
            2,
        )
        if absa_pair_union
        else 100.0
    )

    transitions = [
        {
            "from":
                from_label,

            "to":
                to_label,

            "count":
                count,
        }
        for (
            from_label,
            to_label,
        ), count
        in transition_counter.most_common(
            10
        )
    ]

    return {
        "comments": {
            "count_a":
                count_a,

            "count_b":
                count_b,

            "common":
                common_count,

            "only_a":
                only_a_count,

            "only_b":
                only_b_count,

            "union":
                union_count,

            "overlap_percent":
                overlap_percent,

            "coverage_a_percent":
                coverage_a_percent,

            "coverage_b_percent":
                coverage_b_percent,

            "matched_by_id":
                matched_by_id,

            "matched_by_text_fallback":
                matched_by_text,
        },

        "emotion": {
            "same_label":
                same_emotion,

            "changed_label":
                changed_emotion,

            "agreement_percent":
                emotion_agreement_percent,

            "transitions":
                transitions,

            "changed_examples":
                changed_emotion_examples,
        },

        "absa": {
            "exact_all_common":
                absa_exact_all,

            "all_common_agreement_percent":
                absa_all_agreement_percent,

            "relevant_common":
                absa_relevant_common,

            "exact_relevant":
                absa_exact_relevant,

            "relevant_agreement_percent":
                absa_relevant_agreement_percent,

            "pair_common":
                absa_pair_common,

            "pair_union":
                absa_pair_union,

            "pair_overlap_percent":
                absa_pair_overlap_percent,
        },

        "video_context":
            _compare_context_consistency(
                analysis_a,
                analysis_b,
            ),
    }


def compare_analyses(
    analysis_id_a,
    analysis_id_b,
):
    analysis_id_a = int(
        analysis_id_a
    )

    analysis_id_b = int(
        analysis_id_b
    )

    if (
        analysis_id_a
        == analysis_id_b
    ):
        raise ValueError(
            "Cần chọn 2 phiên khác nhau để so sánh"
        )

    analysis_a = (
        get_history_detail(
            analysis_id_a
        )
    )

    analysis_b = (
        get_history_detail(
            analysis_id_b
        )
    )

    if analysis_a is None:
        raise LookupError(
            f"Không tìm thấy analysis_id={analysis_id_a}"
        )

    if analysis_b is None:
        raise LookupError(
            f"Không tìm thấy analysis_id={analysis_id_b}"
        )

    summary_a = (
        analysis_a.get(
            "summary"
        )
        or {}
    )

    summary_b = (
        analysis_b.get(
            "summary"
        )
        or {}
    )

    emotions_a = (
        analysis_a.get(
            "emotions"
        )
        or {}
    )

    emotions_b = (
        analysis_b.get(
            "emotions"
        )
        or {}
    )

    all_emotion_labels = sorted(
        set(
            emotions_a.keys()
        )
        | set(
            emotions_b.keys()
        )
    )

    emotion_detail = (
        _compare_counts(
            emotions_a,
            emotions_b,
            all_emotion_labels,
        )
    )

    absa_a = (
        analysis_a.get(
            "absa_summary"
        )
        or {}
    )

    absa_b = (
        analysis_b.get(
            "absa_summary"
        )
        or {}
    )

    absa_sentiments_a = (
        absa_a.get(
            "sentiments"
        )
        or {}
    )

    absa_sentiments_b = (
        absa_b.get(
            "sentiments"
        )
        or {}
    )

    return {
        "same_video":
            str(
                analysis_a.get(
                    "video_id"
                )
            )
            == str(
                analysis_b.get(
                    "video_id"
                )
            ),

        "analysis_a":
            _session_summary(
                analysis_a
            ),

        "analysis_b":
            _session_summary(
                analysis_b
            ),

        # Emotion dashboard:
        # positive / neutral / negative
        "emotion_summary":
            _compare_counts(
                summary_a,
                summary_b,
                EMOTION_KEYS,
            ),

        # Fine-grained emotion counts
        "emotions":
            emotion_detail,

        # Compare V1.1:
        # diagnostic tính ổn định dữ liệu / inference
        "consistency":
            _compare_comment_consistency(
                analysis_a,
                analysis_b,
            ),

        "absa": {
            "total_aspects": {
                "a":
                    _to_int(
                        absa_a.get(
                            "total_aspects"
                        )
                    ),

                "b":
                    _to_int(
                        absa_b.get(
                            "total_aspects"
                        )
                    ),

                "delta":
                    _to_int(
                        absa_b.get(
                            "total_aspects"
                        )
                    )
                    - _to_int(
                        absa_a.get(
                            "total_aspects"
                        )
                    ),
            },

            "comments_with_aspects": {
                "a":
                    _to_int(
                        absa_a.get(
                            "comments_with_aspects"
                        )
                    ),

                "b":
                    _to_int(
                        absa_b.get(
                            "comments_with_aspects"
                        )
                    ),

                "delta":
                    _to_int(
                        absa_b.get(
                            "comments_with_aspects"
                        )
                    )
                    - _to_int(
                        absa_a.get(
                            "comments_with_aspects"
                        )
                    ),
            },

            "sentiments":
                _compare_counts(
                    absa_sentiments_a,
                    absa_sentiments_b,
                    ABSA_SENTIMENT_KEYS,
                ),

            "top_aspects":
                _compare_aspects(
                    analysis_a,
                    analysis_b,
                    limit=15,
                ),
        },
    }
