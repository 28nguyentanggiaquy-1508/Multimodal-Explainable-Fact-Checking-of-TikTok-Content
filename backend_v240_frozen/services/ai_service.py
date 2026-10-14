from core.crawler import get_all_comments
from core.model_manager import model_manager

from core.comment_filter import filter_comments_by_context
from core.context_refiner import refine_prediction

from services.absa_service import absa_service
from services.history_service import save_analysis_history
from services.context_cache_service import get_stable_video_context
from services.analyst_cache_service import (
    get_cached_analyst_report,
)


try:
    import emoji
except Exception:
    emoji = None


def preprocess_crawled_comments(comments):
    processed = []

    for comment in comments:
        text = str(
            comment.get(
                "text",
                "",
            )
        )

        if "[Sticker]" in text:
            continue

        if emoji:
            text = emoji.demojize(
                text,
                language="en",
            )

        item = dict(comment)
        item["text"] = text

        processed.append(
            item
        )

    return processed


async def analyze_video(
    url,
    model_id=None,
    refresh_context=False,
):
    try:
        # =========================================================
        # 1. CRAWL COMMENTS
        # =========================================================

        comments, video_id = (
            get_all_comments(
                url,
                50,
            )
        )

        comments = (
            preprocess_crawled_comments(
                comments
            )
        )

        if not comments:
            return {
                "error":
                    "Không lấy được comment"
            }

        # =========================================================
        # 2. EXTRACT RAW TEXT
        # =========================================================

        raw_texts = [
            c["text"]
            for c in comments
            if c.get("text")
        ]

        # =========================================================
        # 3. VIDEO CONTEXT BY GEMINI FLASH VISION
        # =========================================================

        (
            video_context,
            context_cache_meta,
        ) = get_stable_video_context(
            video_url=url,
            video_id=video_id,
            comments=raw_texts,
            refresh=refresh_context,
        )

        print(
            "[CONTEXT CACHE]",
            context_cache_meta["status"],
            "video=",
            video_id,
            "fingerprint=",
            context_cache_meta["fingerprint"],
        )

        # =========================================================
        # 4. FILTER COMMENTS BY CONTEXT
        # =========================================================

        comments = (
            filter_comments_by_context(
                comments,
                video_context,
            )
        )

        if not comments:
            return {
                "error":
                    "Comments bị filter hết"
            }

        # =========================================================
        # 5. PREPARE TEXTS
        # =========================================================

        texts = [
            c["text"]
            for c in comments
        ]

        # =========================================================
        # 6. EMOTION PREDICTION
        # =========================================================

        engine = (
            model_manager.get(
                model_id
            )
        )

        preds = (
            engine.predict_batch(
                texts,
                video_context=
                    video_context,
            )
        )

        refined_preds = []

        for i, pred in enumerate(
            preds
        ):
            refined = (
                refine_prediction(
                    comments[i]["text"],
                    pred,
                    video_context,
                )
            )

            refined_preds.append(
                refined
            )

        preds = refined_preds

        # =========================================================
        # 7. ABSA
        # Chạy song song về mặt logic với emotion classification.
        # Mỗi comment nhận danh sách aspect + sentiment.
        # =========================================================

        try:
            absa_results = (
                absa_service
                .analyze_comments(
                    texts
                )
            )

            absa_summary = (
                absa_service
                .summarize(
                    absa_results
                )
            )

        except Exception as exc:
            # ABSA lỗi không làm hỏng toàn bộ /analyze.
            print(
                "[ABSA SERVICE ERROR]",
                exc,
            )

            absa_results = [
                {
                    "aspect_count": 0,
                    "aspects": [],
                    "error": str(exc),
                }
                for _ in texts
            ]

            absa_summary = {
                "comments_analyzed":
                    len(texts),

                "comments_with_aspects":
                    0,

                "total_aspects":
                    0,

                "sentiments": {
                    "POSITIVE": 0,
                    "NEUTRAL": 0,
                    "NEGATIVE": 0,
                },

                "top_aspects":
                    [],

                "error":
                    str(exc),
            }

        # =========================================================
        # 8. EMOTION SUMMARY
        # =========================================================

        pos = sum(
            1
            for p in preds
            if (
                p["label"]
                == "ENJOYMENT"
            )
        )

        neu = sum(
            1
            for p in preds
            if (
                p["label"]
                == "OTHER"
            )
        )

        neg = sum(
            1
            for p in preds
            if p["label"] in [
                "ANGER",
                "SADNESS",
                "FEAR",
                "DISGUST",
            ]
        )

        # =========================================================
        # 9. EMOTION STATS
        # =========================================================

        emotions_count = {
            label: 0
            for label in [
                "ANGER",
                "DISGUST",
                "ENJOYMENT",
                "FEAR",
                "OTHER",
                "SADNESS",
                "SURPRISE",
            ]
        }

        for p in preds:
            for lb in p[
                "all_labels"
            ]:
                if (
                    lb
                    in emotions_count
                ):
                    emotions_count[
                        lb
                    ] += 1

        # =========================================================
        # 10. AI INSIGHT
        # =========================================================

        analyst_result = (
            get_cached_analyst_report(
                video_id=
                    video_id,

                summary={
                    "positive":
                        pos,

                    "neutral":
                        neu,

                    "negative":
                        neg,
                },

                emotions=
                    emotions_count,

                video_context=
                    video_context,

                comments=
                    texts,

                emotion_model_id=
                    (
                        model_id
                        or "default"
                    ),

                context_fingerprint=
                    (
                        context_cache_meta
                        .get(
                            "fingerprint"
                        )
                        if isinstance(
                            context_cache_meta,
                            dict,
                        )
                        else None
                    ),
            )
        )

        insight = (
            analyst_result[
                "text"
            ]
        )

        analyst_cache_meta = (
            analyst_result[
                "meta"
            ]
        )

        print(
            "[ANALYST CACHE]",
            analyst_cache_meta[
                "status"
            ],
            "video=",
            video_id,
            "request_attempted=",
            analyst_cache_meta[
                "request_attempted"
            ],
        )

        # =========================================================
        # 11. COMMENT RESPONSE
        # =========================================================

        comment_results = []

        for i in range(
            len(preds)
        ):
            item = {
                "text":
                    comments[i][
                        "text"
                    ],

                "label":
                    preds[i][
                        "label"
                    ],

                "all_labels":
                    preds[i][
                        "all_labels"
                    ],

                "confidence":
                    round(
                        preds[i][
                            "confidence"
                        ],
                        4,
                    ),

                "probabilities": {
                    k:
                        round(
                            float(v),
                            4,
                        )
                    for k, v
                    in (
                        preds[i]
                        .get(
                            "probabilities"
                        )
                        or {}
                    ).items()
                },

                # NEW:
                # Kết quả ABSA của chính comment này.
                "absa":
                    absa_results[i],
            }

            # Giữ lại một số metadata crawler nếu có,
            # nhưng không bắt buộc frontend phải dùng.
            for key in [
                "comment_id",
                "id",
                "create_time",
                "created_at",
                "like_count",
                "likes",
                "reply_count",
            ]:
                if (
                    key
                    in comments[i]
                ):
                    item[key] = (
                        comments[i][
                            key
                        ]
                    )

            comment_results.append(
                item
            )

        # =========================================================
        # 12. RETURN
        # =========================================================

        response = {
            "video_id":
                video_id,

            "video_url":
                url,

            "model_id":
                model_id,

            # ABSA runtime/model versions.
            # History service uses these values so V1/V2 sessions
            # remain comparable and rollback-safe.
            "ate_model_version":
                absa_service.ate_model_version,

            "asc_model_version":
                absa_service.asc_model_version,

            "absa_runtime":
                absa_service.runtime_info(),

            # Gemini video context
            "context":
                video_context,

            # Context Stabilization diagnostics
            "context_cache":
                context_cache_meta,

            # Gemini Analyst cache/quota diagnostics
            "analyst_cache":
                analyst_cache_meta,

            # emotion dashboard
            "summary": {
                "positive":
                    pos,
                "neutral":
                    neu,
                "negative":
                    neg,
            },

            # fine-grained emotion chart
            "emotions":
                emotions_count,

            # ABSA tổng hợp
            "absa_summary":
                absa_summary,

            # AI report
            "insight":
                insight,

            # emotion + ABSA theo từng comment
            "comments":
                comment_results,
        }

        # =========================================================
        # 13. SAVE HISTORY
        # Database lỗi không làm hỏng kết quả /analyze.
        # =========================================================

        try:
            analysis_id = (
                save_analysis_history(
                    response
                )
            )

            response[
                "analysis_id"
            ] = analysis_id

            response[
                "history_saved"
            ] = True

        except Exception as exc:
            print(
                "[HISTORY SAVE ERROR]",
                exc,
            )

            response[
                "analysis_id"
            ] = None

            response[
                "history_saved"
            ] = False

            response[
                "history_error"
            ] = str(
                exc
            )

        return response

    except Exception as e:
        print(
            f"[AI SERVICE ERROR] "
            f"{e}"
        )

        return {
            "error":
                str(e)
        }
