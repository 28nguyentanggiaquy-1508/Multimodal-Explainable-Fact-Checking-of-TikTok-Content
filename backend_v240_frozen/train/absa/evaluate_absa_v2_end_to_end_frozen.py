import json

from config_ate_v2 import (
    TEST_FILE as ATE_TEST_FILE,
    BACKEND_DIR,
)

from config_asc_v2 import (
    TEST_FILE as ASC_TEST_FILE,
    LABEL2ID,
)

from metrics_ate_v2 import (
    exact_span_metrics,
    labels_to_char_spans,
)

from metrics_asc_v2 import (
    classification_metrics,
)

from frozen_eval_utils import (
    set_deterministic,
    load_json,
    load_ate,
    load_asc,
    predict_ate_from_frozen_segments,
    predict_asc_from_frozen_text,
)


OUTPUT_DIR = (
    BACKEND_DIR
    / "evaluation_outputs"
    / "absa_v2_end_to_end_frozen"
)


def safe_div(a, b):
    return a / b if b else 0.0


def prf(tp, fp, fn):
    p = safe_div(
        tp,
        tp + fp,
    )

    r = safe_div(
        tp,
        tp + fn,
    )

    f1 = safe_div(
        2 * p * r,
        p + r,
    )

    return {
        "precision":
            p,
        "recall":
            r,
        "f1":
            f1,
        "tp":
            tp,
        "fp":
            fp,
        "fn":
            fn,
    }


def main():
    set_deterministic()

    ate_data = load_json(
        ATE_TEST_FILE
    )

    asc_data = load_json(
        ASC_TEST_FILE
    )

    # Frozen ASC input per exact gold aspect.
    asc_by_key = {
        (
            str(
                item.get(
                    "comment_id"
                )
            ),
            int(
                item[
                    "start"
                ]
            ),
            int(
                item[
                    "end"
                ]
            ),
        ):
            item
        for item
        in asc_data
    }

    (
        ate_tokenizer,
        ate_model,
        threshold,
    ) = load_ate()

    (
        asc_tokenizer,
        asc_model,
    ) = load_asc()

    ate_gold_all = []
    ate_pred_all = []

    oracle_gold = []
    oracle_pred = []

    matched_gold = []
    matched_pred = []

    e2e_tp = 0
    e2e_fp = 0
    e2e_fn = 0

    errors = {
        "ate_false_negatives": [],
        "ate_false_positives": [],
        "matched_span_sentiment_errors": [],
    }

    # ASC oracle from its own frozen preprocessed test file.
    for item in asc_data:
        result = (
            predict_asc_from_frozen_text(
                asc_tokenizer,
                asc_model,
                item[
                    "segmented_text"
                ],
                item[
                    "segmented_aspect"
                ],
            )
        )

        oracle_gold.append(
            LABEL2ID[
                item[
                    "label"
                ]
            ]
        )

        oracle_pred.append(
            LABEL2ID[
                result[
                    "sentiment"
                ]
            ]
        )

    for item in ate_data:
        text = item["text"]

        labels = (
            predict_ate_from_frozen_segments(
                ate_tokenizer,
                ate_model,
                threshold,
                item[
                    "segments"
                ],
            )
        )

        pred_spans = set(
            labels_to_char_spans(
                labels,
                item[
                    "segments"
                ],
            )
        )

        gold_by_span = {
            (
                int(
                    a[
                        "start"
                    ]
                ),
                int(
                    a[
                        "end"
                    ]
                ),
            ):
                str(
                    a[
                        "polarity"
                    ]
                )
                .strip()
                .upper()
            for a
            in (
                item.get(
                    "aspects"
                )
                or []
            )
        }

        gold_spans = set(
            gold_by_span
        )

        ate_gold_all.append(
            gold_spans
        )

        ate_pred_all.append(
            pred_spans
        )

        exact_spans = (
            gold_spans
            & pred_spans
        )

        fp_spans = (
            pred_spans
            - gold_spans
        )

        fn_spans = (
            gold_spans
            - pred_spans
        )

        # Every ATE false span is automatically one end-to-end FP,
        # regardless of the sentiment assigned to it.
        e2e_fp += len(
            fp_spans
        )

        # Every missed gold span is automatically one end-to-end FN.
        e2e_fn += len(
            fn_spans
        )

        for start, end in sorted(
            fp_spans
        ):
            errors[
                "ate_false_positives"
            ].append({
                "comment_id":
                    item.get(
                        "comment_id"
                    ),
                "text":
                    text,
                "term":
                    text[
                        start:
                        end
                    ],
                "start":
                    start,
                "end":
                    end,
            })

        for start, end in sorted(
            fn_spans
        ):
            errors[
                "ate_false_negatives"
            ].append({
                "comment_id":
                    item.get(
                        "comment_id"
                    ),
                "text":
                    text,
                "term":
                    text[
                        start:
                        end
                    ],
                "start":
                    start,
                "end":
                    end,
                "gold_sentiment":
                    gold_by_span[
                        (
                            start,
                            end,
                        )
                    ],
            })

        # ASC is evaluated only where ATE got the exact gold span.
        # For these spans, reuse the frozen ASC processed input that
        # was created before model evaluation.
        for start, end in sorted(
            exact_spans
        ):
            key = (
                str(
                    item.get(
                        "comment_id"
                    )
                ),
                start,
                end,
            )

            asc_item = (
                asc_by_key.get(
                    key
                )
            )

            if asc_item is None:
                raise RuntimeError(
                    "Không tìm thấy frozen ASC sample "
                    f"cho key={key}"
                )

            asc_result = (
                predict_asc_from_frozen_text(
                    asc_tokenizer,
                    asc_model,
                    asc_item[
                        "segmented_text"
                    ],
                    asc_item[
                        "segmented_aspect"
                    ],
                )
            )

            gold_sentiment = (
                gold_by_span[
                    (
                        start,
                        end,
                    )
                ]
            )

            pred_sentiment = (
                asc_result[
                    "sentiment"
                ]
            )

            matched_gold.append(
                LABEL2ID[
                    gold_sentiment
                ]
            )

            matched_pred.append(
                LABEL2ID[
                    pred_sentiment
                ]
            )

            if (
                gold_sentiment
                == pred_sentiment
            ):
                e2e_tp += 1
            else:
                # Correct span but wrong sentiment:
                # one predicted triple is FP and
                # one gold triple is FN.
                e2e_fp += 1
                e2e_fn += 1

                errors[
                    "matched_span_sentiment_errors"
                ].append({
                    "comment_id":
                        item.get(
                            "comment_id"
                        ),
                    "text":
                        text,
                    "term":
                        text[
                            start:
                            end
                        ],
                    "start":
                        start,
                    "end":
                        end,
                    "gold":
                        gold_sentiment,
                    "pred":
                        pred_sentiment,
                    "probabilities":
                        asc_result[
                            "probabilities"
                        ],
                })

    ate_metrics = (
        exact_span_metrics(
            ate_gold_all,
            ate_pred_all,
        )
    )

    asc_oracle = (
        classification_metrics(
            oracle_gold,
            oracle_pred,
        )
    )

    asc_exact = (
        classification_metrics(
            matched_gold,
            matched_pred,
        )
        if matched_gold
        else {
            "accuracy":
                0.0,
            "macro_f1":
                0.0,
        }
    )

    e2e = prf(
        e2e_tp,
        e2e_fp,
        e2e_fn,
    )

    metrics = {
        "dataset": {
            "comments":
                len(
                    ate_data
                ),
            "gold_aspects":
                sum(
                    len(
                        x
                    )
                    for x
                    in ate_gold_all
                ),
        },
        "models": {
            "ate":
                "ate_phobert_v2",
            "asc":
                "asc_phobert_v2",
        },
        "protocol": {
            "ate_input":
                str(
                    ATE_TEST_FILE
                ),
            "asc_input":
                str(
                    ASC_TEST_FILE
                ),
            "segmentation":
                "frozen_preprocessed",
            "device":
                "cpu",
            "ate_threshold":
                threshold,
        },
        "ate_exact_char_span":
            ate_metrics,
        "asc_oracle_gold_aspect":
            asc_oracle,
        "asc_on_exact_ate_matches":
            asc_exact,
        "end_to_end_exact_span_sentiment":
            e2e,
        "error_propagation": {
            "gold_aspects":
                sum(
                    len(
                        x
                    )
                    for x
                    in ate_gold_all
                ),
            "ate_exact_tp":
                ate_metrics[
                    "tp"
                ],
            "ate_false_positive":
                ate_metrics[
                    "fp"
                ],
            "ate_false_negative":
                ate_metrics[
                    "fn"
                ],
            "matched_span_sentiment_correct":
                e2e_tp,
            "matched_span_sentiment_wrong":
                (
                    len(
                        matched_gold
                    )
                    - e2e_tp
                ),
            "end_to_end_tp":
                e2e_tp,
            "end_to_end_fp":
                e2e_fp,
            "end_to_end_fn":
                e2e_fn,
            "asc_accuracy_on_exact_ate_matches":
                safe_div(
                    e2e_tp,
                    len(
                        matched_gold
                    ),
                ),
        },
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_DIR
        / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        OUTPUT_DIR
        / "errors.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            errors,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
