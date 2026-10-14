import json

from config_asc_v2 import (
    TEST_FILE,
    BACKEND_DIR,
    LABEL2ID,
)

from metrics_asc_v2 import (
    classification_metrics,
)

from frozen_eval_utils import (
    set_deterministic,
    load_json,
    load_asc,
    predict_asc_from_frozen_text,
)


OUTPUT_DIR = (
    BACKEND_DIR
    / "evaluation_outputs"
    / "asc_phobert_v2_frozen"
)


def main():
    set_deterministic()

    data = load_json(
        TEST_FILE
    )

    tokenizer, model = load_asc()

    y_true = []
    y_pred = []
    errors = []

    for item in data:
        result = (
            predict_asc_from_frozen_text(
                tokenizer,
                model,
                item[
                    "segmented_text"
                ],
                item[
                    "segmented_aspect"
                ],
            )
        )

        gold = item[
            "label"
        ]

        pred = result[
            "sentiment"
        ]

        y_true.append(
            LABEL2ID[
                gold
            ]
        )

        y_pred.append(
            LABEL2ID[
                pred
            ]
        )

        if gold != pred:
            errors.append({
                "comment_id":
                    item.get(
                        "comment_id"
                    ),
                "text":
                    item[
                        "text"
                    ],
                "aspect_term":
                    item[
                        "aspect_term"
                    ],
                "start":
                    item[
                        "start"
                    ],
                "end":
                    item[
                        "end"
                    ],
                "gold":
                    gold,
                "pred":
                    pred,
                "confidence":
                    result[
                        "confidence"
                    ],
                "probabilities":
                    result[
                        "probabilities"
                    ],
            })

    metrics = (
        classification_metrics(
            y_true,
            y_pred,
        )
    )

    metrics.update({
        "samples":
            len(data),
        "model":
            "asc_phobert_v2",
        "evaluation_input":
            str(TEST_FILE),
        "segmentation":
            "frozen_asc_processed_v2",
        "device":
            "cpu",
    })

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_DIR
        / "test_metrics.json",
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
        / "test_errors.json",
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
