import json

from config_asc_v2 import (
    TEST_FILE,
    EVAL_OUTPUT_DIR,
    LABEL2ID,
    ID2LABEL,
)

from inference_asc_v2 import (
    ASCPhoBERTV2,
)

from metrics_asc_v2 import (
    classification_metrics,
)


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(
            f
        )


def main():
    data = load_json(
        TEST_FILE
    )

    engine = (
        ASCPhoBERTV2()
    )

    y_true = []
    y_pred = []

    errors = []

    for index, item in enumerate(
        data,
        start=1,
    ):
        gold_label = (
            item[
                "label"
            ]
        )

        result = (
            engine.predict(
                item[
                    "text"
                ],
                item[
                    "aspect_term"
                ],
            )
        )

        pred_label = (
            result[
                "sentiment"
            ]
        )

        y_true.append(
            LABEL2ID[
                gold_label
            ]
        )

        y_pred.append(
            LABEL2ID[
                pred_label
            ]
        )

        if (
            gold_label
            != pred_label
        ):
            errors.append({
                "comment_id":
                    item.get(
                        "comment_id"
                    ),

                "video_id":
                    item.get(
                        "video_id"
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
                    gold_label,

                "pred":
                    pred_label,

                "confidence":
                    result[
                        "confidence"
                    ],

                "probabilities":
                    result[
                        "probabilities"
                    ],
            })

        if (
            index % 20
            == 0
            or index
            == len(
                data
            )
        ):
            print(
                f"[{index}/{len(data)}]"
            )

    metrics = (
        classification_metrics(
            y_true,
            y_pred,
        )
    )

    metrics[
        "samples"
    ] = len(
        data
    )

    metrics[
        "model"
    ] = (
        "asc_phobert_v2"
    )

    EVAL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        EVAL_OUTPUT_DIR
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
        EVAL_OUTPUT_DIR
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

    print()
    print("=" * 72)
    print("ASC V2 TEST - GOLD ASPECT / ORACLE ATE")
    print("=" * 72)

    print(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(
        "V1 baseline:"
    )

    print(
        "  Accuracy = 0.6774"
    )

    print(
        "  Macro-F1 = 0.4778"
    )

    print(
        "  NEGATIVE F1 = 0.0000"
    )

    print()
    print(
        "Saved:",
        EVAL_OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
