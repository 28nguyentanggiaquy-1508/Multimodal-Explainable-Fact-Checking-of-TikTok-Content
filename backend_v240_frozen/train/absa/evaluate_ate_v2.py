import json

from config_ate_v2 import (
    TEST_SOURCE,
    EVAL_OUTPUT_DIR,
)

from inference_ate_v2 import (
    ATEPhoBERTV2,
)

from metrics_ate_v2 import (
    exact_span_metrics,
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
        TEST_SOURCE
    )

    engine = (
        ATEPhoBERTV2()
    )

    gold_all = []
    pred_all = []

    errors = []

    for index, item in enumerate(
        data,
        start=1,
    ):
        text = str(
            item.get(
                "text",
                "",
            )
            or ""
        )

        gold = {
            (
                int(
                    aspect[
                        "start"
                    ]
                ),
                int(
                    aspect[
                        "end"
                    ]
                ),
            )
            for aspect in (
                item.get(
                    "aspects"
                )
                or []
            )
        }

        result = engine.predict(
            text
        )

        pred = {
            (
                int(
                    aspect[
                        "start"
                    ]
                ),
                int(
                    aspect[
                        "end"
                    ]
                ),
            )
            for aspect in (
                result[
                    "aspects"
                ]
            )
        }

        gold_all.append(
            gold
        )

        pred_all.append(
            pred
        )

        if gold != pred:
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
                    text,

                "gold": [
                    {
                        "term":
                            text[
                                start:
                                end
                            ],

                        "start":
                            start,

                        "end":
                            end,
                    }
                    for (
                        start,
                        end,
                    )
                    in sorted(
                        gold
                    )
                ],

                "pred":
                    result[
                        "aspects"
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

    metrics = exact_span_metrics(
        gold_all,
        pred_all,
    )

    metrics[
        "comments"
    ] = len(
        data
    )

    metrics[
        "gold_spans"
    ] = sum(
        len(
            value
        )
        for value in gold_all
    )

    metrics[
        "predicted_spans"
    ] = sum(
        len(
            value
        )
        for value in pred_all
    )

    metrics[
        "threshold"
    ] = engine.threshold

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
    print("ATE V2 TEST - EXACT CHARACTER SPAN")
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
        "Saved:",
        EVAL_OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
