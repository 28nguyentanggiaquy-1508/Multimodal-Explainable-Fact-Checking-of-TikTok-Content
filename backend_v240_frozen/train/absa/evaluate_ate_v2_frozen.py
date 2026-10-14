import json

from config_ate_v2 import (
    TEST_FILE,
    BACKEND_DIR,
)

from metrics_ate_v2 import (
    exact_span_metrics,
    labels_to_char_spans,
)

from frozen_eval_utils import (
    set_deterministic,
    load_json,
    load_ate,
    predict_ate_from_frozen_segments,
)


OUTPUT_DIR = (
    BACKEND_DIR
    / "evaluation_outputs"
    / "ate_phobert_v2_frozen"
)


def main():
    set_deterministic()

    data = load_json(
        TEST_FILE
    )

    (
        tokenizer,
        model,
        threshold,
    ) = load_ate()

    gold_all = []
    pred_all = []
    errors = []

    for item in data:
        text = item["text"]
        segments = item["segments"]

        labels = (
            predict_ate_from_frozen_segments(
                tokenizer,
                model,
                threshold,
                segments,
            )
        )

        pred = set(
            labels_to_char_spans(
                labels,
                segments,
            )
        )

        gold = {
            (
                int(a["start"]),
                int(a["end"]),
            )
            for a
            in (
                item.get("aspects")
                or []
            )
        }

        gold_all.append(gold)
        pred_all.append(pred)

        if gold != pred:
            errors.append({
                "comment_id":
                    item.get(
                        "comment_id"
                    ),
                "text":
                    text,
                "gold":
                    [
                        {
                            "term":
                                text[s:e],
                            "start":
                                s,
                            "end":
                                e,
                        }
                        for s, e
                        in sorted(gold)
                    ],
                "pred":
                    [
                        {
                            "term":
                                text[s:e],
                            "start":
                                s,
                            "end":
                                e,
                        }
                        for s, e
                        in sorted(pred)
                    ],
            })

    metrics = exact_span_metrics(
        gold_all,
        pred_all,
    )

    metrics.update({
        "comments":
            len(data),
        "gold_spans":
            sum(
                len(x)
                for x
                in gold_all
            ),
        "predicted_spans":
            sum(
                len(x)
                for x
                in pred_all
            ),
        "threshold":
            threshold,
        "evaluation_input":
            str(TEST_FILE),
        "segmentation":
            "frozen_processed_v2",
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
