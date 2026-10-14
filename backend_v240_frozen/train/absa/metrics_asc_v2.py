from config_asc_v2 import (
    ID2LABEL,
)


def safe_div(num, den):
    return (
        num / den
        if den
        else 0.0
    )


def classification_metrics(
    y_true,
    y_pred,
):
    labels = [
        0,
        1,
        2,
    ]

    per_class = {}

    for label_id in labels:
        tp = sum(
            1
            for gold, pred
            in zip(
                y_true,
                y_pred,
            )
            if (
                gold == label_id
                and pred == label_id
            )
        )

        fp = sum(
            1
            for gold, pred
            in zip(
                y_true,
                y_pred,
            )
            if (
                gold != label_id
                and pred == label_id
            )
        )

        fn = sum(
            1
            for gold, pred
            in zip(
                y_true,
                y_pred,
            )
            if (
                gold == label_id
                and pred != label_id
            )
        )

        support = sum(
            1
            for gold in y_true
            if gold == label_id
        )

        precision = safe_div(
            tp,
            tp + fp,
        )

        recall = safe_div(
            tp,
            tp + fn,
        )

        f1 = safe_div(
            2
            * precision
            * recall,
            precision
            + recall,
        )

        per_class[
            ID2LABEL[
                label_id
            ]
        ] = {
            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "support":
                support,

            "tp":
                tp,

            "fp":
                fp,

            "fn":
                fn,
        }

    accuracy = safe_div(
        sum(
            1
            for gold, pred
            in zip(
                y_true,
                y_pred,
            )
            if gold == pred
        ),
        len(
            y_true
        ),
    )

    macro_precision = (
        sum(
            row[
                "precision"
            ]
            for row
            in per_class.values()
        )
        / 3
    )

    macro_recall = (
        sum(
            row[
                "recall"
            ]
            for row
            in per_class.values()
        )
        / 3
    )

    macro_f1 = (
        sum(
            row[
                "f1"
            ]
            for row
            in per_class.values()
        )
        / 3
    )

    total_support = sum(
        row[
            "support"
        ]
        for row
        in per_class.values()
    )

    weighted_f1 = safe_div(
        sum(
            row[
                "f1"
            ]
            * row[
                "support"
            ]
            for row
            in per_class.values()
        ),
        total_support,
    )

    matrix = [
        [
            0,
            0,
            0,
        ]
        for _ in range(
            3
        )
    ]

    for gold, pred in zip(
        y_true,
        y_pred,
    ):
        matrix[
            gold
        ][
            pred
        ] += 1

    return {
        "accuracy":
            accuracy,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1,

        "per_class":
            per_class,

        "confusion_matrix":
            matrix,

        "confusion_matrix_labels":
            [
                ID2LABEL[
                    i
                ]
                for i
                in range(
                    3
                )
            ],
    }
