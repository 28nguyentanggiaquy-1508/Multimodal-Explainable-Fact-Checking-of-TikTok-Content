def safe_div(num, den):
    return (
        num / den
        if den
        else 0.0
    )


def exact_span_metrics(
    gold_spans,
    pred_spans,
):
    """
    gold_spans / pred_spans:
        iterable of per-comment set[(start, end)]
    """
    tp = 0
    fp = 0
    fn = 0

    for gold, pred in zip(
        gold_spans,
        pred_spans,
    ):
        gold = set(
            gold
        )

        pred = set(
            pred
        )

        tp += len(
            gold
            & pred
        )

        fp += len(
            pred
            - gold
        )

        fn += len(
            gold
            - pred
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
        2 * precision * recall,
        precision + recall,
    )

    return {
        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "tp":
            tp,

        "fp":
            fp,

        "fn":
            fn,
    }


def repair_bio(
    labels,
):
    """
    I-ASP after O is converted to B-ASP.
    """
    output = []

    inside = False

    for label in labels:
        if label == "B-ASP":
            output.append(
                "B-ASP"
            )

            inside = True

        elif label == "I-ASP":
            if inside:
                output.append(
                    "I-ASP"
                )
            else:
                output.append(
                    "B-ASP"
                )

                inside = True

        else:
            output.append(
                "O"
            )

            inside = False

    return output


def labels_to_char_spans(
    labels,
    segments,
):
    labels = repair_bio(
        labels
    )

    spans = []

    current_start = None
    current_end = None

    for label, segment in zip(
        labels,
        segments,
    ):
        start = int(
            segment[
                "start"
            ]
        )

        end = int(
            segment[
                "end"
            ]
        )

        if label == "B-ASP":
            if current_start is not None:
                spans.append(
                    (
                        current_start,
                        current_end,
                    )
                )

            current_start = start
            current_end = end

        elif label == "I-ASP":
            if current_start is None:
                current_start = start

            current_end = end

        else:
            if current_start is not None:
                spans.append(
                    (
                        current_start,
                        current_end,
                    )
                )

                current_start = None
                current_end = None

    if current_start is not None:
        spans.append(
            (
                current_start,
                current_end,
            )
        )

    return spans
