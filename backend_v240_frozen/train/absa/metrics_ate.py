def bio_to_spans(labels):

    spans = []

    start = None

    for index, label in enumerate(
        labels
    ):

        if label == "B-ASP":

            if start is not None:

                spans.append(
                    (
                        start,
                        index,
                    )
                )

            start = index

        elif label == "I-ASP":

            if start is None:

                # I xuất hiện không có B
                # coi như bắt đầu aspect mới
                start = index

        else:

            if start is not None:

                spans.append(
                    (
                        start,
                        index,
                    )
                )

                start = None

    if start is not None:

        spans.append(
            (
                start,
                len(labels),
            )
        )

    return spans


def span_metrics(
    gold_sequences,
    pred_sequences,
):

    tp = 0
    fp = 0
    fn = 0

    for gold, pred in zip(
        gold_sequences,
        pred_sequences,
    ):

        gold_spans = set(
            bio_to_spans(
                gold
            )
        )

        pred_spans = set(
            bio_to_spans(
                pred
            )
        )

        tp += len(
            gold_spans
            & pred_spans
        )

        fp += len(
            pred_spans
            - gold_spans
        )

        fn += len(
            gold_spans
            - pred_spans
        )

    precision = (
        tp
        / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp
        / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if precision + recall > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }