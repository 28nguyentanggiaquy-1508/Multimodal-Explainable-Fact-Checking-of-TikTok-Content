import json

import torch

from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)

from config_ate import (
    TEST_FILE,
    MODEL_OUTPUT_DIR,
    BATCH_SIZE,
    DEVICE,
    ID2LABEL,
)

from dataset_ate import (
    ABSATokenDataset,
)

from metrics_ate import (
    span_metrics,
)


def collapse_to_words(
    pred_ids,
    gold_ids,
    word_ids,
):

    pred_sequence = []
    gold_sequence = []

    seen_words = set()

    for pred, gold, word_id in zip(
        pred_ids,
        gold_ids,
        word_ids,
    ):

        word_id = int(word_id)

        if word_id < 0:
            continue

        if word_id in seen_words:
            continue

        seen_words.add(word_id)

        if int(gold) == -100:
            continue

        pred_sequence.append(
            ID2LABEL[int(pred)]
        )

        gold_sequence.append(
            ID2LABEL[int(gold)]
        )

    return (
        pred_sequence,
        gold_sequence,
    )


@torch.no_grad()
def evaluate():

    print("=" * 70)
    print("PhoBERT ATE V1 - TEST EVALUATION")
    print("=" * 70)

    print(
        "Device:",
        DEVICE,
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_OUTPUT_DIR
        )
    )

    model = (
        AutoModelForTokenClassification
        .from_pretrained(
            MODEL_OUTPUT_DIR
        )
    )

    model.to(
        DEVICE
    )

    model.eval()

    dataset = ABSATokenDataset(
        TEST_FILE,
        tokenizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    all_predictions = []
    all_gold = []

    for batch in loader:

        input_ids = (
            batch["input_ids"]
            .to(DEVICE)
        )

        attention_mask = (
            batch["attention_mask"]
            .to(DEVICE)
        )

        labels = (
            batch["labels"]
            .cpu()
            .numpy()
        )

        word_ids = (
            batch["word_ids"]
            .cpu()
            .numpy()
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        predictions = (
            outputs.logits
            .argmax(dim=-1)
            .cpu()
            .numpy()
        )

        for i in range(
            predictions.shape[0]
        ):

            pred_seq, gold_seq = (
                collapse_to_words(
                    predictions[i],
                    labels[i],
                    word_ids[i],
                )
            )

            all_predictions.append(
                pred_seq
            )

            all_gold.append(
                gold_seq
            )

    metrics = span_metrics(
        all_gold,
        all_predictions,
    )

    predicted_spans = (
        metrics["tp"]
        + metrics["fp"]
    )

    gold_spans = (
        metrics["tp"]
        + metrics["fn"]
    )

    print()
    print("Exact Span Metrics")
    print("-" * 40)

    print(
        f"Precision : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{metrics['f1']:.4f}"
    )

    print()
    print(
        f"TP        : "
        f"{metrics['tp']}"
    )

    print(
        f"FP        : "
        f"{metrics['fp']}"
    )

    print(
        f"FN        : "
        f"{metrics['fn']}"
    )

    print(
        f"Gold spans: "
        f"{gold_spans}"
    )

    print(
        f"Pred spans: "
        f"{predicted_spans}"
    )

    result = {
        **metrics,

        "gold_spans":
            gold_spans,

        "predicted_spans":
            predicted_spans,
    }

    output_file = (
        MODEL_OUTPUT_DIR
        / "test_metrics.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "Saved:",
        output_file,
    )


if __name__ == "__main__":
    evaluate()