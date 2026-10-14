import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from config_ate_v2 import (
    TRAIN_FILE,
    VAL_FILE,
    MODEL_NAME,
    MODEL_OUTPUT_DIR,
    LABEL2ID,
    ID2LABEL,
    BATCH_SIZE,
    GRAD_ACCUM_STEPS,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    EARLY_STOPPING_PATIENCE,
    LABEL_SMOOTHING,
    MAX_ASPECT_CLASS_WEIGHT,
    THRESHOLD_CANDIDATES,
    RANDOM_SEED,
    DEVICE,
)

from dataset_ate_v2 import (
    ATEV2Dataset,
)

from metrics_ate_v2 import (
    exact_span_metrics,
    labels_to_char_spans,
)


def set_seed(seed):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
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


def calculate_class_weights():
    data = load_json(
        TRAIN_FILE
    )

    counts = Counter()

    for sample in data:
        for (
            label,
            ignored,
        ) in zip(
            sample[
                "segment_labels"
            ],
            sample[
                "segment_ignore"
            ],
        ):
            if ignored:
                continue

            counts[
                label
            ] += 1

    o_count = max(
        counts[
            "O"
        ],
        1,
    )

    weights = [
        1.0
    ]

    for label in [
        "B-ASP",
        "I-ASP",
    ]:
        count = max(
            counts[
                label
            ],
            1,
        )

        weight = math.log1p(
            o_count
            / count
        )

        weight = min(
            weight,
            MAX_ASPECT_CLASS_WEIGHT,
        )

        weights.append(
            weight
        )

    print(
        "V2 segment label counts:",
        dict(
            counts
        ),
    )

    print(
        "V2 class weights:",
        weights,
    )

    return torch.tensor(
        weights,
        dtype=torch.float,
        device=DEVICE,
    )


def collapse_segment_probabilities(
    logits,
    segment_ids,
):
    """
    Use first BPE piece of each VnCoreNLP word for word-level decoding.
    """
    probs = torch.softmax(
        logits,
        dim=-1,
    )

    seen = set()

    output = {}

    for token_index, segment_id in enumerate(
        segment_ids
    ):
        segment_id = int(
            segment_id
        )

        if (
            segment_id < 0
            or segment_id in seen
        ):
            continue

        seen.add(
            segment_id
        )

        output[
            segment_id
        ] = (
            probs[
                token_index
            ]
            .detach()
            .cpu()
            .tolist()
        )

    return output


def decode_segment_labels(
    segment_probs,
    segment_count,
    threshold,
):
    labels = []

    for segment_id in range(
        segment_count
    ):
        probs = segment_probs.get(
            segment_id
        )

        if probs is None:
            labels.append(
                "O"
            )
            continue

        o_prob = probs[
            LABEL2ID[
                "O"
            ]
        ]

        b_prob = probs[
            LABEL2ID[
                "B-ASP"
            ]
        ]

        i_prob = probs[
            LABEL2ID[
                "I-ASP"
            ]
        ]

        aspect_prob = max(
            b_prob,
            i_prob,
        )

        if (
            aspect_prob
            < threshold
            or o_prob
            >= aspect_prob
        ):
            labels.append(
                "O"
            )

        elif b_prob >= i_prob:
            labels.append(
                "B-ASP"
            )

        else:
            labels.append(
                "I-ASP"
            )

    return labels


@torch.no_grad()
def collect_validation_predictions(
    model,
    loader,
    raw_samples,
):
    model.eval()

    all_segment_probs = [
        {}
        for _ in raw_samples
    ]

    for batch in loader:
        input_ids = (
            batch[
                "input_ids"
            ]
            .to(
                DEVICE
            )
        )

        attention_mask = (
            batch[
                "attention_mask"
            ]
            .to(
                DEVICE
            )
        )

        sample_indices = (
            batch[
                "sample_index"
            ]
            .cpu()
            .tolist()
        )

        segment_ids = (
            batch[
                "segment_ids"
            ]
            .cpu()
            .tolist()
        )

        outputs = model(
            input_ids=
                input_ids,

            attention_mask=
                attention_mask,
        )

        for row_index, sample_index in enumerate(
            sample_indices
        ):
            all_segment_probs[
                sample_index
            ] = (
                collapse_segment_probabilities(
                    outputs.logits[
                        row_index
                    ],
                    segment_ids[
                        row_index
                    ],
                )
            )

    return all_segment_probs


def evaluate_threshold(
    raw_samples,
    all_segment_probs,
    threshold,
):
    gold_spans = []
    pred_spans = []

    for sample, segment_probs in zip(
        raw_samples,
        all_segment_probs,
    ):
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
                sample.get(
                    "aspects"
                )
                or []
            )
        }

        labels = (
            decode_segment_labels(
                segment_probs,
                len(
                    sample[
                        "segments"
                    ]
                ),
                threshold,
            )
        )

        pred = set(
            labels_to_char_spans(
                labels,
                sample[
                    "segments"
                ],
            )
        )

        gold_spans.append(
            gold
        )

        pred_spans.append(
            pred
        )

    metrics = exact_span_metrics(
        gold_spans,
        pred_spans,
    )

    metrics[
        "threshold"
    ] = threshold

    return metrics


def main():
    set_seed(
        RANDOM_SEED
    )

    print("=" * 72)
    print("PhoBERT ATE V2")
    print("VnCoreNLP segmentation + exact char-span validation")
    print("=" * 72)

    print(
        "Device:",
        DEVICE,
    )

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            "Chưa có processed_v2 data. "
            "Chạy prepare_ate_v2_data.py trước."
        )

    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    train_dataset = (
        ATEV2Dataset(
            TRAIN_FILE,
            tokenizer,
        )
    )

    val_dataset = (
        ATEV2Dataset(
            VAL_FILE,
            tokenizer,
        )
    )

    raw_val_samples = load_json(
        VAL_FILE
    )

    generator = (
        torch.Generator()
        .manual_seed(
            RANDOM_SEED
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=
            BATCH_SIZE,
        shuffle=True,
        generator=
            generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=
            BATCH_SIZE,
        shuffle=False,
    )

    model = (
        AutoModelForTokenClassification
        .from_pretrained(
            MODEL_NAME,

            num_labels=
                len(
                    LABEL2ID
                ),

            id2label=
                ID2LABEL,

            label2id=
                LABEL2ID,

            ignore_mismatched_sizes=
                True,
        )
    )

    model.to(
        DEVICE
    )

    class_weights = (
        calculate_class_weights()
    )

    criterion = (
        torch.nn.CrossEntropyLoss(
            weight=
                class_weights,

            ignore_index=
                -100,

            label_smoothing=
                LABEL_SMOOTHING,
        )
    )

    optimizer = AdamW(
        model.parameters(),
        lr=
            LEARNING_RATE,
        weight_decay=
            WEIGHT_DECAY,
    )

    update_steps_per_epoch = (
        math.ceil(
            len(
                train_loader
            )
            / GRAD_ACCUM_STEPS
        )
    )

    total_steps = (
        update_steps_per_epoch
        * EPOCHS
    )

    warmup_steps = int(
        total_steps
        * WARMUP_RATIO
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=
                warmup_steps,
            num_training_steps=
                total_steps,
        )
    )

    best_f1 = -1.0
    best_threshold = 0.50
    patience = 0
    history = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        print()
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        model.train()

        optimizer.zero_grad()

        total_loss = 0.0

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):
            input_ids = (
                batch[
                    "input_ids"
                ]
                .to(
                    DEVICE
                )
            )

            attention_mask = (
                batch[
                    "attention_mask"
                ]
                .to(
                    DEVICE
                )
            )

            labels = (
                batch[
                    "labels"
                ]
                .to(
                    DEVICE
                )
            )

            outputs = model(
                input_ids=
                    input_ids,

                attention_mask=
                    attention_mask,
            )

            logits = outputs.logits

            loss = criterion(
                logits.view(
                    -1,
                    len(
                        LABEL2ID
                    ),
                ),

                labels.view(
                    -1
                ),
            )

            scaled_loss = (
                loss
                / GRAD_ACCUM_STEPS
            )

            scaled_loss.backward()

            total_loss += (
                loss.item()
            )

            if (
                step
                % GRAD_ACCUM_STEPS
                == 0
                or step
                == len(
                    train_loader
                )
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = (
            total_loss
            / max(
                len(
                    train_loader
                ),
                1,
            )
        )

        all_segment_probs = (
            collect_validation_predictions(
                model,
                val_loader,
                raw_val_samples,
            )
        )

        threshold_results = [
            evaluate_threshold(
                raw_val_samples,
                all_segment_probs,
                threshold,
            )
            for threshold
            in THRESHOLD_CANDIDATES
        ]

        epoch_best = max(
            threshold_results,
            key=lambda item: (
                item[
                    "f1"
                ],
                item[
                    "precision"
                ],
                -abs(
                    item[
                        "threshold"
                    ]
                    - 0.5
                ),
            ),
        )

        print(
            f"Train loss: "
            f"{avg_loss:.4f}"
        )

        print(
            "Validation exact char-span:"
        )

        print(
            f"  threshold: "
            f"{epoch_best['threshold']:.2f}"
        )

        print(
            f"  Precision: "
            f"{epoch_best['precision']:.4f}"
        )

        print(
            f"  Recall: "
            f"{epoch_best['recall']:.4f}"
        )

        print(
            f"  F1: "
            f"{epoch_best['f1']:.4f}"
        )

        history.append({
            "epoch":
                epoch,

            "train_loss":
                avg_loss,

            "best_validation":
                epoch_best,

            "threshold_results":
                threshold_results,
        })

        if (
            epoch_best[
                "f1"
            ]
            > best_f1
        ):
            best_f1 = (
                epoch_best[
                    "f1"
                ]
            )

            best_threshold = (
                epoch_best[
                    "threshold"
                ]
            )

            patience = 0

            print(
                "[+] New best V2 model"
            )

            model.save_pretrained(
                MODEL_OUTPUT_DIR
            )

            tokenizer.save_pretrained(
                MODEL_OUTPUT_DIR
            )

            with open(
                MODEL_OUTPUT_DIR
                / "inference_config.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "aspect_threshold":
                            best_threshold,

                        "validation_f1":
                            best_f1,

                        "segmentation":
                            "VnCoreNLP-RDRSegmenter",

                        "metric":
                            "exact_character_span_f1",
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            with open(
                MODEL_OUTPUT_DIR
                / "training_history.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    history,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        else:
            patience += 1

            print(
                "No improvement "
                f"({patience}/"
                f"{EARLY_STOPPING_PATIENCE})"
            )

        if (
            patience
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "[!] Early stopping"
            )

            break

    print()
    print("=" * 72)

    print(
        "Best validation exact-span F1:",
        f"{best_f1:.4f}",
    )

    print(
        "Best aspect threshold:",
        best_threshold,
    )

    print(
        "Saved:",
        MODEL_OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
