import json
import math
import random
from collections import Counter

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from config_asc_v2 import (
    MODEL_NAME,
    TRAIN_FILE,
    VAL_FILE,
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
    MAX_CLASS_WEIGHT,
    RANDOM_SEED,
    DEVICE,
)

from dataset_asc_v2 import (
    ASCV2Dataset,
)

from metrics_asc_v2 import (
    classification_metrics,
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

    counts = Counter(
        sample[
            "label"
        ]
        for sample in data
    )

    max_count = max(
        counts.values()
    )

    weights = []

    for label_id in range(
        3
    ):
        label = ID2LABEL[
            label_id
        ]

        count = max(
            counts[
                label
            ],
            1,
        )

        # sqrt inverse frequency:
        # moderate correction, less aggressive than full inverse.
        weight = math.sqrt(
            max_count
            / count
        )

        weight = min(
            weight,
            MAX_CLASS_WEIGHT,
        )

        weights.append(
            weight
        )

    # Normalize so average weight is roughly 1.
    mean_weight = (
        sum(
            weights
        )
        / len(
            weights
        )
    )

    weights = [
        weight
        / mean_weight
        for weight in weights
    ]

    print(
        "ASC V2 train counts:",
        dict(
            counts
        ),
    )

    print(
        "ASC V2 class weights:",
        {
            ID2LABEL[i]:
                weights[i]
            for i in range(
                3
            )
        },
    )

    return torch.tensor(
        weights,
        dtype=torch.float,
        device=DEVICE,
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
):
    model.eval()

    y_true = []
    y_pred = []

    total_loss = 0.0

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

        logits = (
            outputs.logits
        )

        loss = criterion(
            logits,
            labels,
        )

        total_loss += (
            loss.item()
        )

        predictions = (
            torch.argmax(
                logits,
                dim=-1,
            )
        )

        y_true.extend(
            labels
            .cpu()
            .tolist()
        )

        y_pred.extend(
            predictions
            .cpu()
            .tolist()
        )

    metrics = (
        classification_metrics(
            y_true,
            y_pred,
        )
    )

    metrics[
        "val_loss"
    ] = (
        total_loss
        / max(
            len(
                loader
            ),
            1,
        )
    )

    return metrics


def is_better(
    current,
    best,
):
    if best is None:
        return True

    # Primary metric: Macro-F1.
    if (
        current[
            "macro_f1"
        ]
        > best[
            "macro_f1"
        ]
        + 1e-12
    ):
        return True

    if (
        abs(
            current[
                "macro_f1"
            ]
            - best[
                "macro_f1"
            ]
        )
        <= 1e-12
    ):
        current_neg = (
            current[
                "per_class"
            ][
                "NEGATIVE"
            ][
                "f1"
            ]
        )

        best_neg = (
            best[
                "per_class"
            ][
                "NEGATIVE"
            ][
                "f1"
            ]
        )

        # Tie-break: do not choose a model that ignores NEGATIVE.
        if current_neg > best_neg:
            return True

        if (
            current_neg
            == best_neg
            and current[
                "accuracy"
            ]
            > best[
                "accuracy"
            ]
        ):
            return True

    return False


def main():
    set_seed(
        RANDOM_SEED
    )

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            "Chưa có ASC V2 processed data. "
            "Chạy prepare_asc_v2_data.py trước."
        )

    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("PhoBERT ASC V2")
    print("VnCoreNLP + explicit aspect pair + weighted CE")
    print("=" * 72)
    print(
        "Device:",
        DEVICE,
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    train_dataset = (
        ASCV2Dataset(
            TRAIN_FILE,
            tokenizer,
        )
    )

    val_dataset = (
        ASCV2Dataset(
            VAL_FILE,
            tokenizer,
        )
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
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,

            num_labels=
                3,

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
                class_weights
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

    history = []

    best_metrics = None
    patience = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        optimizer.zero_grad()

        total_train_loss = 0.0

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

            loss = criterion(
                outputs.logits,
                labels,
            )

            total_train_loss += (
                loss.item()
            )

            (
                loss
                / GRAD_ACCUM_STEPS
            ).backward()

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

        train_loss = (
            total_train_loss
            / max(
                len(
                    train_loader
                ),
                1,
            )
        )

        metrics = evaluate(
            model,
            val_loader,
            criterion,
        )

        epoch_result = {
            "epoch":
                epoch,

            "train_loss":
                train_loss,

            **metrics,
        }

        history.append(
            epoch_result
        )

        # Always persist full history, not only on new best.
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

        neg = (
            metrics[
                "per_class"
            ][
                "NEGATIVE"
            ]
        )

        print()
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Val loss: "
            f"{metrics['val_loss']:.4f}"
        )

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Macro-F1: "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            "NEGATIVE: "
            f"P={neg['precision']:.4f} "
            f"R={neg['recall']:.4f} "
            f"F1={neg['f1']:.4f} "
            f"TP={neg['tp']} "
            f"FP={neg['fp']} "
            f"FN={neg['fn']}"
        )

        if is_better(
            metrics,
            best_metrics,
        ):
            best_metrics = (
                metrics
            )

            patience = 0

            print(
                "[+] New best ASC V2"
            )

            model.save_pretrained(
                MODEL_OUTPUT_DIR
            )

            tokenizer.save_pretrained(
                MODEL_OUTPUT_DIR
            )

            with open(
                MODEL_OUTPUT_DIR
                / "best_validation_metrics.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    metrics,
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
    print("ASC V2 training finished")
    print("=" * 72)

    if best_metrics:
        print(
            "Best val Macro-F1:",
            f"{best_metrics['macro_f1']:.4f}",
        )

        print(
            "Best val NEGATIVE F1:",
            f"{best_metrics['per_class']['NEGATIVE']['f1']:.4f}",
        )

    print(
        "Saved:",
        MODEL_OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
