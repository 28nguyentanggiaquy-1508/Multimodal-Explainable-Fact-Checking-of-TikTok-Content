import json
import math
import random

import numpy as np
import torch

from torch.optim import AdamW

from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)

from config_ate import (
    TRAIN_FILE,
    VAL_FILE,
    MODEL_OUTPUT_DIR,
    MODEL_NAME,
    LABEL2ID,
    ID2LABEL,
    BATCH_SIZE,
    GRAD_ACCUM_STEPS,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED,
    DEVICE,
)

from dataset_ate import (
    ABSATokenDataset,
)

from metrics_ate import (
    span_metrics,
)


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights():

    with open(
        TRAIN_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    counts = {
        "O": 0,
        "B-ASP": 0,
        "I-ASP": 0,
    }

    for sample in data:

        for label in sample[
            "bio_labels"
        ]:

            counts[label] += 1

    o_count = max(
        counts["O"],
        1,
    )

    weights = [
        1.0,

        math.sqrt(
            o_count
            / max(
                counts["B-ASP"],
                1,
            )
        ),

        math.sqrt(
            o_count
            / max(
                counts["I-ASP"],
                1,
            )
        ),
    ]

    print(
        "BIO counts:",
        counts,
    )

    print(
        "Class weights:",
        weights,
    )

    return torch.tensor(
        weights,
        dtype=torch.float,
        device=DEVICE,
    )


# ============================================================
# COLLAPSE SUBWORD → WORD
# ============================================================

def collapse_to_words(
    pred_ids,
    gold_ids,
    word_ids,
):

    predictions = []

    gold = []

    seen_words = set()

    for pred, target, word_id in zip(
        pred_ids,
        gold_ids,
        word_ids,
    ):

        if word_id < 0:
            continue

        if word_id in seen_words:
            continue

        seen_words.add(
            word_id
        )

        if target == -100:
            continue

        predictions.append(
            ID2LABEL[int(pred)]
        )

        gold.append(
            ID2LABEL[int(target)]
        )

    return (
        predictions,
        gold,
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
):

    model.eval()

    pred_sequences = []

    gold_sequences = []

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
            .to(DEVICE)
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
            .argmax(
                dim=-1
            )
            .cpu()
            .numpy()
        )

        labels_np = (
            labels
            .cpu()
            .numpy()
        )

        for i in range(
            predictions.shape[0]
        ):

            pred_seq, gold_seq = (
                collapse_to_words(
                    predictions[i],
                    labels_np[i],
                    word_ids[i],
                )
            )

            pred_sequences.append(
                pred_seq
            )

            gold_sequences.append(
                gold_seq
            )

    return span_metrics(
        gold_sequences,
        pred_sequences,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        RANDOM_SEED
    )

    print("=" * 70)
    print("PhoBERT ATE V1")
    print("=" * 70)

    print(
        "Device:",
        DEVICE
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    train_dataset = (
        ABSATokenDataset(
            TRAIN_FILE,
            tokenizer,
        )
    )

    val_dataset = (
        ABSATokenDataset(
            VAL_FILE,
            tokenizer,
        )
    )

    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    )

    model = (
        AutoModelForTokenClassification
        .from_pretrained(
            MODEL_NAME,

            num_labels=3,

            id2label=ID2LABEL,

            label2id=LABEL2ID,

            ignore_mismatched_sizes=True,
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
            weight=class_weights,
            ignore_index=-100,
        )
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    update_steps_per_epoch = math.ceil(
        len(train_loader)
        / GRAD_ACCUM_STEPS
    )

    total_training_steps = (
        update_steps_per_epoch
        * EPOCHS
    )

    warmup_steps = int(
        total_training_steps
        * WARMUP_RATIO
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,

            num_warmup_steps=
                warmup_steps,

            num_training_steps=
                total_training_steps,
        )
    )

    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_f1 = -1.0

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
                batch["input_ids"]
                .to(DEVICE)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(DEVICE)
            )

            labels = (
                batch["labels"]
                .to(DEVICE)
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=
                    attention_mask,
            )

            logits = (
                outputs.logits
            )

            loss = criterion(
                logits.view(
                    -1,
                    3,
                ),

                labels.view(
                    -1
                ),
            )

            loss = (
                loss
                / GRAD_ACCUM_STEPS
            )

            loss.backward()

            total_loss += (
                loss.item()
                * GRAD_ACCUM_STEPS
            )

            if (
                step
                % GRAD_ACCUM_STEPS
                == 0
                or step
                == len(train_loader)
            ):

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                scheduler.step()

                optimizer.zero_grad()

            if step % 20 == 0:

                print(
                    f"  step "
                    f"{step}/"
                    f"{len(train_loader)}"
                    f" loss="
                    f"{loss.item() * GRAD_ACCUM_STEPS:.4f}"
                )

        avg_loss = (
            total_loss
            / len(train_loader)
        )

        metrics = evaluate(
            model,
            val_loader,
        )

        print(
            f"Train loss: "
            f"{avg_loss:.4f}"
        )

        print(
            "Validation:"
        )

        print(
            f"  Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"  Recall: "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"  F1: "
            f"{metrics['f1']:.4f}"
        )

        history.append({
            "epoch":
                epoch,

            "train_loss":
                avg_loss,

            **metrics,
        })

        # ====================================================
        # SAVE BEST
        # ====================================================

        if metrics["f1"] > best_f1:

            best_f1 = (
                metrics["f1"]
            )

            patience = 0

            print(
                "[+] New best model"
            )

            model.save_pretrained(
                MODEL_OUTPUT_DIR
            )

            tokenizer.save_pretrained(
                MODEL_OUTPUT_DIR
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
                f"No improvement "
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
    print("=" * 70)

    print(
        f"Best validation F1: "
        f"{best_f1:.4f}"
    )

    print(
        "Saved:",
        MODEL_OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()