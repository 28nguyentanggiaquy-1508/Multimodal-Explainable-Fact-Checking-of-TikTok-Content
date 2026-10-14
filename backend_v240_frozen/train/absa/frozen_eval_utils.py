import json
import random

import numpy as np
import torch

from transformers import (
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from config_ate_v2 import (
    MODEL_OUTPUT_DIR as ATE_MODEL_DIR,
    LABEL2ID as ATE_LABEL2ID,
    MAX_LENGTH as ATE_MAX_LENGTH,
)

from config_asc_v2 import (
    MODEL_OUTPUT_DIR as ASC_MODEL_DIR,
    ID2LABEL as ASC_ID2LABEL,
    MAX_LENGTH as ASC_MAX_LENGTH,
)


DEVICE = torch.device("cpu")


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ate():
    tokenizer = AutoTokenizer.from_pretrained(
        ATE_MODEL_DIR
    )

    model = (
        AutoModelForTokenClassification
        .from_pretrained(
            ATE_MODEL_DIR
        )
        .to(DEVICE)
    )

    model.eval()

    config_path = (
        ATE_MODEL_DIR
        / "inference_config.json"
    )

    threshold = 0.50

    if config_path.exists():
        cfg = json.loads(
            config_path.read_text(
                encoding="utf-8"
            )
        )

        threshold = float(
            cfg.get(
                "aspect_threshold",
                0.50,
            )
        )

    return (
        tokenizer,
        model,
        threshold,
    )


def encode_ate_segments(
    tokenizer,
    segments,
):
    token_ids = []
    segment_ids = []

    for segment_index, segment in enumerate(
        segments
    ):
        pieces = tokenizer.encode(
            segment["word_form"],
            add_special_tokens=False,
        )

        if not pieces:
            pieces = [
                tokenizer.unk_token_id
            ]

        for piece_id in pieces:
            token_ids.append(piece_id)
            segment_ids.append(
                segment_index
            )

    max_body = (
        ATE_MAX_LENGTH
        - tokenizer.num_special_tokens_to_add(
            pair=False
        )
    )

    token_ids = token_ids[:max_body]
    segment_ids = segment_ids[:max_body]

    input_ids = (
        tokenizer
        .build_inputs_with_special_tokens(
            token_ids
        )
    )

    special_mask = (
        tokenizer
        .get_special_tokens_mask(
            token_ids,
            already_has_special_tokens=False,
        )
    )

    final_segment_ids = []
    body_cursor = 0

    for is_special in special_mask:
        if is_special:
            final_segment_ids.append(-1)
        else:
            final_segment_ids.append(
                segment_ids[
                    body_cursor
                ]
            )
            body_cursor += 1

    attention_mask = [
        1
    ] * len(input_ids)

    return (
        torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=DEVICE,
        ),
        torch.tensor(
            [attention_mask],
            dtype=torch.long,
            device=DEVICE,
        ),
        final_segment_ids,
    )


@torch.no_grad()
def predict_ate_from_frozen_segments(
    tokenizer,
    model,
    threshold,
    segments,
):
    (
        input_ids,
        attention_mask,
        segment_ids,
    ) = encode_ate_segments(
        tokenizer,
        segments,
    )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    probs = (
        torch.softmax(
            outputs.logits[0],
            dim=-1,
        )
        .cpu()
    )

    first_piece_probs = {}

    for token_index, segment_id in enumerate(
        segment_ids
    ):
        if (
            segment_id < 0
            or segment_id in first_piece_probs
        ):
            continue

        first_piece_probs[
            segment_id
        ] = probs[
            token_index
        ].tolist()

    labels = []

    for segment_index in range(
        len(segments)
    ):
        values = first_piece_probs.get(
            segment_index
        )

        if values is None:
            labels.append("O")
            continue

        o_prob = values[
            ATE_LABEL2ID["O"]
        ]

        b_prob = values[
            ATE_LABEL2ID["B-ASP"]
        ]

        i_prob = values[
            ATE_LABEL2ID["I-ASP"]
        ]

        aspect_prob = max(
            b_prob,
            i_prob,
        )

        if (
            aspect_prob < threshold
            or o_prob >= aspect_prob
        ):
            labels.append("O")
        elif b_prob >= i_prob:
            labels.append("B-ASP")
        else:
            labels.append("I-ASP")

    return labels


def load_asc():
    tokenizer = AutoTokenizer.from_pretrained(
        ASC_MODEL_DIR
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            ASC_MODEL_DIR
        )
        .to(DEVICE)
    )

    model.eval()

    return tokenizer, model


@torch.no_grad()
def predict_asc_from_frozen_text(
    tokenizer,
    model,
    segmented_text,
    segmented_aspect,
):
    encoded = tokenizer(
        segmented_text,
        segmented_aspect,
        max_length=ASC_MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value
        in encoded.items()
    }

    outputs = model(
        **encoded
    )

    probs = (
        torch.softmax(
            outputs.logits[0],
            dim=-1,
        )
        .cpu()
        .tolist()
    )

    pred_id = int(
        torch.argmax(
            outputs.logits[0]
        ).item()
    )

    return {
        "sentiment":
            ASC_ID2LABEL[
                pred_id
            ],
        "confidence":
            probs[pred_id],
        "probabilities": {
            ASC_ID2LABEL[i]:
                value
            for i, value
            in enumerate(probs)
        },
    }
