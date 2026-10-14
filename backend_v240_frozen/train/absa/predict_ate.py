import re

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)

from config_ate import (
    MODEL_OUTPUT_DIR,
    MAX_LENGTH,
    DEVICE,
    ID2LABEL,
)


TOKEN_PATTERN = re.compile(
    r"\w+|[^\w\s]",
    flags=re.UNICODE,
)


def tokenize_offsets(text):

    words = []

    offsets = []

    for match in TOKEN_PATTERN.finditer(
        text
    ):

        words.append(
            match.group()
        )

        offsets.append(
            (
                match.start(),
                match.end(),
            )
        )

    return words, offsets


def load_model():

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

    return (
        tokenizer,
        model,
    )


def predict_aspects(
    text,
    tokenizer,
    model,
):

    words, offsets = (
        tokenize_offsets(
            text
        )
    )

    piece_ids = []

    piece_word_ids = []

    for word_index, word in enumerate(
        words
    ):

        pieces = tokenizer.encode(
            word,
            add_special_tokens=False,
        )

        if not pieces:

            pieces = [
                tokenizer.unk_token_id
            ]

        for piece in pieces:

            piece_ids.append(
                piece
            )

            piece_word_ids.append(
                word_index
            )

    max_content = (
        MAX_LENGTH
        - tokenizer
        .num_special_tokens_to_add(
            pair=False
        )
    )

    piece_ids = (
        piece_ids[:max_content]
    )

    piece_word_ids = (
        piece_word_ids[:max_content]
    )

    input_ids = (
        tokenizer
        .build_inputs_with_special_tokens(
            piece_ids
        )
    )

    special_mask = (
        tokenizer
        .get_special_tokens_mask(
            piece_ids,
            already_has_special_tokens=False,
        )
    )

    full_word_ids = []

    pos = 0

    for special in special_mask:

        if special:

            full_word_ids.append(
                -1
            )

        else:

            full_word_ids.append(
                piece_word_ids[pos]
            )

            pos += 1

    attention_mask = [
        1
        for _ in input_ids
    ]

    input_tensor = (
        torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=DEVICE,
        )
    )

    mask_tensor = (
        torch.tensor(
            [attention_mask],
            dtype=torch.long,
            device=DEVICE,
        )
    )

    with torch.no_grad():

        logits = model(
            input_ids=input_tensor,
            attention_mask=mask_tensor,
        ).logits

    predictions = (
        logits
        .argmax(
            dim=-1
        )[0]
        .cpu()
        .tolist()
    )

    word_labels = [
        "O"
        for _ in words
    ]

    seen = set()

    for pred, word_id in zip(
        predictions,
        full_word_ids,
    ):

        if word_id < 0:

            continue

        if word_id in seen:

            continue

        seen.add(
            word_id
        )

        word_labels[
            word_id
        ] = ID2LABEL[
            pred
        ]

    aspects = []

    start_word = None

    for i, label in enumerate(
        word_labels
    ):

        if label == "B-ASP":

            if start_word is not None:

                start_char = (
                    offsets[
                        start_word
                    ][0]
                )

                end_char = (
                    offsets[
                        i - 1
                    ][1]
                )

                aspects.append({
                    "term":
                        text[
                            start_char:
                            end_char
                        ],

                    "start":
                        start_char,

                    "end":
                        end_char,
                })

            start_word = i

        elif label == "I-ASP":

            if start_word is None:

                start_word = i

        else:

            if start_word is not None:

                start_char = (
                    offsets[
                        start_word
                    ][0]
                )

                end_char = (
                    offsets[
                        i - 1
                    ][1]
                )

                aspects.append({
                    "term":
                        text[
                            start_char:
                            end_char
                        ],

                    "start":
                        start_char,

                    "end":
                        end_char,
                })

                start_word = None

    if start_word is not None:

        start_char = (
            offsets[
                start_word
            ][0]
        )

        end_char = (
            offsets[-1][1]
        )

        aspects.append({
            "term":
                text[
                    start_char:
                    end_char
                ],

            "start":
                start_char,

            "end":
                end_char,
        })

    return {
        "text": text,
        "tokens": words,
        "labels": word_labels,
        "aspects": aspects,
    }


def main():

    tokenizer, model = (
        load_model()
    )

    print(
        "PhoBERT ATE V1"
    )

    print(
        "Device:",
        DEVICE
    )

    while True:

        text = input(
            "\nComment: "
        ).strip()

        if not text:
            break

        result = predict_aspects(
            text,
            tokenizer,
            model,
        )

        print(
            "Tokens:"
        )

        for token, label in zip(
            result["tokens"],
            result["labels"],
        ):

            print(
                f"{token:20}"
                f"{label}"
            )

        print(
            "\nAspects:"
        )

        print(
            result["aspects"]
        )


if __name__ == "__main__":
    main()