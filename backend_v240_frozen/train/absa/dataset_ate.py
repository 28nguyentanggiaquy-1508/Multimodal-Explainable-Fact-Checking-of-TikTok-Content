import json
import re

import torch

from torch.utils.data import Dataset

from config_ate import (
    LABEL2ID,
    MAX_LENGTH,
)


TOKEN_PATTERN = re.compile(
    r"\w+|[^\w\s]",
    flags=re.UNICODE,
)


def tokenize_text_with_offsets(text):

    tokens = []

    offsets = []

    for match in TOKEN_PATTERN.finditer(text):

        tokens.append(
            match.group()
        )

        offsets.append(
            (
                match.start(),
                match.end(),
            )
        )

    return tokens, offsets


class ABSATokenDataset(Dataset):

    def __init__(
        self,
        file_path,
        tokenizer,
        max_length=MAX_LENGTH,
    ):

        self.tokenizer = tokenizer

        self.max_length = max_length

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as f:

            self.samples = json.load(f)

    def __len__(self):

        return len(
            self.samples
        )

    def _encode_words(
        self,
        words,
        bio_labels,
    ):

        token_ids = []

        label_ids = []

        word_ids = []

        for word_index, (
            word,
            label,
        ) in enumerate(
            zip(
                words,
                bio_labels,
            )
        ):

            pieces = (
                self.tokenizer.encode(
                    word,
                    add_special_tokens=False,
                )
            )

            if not pieces:

                pieces = [
                    self.tokenizer.unk_token_id
                ]

            original_label_id = (
                LABEL2ID[label]
            )

            for piece_index, piece_id in enumerate(
                pieces
            ):

                token_ids.append(
                    piece_id
                )

                word_ids.append(
                    word_index
                )

                if piece_index == 0:

                    label_ids.append(
                        original_label_id
                    )

                else:

                    # B-ASP tiếp tục thành I-ASP
                    if label == "B-ASP":

                        label_ids.append(
                            LABEL2ID["I-ASP"]
                        )

                    else:

                        label_ids.append(
                            original_label_id
                        )

        return (
            token_ids,
            label_ids,
            word_ids,
        )

    def __getitem__(
        self,
        index,
    ):

        sample = self.samples[
            index
        ]

        words = sample[
            "tokens"
        ]

        labels = sample[
            "bio_labels"
        ]

        (
            content_ids,
            content_labels,
            content_word_ids,
        ) = self._encode_words(
            words,
            labels,
        )

        special_count = (
            self.tokenizer
            .num_special_tokens_to_add(
                pair=False
            )
        )

        max_content_length = (
            self.max_length
            - special_count
        )

        content_ids = (
            content_ids[
                :max_content_length
            ]
        )

        content_labels = (
            content_labels[
                :max_content_length
            ]
        )

        content_word_ids = (
            content_word_ids[
                :max_content_length
            ]
        )

        input_ids = (
            self.tokenizer
            .build_inputs_with_special_tokens(
                content_ids
            )
        )

        special_mask = (
            self.tokenizer
            .get_special_tokens_mask(
                content_ids,
                already_has_special_tokens=False,
            )
        )

        final_labels = []

        final_word_ids = []

        content_position = 0

        for is_special in special_mask:

            if is_special:

                final_labels.append(
                    -100
                )

                final_word_ids.append(
                    -1
                )

            else:

                final_labels.append(
                    content_labels[
                        content_position
                    ]
                )

                final_word_ids.append(
                    content_word_ids[
                        content_position
                    ]
                )

                content_position += 1

        attention_mask = [
            1
            for _ in input_ids
        ]

        padding_length = (
            self.max_length
            - len(input_ids)
        )

        if padding_length > 0:

            input_ids += [
                self.tokenizer.pad_token_id
            ] * padding_length

            attention_mask += [
                0
            ] * padding_length

            final_labels += [
                -100
            ] * padding_length

            final_word_ids += [
                -1
            ] * padding_length

        return {
            "input_ids":
                torch.tensor(
                    input_ids,
                    dtype=torch.long,
                ),

            "attention_mask":
                torch.tensor(
                    attention_mask,
                    dtype=torch.long,
                ),

            "labels":
                torch.tensor(
                    final_labels,
                    dtype=torch.long,
                ),

            "word_ids":
                torch.tensor(
                    final_word_ids,
                    dtype=torch.long,
                ),
        }