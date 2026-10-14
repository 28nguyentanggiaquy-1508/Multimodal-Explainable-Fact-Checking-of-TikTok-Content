import json

import torch
from torch.utils.data import Dataset

from config_ate_v2 import (
    LABEL2ID,
    MAX_LENGTH,
)


class ATEV2Dataset(Dataset):
    """
    VnCoreNLP word-level supervision -> PhoBERT BPE pieces.

    Important:
    - each `word_form` is already Vietnamese word-segmented
      (e.g. "màn_hình", "điện_thoại")
    - we encode each segmented word with PhoBERT BPE
    - first BPE piece inherits the word BIO label
    - subsequent pieces:
        B-ASP -> I-ASP
        I-ASP -> I-ASP
        O     -> O
    """

    def __init__(
        self,
        path,
        tokenizer,
    ):
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            self.samples = json.load(
                f
            )

        self.tokenizer = tokenizer

        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id
            is not None
            else 1
        )

    def __len__(self):
        return len(
            self.samples
        )

    def _encode_word(
        self,
        word_form,
    ):
        ids = self.tokenizer.encode(
            word_form,
            add_special_tokens=False,
        )

        if not ids:
            ids = [
                self.tokenizer.unk_token_id
            ]

        return ids

    def __getitem__(
        self,
        index,
    ):
        sample = self.samples[
            index
        ]

        segments = sample[
            "segments"
        ]

        segment_labels = sample[
            "segment_labels"
        ]

        segment_ignore = sample[
            "segment_ignore"
        ]

        token_ids = []

        labels = []

        segment_ids = []

        for (
            segment_index,
            (
                segment,
                label,
                ignored,
            ),
        ) in enumerate(
            zip(
                segments,
                segment_labels,
                segment_ignore,
            )
        ):
            pieces = self._encode_word(
                segment[
                    "word_form"
                ]
            )

            for piece_index, piece_id in enumerate(
                pieces
            ):
                token_ids.append(
                    piece_id
                )

                segment_ids.append(
                    segment_index
                )

                if ignored:
                    labels.append(
                        -100
                    )
                    continue

                current_label = label

                if (
                    piece_index > 0
                    and label == "B-ASP"
                ):
                    current_label = (
                        "I-ASP"
                    )

                labels.append(
                    LABEL2ID[
                        current_label
                    ]
                )

        # Reserve PhoBERT special tokens.
        max_body_length = (
            MAX_LENGTH
            - self.tokenizer.num_special_tokens_to_add(
                pair=False
            )
        )

        token_ids = token_ids[
            :max_body_length
        ]

        labels = labels[
            :max_body_length
        ]

        segment_ids = segment_ids[
            :max_body_length
        ]

        input_ids = (
            self.tokenizer
            .build_inputs_with_special_tokens(
                token_ids
            )
        )

        special_mask = (
            self.tokenizer
            .get_special_tokens_mask(
                token_ids,
                already_has_special_tokens=False,
            )
        )

        final_labels = []
        final_segment_ids = []

        body_cursor = 0

        for is_special in special_mask:
            if is_special:
                final_labels.append(
                    -100
                )

                final_segment_ids.append(
                    -1
                )
            else:
                final_labels.append(
                    labels[
                        body_cursor
                    ]
                )

                final_segment_ids.append(
                    segment_ids[
                        body_cursor
                    ]
                )

                body_cursor += 1

        attention_mask = [
            1
        ] * len(
            input_ids
        )

        pad_length = (
            MAX_LENGTH
            - len(
                input_ids
            )
        )

        if pad_length > 0:
            input_ids.extend(
                [
                    self.pad_id
                ]
                * pad_length
            )

            attention_mask.extend(
                [0]
                * pad_length
            )

            final_labels.extend(
                [-100]
                * pad_length
            )

            final_segment_ids.extend(
                [-1]
                * pad_length
            )

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

            "segment_ids":
                torch.tensor(
                    final_segment_ids,
                    dtype=torch.long,
                ),

            "sample_index":
                torch.tensor(
                    index,
                    dtype=torch.long,
                ),
        }
