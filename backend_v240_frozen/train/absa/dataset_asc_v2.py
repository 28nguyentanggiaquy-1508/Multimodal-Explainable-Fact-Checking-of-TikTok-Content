import json

import torch
from torch.utils.data import Dataset

from config_asc_v2 import (
    LABEL2ID,
    MAX_LENGTH,
)


class ASCV2Dataset(Dataset):
    """
    PhoBERT pair input:

        sequence A = segmented comment
        sequence B = segmented aspect term

    This makes the target explicit when one comment has
    multiple aspects with different sentiments.
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

    def __len__(self):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):
        sample = self.samples[
            index
        ]

        encoded = self.tokenizer(
            sample[
                "segmented_text"
            ],

            sample[
                "segmented_aspect"
            ],

            max_length=
                MAX_LENGTH,

            truncation=True,

            padding=
                "max_length",

            return_tensors=
                "pt",
        )

        return {
            "input_ids":
                encoded[
                    "input_ids"
                ][
                    0
                ],

            "attention_mask":
                encoded[
                    "attention_mask"
                ][
                    0
                ],

            "labels":
                torch.tensor(
                    LABEL2ID[
                        sample[
                            "label"
                        ]
                    ],
                    dtype=torch.long,
                ),

            "sample_index":
                torch.tensor(
                    index,
                    dtype=torch.long,
                ),
        }
