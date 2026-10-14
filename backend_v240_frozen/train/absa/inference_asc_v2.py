from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from config_asc_v2 import (
    MODEL_OUTPUT_DIR,
    VNCORENLP_DIR,
    ID2LABEL,
    MAX_LENGTH,
    DEVICE,
)

from vncorenlp_segmenter import (
    VnCoreNLPSegmenter,
)


class ASCPhoBERTV2:
    def __init__(
        self,
        model_dir=MODEL_OUTPUT_DIR,
    ):
        self.model_dir = Path(
            model_dir
        )

        # ==============================
        # TOKENIZER
        # ==============================
        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                self.model_dir
            )
        )

        # ==============================
        # MODEL
        # ==============================
        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                self.model_dir
            )
            .to(
                DEVICE
            )
        )

        self.model.eval()

        # ==============================
        # VNCORENLP
        # ==============================
        self.segmenter = (
            VnCoreNLPSegmenter(
                VNCORENLP_DIR
            )
        )

    def _segment(self, text):
        """
        Word segmentation giống preprocessing
        lúc train ASC V2.
        """

        segments = (
            self.segmenter
            .segment_with_offsets(
                text
            )
        )

        return " ".join(
            token["word_form"]
            for token in segments
        )

    @torch.no_grad()
    def predict(
        self,
        text,
        aspect_term,
    ):
        # ==============================
        # SEGMENT INPUT
        # ==============================
        segmented_text = (
            self._segment(
                text
            )
        )

        segmented_aspect = (
            self._segment(
                aspect_term
            )
        )

        # ==============================
        # TOKENIZE
        #
        # sequence A = comment
        # sequence B = aspect
        # ==============================
        encoded = (
            self.tokenizer(
                segmented_text,
                segmented_aspect,

                max_length=
                    MAX_LENGTH,

                truncation=True,

                padding=True,

                return_tensors=
                    "pt",
            )
        )

        encoded = {
            key:
                value.to(
                    DEVICE
                )
            for key, value
            in encoded.items()
        }

        # ==============================
        # INFERENCE
        # ==============================
        outputs = self.model(
            **encoded
        )

        logits = outputs.logits[0]

        probabilities = (
            torch.softmax(
                logits,
                dim=-1,
            )
            .cpu()
            .tolist()
        )

        pred_id = int(
            torch.argmax(
                logits
            ).item()
        )

        return {
            "sentiment":
                ID2LABEL[
                    pred_id
                ],

            "confidence":
                float(
                    probabilities[
                        pred_id
                    ]
                ),

            "probabilities": {
                ID2LABEL[index]:
                    float(
                        probability
                    )

                for index, probability
                in enumerate(
                    probabilities
                )
            },
        }