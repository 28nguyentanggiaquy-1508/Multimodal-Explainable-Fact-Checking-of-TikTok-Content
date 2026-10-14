import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from config_ate_v2 import (
    MODEL_OUTPUT_DIR,
    VNCORENLP_DIR,
    LABEL2ID,
    ID2LABEL,
    MAX_LENGTH,
    DEVICE,
)

from metrics_ate_v2 import (
    labels_to_char_spans,
)

from vncorenlp_segmenter import (
    VnCoreNLPSegmenter,
)


class ATEPhoBERTV2:
    def __init__(
        self,
        model_dir=MODEL_OUTPUT_DIR,
    ):
        self.model_dir = Path(
            model_dir
        )

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                self.model_dir
            )
        )

        self.model = (
            AutoModelForTokenClassification
            .from_pretrained(
                self.model_dir
            )
            .to(
                DEVICE
            )
        )

        self.model.eval()

        config_path = (
            self.model_dir
            / "inference_config.json"
        )

        if config_path.exists():
            config = json.loads(
                config_path.read_text(
                    encoding="utf-8"
                )
            )

            self.threshold = float(
                config.get(
                    "aspect_threshold",
                    0.50,
                )
            )

        else:
            self.threshold = 0.50

        self.segmenter = (
            VnCoreNLPSegmenter(
                VNCORENLP_DIR
            )
        )

    def _encode(
        self,
        segments,
    ):
        token_ids = []
        segment_ids = []

        for (
            segment_index,
            segment,
        ) in enumerate(
            segments
        ):
            pieces = (
                self.tokenizer
                .encode(
                    segment[
                        "word_form"
                    ],
                    add_special_tokens=False,
                )
            )

            if not pieces:
                pieces = [
                    self.tokenizer
                    .unk_token_id
                ]

            for piece_id in pieces:
                token_ids.append(
                    piece_id
                )

                segment_ids.append(
                    segment_index
                )

        max_body = (
            MAX_LENGTH
            - self.tokenizer
            .num_special_tokens_to_add(
                pair=False
            )
        )

        token_ids = token_ids[
            :max_body
        ]

        segment_ids = segment_ids[
            :max_body
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

        final_segment_ids = []
        body_cursor = 0

        for is_special in special_mask:
            if is_special:
                final_segment_ids.append(
                    -1
                )
            else:
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

        return (
            torch.tensor(
                [
                    input_ids
                ],
                dtype=torch.long,
                device=DEVICE,
            ),

            torch.tensor(
                [
                    attention_mask
                ],
                dtype=torch.long,
                device=DEVICE,
            ),

            final_segment_ids,
        )

    @torch.no_grad()
    def predict(
        self,
        text,
    ):
        segments = (
            self.segmenter
            .segment_with_offsets(
                text
            )
        )

        (
            input_ids,
            attention_mask,
            segment_ids,
        ) = self._encode(
            segments
        )

        outputs = self.model(
            input_ids=
                input_ids,

            attention_mask=
                attention_mask,
        )

        probs = (
            torch.softmax(
                outputs.logits[
                    0
                ],
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
                or segment_id
                in first_piece_probs
            ):
                continue

            first_piece_probs[
                segment_id
            ] = (
                probs[
                    token_index
                ]
                .tolist()
            )

        labels = []

        confidences = []

        for segment_index in range(
            len(
                segments
            )
        ):
            values = (
                first_piece_probs
                .get(
                    segment_index
                )
            )

            if values is None:
                labels.append(
                    "O"
                )

                confidences.append(
                    0.0
                )

                continue

            o_prob = values[
                LABEL2ID[
                    "O"
                ]
            ]

            b_prob = values[
                LABEL2ID[
                    "B-ASP"
                ]
            ]

            i_prob = values[
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
                < self.threshold
                or o_prob
                >= aspect_prob
            ):
                labels.append(
                    "O"
                )

                confidences.append(
                    o_prob
                )

            elif b_prob >= i_prob:
                labels.append(
                    "B-ASP"
                )

                confidences.append(
                    b_prob
                )

            else:
                labels.append(
                    "I-ASP"
                )

                confidences.append(
                    i_prob
                )

        spans = (
            labels_to_char_spans(
                labels,
                segments,
            )
        )

        aspects = []

        for start, end in spans:
            segment_confidences = [
                confidence
                for label, confidence, segment
                in zip(
                    labels,
                    confidences,
                    segments,
                )
                if (
                    label != "O"
                    and segment[
                        "start"
                    ] >= start
                    and segment[
                        "end"
                    ] <= end
                )
            ]

            confidence = (
                sum(
                    segment_confidences
                )
                / len(
                    segment_confidences
                )
                if segment_confidences
                else 0.0
            )

            aspects.append({
                "term":
                    text[
                        start:
                        end
                    ],

                "start":
                    start,

                "end":
                    end,

                "confidence":
                    confidence,
            })

        return {
            "text":
                text,

            "threshold":
                self.threshold,

            "segments":
                segments,

            "labels":
                labels,

            "aspects":
                aspects,
        }
