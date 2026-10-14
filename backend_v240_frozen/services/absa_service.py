import json
import os
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)


# ============================================================
# PATHS / RUNTIME VERSION
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]

VNCORENLP_DIR = (
    BACKEND_DIR
    / "assets"
    / "vncorenlp"
)

# Default deploy = V2.
#
# Rollback without deleting any model:
#   PowerShell:
#       $env:ABSA_RUNTIME_VERSION="v1"
#       uvicorn main:app --reload
#
# Restore V2:
#       $env:ABSA_RUNTIME_VERSION="v2"
#
ABSA_RUNTIME_VERSION = (
    os.getenv(
        "ABSA_RUNTIME_VERSION",
        "v2",
    )
    .strip()
    .lower()
)

if ABSA_RUNTIME_VERSION not in {
    "v1",
    "v2",
}:
    raise ValueError(
        "ABSA_RUNTIME_VERSION chỉ hỗ trợ "
        "'v1' hoặc 'v2'."
    )

ATE_MODEL_VERSION = (
    "ate_phobert_v2"
    if ABSA_RUNTIME_VERSION == "v2"
    else "ate_phobert_v1"
)

ASC_MODEL_VERSION = (
    "asc_phobert_v2"
    if ABSA_RUNTIME_VERSION == "v2"
    else "asc_phobert_v1"
)

ATE_MODEL_DIR = (
    BACKEND_DIR
    / "model"
    / "absa"
    / ATE_MODEL_VERSION
)

ASC_MODEL_DIR = (
    BACKEND_DIR
    / "model"
    / "absa"
    / ASC_MODEL_VERSION
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MAX_LENGTH_V1 = 128
MAX_LENGTH_V2 = 256

# Official ATE V2 validation-selected threshold.
# If inference_config.json exists, its value takes precedence.
ATE_V2_FALLBACK_THRESHOLD = 0.45


# ============================================================
# V1 PREPROCESSING
# Keep exact legacy behavior for rollback.
# ============================================================

WORD_PATTERN = re.compile(
    r"\w+|[^\w\s]",
    flags=re.UNICODE,
)


def _regex_tokenize_with_offsets(
    text,
):
    items = []

    for match in WORD_PATTERN.finditer(
        text
    ):
        items.append({
            "word":
                match.group(0),

            "start":
                match.start(),

            "end":
                match.end(),
        })

    return items


def _build_ate_inputs_v1(
    text,
    tokenizer,
    max_length=MAX_LENGTH_V1,
):
    word_items = (
        _regex_tokenize_with_offsets(
            text
        )
    )

    token_ids = []
    word_ids = []

    for word_index, item in enumerate(
        word_items
    ):
        pieces = tokenizer.encode(
            item["word"],
            add_special_tokens=False,
        )

        for piece_id in pieces:
            token_ids.append(
                piece_id
            )

            word_ids.append(
                word_index
            )

    # Legacy V1 used max_length - 2.
    token_ids = token_ids[
        : max_length - 2
    ]

    word_ids = word_ids[
        : max_length - 2
    ]

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

    expanded_word_ids = []
    source_index = 0

    for is_special in special_mask:
        if is_special:
            expanded_word_ids.append(
                -1
            )
        else:
            expanded_word_ids.append(
                word_ids[
                    source_index
                ]
            )
            source_index += 1

    attention_mask = [
        1
    ] * len(
        input_ids
    )

    pad_token_id = (
        tokenizer.pad_token_id
    )

    while len(
        input_ids
    ) < max_length:
        input_ids.append(
            pad_token_id
        )

        attention_mask.append(
            0
        )

        expanded_word_ids.append(
            -1
        )

    return {
        "input_ids":
            torch.tensor(
                [
                    input_ids[
                        :max_length
                    ]
                ],
                dtype=torch.long,
            ),

        "attention_mask":
            torch.tensor(
                [
                    attention_mask[
                        :max_length
                    ]
                ],
                dtype=torch.long,
            ),

        "word_ids":
            expanded_word_ids[
                :max_length
            ],

        "word_items":
            word_items,
    }


def _bio_to_aspects_v1(
    labels,
    word_items,
):
    aspects = []

    start_word = None
    end_word = None

    def close_current():
        nonlocal start_word
        nonlocal end_word

        if start_word is None:
            return

        start = word_items[
            start_word
        ]["start"]

        end = word_items[
            end_word
        ]["end"]

        aspects.append({
            "start":
                start,

            "end":
                end,
        })

        start_word = None
        end_word = None

    for word_index, label in enumerate(
        labels
    ):
        if label == "B-ASP":
            close_current()

            start_word = (
                word_index
            )

            end_word = (
                word_index
            )

        elif label == "I-ASP":
            # Legacy BIO repair.
            if start_word is None:
                start_word = (
                    word_index
                )

            end_word = (
                word_index
            )

        else:
            close_current()

    close_current()

    return aspects


# ============================================================
# V2 VnCoreNLP SEGMENTATION
# ============================================================

class VnCoreNLPSegmenter:
    """
    VnCoreNLP word segmentation + offsets in ORIGINAL raw text.

    PhoBERT expects Vietnamese word segmentation, e.g.
        "Tỉ_lệ", "cơ_thể", "điện_thoại"

    This implementation also preserves non-whitespace raw gaps
    (punctuation / emoji / symbols) as raw tokens so live runtime
    follows the V2 preprocessing style used to prepare the dataset.
    """

    def __init__(
        self,
        model_dir,
    ):
        try:
            import py_vncorenlp
        except ImportError as exc:
            raise RuntimeError(
                "Thiếu py_vncorenlp. "
                "Chạy: pip install py_vncorenlp"
            ) from exc

        model_dir = Path(
            model_dir
        ).resolve()

        jar_path = (
            model_dir
            / "VnCoreNLP-1.2.jar"
        )

        if not jar_path.exists():
            raise RuntimeError(
                "Chưa có VnCoreNLP model tại: "
                f"{model_dir}. "
                "Cần có VnCoreNLP-1.2.jar."
            )

        self.model = (
            py_vncorenlp.VnCoreNLP(
                annotators=[
                    "wseg"
                ],
                save_dir=str(
                    model_dir
                ),
            )
        )

    @staticmethod
    def _java_truncated_shadow(
        text,
    ):
        """
        Some Windows/Java + VnCoreNLP combinations can return a
        supplementary-plane Unicode character with only its low 16 bits.

        Example:
            raw       : 𝓜𝓮𝓵𝓸𝓭𝔂
            VnCoreNLP : 퓜퓮퓵퓸퓭픂

        The shadow keeps exactly one Python character per raw character,
        therefore offsets still map to the original raw string.
        """
        output = []

        for ch in str(
            text
        ):
            codepoint = ord(
                ch
            )

            if codepoint <= 0xFFFF:
                output.append(
                    ch
                )
                continue

            low16 = (
                codepoint
                & 0xFFFF
            )

            # Avoid isolated surrogate characters.
            if (
                0xD800
                <= low16
                <= 0xDFFF
            ):
                output.append(
                    ch
                )
            else:
                output.append(
                    chr(
                        low16
                    )
                )

        return "".join(
            output
        )

    @staticmethod
    def _find_surface(
        text,
        surface,
        cursor,
    ):
        surface = str(
            surface
        )

        stripped_surface = (
            surface.strip()
        )

        if not stripped_surface:
            raise ValueError(
                "Empty segmented surface."
            )

        parts = re.split(
            r"\s+",
            stripped_surface,
        )

        pattern = (
            r"\s+".join(
                re.escape(
                    part
                )
                for part
                in parts
            )
        )

        # 1) Normal raw-text alignment.
        match = re.search(
            pattern,
            text[
                cursor:
            ],
            flags=re.IGNORECASE,
        )

        if match is not None:
            start = (
                cursor
                + match.start()
            )

            end = (
                cursor
                + match.end()
            )

        else:
            # 2) Recover Java-truncated supplementary Unicode.
            shadow_text = (
                VnCoreNLPSegmenter
                ._java_truncated_shadow(
                    text
                )
            )

            shadow_match = re.search(
                pattern,
                shadow_text[
                    cursor:
                ],
                flags=re.IGNORECASE,
            )

            if shadow_match is not None:
                start = (
                    cursor
                    + shadow_match.start()
                )

                end = (
                    cursor
                    + shadow_match.end()
                )

            else:
                match_any = re.search(
                    pattern,
                    text,
                    flags=re.IGNORECASE,
                )

                shadow_any = re.search(
                    pattern,
                    shadow_text,
                    flags=re.IGNORECASE,
                )

                candidate = (
                    match_any
                    if match_any is not None
                    else shadow_any
                )

                if candidate is None:
                    raise ValueError(
                        "Không align được VnCoreNLP token "
                        "với raw text. "
                        f"token={stripped_surface!r}, "
                        f"cursor={cursor}, "
                        f"text={text!r}"
                    )

                if (
                    candidate.start()
                    < cursor
                ):
                    raise ValueError(
                        "VnCoreNLP token bị align "
                        "ngược thứ tự. "
                        f"token={stripped_surface!r}, "
                        f"cursor={cursor}"
                    )

                start = (
                    candidate.start()
                )

                end = (
                    candidate.end()
                )

        while (
            start < end
            and text[
                start
            ].isspace()
        ):
            start += 1

        while (
            end > start
            and text[
                end - 1
            ].isspace()
        ):
            end -= 1

        return (
            start,
            end,
        )

    @staticmethod
    def _raw_gap_tokens(
        text,
        start,
        end,
    ):
        """
        Preserve non-whitespace material skipped by VnCoreNLP.

        Example:
            "."       -> "."
            "/"       -> "/"
            "😂😂"    -> "😂😂"

        Spaces are not emitted as model tokens.
        """
        output = []

        if end <= start:
            return output

        gap = text[
            start:
            end
        ]

        for match in re.finditer(
            r"\S+",
            gap,
            flags=re.UNICODE,
        ):
            token_start = (
                start
                + match.start()
            )

            token_end = (
                start
                + match.end()
            )

            surface = text[
                token_start:
                token_end
            ]

            output.append({
                "word_form":
                    surface,

                "surface":
                    surface,

                "start":
                    token_start,

                "end":
                    token_end,

                "source":
                    "raw",
            })

        return output

    @staticmethod
    def _fallback_segments(
        text,
    ):
        """
        Safe per-comment fallback for rare social-media Unicode that
        VnCoreNLP cannot align. Exact raw offsets are preserved.
        """
        output = []

        for match in re.finditer(
            r"\w+|[^\w\s]",
            str(
                text
            ),
            flags=re.UNICODE,
        ):
            surface = (
                match.group(0)
            )

            output.append({
                "word_form":
                    surface,

                "surface":
                    surface,

                "start":
                    match.start(),

                "end":
                    match.end(),

                "source":
                    "fallback",
            })

        return output

    def segment_with_offsets(
        self,
        text,
    ):
        text = str(
            text or ""
        )

        if not text:
            return []

        try:
            sentences = (
                self.model
                .word_segment(
                    text
                )
            )

            aligned = []
            cursor = 0

            for sentence in sentences:
                for word_form in str(
                    sentence
                ).split():

                    surface = (
                        word_form
                        .replace(
                            "_",
                            " ",
                        )
                    )

                    # A literal "_" token would become whitespace after
                    # replace("_", " "). Keep it literal.
                    if not surface.strip():
                        surface = (
                            word_form
                        )

                    start, end = (
                        self._find_surface(
                            text,
                            surface,
                            cursor,
                        )
                    )

                    aligned.extend(
                        self._raw_gap_tokens(
                            text,
                            cursor,
                            start,
                        )
                    )

                    aligned.append({
                        "word_form":
                            word_form,

                        "surface":
                            text[
                                start:
                                end
                            ],

                        "start":
                            start,

                        "end":
                            end,

                        "source":
                            "vncorenlp",
                    })

                    cursor = end

            aligned.extend(
                self._raw_gap_tokens(
                    text,
                    cursor,
                    len(
                        text
                    ),
                )
            )

            if (
                text.strip()
                and not aligned
            ):
                raise ValueError(
                    "VnCoreNLP returned no aligned tokens."
                )

            return aligned

        except Exception as exc:
            # Do not drop ABSA for the entire comment because of one
            # Unicode/alignment edge case.
            print(
                "[ABSA SEGMENT FALLBACK]",
                type(
                    exc
                ).__name__,
                str(
                    exc
                ),
                "| text=",
                repr(
                    text[:120]
                ),
            )

            return (
                self._fallback_segments(
                    text
                )
            )


def _segments_to_segmented_text(
    segments,
):
    return " ".join(
        str(
            token[
                "word_form"
            ]
        )
        for token in segments
    )


# ============================================================
# V2 ATE PREPROCESSING / BIO REPAIR
# ============================================================

def _build_ate_inputs_v2(
    segments,
    tokenizer,
    max_length=MAX_LENGTH_V2,
):
    token_ids = []
    segment_ids = []

    for (
        segment_index,
        segment,
    ) in enumerate(
        segments
    ):
        pieces = tokenizer.encode(
            segment[
                "word_form"
            ],
            add_special_tokens=False,
        )

        if not pieces:
            pieces = [
                tokenizer.unk_token_id
            ]

        for piece_id in pieces:
            token_ids.append(
                piece_id
            )

            segment_ids.append(
                segment_index
            )

    max_body = (
        max_length
        - tokenizer
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
        ),

        torch.tensor(
            [
                attention_mask
            ],
            dtype=torch.long,
        ),

        final_segment_ids,
    )


def _repair_bio_v2(
    labels,
):
    output = []
    inside = False

    for label in labels:
        if label == "B-ASP":
            output.append(
                "B-ASP"
            )
            inside = True

        elif label == "I-ASP":
            if inside:
                output.append(
                    "I-ASP"
                )
            else:
                output.append(
                    "B-ASP"
                )
                inside = True

        else:
            output.append(
                "O"
            )
            inside = False

    return output


def _labels_to_char_spans_v2(
    labels,
    segments,
):
    labels = (
        _repair_bio_v2(
            labels
        )
    )

    spans = []

    current_start = None
    current_end = None

    for label, segment in zip(
        labels,
        segments,
    ):
        start = int(
            segment[
                "start"
            ]
        )

        end = int(
            segment[
                "end"
            ]
        )

        if label == "B-ASP":
            if (
                current_start
                is not None
            ):
                spans.append(
                    (
                        current_start,
                        current_end,
                    )
                )

            current_start = (
                start
            )
            current_end = (
                end
            )

        elif label == "I-ASP":
            if (
                current_start
                is None
            ):
                current_start = (
                    start
                )

            current_end = (
                end
            )

        else:
            if (
                current_start
                is not None
            ):
                spans.append(
                    (
                        current_start,
                        current_end,
                    )
                )

                current_start = None
                current_end = None

    if (
        current_start
        is not None
    ):
        spans.append(
            (
                current_start,
                current_end,
            )
        )

    return spans


# ============================================================
# SERVICE
# ============================================================

class ABSAService:
    def __init__(
        self,
        runtime_version=
            ABSA_RUNTIME_VERSION,
    ):
        self.runtime_version = (
            runtime_version
        )

        self.ate_model_version = (
            "ate_phobert_v2"
            if runtime_version == "v2"
            else "ate_phobert_v1"
        )

        self.asc_model_version = (
            "asc_phobert_v2"
            if runtime_version == "v2"
            else "asc_phobert_v1"
        )

        self.ate_model_dir = (
            BACKEND_DIR
            / "model"
            / "absa"
            / self.ate_model_version
        )

        self.asc_model_dir = (
            BACKEND_DIR
            / "model"
            / "absa"
            / self.asc_model_version
        )

        self.ready = False

        self.ate_tokenizer = None
        self.ate_model = None

        self.asc_tokenizer = None
        self.asc_model = None

        self.segmenter = None

        self.ate_threshold = (
            ATE_V2_FALLBACK_THRESHOLD
            if runtime_version == "v2"
            else None
        )

    def runtime_info(
        self,
    ):
        return {
            "runtime_version":
                self.runtime_version,

            "ate_model_version":
                self.ate_model_version,

            "asc_model_version":
                self.asc_model_version,

            "ate_threshold":
                self.ate_threshold,

            "device":
                str(
                    DEVICE
                ),

            "segmentation":
                (
                    "vncorenlp"
                    if self.runtime_version
                    == "v2"
                    else "regex_v1"
                ),
        }

    def load(
        self,
    ):
        if self.ready:
            return

        if not (
            self.ate_model_dir
            .exists()
        ):
            raise FileNotFoundError(
                "Không tìm thấy ATE model: "
                f"{self.ate_model_dir}"
            )

        if not (
            self.asc_model_dir
            .exists()
        ):
            raise FileNotFoundError(
                "Không tìm thấy ASC model: "
                f"{self.asc_model_dir}"
            )

        print(
            "[ABSA] Runtime:",
            self.runtime_version,
        )

        print(
            "[ABSA] Loading ATE model:",
            self.ate_model_dir,
        )

        self.ate_tokenizer = (
            AutoTokenizer
            .from_pretrained(
                self.ate_model_dir
            )
        )

        self.ate_model = (
            AutoModelForTokenClassification
            .from_pretrained(
                self.ate_model_dir
            )
        )

        self.ate_model.to(
            DEVICE
        )

        self.ate_model.eval()

        print(
            "[ABSA] Loading ASC model:",
            self.asc_model_dir,
        )

        self.asc_tokenizer = (
            AutoTokenizer
            .from_pretrained(
                self.asc_model_dir
            )
        )

        self.asc_model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                self.asc_model_dir
            )
        )

        self.asc_model.to(
            DEVICE
        )

        self.asc_model.eval()

        if (
            self.runtime_version
            == "v2"
        ):
            inference_config = (
                self.ate_model_dir
                / "inference_config.json"
            )

            if inference_config.exists():
                try:
                    config = json.loads(
                        inference_config
                        .read_text(
                            encoding="utf-8"
                        )
                    )

                    self.ate_threshold = (
                        float(
                            config.get(
                                "aspect_threshold",
                                ATE_V2_FALLBACK_THRESHOLD,
                            )
                        )
                    )

                except Exception as exc:
                    print(
                        "[ABSA] Warning: "
                        "không đọc được "
                        "inference_config.json:",
                        exc,
                    )

                    self.ate_threshold = (
                        ATE_V2_FALLBACK_THRESHOLD
                    )

            self.segmenter = (
                VnCoreNLPSegmenter(
                    VNCORENLP_DIR
                )
            )

        self.ready = True

        print(
            "[ABSA] Ready:",
            self.runtime_info(),
        )

    # --------------------------------------------------------
    # ATE
    # --------------------------------------------------------

    @torch.no_grad()
    def _extract_aspects_v1(
        self,
        text,
    ):
        prepared = (
            _build_ate_inputs_v1(
                text,
                self.ate_tokenizer,
                MAX_LENGTH_V1,
            )
        )

        input_ids = (
            prepared[
                "input_ids"
            ]
            .to(
                DEVICE
            )
        )

        attention_mask = (
            prepared[
                "attention_mask"
            ]
            .to(
                DEVICE
            )
        )

        outputs = (
            self.ate_model(
                input_ids=
                    input_ids,

                attention_mask=
                    attention_mask,
            )
        )

        pred_ids = (
            outputs.logits
            .argmax(
                dim=-1
            )[
                0
            ]
            .cpu()
            .tolist()
        )

        word_predictions = {}

        for pred_id, word_id in zip(
            pred_ids,
            prepared[
                "word_ids"
            ],
        ):
            if word_id < 0:
                continue

            if (
                word_id
                in word_predictions
            ):
                continue

            label = (
                self.ate_model
                .config
                .id2label[
                    int(
                        pred_id
                    )
                ]
            )

            word_predictions[
                word_id
            ] = label

        word_items = (
            prepared[
                "word_items"
            ]
        )

        labels = [
            word_predictions.get(
                i,
                "O",
            )
            for i in range(
                len(
                    word_items
                )
            )
        ]

        aspects = (
            _bio_to_aspects_v1(
                labels,
                word_items,
            )
        )

        for aspect in aspects:
            aspect[
                "term"
            ] = text[
                aspect[
                    "start"
                ]:
                aspect[
                    "end"
                ]
            ]

        return aspects

    @torch.no_grad()
    def _extract_aspects_v2(
        self,
        text,
        segments,
    ):
        (
            input_ids,
            attention_mask,
            segment_ids,
        ) = _build_ate_inputs_v2(
            segments,
            self.ate_tokenizer,
            MAX_LENGTH_V2,
        )

        input_ids = (
            input_ids.to(
                DEVICE
            )
        )

        attention_mask = (
            attention_mask.to(
                DEVICE
            )
        )

        outputs = (
            self.ate_model(
                input_ids=
                    input_ids,

                attention_mask=
                    attention_mask,
            )
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

        for (
            token_index,
            segment_id,
        ) in enumerate(
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

        # Training config:
        # O=0, B-ASP=1, I-ASP=2.
        label2id = {
            "O": 0,
            "B-ASP": 1,
            "I-ASP": 2,
        }

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
                label2id[
                    "O"
                ]
            ]

            b_prob = values[
                label2id[
                    "B-ASP"
                ]
            ]

            i_prob = values[
                label2id[
                    "I-ASP"
                ]
            ]

            aspect_prob = max(
                b_prob,
                i_prob,
            )

            if (
                aspect_prob
                < self.ate_threshold
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
            _labels_to_char_spans_v2(
                labels,
                segments,
            )
        )

        aspects = []

        for start, end in spans:
            segment_confidences = [
                confidence
                for (
                    label,
                    confidence,
                    segment,
                )
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
                (
                    sum(
                        segment_confidences
                    )
                    / len(
                        segment_confidences
                    )
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

                "ate_confidence":
                    round(
                        float(
                            confidence
                        ),
                        4,
                    ),
            })

        return aspects

    def extract_aspects(
        self,
        text,
        segments=None,
    ):
        self.load()

        if (
            self.runtime_version
            == "v1"
        ):
            return (
                self._extract_aspects_v1(
                    text
                )
            )

        if segments is None:
            segments = (
                self.segmenter
                .segment_with_offsets(
                    text
                )
            )

        return (
            self._extract_aspects_v2(
                text,
                segments,
            )
        )

    # --------------------------------------------------------
    # ASC
    # --------------------------------------------------------

    @torch.no_grad()
    def _classify_sentiment_v1(
        self,
        text,
        aspect_term,
    ):
        encoded = (
            self.asc_tokenizer(
                text,
                aspect_term,

                truncation=
                    "only_first",

                max_length=
                    MAX_LENGTH_V1,

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

        outputs = (
            self.asc_model(
                **encoded
            )
        )

        probabilities = (
            torch.softmax(
                outputs.logits,
                dim=-1,
            )[
                0
            ]
            .cpu()
            .tolist()
        )

        pred_id = int(
            outputs.logits
            .argmax(
                dim=-1
            )[
                0
            ]
            .item()
        )

        label = (
            self.asc_model
            .config
            .id2label[
                pred_id
            ]
        )

        return {
            "sentiment":
                label,

            "confidence":
                round(
                    float(
                        probabilities[
                            pred_id
                        ]
                    ),
                    4,
                ),

            "probabilities": {
                self.asc_model
                .config
                .id2label[i]:
                    round(
                        float(
                            probabilities[i]
                        ),
                        4,
                    )
                for i in range(
                    len(
                        probabilities
                    )
                )
            },
        }

    @torch.no_grad()
    def _classify_sentiment_v2(
        self,
        text,
        aspect_term,
        segmented_text=None,
    ):
        if segmented_text is None:
            text_segments = (
                self.segmenter
                .segment_with_offsets(
                    text
                )
            )

            segmented_text = (
                _segments_to_segmented_text(
                    text_segments
                )
            )

        aspect_segments = (
            self.segmenter
            .segment_with_offsets(
                aspect_term
            )
        )

        segmented_aspect = (
            _segments_to_segmented_text(
                aspect_segments
            )
        )

        encoded = (
            self.asc_tokenizer(
                segmented_text,
                segmented_aspect,

                max_length=
                    MAX_LENGTH_V2,

                truncation=True,

                # Match ASC V2 training/evaluation.
                padding=
                    "max_length",

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

        outputs = (
            self.asc_model(
                **encoded
            )
        )

        probabilities = (
            torch.softmax(
                outputs.logits[
                    0
                ],
                dim=-1,
            )
            .cpu()
            .tolist()
        )

        pred_id = int(
            torch.argmax(
                outputs.logits[
                    0
                ]
            ).item()
        )

        # Training config:
        # NEGATIVE=0, NEUTRAL=1, POSITIVE=2.
        id2label = {
            0: "NEGATIVE",
            1: "NEUTRAL",
            2: "POSITIVE",
        }

        label = (
            id2label[
                pred_id
            ]
        )

        return {
            "sentiment":
                label,

            "confidence":
                round(
                    float(
                        probabilities[
                            pred_id
                        ]
                    ),
                    4,
                ),

            "probabilities": {
                id2label[
                    index
                ]:
                    round(
                        float(
                            probability
                        ),
                        4,
                    )
                for (
                    index,
                    probability,
                )
                in enumerate(
                    probabilities
                )
            },
        }

    def classify_sentiment(
        self,
        text,
        aspect_term,
        segmented_text=None,
    ):
        self.load()

        if (
            self.runtime_version
            == "v1"
        ):
            return (
                self._classify_sentiment_v1(
                    text,
                    aspect_term,
                )
            )

        return (
            self._classify_sentiment_v2(
                text,
                aspect_term,
                segmented_text=
                    segmented_text,
            )
        )

    # --------------------------------------------------------
    # COMMENT / BATCH
    # --------------------------------------------------------

    def analyze_comment(
        self,
        text,
    ):
        text = str(
            text or ""
        ).strip()

        if not text:
            return {
                "aspect_count":
                    0,

                "aspects":
                    [],
            }

        self.load()

        segments = None
        segmented_text = None

        if (
            self.runtime_version
            == "v2"
        ):
            # Segment one time per comment and reuse for
            # ATE + every ASC prediction of that comment.
            segments = (
                self.segmenter
                .segment_with_offsets(
                    text
                )
            )

            segmented_text = (
                _segments_to_segmented_text(
                    segments
                )
            )

        extracted = (
            self.extract_aspects(
                text,
                segments=
                    segments,
            )
        )

        results = []

        for aspect in extracted:
            sentiment = (
                self.classify_sentiment(
                    text,
                    aspect[
                        "term"
                    ],
                    segmented_text=
                        segmented_text,
                )
            )

            result = {
                "term":
                    aspect[
                        "term"
                    ],

                "start":
                    aspect[
                        "start"
                    ],

                "end":
                    aspect[
                        "end"
                    ],
            }

            if (
                "ate_confidence"
                in aspect
            ):
                result[
                    "ate_confidence"
                ] = aspect[
                    "ate_confidence"
                ]

            result.update(
                sentiment
            )

            results.append(
                result
            )

        return {
            "aspect_count":
                len(
                    results
                ),

            "aspects":
                results,
        }

    def analyze_comments(
        self,
        texts,
    ):
        results = []

        for text in texts:
            try:
                result = (
                    self.analyze_comment(
                        text
                    )
                )

            except Exception as exc:
                print(
                    "[ABSA COMMENT ERROR]",
                    exc,
                )

                result = {
                    "aspect_count":
                        0,

                    "aspects":
                        [],

                    "error":
                        str(
                            exc
                        ),
                }

            results.append(
                result
            )

        return results

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    @staticmethod
    def summarize(
        absa_results,
    ):
        sentiment_counts = {
            "POSITIVE":
                0,

            "NEUTRAL":
                0,

            "NEGATIVE":
                0,
        }

        aspect_counter = (
            Counter()
        )

        aspect_sentiments = {}

        total_aspects = 0
        comments_with_aspects = 0

        for result in absa_results:
            aspects = (
                result.get(
                    "aspects"
                )
                or []
            )

            if aspects:
                comments_with_aspects += 1

            for aspect in aspects:
                total_aspects += 1

                term = str(
                    aspect.get(
                        "term",
                        "",
                    )
                ).strip()

                sentiment = (
                    aspect.get(
                        "sentiment"
                    )
                )

                if (
                    sentiment
                    in sentiment_counts
                ):
                    sentiment_counts[
                        sentiment
                    ] += 1

                if term:
                    key = (
                        term.lower()
                    )

                    aspect_counter[
                        key
                    ] += 1

                    if (
                        key
                        not in aspect_sentiments
                    ):
                        aspect_sentiments[
                            key
                        ] = {
                            "POSITIVE":
                                0,

                            "NEUTRAL":
                                0,

                            "NEGATIVE":
                                0,
                        }

                    if (
                        sentiment
                        in aspect_sentiments[
                            key
                        ]
                    ):
                        aspect_sentiments[
                            key
                        ][
                            sentiment
                        ] += 1

        top_aspects = []

        for term, count in (
            aspect_counter
            .most_common(
                10
            )
        ):
            top_aspects.append({
                "term":
                    term,

                "count":
                    count,

                "sentiments":
                    aspect_sentiments[
                        term
                    ],
            })

        return {
            "comments_analyzed":
                len(
                    absa_results
                ),

            "comments_with_aspects":
                comments_with_aspects,

            "total_aspects":
                total_aspects,

            "sentiments":
                sentiment_counts,

            "top_aspects":
                top_aspects,
        }


# Lazy singleton:
# model loads only on the first ABSA call.
absa_service = ABSAService()
