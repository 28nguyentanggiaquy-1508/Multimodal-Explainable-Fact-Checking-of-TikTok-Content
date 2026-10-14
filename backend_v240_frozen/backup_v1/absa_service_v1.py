import re
from collections import Counter
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)


# ============================================================
# PATHS
# backend/services/absa_service.py -> parents[1] = backend/
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]

ATE_MODEL_DIR = (
    BACKEND_DIR
    / "model"
    / "absa"
    / "ate_phobert_v1"
)

ASC_MODEL_DIR = (
    BACKEND_DIR
    / "model"
    / "absa"
    / "asc_phobert_v1"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MAX_LENGTH = 128

WORD_PATTERN = re.compile(
    r"\w+|[^\w\s]",
    flags=re.UNICODE,
)


# ============================================================
# ATE PREPROCESSING
# Giữ đúng preprocessing của ATE V1 đã train.
# ============================================================

def _regex_tokenize_with_offsets(text):
    items = []

    for match in WORD_PATTERN.finditer(text):
        items.append({
            "word": match.group(0),
            "start": match.start(),
            "end": match.end(),
        })

    return items


def _build_ate_inputs(
    text,
    tokenizer,
    max_length=MAX_LENGTH,
):
    word_items = _regex_tokenize_with_offsets(
        text
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

    # PhoBERT/Roberta: chừa chỗ cho special tokens.
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
                word_ids[source_index]
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
                [input_ids[:max_length]],
                dtype=torch.long,
            ),

        "attention_mask":
            torch.tensor(
                [attention_mask[:max_length]],
                dtype=torch.long,
            ),

        "word_ids":
            expanded_word_ids[
                :max_length
            ],

        "word_items":
            word_items,
    }


def _bio_to_aspects(
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
            "start": start,
            "end": end,
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
            # BIO repair:
            # I-ASP không có B trước -> coi như B.
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
# SERVICE
# ============================================================

class ABSAService:

    def __init__(self):
        self.ready = False

        self.ate_tokenizer = None
        self.ate_model = None

        self.asc_tokenizer = None
        self.asc_model = None

    def load(self):
        if self.ready:
            return

        if not ATE_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Không tìm thấy ATE model: "
                f"{ATE_MODEL_DIR}"
            )

        if not ASC_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Không tìm thấy ASC model: "
                f"{ASC_MODEL_DIR}"
            )

        print(
            "[ABSA] Loading ATE model:",
            ATE_MODEL_DIR,
        )

        self.ate_tokenizer = (
            AutoTokenizer
            .from_pretrained(
                ATE_MODEL_DIR
            )
        )

        self.ate_model = (
            AutoModelForTokenClassification
            .from_pretrained(
                ATE_MODEL_DIR
            )
        )

        self.ate_model.to(
            DEVICE
        )

        self.ate_model.eval()

        print(
            "[ABSA] Loading ASC model:",
            ASC_MODEL_DIR,
        )

        self.asc_tokenizer = (
            AutoTokenizer
            .from_pretrained(
                ASC_MODEL_DIR
            )
        )

        self.asc_model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                ASC_MODEL_DIR
            )
        )

        self.asc_model.to(
            DEVICE
        )

        self.asc_model.eval()

        self.ready = True

        print(
            "[ABSA] Ready on:",
            DEVICE,
        )

    @torch.no_grad()
    def extract_aspects(
        self,
        text,
    ):
        self.load()

        prepared = _build_ate_inputs(
            text,
            self.ate_tokenizer,
            MAX_LENGTH,
        )

        input_ids = (
            prepared["input_ids"]
            .to(DEVICE)
        )

        attention_mask = (
            prepared[
                "attention_mask"
            ]
            .to(DEVICE)
        )

        outputs = self.ate_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pred_ids = (
            outputs.logits
            .argmax(
                dim=-1
            )[0]
            .cpu()
            .tolist()
        )

        word_predictions = {}

        for pred_id, word_id in zip(
            pred_ids,
            prepared["word_ids"],
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
                    int(pred_id)
                ]
            )

            word_predictions[
                word_id
            ] = label

        word_items = prepared[
            "word_items"
        ]

        labels = [
            word_predictions.get(
                i,
                "O",
            )
            for i in range(
                len(word_items)
            )
        ]

        aspects = _bio_to_aspects(
            labels,
            word_items,
        )

        for aspect in aspects:
            aspect["term"] = text[
                aspect["start"]:
                aspect["end"]
            ]

        return aspects

    @torch.no_grad()
    def classify_sentiment(
        self,
        text,
        aspect_term,
    ):
        self.load()

        encoded = (
            self.asc_tokenizer(
                text,
                aspect_term,
                truncation="only_first",
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
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
            )[0]
            .cpu()
            .tolist()
        )

        pred_id = int(
            outputs.logits
            .argmax(
                dim=-1
            )[0]
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

    def analyze_comment(
        self,
        text,
    ):
        text = str(
            text or ""
        ).strip()

        if not text:
            return {
                "aspect_count": 0,
                "aspects": [],
            }

        extracted = (
            self.extract_aspects(
                text
            )
        )

        results = []

        for aspect in extracted:
            sentiment = (
                self.classify_sentiment(
                    text,
                    aspect["term"],
                )
            )

            results.append({
                "term":
                    aspect["term"],

                "start":
                    aspect["start"],

                "end":
                    aspect["end"],

                **sentiment,
            })

        return {
            "aspect_count":
                len(results),

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
                    "aspect_count": 0,
                    "aspects": [],
                    "error": str(exc),
                }

            results.append(
                result
            )

        return results

    @staticmethod
    def summarize(
        absa_results,
    ):
        sentiment_counts = {
            "POSITIVE": 0,
            "NEUTRAL": 0,
            "NEGATIVE": 0,
        }

        aspect_counter = Counter()
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
                    key = term.lower()

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
                            "POSITIVE": 0,
                            "NEUTRAL": 0,
                            "NEGATIVE": 0,
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
            .most_common(10)
        ):
            top_aspects.append({
                "term": term,
                "count": count,
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
# model chỉ load ở lần đầu tiên ABSA thực sự được gọi.
absa_service = ABSAService()
