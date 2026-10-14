import json
import re
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)

BACKEND_DIR = Path(__file__).resolve().parents[2]

ATE_MODEL_DIR = (
    BACKEND_DIR / "model" / "absa" / "ate_phobert_v1"
)

ASC_MODEL_DIR = (
    BACKEND_DIR / "model" / "absa" / "asc_phobert_v1"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WORD_PATTERN = re.compile(
    r"\w+|[^\w\s]",
    flags=re.UNICODE,
)


def regex_tokenize_with_offsets(text):
    result = []

    for match in WORD_PATTERN.finditer(text):
        result.append({
            "word": match.group(0),
            "start": match.start(),
            "end": match.end(),
        })

    return result


def build_ate_inputs(
    text,
    tokenizer,
    max_length=128,
):
    word_items = regex_tokenize_with_offsets(text)

    token_ids = []
    word_ids = []

    for word_index, item in enumerate(word_items):
        pieces = tokenizer.encode(
            item["word"],
            add_special_tokens=False,
        )

        for piece_id in pieces:
            token_ids.append(piece_id)
            word_ids.append(word_index)

    token_ids = token_ids[: max_length - 2]
    word_ids = word_ids[: max_length - 2]

    input_ids = tokenizer.build_inputs_with_special_tokens(
        token_ids
    )

    special_mask = tokenizer.get_special_tokens_mask(
        token_ids,
        already_has_special_tokens=False,
    )

    expanded_word_ids = []
    source_index = 0

    for is_special in special_mask:
        if is_special:
            expanded_word_ids.append(-1)
        else:
            expanded_word_ids.append(
                word_ids[source_index]
            )
            source_index += 1

    attention_mask = [1] * len(input_ids)
    pad_token_id = tokenizer.pad_token_id

    while len(input_ids) < max_length:
        input_ids.append(pad_token_id)
        attention_mask.append(0)
        expanded_word_ids.append(-1)

    return {
        "input_ids": torch.tensor(
            [input_ids[:max_length]],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [attention_mask[:max_length]],
            dtype=torch.long,
        ),
        "word_ids": expanded_word_ids[:max_length],
        "word_items": word_items,
    }


def bio_words_to_aspects(
    labels,
    word_items,
):
    aspects = []

    current_start_word = None
    current_end_word = None

    def close_current():
        nonlocal current_start_word
        nonlocal current_end_word

        if current_start_word is None:
            return

        start = word_items[current_start_word]["start"]
        end = word_items[current_end_word]["end"]

        aspects.append({
            "term": None,
            "start": start,
            "end": end,
        })

        current_start_word = None
        current_end_word = None

    for word_index, label in enumerate(labels):
        if label == "B-ASP":
            close_current()
            current_start_word = word_index
            current_end_word = word_index

        elif label == "I-ASP":
            if current_start_word is None:
                current_start_word = word_index
                current_end_word = word_index
            else:
                current_end_word = word_index

        else:
            close_current()

    close_current()

    return aspects


class ABSAPipelineV1:

    def __init__(
        self,
        ate_model_dir=ATE_MODEL_DIR,
        asc_model_dir=ASC_MODEL_DIR,
        max_length=128,
    ):
        self.max_length = max_length

        print("=" * 70)
        print("Loading ABSA Pipeline V1")
        print("=" * 70)
        print("Device:", DEVICE)
        print("ATE model:", ate_model_dir)
        print("ASC model:", asc_model_dir)

        self.ate_tokenizer = (
            AutoTokenizer.from_pretrained(
                ate_model_dir
            )
        )

        self.ate_model = (
            AutoModelForTokenClassification
            .from_pretrained(
                ate_model_dir
            )
        )

        self.ate_model.to(DEVICE)
        self.ate_model.eval()

        self.asc_tokenizer = (
            AutoTokenizer.from_pretrained(
                asc_model_dir
            )
        )

        self.asc_model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                asc_model_dir
            )
        )

        self.asc_model.to(DEVICE)
        self.asc_model.eval()

        print("[OK] ABSA Pipeline V1 loaded.")

    @torch.no_grad()
    def extract_aspects(self, text):
        prepared = build_ate_inputs(
            text,
            self.ate_tokenizer,
            self.max_length,
        )

        input_ids = prepared["input_ids"].to(DEVICE)
        attention_mask = prepared[
            "attention_mask"
        ].to(DEVICE)

        outputs = self.ate_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        token_predictions = (
            outputs.logits
            .argmax(dim=-1)[0]
            .cpu()
            .tolist()
        )

        word_ids = prepared["word_ids"]
        word_items = prepared["word_items"]

        word_predictions = {}

        for pred_id, word_id in zip(
            token_predictions,
            word_ids,
        ):
            if word_id < 0:
                continue

            if word_id in word_predictions:
                continue

            label = self.ate_model.config.id2label[
                int(pred_id)
            ]

            word_predictions[word_id] = label

        labels = [
            word_predictions.get(
                word_index,
                "O",
            )
            for word_index in range(
                len(word_items)
            )
        ]

        aspects = bio_words_to_aspects(
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
        aspect,
    ):
        encoded = self.asc_tokenizer(
            text,
            aspect,
            truncation="only_first",
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        outputs = self.asc_model(
            **encoded
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
            .argmax(dim=-1)[0]
            .item()
        )

        label = self.asc_model.config.id2label[
            pred_id
        ]

        return {
            "sentiment": label,
            "confidence": float(
                probabilities[pred_id]
            ),
            "probabilities": {
                self.asc_model.config.id2label[i]:
                    float(probabilities[i])
                for i in range(
                    len(probabilities)
                )
            },
        }

    def analyze(self, text):
        text = text.strip()

        if not text:
            return {
                "text": text,
                "aspect_count": 0,
                "aspects": [],
            }

        aspects = self.extract_aspects(text)

        results = []

        for aspect in aspects:
            sentiment_result = (
                self.classify_sentiment(
                    text,
                    aspect["term"],
                )
            )

            results.append({
                "term": aspect["term"],
                "start": aspect["start"],
                "end": aspect["end"],
                **sentiment_result,
            })

        return {
            "text": text,
            "aspect_count": len(results),
            "aspects": results,
        }


def main():
    pipeline = ABSAPipelineV1()

    print()
    print("Nhập comment để test.")
    print("Enter rỗng để thoát.")

    while True:
        text = input(
            "\nComment: "
        ).strip()

        if not text:
            break

        result = pipeline.analyze(text)

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
