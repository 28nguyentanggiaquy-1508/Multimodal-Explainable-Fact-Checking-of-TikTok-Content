import os
import json
import torch
import torch.nn as nn

# ===== CACHE HUGGINGFACE =====
os.environ["HF_HOME"] = "C:/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "C:/huggingface_cache"

from transformers import AutoModel, AutoTokenizer
from pyvi import ViTokenizer

from config import (
    MODEL_PATH,
    MAX_SEQ_LEN,
    BATCH_SIZE,
    DICT_PATH
)

# ===== LABELS =====
LABEL_COLS = [
    "ANGER",
    "DISGUST",
    "ENJOYMENT",
    "FEAR",
    "OTHER",
    "SADNESS",
    "SURPRISE"
]

# ===== THRESHOLDS (tuned for multi-label output) =====
BASE_THRESHOLDS = {
    "ANGER": 0.47,
    "DISGUST": 0.09,
    "ENJOYMENT": 0.73,
    "FEAR": 0.93,
    "OTHER": 0.50,
    "SADNESS": 0.67,
    "SURPRISE": 0.81
}


# =========================================================
# MODEL ARCHITECTURE
# =========================================================
class PhoBERT_Large_Arch(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.bert = AutoModel.from_pretrained(
            "vinai/phobert-large"
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),

            nn.Dropout(0.3),

            nn.Linear(512, n_classes)
        )

    def forward(self, ids, mask, token_type_ids):
        output = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )

        pooled_output = output[1]

        return self.classifier(pooled_output)


# =========================================================
# ENGINE
# =========================================================
class PhoBERT_Engine:

    def __init__(
        self,
        model_path=MODEL_PATH,
        dict_path=DICT_PATH,
        device=None
    ):

        self.device = (
            device
            if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"[*] PhoBERT Engine running on: {self.device}")

        # =====================================================
        # LOAD TOKENIZER
        # =====================================================
        self.tokenizer = AutoTokenizer.from_pretrained(
            "vinai/phobert-large"
        )

        print("[+] PhoBERT tokenizer loaded")

        # =====================================================
        # LOAD CUSTOM DICTIONARY
        # =====================================================
        self.custom_dict = {}

        try:
            if os.path.exists(dict_path):

                with open(dict_path, "r", encoding="utf-8") as f:
                    self.custom_dict = json.load(f)

                print(
                    f"[+] Loaded custom dictionary: "
                    f"{len(self.custom_dict)} entries"
                )
            else:
                print("[!] Dictionary file not found")

        except Exception as e:
            print(f"[!] Dictionary load error: {e}")

        # =====================================================
        # MODEL
        # =====================================================
        self.model = PhoBERT_Large_Arch(
            len(LABEL_COLS)
        )

        # =====================================================
        # LOAD WEIGHTS
        # =====================================================
        if os.path.exists(model_path):

            try:
                state_dict = torch.load(
                    model_path,
                    map_location=self.device,
                    weights_only=False
                )

                self.model.load_state_dict(
                    state_dict
                )

                print("[+] PhoBERT weights loaded")

            except Exception as e:
                print(f"[x] Model loading error: {e}")

        else:
            print(f"[!] Model not found: {model_path}")

        self.model.to(self.device)
        self.model.eval()

        print("[+] Engine ready")


    # =========================================================
    # CLEAN COMMENT
    # =========================================================
    def clean_comment(self, text):

        if not text:
            return ""

        text = str(text).strip().lower()

        # remove spam chars
        text = text.replace("\n", " ")
        text = text.replace("\r", " ")

        while "  " in text:
            text = text.replace("  ", " ")

        return text


    # =========================================================
    # APPLY CONTEXT BOOST
    # =========================================================
    def apply_context_boost(
        self,
        probs,
        context_tone
    ):

        if not context_tone:
            return probs

        # SAD CONTEXT
        if context_tone == "sad":

            probs[5] += 0.12  # sadness
            probs[2] -= 0.10  # enjoyment

        # HAPPY CONTEXT
        elif context_tone == "happy":

            probs[2] += 0.12
            probs[5] -= 0.08

        # ANGRY CONTEXT
        elif context_tone == "angry":

            probs[0] += 0.10

        return probs


    # =========================================================
    # DICTIONARY BOOST
    # =========================================================
    def apply_dictionary_boost(
        self,
        text,
        probs
    ):

        if not self.custom_dict:
            return probs

        text_lower = text.lower()

        for keyword, label in self.custom_dict.items():

            if keyword.lower() in text_lower:

                if label.upper() in LABEL_COLS:

                    idx = LABEL_COLS.index(
                        label.upper()
                    )

                    probs[idx] += 0.20

        return probs


    # =========================================================
    # MAIN PREDICT
    # =========================================================
    def predict_batch(
        self,
        comments,
        video_context=None
    ):

        if not comments:
            return []

        # =====================================================
        # VIDEO CONTEXT
        # =====================================================
        context_text = ""
        context_tone = None

        if isinstance(video_context, dict):

            context_text = (
                video_context.get("topic", "")
            )

            context_tone = (
                video_context.get("tone", "")
            )

        elif isinstance(video_context, str):

            context_text = video_context

        context_text = ViTokenizer.tokenize(
            context_text.lower()
        ) if context_text else ""

        all_predictions = []

        # =====================================================
        # BATCH
        # =====================================================
        for i in range(
            0,
            len(comments),
            BATCH_SIZE
        ):

            batch_comments = comments[
                i:i + BATCH_SIZE
            ]

            cleaned_comments = [
                self.clean_comment(c)
                for c in batch_comments
            ]

            tokenized_comments = [
                ViTokenizer.tokenize(c)
                for c in cleaned_comments
            ]

            # =================================================
            # TOKENIZE
            # =================================================
            if context_text:

                inputs = self.tokenizer(
                    [context_text] * len(tokenized_comments),
                    tokenized_comments,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_SEQ_LEN,
                    return_token_type_ids=True
                )

            else:

                inputs = self.tokenizer(
                    tokenized_comments,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_SEQ_LEN,
                    return_token_type_ids=True
                )

            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
            }

            # =================================================
            # MODEL INFERENCE
            # =================================================
            with torch.no_grad():

                outputs = self.model(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    inputs["token_type_ids"]
                )

                probs_batch = torch.sigmoid(
                    outputs
                ).cpu().numpy()

            # =================================================
            # POST PROCESS
            # =================================================
            for idx, probs in enumerate(probs_batch):

                original_comment = cleaned_comments[idx]

                # ---------------------------------------------
                # CONTEXT BOOST
                # ---------------------------------------------
                probs = self.apply_context_boost(
                    probs,
                    context_tone
                )

                # ---------------------------------------------
                # DICTIONARY BOOST
                # ---------------------------------------------
                probs = self.apply_dictionary_boost(
                    original_comment,
                    probs
                )

                # ---------------------------------------------
                # FIND LABELS
                # ---------------------------------------------
                result_labels = []

                for k, label in enumerate(LABEL_COLS):

                    threshold = BASE_THRESHOLDS[label]

                    if probs[k] >= threshold:
                        result_labels.append(label)

                # fallback
                if not result_labels:

                    max_idx = probs.argmax()

                    result_labels = [
                        LABEL_COLS[max_idx]
                    ]

                # OTHER-collapse rule: if OTHER fires alongside any emotion,
                # treat the others as noise and keep OTHER only.
                if "OTHER" in result_labels and len(result_labels) > 1:
                    result_labels = ["OTHER"]

                # MAIN LABEL
                main_label = result_labels[0]

                # =============================================
                # FINAL FIX RULES
                # =============================================

                # Sad context nhưng predict enjoyment
                if (
                    context_tone == "sad"
                    and main_label == "ENJOYMENT"
                    and probs[5] > 0.50
                ):
                    main_label = "SADNESS"

                # Angry context
                if (
                    context_tone == "angry"
                    and probs[0] > 0.55
                ):
                    main_label = "ANGER"

                # =============================================
                # SAVE
                # =============================================
                all_predictions.append({

                    "label": main_label,

                    "confidence": float(
                        probs.max()
                    ),

                    "all_labels": result_labels,

                    "comment": original_comment
                })

        return all_predictions