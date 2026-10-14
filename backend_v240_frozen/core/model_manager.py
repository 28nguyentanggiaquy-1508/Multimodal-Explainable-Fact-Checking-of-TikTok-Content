import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from config import BATCH_SIZE, MAX_SEQ_LEN


LABEL_COLS = [
    "ANGER",
    "DISGUST",
    "ENJOYMENT",
    "FEAR",
    "OTHER",
    "SADNESS",
    "SURPRISE",
]

SEQUENCE_LABELS = [
    "ENJOYMENT",
    "SADNESS",
    "ANGER",
    "SURPRISE",
    "FEAR",
    "DISGUST",
]

# Per-label thresholds tuned for multi-label emotion output.
# Labels are matched case-insensitively. Labels absent from this dict
# (e.g. OTHER) use DEFAULT_LABEL_THRESHOLD.
LABEL_THRESHOLDS = {
    "ENJOYMENT": 0.73,
    "SADNESS": 0.67,
    "ANGER": 0.47,
    "SURPRISE": 0.81,
    "FEAR": 0.93,
    "DISGUST": 0.09,
}
DEFAULT_LABEL_THRESHOLD = 0.5


def _threshold_for(label: str) -> float:
    return LABEL_THRESHOLDS.get(str(label).upper(), DEFAULT_LABEL_THRESHOLD)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = Path(__file__).resolve().parent
INFERENCE_DIR = PROJECT_ROOT / "model" / "inference_gr"
SEQUENCE_DIR = PROJECT_ROOT / "model" / "sequence_gr"


def _model_id(group: str, path: Path) -> str:
    return f"{group}:{path.stem}"


def _apply_other_collapse(labels):
    """
    Multi-label rule: when OTHER co-occurs with any emotion label, collapse to OTHER only.
    OTHER means "no specific emotion / off-topic", so any emotion firing alongside it
    is treated as noise.
    """
    upper = [str(l).upper() for l in labels]
    if "OTHER" in upper and len(upper) > 1:
        return ["OTHER"]
    return upper


def _context_to_text(video_context) -> str:
    if isinstance(video_context, dict):
        parts = [
            str(video_context.get("topic", "")).strip(),
            str(video_context.get("tone", "")).strip(),
            str(video_context.get("visual_summary", "")).strip(),
            str(video_context.get("audience_context", "")).strip(),
        ]
        return " ".join(part for part in parts if part)

    if isinstance(video_context, str):
        return video_context

    return ""


def list_model_options() -> List[Dict[str, str]]:
    options: List[Dict[str, str]] = []

    for path in sorted(INFERENCE_DIR.glob("*.pt")):
        is_full_checkpoint = "full" in path.stem.lower()
        options.append({
            "id": _model_id("inference", path),
            "name": path.stem,
            "group": "inference",
            "filename": path.name,
            "available": not is_full_checkpoint,
            "note": (
                "Full pickle checkpoint is hidden from execution; use a state checkpoint."
                if is_full_checkpoint else ""
            ),
        })

    for path in sorted(SEQUENCE_DIR.glob("*.pt")):
        options.append({
            "id": _model_id("sequence", path),
            "name": path.stem,
            "group": "sequence",
            "filename": path.name,
            "available": True,
            "note": "",
        })

    return options


def get_default_model_id() -> Optional[str]:
    options = list_model_options()
    for option in options:
        if option.get("available", True):
            return option["id"]
    return options[0]["id"] if options else None


class SoftLogicInferenceAdapter:
    def __init__(self, checkpoint_path: Path, device: Optional[str] = None):
        if str(CORE_ROOT) not in sys.path:
            sys.path.insert(0, str(CORE_ROOT))

        from softlogic_vibert.inference import SentimentPredictor

        self.predictor = SentimentPredictor.load(
            str(checkpoint_path),
            device=device,
            weights_only=True,
        )

    def predict_batch(self, comments, video_context=None):
        context = _context_to_text(video_context)
        predictions = []

        for comment in comments:
            result = self.predictor.predict(
                comment=comment,
                context=context or None,
                return_probs=True,
            )

            probabilities = {
                str(label).upper(): float(prob)
                for label, prob in result.get("probabilities", {}).items()
            }

            if probabilities:
                # Recompute multi-label set using per-label thresholds
                labels = [
                    name for name, prob in probabilities.items()
                    if prob >= _threshold_for(name)
                ]
                best_label = max(probabilities, key=probabilities.get)
                if not labels:
                    labels = [best_label]

                # Order labels by probability (highest first) so main_label
                # is the most confident one, then apply OTHER-collapse.
                labels = sorted(labels, key=lambda n: probabilities.get(n, 0.0), reverse=True)
                labels = _apply_other_collapse(labels)

                main_label = labels[0]
                confidence = probabilities.get(main_label, probabilities[best_label])
            else:
                main_label = "OTHER"
                labels = ["OTHER"]
                confidence = 0.0

            predictions.append({
                "label": main_label,
                "confidence": float(confidence),
                "all_labels": labels,
                "probabilities": probabilities,
                "comment": comment,
            })

        return predictions


class MultiVectorAttention(nn.Module):
    def __init__(self, num_labels: int, hidden_size: int, num_heads: int):
        super().__init__()
        self.label_queries = nn.Parameter(torch.randn(1, num_labels, hidden_size))
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, hidden_states):
        batch_size = hidden_states.size(0)
        query = self.label_queries.expand(batch_size, -1, -1)
        output, _ = self.attention(query=query, key=hidden_states, value=hidden_states)
        return output


class PhoBERTMVAKAN1D(nn.Module):
    def __init__(
        self,
        num_labels: int,
        num_heads: int = 8,
        dropout: float = 0.3,
        grid_size: int = 5,
    ):
        super().__init__()
        if str(CORE_ROOT) not in sys.path:
            sys.path.insert(0, str(CORE_ROOT))

        from kan_Arch_1D import KAN, KAN_Convolutional_Layer_1D

        self.phobert = AutoModel.from_pretrained(
            "vinai/phobert-base-v2",
            add_pooling_layer=False,
        )
        hidden_size = self.phobert.config.hidden_size

        self.multi_vector = MultiVectorAttention(num_labels, hidden_size, num_heads)
        self.mva_proj = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
        )

        self.cnn_bottleneck = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=16,
            kernel_size=1,
        )
        self.kan_conv = nn.Sequential(
            KAN_Convolutional_Layer_1D(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding=1,
                grid_size=grid_size,
            ),
            nn.AdaptiveMaxPool1d(1),
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = KAN(
            layers_hidden=[(num_labels * 64) + 16, 64, num_labels],
            grid_size=grid_size,
            spline_order=3,
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        multi_vector = self.multi_vector(hidden)
        multi_vector = self.mva_proj(multi_vector)
        multi_vector = multi_vector.reshape(multi_vector.size(0), -1)

        cnn_in = hidden.transpose(1, 2)
        cnn_vec = self.cnn_bottleneck(cnn_in)
        cnn_vec = self.kan_conv(cnn_vec).squeeze(-1)

        fused = self.dropout(torch.cat([multi_vector, cnn_vec], dim=1))
        return self.classifier(fused, update_grid=False)


class SequenceAdapter:
    def __init__(self, checkpoint_path: Path, device: Optional[str] = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.model = PhoBERTMVAKAN1D(num_labels=len(SEQUENCE_LABELS)).to(self.device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.thresholds = checkpoint.get("thresholds")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def _build_input(self, comment: str, video_context) -> str:
        context = _context_to_text(video_context)
        return f"Context:{context} [SEP]Comment:{comment}"

    def predict_batch(self, comments, video_context=None):
        predictions = []

        for start in range(0, len(comments), BATCH_SIZE):
            batch_comments = comments[start:start + BATCH_SIZE]
            texts = [
                self._build_input(comment, video_context)
                for comment in batch_comments
            ]

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=MAX_SEQ_LEN,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                logits = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                probs = torch.sigmoid(logits).cpu()

            thresholds = [_threshold_for(name) for name in SEQUENCE_LABELS]

            for comment, row in zip(batch_comments, probs):
                row_values = row.tolist()
                labels = [
                    SEQUENCE_LABELS[index]
                    for index, prob in enumerate(row_values)
                    if prob >= thresholds[index]
                ]

                best_index = int(row.argmax().item())
                if not labels:
                    labels = [SEQUENCE_LABELS[best_index]]

                # Order labels by probability (highest first)
                labels = sorted(
                    labels,
                    key=lambda name: row_values[SEQUENCE_LABELS.index(name)],
                    reverse=True,
                )
                labels = _apply_other_collapse(labels)

                probabilities = {
                    SEQUENCE_LABELS[i]: float(row_values[i])
                    for i in range(len(SEQUENCE_LABELS))
                }

                predictions.append({
                    "label": labels[0],
                    "confidence": float(row_values[best_index]),
                    "all_labels": labels,
                    "probabilities": probabilities,
                    "comment": comment,
                })

        return predictions


class ModelManager:
    def __init__(self):
        self._cache = {}

    def get(self, model_id: Optional[str]):
        if not model_id:
            model_id = get_default_model_id()

        if not model_id:
            raise ValueError("No model checkpoints found")

        if model_id in self._cache:
            return self._cache[model_id]

        selected = next(
            (option for option in list_model_options() if option["id"] == model_id),
            None,
        )
        if selected and not selected.get("available", True):
            raise ValueError(selected.get("note") or f"Model is not available: {model_id}")

        group, name = model_id.split(":", 1)
        if group == "inference":
            path = INFERENCE_DIR / f"{name}.pt"
            adapter = SoftLogicInferenceAdapter(path)
        elif group == "sequence":
            path = SEQUENCE_DIR / f"{name}.pt"
            adapter = SequenceAdapter(path)
        else:
            raise ValueError(f"Unsupported model group: {group}")

        self._cache[model_id] = adapter
        return adapter


model_manager = ModelManager()
