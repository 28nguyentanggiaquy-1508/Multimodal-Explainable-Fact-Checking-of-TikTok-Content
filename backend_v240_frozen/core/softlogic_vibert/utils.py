"""
Utility functions for SoftLogic ViBERT.

Includes:
- Pragmatic feature extraction for GenZ text
- Label encoding utilities
- Reproducibility helpers
- Visualization utilities
"""

import json
import random
import re
from typing import Dict, Iterable, List, Optional, Tuple, Any
import unicodedata

import numpy as np
import torch


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# LABEL UTILITIES
# ============================================================================

def ensure_list(x: Any) -> List[str]:
    """
    Convert various label formats to a standardized list of strings.
    
    Handles:
    - None -> []
    - List/tuple -> flatten and normalize
    - String -> split by comma if needed
    - JSON string -> parse and normalize
    """
    if x is None:
        return []
    
    if isinstance(x, list):
        vals = x
    elif isinstance(x, tuple):
        vals = list(x)
    elif isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = json.loads(s)
            vals = v if isinstance(v, list) else [s]
        except Exception:
            vals = [s]
    else:
        vals = [x]
    
    out = []
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        # Handle comma-separated labels
        parts = [p.strip() for p in s.split(",")] if "," in s else [s]
        for p in parts:
            if p:
                out.append(p.lower())
    
    return out


def build_label_map(labels_iter: Iterable[List[str]]) -> Tuple[Dict[str, int], List[str]]:
    """
    Build label-to-index mapping from an iterable of label lists.
    
    Returns:
        label_map: Dict mapping label string to index
        label_list: List of labels in sorted order
    """
    label_set = set()
    for labels in labels_iter:
        for lab in labels:
            label_set.add(lab.lower())
    
    label_list = sorted(label_set)
    label_map = {lab: i for i, lab in enumerate(label_list)}
    
    return label_map, label_list


def labels_to_multi_hot(labels: List[str], label_map: Dict[str, int]) -> List[int]:
    """Convert label list to multi-hot vector."""
    vec = [0] * len(label_map)
    for lab in labels:
        idx = label_map.get(lab.lower())
        if idx is not None:
            vec[idx] = 1
    return vec


# ============================================================================
# PRAGMATIC FEATURE EXTRACTION
# ============================================================================

# Comprehensive emoji regex pattern
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F680-\U0001F6FF"  # Transport and Map
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U0001F1E0-\U0001F1FF"  # Flags
    "]",
    flags=re.UNICODE,
)

# GenZ slang intensifiers
_GENZ_INTENSIFIERS = [
    "quá", "lắm", "cực", "siêu", "vãi", "vcl", "vl", "đỉnh", "peak",
    "real", "slay", "iconic", "iu", "yupp", "yess", "nooo", "bruh",
    "ớt", "chết", "xiềng", "xỉu", "khóc", "hú", "mlem", "hehe", "hihi",
]

# Punctuation patterns
_EXCLAIM_PATTERN = re.compile(r"!+")
_QUESTION_PATTERN = re.compile(r"\?+")
_ELLIPSIS_PATTERN = re.compile(r"\.{2,}|…+")


def count_repetitions(text: str, min_repeat: int = 3) -> int:
    """Count character repetitions like 'niceeeee' or 'hahahaha'."""
    if len(text) < min_repeat:
        return 0
    
    count = 0
    i = 0
    while i < len(text):
        j = i + 1
        while j < len(text) and text[j] == text[i]:
            j += 1
        if j - i >= min_repeat:
            count += 1
        i = j
    
    return count


def count_word_repetitions(text: str) -> int:
    """Count repeated word patterns like 'ha ha ha' or 'đi đi'."""
    words = text.lower().split()
    if len(words) < 2:
        return 0
    
    count = 0
    i = 0
    while i < len(words) - 1:
        if words[i] == words[i + 1]:
            repeat_len = 1
            j = i + 1
            while j < len(words) and words[j] == words[i]:
                repeat_len += 1
                j += 1
            if repeat_len >= 2:
                count += 1
            i = j
        else:
            i += 1
    
    return count


def extract_prag_features(text: str) -> List[float]:
    """
    Extract pragmatic/affective features from GenZ text.
    
    Features:
    1. emoji_ratio: Emoji count / text length
    2. punct_ratio: Exclamation/question mark count / text length
    3. upper_ratio: Uppercase character ratio
    4. char_repeat_ratio: Character repetition ratio
    5. length_norm: Normalized text length (log scale)
    6. emoji_count: Raw emoji count
    7. genz_intensifier_count: Count of GenZ slang intensifiers
    8. word_repeat_count: Count of word repetitions
    """
    if text is None:
        text = ""
    
    text = str(text)
    length = max(len(text), 1)
    
    # Emoji features
    emoji_matches = _EMOJI_PATTERN.findall(text)
    emoji_count = len(emoji_matches)
    emoji_ratio = emoji_count / length
    
    # Punctuation intensity
    exclaim_count = sum(len(m.group()) for m in _EXCLAIM_PATTERN.finditer(text))
    question_count = sum(len(m.group()) for m in _QUESTION_PATTERN.finditer(text))
    ellipsis_count = len(_ELLIPSIS_PATTERN.findall(text))
    punct_count = exclaim_count + question_count + ellipsis_count
    punct_ratio = punct_count / length
    
    # Capitalization
    upper_count = sum(1 for ch in text if ch.isupper())
    upper_ratio = upper_count / length
    
    # Character repetition
    char_repeat = count_repetitions(text)
    char_repeat_ratio = char_repeat / (length / 3 + 1)  # Normalize
    
    # Word repetition
    word_repeat = count_word_repetitions(text)
    
    # GenZ intensifiers
    text_lower = text.lower()
    intensifier_count = sum(1 for word in _GENZ_INTENSIFIERS if word in text_lower)
    
    # Length (log normalized)
    length_norm = np.log1p(length) / 10.0
    
    return [
        float(emoji_ratio),
        float(punct_ratio),
        float(upper_ratio),
        float(char_repeat_ratio),
        float(length_norm),
        float(emoji_count),
        float(intensifier_count),
        float(word_repeat),
    ]


# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def normalize_vietnamese_text(text: str) -> str:
    """Normalize Vietnamese text with proper Unicode handling."""
    if not text:
        return ""
    
    # Normalize Unicode
    text = unicodedata.normalize("NFC", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


# ============================================================================
# TF-IDF UTILITIES
# ============================================================================

def _row_to_text(row: Dict, normalize_text: bool = True) -> str:
    comment = str(row.get("comment", row.get("text", "")))
    context = row.get("context", None)

    if normalize_text:
        comment = normalize_vietnamese_text(comment)
        if context:
            context = normalize_vietnamese_text(str(context))
    elif context:
        context = str(context)

    if context:
        return f"{comment} {context}"
    return comment


def build_tfidf_cache(
    train_rows: List[Dict],
    tokenizer,
    cache_path: str,
    default_score: float = 0.1,
    normalize_text: bool = True,
    max_features: Optional[int] = None,
) -> torch.Tensor:
    """Build TF-IDF cache mapping input_id -> tfidf score."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for TF-IDF. Install with: pip install scikit-learn"
        ) from exc

    texts = [_row_to_text(row, normalize_text=normalize_text) for row in train_rows]
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(texts)

    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    scores = torch.full((vocab_size,), float(default_score), dtype=torch.float32)

    matched_ids: List[int] = []
    for term, idx in vectorizer.vocabulary_.items():
        token_id = vocab.get(term)
        if token_id is None:
            tokens = tokenizer.tokenize(term)
            if len(tokens) == 1:
                token_id = vocab.get(tokens[0])
        if token_id is not None:
            scores[token_id] = float(vectorizer.idf_[idx])
            matched_ids.append(token_id)

    if matched_ids:
        matched_tensor = scores[torch.tensor(matched_ids, dtype=torch.long)]
        min_val = matched_tensor.min()
        max_val = matched_tensor.max()
        if max_val > min_val:
            norm_vals = (matched_tensor - min_val) / (max_val - min_val)
            scores[torch.tensor(matched_ids, dtype=torch.long)] = norm_vals

    for special_id in [
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
    ]:
        if special_id is not None:
            scores[int(special_id)] = float(default_score)

    from pathlib import Path
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"scores": scores, "default_score": float(default_score)}, cache_path)

    return scores


def clean_genz_text(text: str, preserve_emojis: bool = True) -> str:
    """
    Clean GenZ text while preserving important signals.
    
    Args:
        text: Input text
        preserve_emojis: Whether to keep emojis
    """
    if not text:
        return ""
    
    text = normalize_vietnamese_text(text)
    
    if not preserve_emojis:
        text = _EMOJI_PATTERN.sub("", text)
    
    # Normalize repeated punctuation (keep some intensity)
    text = re.sub(r"!{4,}", "!!!", text)
    text = re.sub(r"\?{4,}", "???", text)
    
    return text.strip()


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def visualize_token_masks(
    tokenizer,
    input_ids: torch.Tensor,
    mask_vals: torch.Tensor,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Visualize token-level mask values.
    
    Args:
        tokenizer: Tokenizer instance
        input_ids: Token IDs [L]
        mask_vals: Mask values [L, 1]
    
    Returns:
        List of dicts with token, mask value, and importance
    """
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    mask_vals = mask_vals.squeeze(-1).tolist() if mask_vals.dim() > 1 else mask_vals.tolist()
    
    result = []
    for token, mask_val in zip(tokens, mask_vals):
        if token in ["[PAD]", "<pad>"]:
            continue
        # ============================================================================
        # TF-IDF UTILITIES
        # ============================================================================

        
        result.append({
            "token": token,
            "mask_value": mask_val,
            "importance": "high" if mask_val >= threshold else "low",
        })
    
    return result


def format_rule_activations(rules: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Format rule activations as plain floats for display."""
    return {
        k: v.item() if isinstance(v, torch.Tensor) else float(v)
        for k, v in rules.items()
    }


def interpret_rules(rules: Dict[str, float], threshold: float = 0.5) -> List[str]:
    """
    Generate human-readable interpretation of rule activations.
    
    Args:
        rules: Dict of rule name to activation value
        threshold: Activation threshold for significance
    
    Returns:
        List of interpretation strings
    """
    interpretations = []
    
    rule_meanings = {
        "r1": ("Sarcasm/Contradiction", "Surface positive + underlying negative detected"),
        "r2": ("Strong Negative", "Both surface and semantic signals are negative"),
        "r3": ("Mild Positive", "Genuine positive without strong intensity"),
        "r4": ("Intense Negative", "High emotional intensity with negative semantics"),
        "r5": ("Inconsistency/Surprise", "Mismatch between lexical and semantic polarity"),
    }
    
    for rule_key, (name, desc) in rule_meanings.items():
        value = rules.get(rule_key, 0.0)
        if value >= threshold:
            interpretations.append(f"[{name}] (score: {value:.2f}): {desc}")
    
    return interpretations if interpretations else ["No significant reasoning patterns detected"]


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_summary(model: torch.nn.Module) -> Dict[str, int]:
    """Get summary of model parameters by component."""
    summary = {}
    
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        summary[name] = {
            "total": params,
            "trainable": trainable,
        }
    
    summary["total"] = {
        "total": count_parameters(model, trainable_only=False),
        "trainable": count_parameters(model, trainable_only=True),
    }
    
    return summary


# ============================================================================
# DATA UTILITIES
# ============================================================================

def compute_class_weights(
    labels_list: List[List[str]],
    label_map: Dict[str, int],
    smoothing: float = 1.0,
) -> torch.Tensor:
    """
    Compute class weights for imbalanced multi-label data.
    
    Uses inverse frequency weighting with smoothing.
    """
    num_labels = len(label_map)
    counts = np.zeros(num_labels)
    
    for labels in labels_list:
        for lab in labels:
            idx = label_map.get(lab.lower())
            if idx is not None:
                counts[idx] += 1
    
    total = len(labels_list)
    weights = (total + smoothing) / (counts + smoothing)
    
    # Normalize
    weights = weights / weights.sum() * num_labels
    
    return torch.tensor(weights, dtype=torch.float32)


def stratified_split_multilabel(
    data: List[Dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stratified split for multi-label data.
    
    Uses label powerset approximation for stratification.
    """
    random.seed(seed)
    
    # Group by label combination
    label_groups = {}
    for item in data:
        labels = tuple(sorted(ensure_list(item.get("labels", []))))
        if labels not in label_groups:
            label_groups[labels] = []
        label_groups[labels].append(item)
    
    train_data = []
    val_data = []
    
    for labels, items in label_groups.items():
        random.shuffle(items)
        val_size = max(1, int(len(items) * val_ratio))
        val_data.extend(items[:val_size])
        train_data.extend(items[val_size:])
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    return train_data, val_data
