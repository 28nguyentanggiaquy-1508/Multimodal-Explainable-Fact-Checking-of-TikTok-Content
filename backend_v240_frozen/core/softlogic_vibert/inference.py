"""
Inference script for SoftLogic ViBERT.

Features:
- Single sample and batch inference
- Token mask visualization
- Rule activation interpretation
- Predicate analysis
- Comprehensive output formatting
"""

import argparse
import json
from typing import Dict, List, Optional, Tuple, Union, cast
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .config import ModelConfig
from .model import SoftLogicViBERT
from .utils import (
    extract_prag_features,
    visualize_token_masks,
    format_rule_activations,
    interpret_rules,
    normalize_vietnamese_text,
)


# ============================================================================
# SIMPLE PREDICTOR CLASS
# ============================================================================

class SentimentPredictor:
    """
    Simple wrapper for easy inference.
    
    Usage:
        model = SentimentPredictor.load(path, device)
        pred = model.predict(comment=comment, context=context)
        print("Nhãn là:", pred)
    """
    
    def __init__(
        self,
        model: SoftLogicViBERT,
        tokenizer,
        label_list: List[str],
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.device = device
        
        self.special_token_ids = torch.tensor([
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
        ], device=device)
    
    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        device: Optional[str] = None,
        weights_only: bool = True,
    ) -> "SentimentPredictor":
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to .pt checkpoint file
            device: "cuda" or "cpu" (auto-detect if None)
            weights_only: Only load state dict (safer, default True)
        
        Returns:
            SentimentPredictor instance
        
        Example:
            model = SentimentPredictor.load("outputs/softlogic_vibert_state.pt")
        """
        device_str = device
        if device_str is None:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device_t = torch.device(device_str)
        
        # PyTorch 2.6+ defaults to weights_only=True, which can fail on older
        # checkpoints that include pickled numpy scalars/objects. We try the
        # requested mode first, then fall back to full load if needed.
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=weights_only)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Reconstruct model config
        model_config = ModelConfig(**checkpoint["model_config"])
        
        # Load model
        model = SoftLogicViBERT(model_config)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model = model.to(device_t)
        model.eval()
        
        # Load tokenizer
        tokenizer_name = checkpoint.get("tokenizer_name", model_config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Get label list
        label_list = checkpoint.get("label_list", [])
        
        return cls(model, tokenizer, label_list, device_t)
    
    def predict(
        self,
        comment: str,
        context: Optional[str] = None,
        threshold: float = 0.5,
        return_probs: bool = False,
    ) -> Union[List[str], Dict]:
        """
        Predict sentiment labels for a comment.
        
        Args:
            comment: The comment text to analyze
            context: Optional context text
            threshold: Classification threshold (default 0.5)
            return_probs: If True, return dict with labels and probabilities
        
        Returns:
            List of predicted label names, or dict with details if return_probs=True
        
        Example:
            labels = model.predict("ngon thật đấy 🙄")
            # => ['Disgust']
            
            result = model.predict("ngon thật đấy 🙄", return_probs=True)
            # => {'labels': ['Disgust'], 'probabilities': {...}}
        """
        # Normalize text
        comment = normalize_vietnamese_text(comment)
        if context:
            context = normalize_vietnamese_text(context)
        
        # Tokenize
        tok = self.tokenizer(
            comment,
            text_pair=context,
            max_length=self.model.config.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Extract pragmatic features
        prag = torch.tensor([extract_prag_features(comment)], dtype=torch.float)
        
        # Forward pass
        with torch.no_grad():
            input_ids = tok["input_ids"].to(self.device)
            attention_mask = tok["attention_mask"].to(self.device)
            token_type_ids = tok.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            
            logits, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                prag_features=prag.to(self.device),
                special_token_ids=self.special_token_ids,
                return_extras=True,
            )
        
        # Process predictions
        probs = torch.sigmoid(logits).squeeze(0).cpu()
        preds = (probs >= threshold).int()
        
        # Get predicted labels
        predicted_labels = [
            self.label_list[i] for i, v in enumerate(preds.tolist()) if v == 1
        ]
        
        if not return_probs:
            return predicted_labels
        
        # Return detailed result
        label_probs = {
            self.label_list[i]: prob
            for i, prob in enumerate(probs.tolist())
        }
        
        return {
            "labels": predicted_labels,
            "probabilities": label_probs,
        }
    
    def predict_batch(
        self,
        comments: List[str],
        contexts: Optional[List[Optional[str]]] = None,
        threshold: float = 0.5,
    ) -> List[List[str]]:
        """
        Predict labels for multiple comments.
        
        Args:
            comments: List of comment texts
            contexts: Optional list of context texts
            threshold: Classification threshold
        
        Returns:
            List of predicted label lists
        """
        contexts_list: List[Optional[str]]
        if contexts is None:
            contexts_list = [cast(Optional[str], None) for _ in comments]
        else:
            contexts_list = contexts
        
        results = []
        for comment, context in zip(comments, contexts_list):
            labels = self.predict(comment, context, threshold)
            results.append(labels)
        
        return results
    
    def analyze(
        self,
        comment: str,
        context: Optional[str] = None,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Get detailed analysis including rules and tokens.
        
        Args:
            comment: Comment text
            context: Optional context text
            threshold: Classification threshold
        
        Returns:
            Dict with labels, probabilities, rules, and token masks
        """
        # Normalize text
        comment = normalize_vietnamese_text(comment)
        if context:
            context = normalize_vietnamese_text(context)
        
        # Tokenize
        tok = self.tokenizer(
            comment,
            text_pair=context,
            max_length=self.model.config.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Extract pragmatic features
        prag = torch.tensor([extract_prag_features(comment)], dtype=torch.float)
        
        # Forward pass
        with torch.no_grad():
            input_ids = tok["input_ids"].to(self.device)
            attention_mask = tok["attention_mask"].to(self.device)
            token_type_ids = tok.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            
            logits, extras = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                prag_features=prag.to(self.device),
                special_token_ids=self.special_token_ids,
                return_extras=True,
            )
        
        # Process predictions
        probs = torch.sigmoid(logits).squeeze(0).cpu()
        preds = (probs >= threshold).int()
        
        predicted_labels = [
            self.label_list[i] for i, v in enumerate(preds.tolist()) if v == 1
        ]
        
        label_probs = {
            self.label_list[i]: round(prob, 4)
            for i, prob in enumerate(probs.tolist())
        }
        
        # Get rules
        rules = {}
        if "rules" in extras:
            for k, v in extras["rules"].items():
                rules[k] = round(v.mean().item(), 4)
        
        return {
            "comment": comment,
            "context": context,
            "labels": predicted_labels,
            "probabilities": label_probs,
            "rules": rules,
            "mask_mean": round(extras.get("mask_mean", torch.tensor(0)).item(), 4),
        }



# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    weights_only: bool = True,
) -> Tuple[SoftLogicViBERT, AutoTokenizer, Dict]:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Loaded model
        tokenizer: Associated tokenizer
        checkpoint: Full checkpoint dict
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=weights_only)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Reconstruct model config
    model_config = ModelConfig(**checkpoint["model_config"])
    
    # Load model
    if "model" in checkpoint and not weights_only:
        # Full model saved
        model = checkpoint["model"]
        model.config = model_config
    else:
        # State dict saved
        model = SoftLogicViBERT(model_config)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer_name = checkpoint.get("tokenizer_name", model_config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    return model, tokenizer, checkpoint


# ============================================================================
# INFERENCE
# ============================================================================

def predict_single(
    model: SoftLogicViBERT,
    tokenizer,
    comment: str,
    context: Optional[str] = None,
    label_list: Optional[List[str]] = None,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
    return_details: bool = True,
) -> Dict:
    """
    Make prediction for a single sample.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        comment: Comment text
        context: Optional context text
        label_list: List of label names
        threshold: Classification threshold
        device: Device
        return_details: Whether to return detailed interpretability info
    
    Returns:
        Dictionary with predictions and optional interpretability details
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Normalize text
    comment = normalize_vietnamese_text(comment)
    if context:
        context = normalize_vietnamese_text(context)
    
    # Tokenize
    tok = tokenizer(
        comment,
        text_pair=context,
        max_length=model.config.max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    # Extract pragmatic features
    prag = torch.tensor([extract_prag_features(comment)], dtype=torch.float)
    
    # Special tokens
    special_token_ids = torch.tensor([
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
    ], device=device)
    
    # Forward pass
    with torch.no_grad():
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)
        token_type_ids = tok.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        
        logits, extras = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            prag_features=prag.to(device),
            special_token_ids=special_token_ids,
            return_extras=True,
        )
    
    # Process predictions
    probs = torch.sigmoid(logits).squeeze(0)
    preds = (probs >= threshold).int()
    
    # Build result
    result = {
        "comment": comment,
        "context": context,
        "probabilities": probs.cpu().tolist(),
        "predictions": preds.cpu().tolist(),
        "threshold": threshold,
    }
    
    # Add predicted labels
    if label_list:
        result["predicted_labels"] = [
            label_list[i] for i, v in enumerate(preds.cpu().tolist()) if v == 1
        ]
        result["label_probabilities"] = {
            label_list[i]: prob
            for i, prob in enumerate(probs.cpu().tolist())
        }
    
    if not return_details:
        return result
    
    # Add interpretability details
    
    # Mask visualization
    if "mask_vals" in extras and extras["mask_vals"] is not None:
        mask_vals = extras["mask_vals"].squeeze(0)
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().tolist())
        attention = attention_mask.squeeze(0).cpu()
        
        token_masks = []
        for i, (token, mask_val, attn) in enumerate(zip(tokens, mask_vals.cpu(), attention)):
            if attn.item() > 0 and token not in ["[PAD]", "<pad>"]:
                token_masks.append({
                    "token": token,
                    "mask_value": mask_val.item(),
                    "importance": "high" if mask_val.item() >= 0.5 else "low",
                })
        
        result["token_masks"] = token_masks
        result["mask_mean"] = extras.get("mask_mean", torch.tensor(0.0)).item()
    
    # Rule activations
    if "rules" in extras and extras["rules"]:
        rules = format_rule_activations(extras["rules"])
        result["rule_activations"] = rules
        result["rule_interpretations"] = interpret_rules(rules, threshold=0.5)
    
    # Predicate values
    if "predicates" in extras and extras["predicates"]:
        result["predicates"] = format_rule_activations(extras["predicates"])
    
    return result


def predict_batch(
    model: SoftLogicViBERT,
    tokenizer,
    comments: List[str],
    contexts: Optional[List[Optional[str]]] = None,
    label_list: Optional[List[str]] = None,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> List[Dict]:
    """
    Make predictions for a batch of samples.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        comments: List of comment texts
        contexts: Optional list of context texts
        label_list: List of label names
        threshold: Classification threshold
        device: Device
        batch_size: Batch size for inference
    
    Returns:
        List of prediction dictionaries
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    contexts_list: List[Optional[str]]
    if contexts is None:
        contexts_list = [cast(Optional[str], None) for _ in comments]
    else:
        contexts_list = contexts
    
    all_results = []
    
    for start_idx in range(0, len(comments), batch_size):
        end_idx = min(start_idx + batch_size, len(comments))
        batch_comments = comments[start_idx:end_idx]
        batch_contexts = contexts_list[start_idx:end_idx]
        
        # Normalize texts
        batch_comments = [normalize_vietnamese_text(c) for c in batch_comments]
        batch_contexts = [
            normalize_vietnamese_text(c) if c else None
            for c in batch_contexts
        ]
        
        # Tokenize batch
        encodings = []
        for comment, context in zip(batch_comments, batch_contexts):
            tok = tokenizer(
                comment,
                text_pair=context,
                max_length=model.config.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            encodings.append(tok)
        
        # Stack tensors
        input_ids = torch.cat([e["input_ids"] for e in encodings], dim=0).to(device)
        attention_mask = torch.cat([e["attention_mask"] for e in encodings], dim=0).to(device)
        
        if "token_type_ids" in encodings[0]:
            token_type_ids = torch.cat(
                [e["token_type_ids"] for e in encodings], dim=0
            ).to(device)
        else:
            token_type_ids = None
        
        # Pragmatic features
        prag = torch.tensor(
            [extract_prag_features(c) for c in batch_comments],
            dtype=torch.float,
        ).to(device)
        
        # Special tokens
        special_token_ids = torch.tensor([
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
        ], device=device)
        
        # Forward pass
        with torch.no_grad():
            logits, extras = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                prag_features=prag,
                special_token_ids=special_token_ids,
                return_extras=True,
            )
        
        # Process predictions
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).int()
        
        # Build results for this batch
        for i in range(len(batch_comments)):
            result = {
                "comment": batch_comments[i],
                "context": batch_contexts[i],
                "probabilities": probs[i].cpu().tolist(),
                "predictions": preds[i].cpu().tolist(),
            }
            
            if label_list:
                result["predicted_labels"] = [
                    label_list[j] for j, v in enumerate(preds[i].cpu().tolist()) if v == 1
                ]
            
            all_results.append(result)
    
    return all_results


# ============================================================================
# ANALYSIS UTILITIES
# ============================================================================

def analyze_prediction(
    result: Dict,
    verbose: bool = True,
) -> None:
    """
    Print detailed analysis of a prediction.
    
    Args:
        result: Prediction result dictionary
        verbose: Whether to print detailed output
    """
    print("\n" + "=" * 60)
    print("PREDICTION ANALYSIS")
    print("=" * 60)
    
    print("\nInput:")
    print(f"   Comment: {result['comment']}")
    if result.get("context"):
        print(f"   Context: {result['context']}")
    
    print("\nPredictions:")
    if "predicted_labels" in result:
        if result["predicted_labels"]:
            for label in result["predicted_labels"]:
                prob = result.get("label_probabilities", {}).get(label, 0)
                print(f"   - {label}: {prob:.2%}")
        else:
            print("   (No labels predicted above threshold)")
    
    print("\nAll Probabilities:")
    if "label_probabilities" in result:
        for label, prob in sorted(
            result["label_probabilities"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            print(f"   {label:15s} [{bar}] {prob:.2%}")
    
    if verbose:
        # Rule interpretations
        if "rule_interpretations" in result:
            print("\nReasoning Analysis:")
            for interp in result["rule_interpretations"]:
                print(f"   {interp}")
        
        # Rule activations
        if "rule_activations" in result:
            print(f"\nRule Activations:")
            for rule, value in result["rule_activations"].items():
                bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
                print(f"   {rule}: [{bar}] {value:.3f}")
        
        # Predicate values
        if "predicates" in result:
            print("\nSoft Predicates:")
            for pred, value in result["predicates"].items():
                bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
                print(f"   {pred}: [{bar}] {value:.3f}")
        
        # Token masks
        if "token_masks" in result and result["token_masks"]:
            print("\nToken Importance (top tokens):")
            sorted_tokens = sorted(
                result["token_masks"],
                key=lambda x: x["mask_value"],
                reverse=True,
            )
            for tok in sorted_tokens[:10]:
                bar = "█" * int(tok["mask_value"] * 10) + "░" * (10 - int(tok["mask_value"] * 10))
                print(f"   {tok['token']:15s} [{bar}] {tok['mask_value']:.3f}")
            
            print(f"\n   Average mask: {result.get('mask_mean', 0):.3f}")
    
    print("\n" + "=" * 60)


def get_important_tokens(
    result: Dict,
    top_k: int = 10,
    threshold: float = 0.5,
) -> List[Dict]:
    """
    Get the most important tokens based on mask values.
    
    Args:
        result: Prediction result with token_masks
        top_k: Number of top tokens to return
        threshold: Minimum mask value threshold
    
    Returns:
        List of important token info dicts
    """
    if "token_masks" not in result:
        return []
    
    tokens = result["token_masks"]
    important = [t for t in tokens if t["mask_value"] >= threshold]
    important.sort(key=lambda x: x["mask_value"], reverse=True)
    
    return important[:top_k]


def get_reasoning_summary(result: Dict) -> Dict:
    """
    Get a summary of the reasoning process.
    
    Args:
        result: Prediction result
    
    Returns:
        Dictionary with reasoning summary
    """
    summary = {
        "dominant_signal": None,
        "contradiction_detected": False,
        "intensity_level": "normal",
        "confidence": "low",
    }
    
    if "rule_activations" not in result:
        return summary
    
    rules = result["rule_activations"]
    
    # Check for contradiction (r1 or r5 high)
    if rules.get("r1", 0) > 0.5 or rules.get("r5", 0) > 0.5:
        summary["contradiction_detected"] = True
    
    # Determine dominant signal
    if rules.get("r2", 0) > 0.6:
        summary["dominant_signal"] = "strong_negative"
    elif rules.get("r4", 0) > 0.6:
        summary["dominant_signal"] = "intense_negative"
    elif rules.get("r3", 0) > 0.6:
        summary["dominant_signal"] = "mild_positive"
    elif rules.get("r1", 0) > 0.6:
        summary["dominant_signal"] = "sarcasm"
    
    # Intensity level
    if "predicates" in result:
        preds = result["predicates"]
        if preds.get("p_high_int", 0) > 0.7:
            summary["intensity_level"] = "high"
        elif preds.get("p_high_int", 0) < 0.3:
            summary["intensity_level"] = "low"
    
    # Confidence based on prediction probabilities
    if "probabilities" in result:
        probs = result["probabilities"]
        max_prob = max(probs)
        if max_prob > 0.8:
            summary["confidence"] = "high"
        elif max_prob > 0.5:
            summary["confidence"] = "medium"
    
    return summary


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SoftLogic ViBERT Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--ckpt", "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--comment",
        type=str,
        required=True,
        help="Comment text to analyze",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Optional context text",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed analysis",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.ckpt}...")
    model, tokenizer, checkpoint = load_model(args.ckpt)
    label_list = checkpoint.get("label_list", [])
    
    # Make prediction
    result = predict_single(
        model=model,
        tokenizer=tokenizer,
        comment=args.comment,
        context=args.context,
        label_list=label_list,
        threshold=args.threshold,
        return_details=True,
    )
    
    if args.json:
        # Clean up tensors for JSON serialization
        clean_result = {}
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                clean_result[k] = v.tolist()
            else:
                clean_result[k] = v
        print(json.dumps(clean_result, ensure_ascii=False, indent=2))
    else:
        analyze_prediction(result, verbose=args.verbose)


if __name__ == "__main__":
    main()
