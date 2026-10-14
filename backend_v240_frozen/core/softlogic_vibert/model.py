"""
Statistically-Guided ViDeBERTa: A TF-IDF guided architecture for Vietnamese GenZ Sentiment Analysis.

Architecture Overview:
    (context, comment)
           ↓
    Token-level TF-IDF Gating (statistical)
           ↓
    ViDeBERTa Encoder (pretrained, fine-tuned)
           ↓
    Multi-view Projection Heads (semantic, lexical, pragmatic)
           ↓
    Soft Logic Inference Module (feature concatenation)
           ↓
    Sentiment Inference Subnetwork (MLP)
           ↓
    Multi-label Classification Head

Key Innovations:
1. Token-level TF-IDF gating: Suppresses low-salience tokens via statistics
2. Multi-view representations: Captures semantic, lexical, and pragmatic signals
3. Soft logic inference: Differentiable fuzzy logic for reasoning about sentiment
4. Conflict awareness: Detects contradictions (e.g., positive surface + negative intent)
"""

from typing import Dict, List, Optional, Tuple, Union
import math
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


# ============================================================================
# TF-IDF GATING MODULE
# ============================================================================

class TFIDFGatingLayer(nn.Module):
    """
    Token-level gating using precomputed TF-IDF scores.
    """

    def __init__(self, cache_path: str, default_score: float = 0.1):
        super().__init__()

        cache = torch.load(cache_path, map_location="cpu")
        if isinstance(cache, dict) and "scores" in cache:
            scores = cache["scores"].float()
            default_score = float(cache.get("default_score", default_score))
        else:
            scores = cache.float()

        self.register_buffer("tfidf_scores", scores)
        self.default_score = float(default_score)

    def forward(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_token_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        scores = self.tfidf_scores[input_ids]

        if special_token_ids is not None:
            is_special = torch.isin(input_ids, special_token_ids)
            scores = torch.where(is_special, torch.full_like(scores, self.default_score), scores)

        if attention_mask is not None:
            scores = scores * attention_mask

        mask_values = scores.unsqueeze(-1)
        masked_embeddings = embeddings * mask_values

        if attention_mask is not None:
            valid_count = attention_mask.sum() + 1e-8
            mask_mean = mask_values.sum() / valid_count
        else:
            mask_mean = mask_values.mean()

        mask_stats = {"mask_mean": mask_mean}

        if token_type_ids is not None:
            ctx_mask = (token_type_ids == 1).float().unsqueeze(-1)
            cmt_mask = (token_type_ids == 0).float().unsqueeze(-1)
            if attention_mask is not None:
                ctx_count = (ctx_mask * attention_mask.unsqueeze(-1)).sum() + 1e-8
                cmt_count = (cmt_mask * attention_mask.unsqueeze(-1)).sum() + 1e-8
            else:
                ctx_count = ctx_mask.sum() + 1e-8
                cmt_count = cmt_mask.sum() + 1e-8
            mask_stats["mask_context_mean"] = (mask_values * ctx_mask).sum() / ctx_count
            mask_stats["mask_comment_mean"] = (mask_values * cmt_mask).sum() / cmt_count

        return masked_embeddings, mask_values, mask_stats


def _resolve_tfidf_cache_path(cache_path: str) -> Optional[str]:
    """Resolve a TF-IDF cache path robustly across different working directories.

    Some checkpoints store paths like `softlogic_vibert/tfidf_cache.pt`, which only
    resolve when executed from a particular project root. This helper tries a few
    safe alternatives and returns the first existing path.
    """
    if not cache_path:
        return None

    p = Path(cache_path)
    candidates: List[Path] = []

    # As-is (relative to current working dir) and absolute.
    candidates.append(p)

    # Relative to the installed/source package location.
    module_dir = Path(__file__).resolve().parent
    candidates.append(module_dir / p)
    candidates.append(module_dir.parent / p)

    # If path is like "softlogic_vibert/tfidf_cache.pt", also try dropping prefix.
    try:
        parts = p.parts
        if len(parts) >= 2 and parts[0] == "softlogic_vibert":
            stripped = Path(*parts[1:])
            candidates.append(stripped)
            candidates.append(module_dir / stripped)
            candidates.append(module_dir.parent / stripped)
    except Exception:
        pass

    for cand in candidates:
        try:
            if cand.is_file():
                return str(cand)
        except OSError:
            continue
    return None


# ============================================================================
# MULTI-VIEW REPRESENTATION MODULES
# ============================================================================

class LexicalView(nn.Module):
    """
    Lexical/Surface View via Multi-kernel 1D CNN.
    
    Captures:
    - Surface polarity signals
    - Keywords and n-grams
    - Emoji and slang patterns
    
    Uses multiple kernel sizes (1, 2, 3) for different n-gram patterns.
    
    Args:
        hidden_size: Input embedding dimension
        num_channels: Number of output channels per kernel
        kernel_sizes: List of kernel sizes for conv layers
        proj_size: Final projection dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_channels: int = 128,
        kernel_sizes: List[int] = [1, 2, 3],
        proj_size: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, num_channels, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        
        self.proj = nn.Sequential(
            nn.Linear(num_channels * len(kernel_sizes), proj_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract lexical view representation.
        
        Args:
            embeddings: Token embeddings [B, L, d]
            attention_mask: Attention mask [B, L]
        
        Returns:
            z_lex: Lexical representation [B, proj_size]
        """
        # Transpose for Conv1d: [B, L, d] -> [B, d, L]
        x = embeddings.transpose(1, 2)
        seq_len = x.size(2)
        
        # Apply each conv layer
        conv_outputs = []
        for conv in self.convs:
            h = F.gelu(conv(x))  # [B, C, L'] where L' may differ from L
            
            # Ensure output matches input length for masking
            if h.size(2) != seq_len:
                # Pad or truncate to match
                if h.size(2) > seq_len:
                    h = h[:, :, :seq_len]
                else:
                    h = F.pad(h, (0, seq_len - h.size(2)), value=float("-inf"))
            
            # Max pooling with mask
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(1).float()  # [B, 1, L]
                # Set masked positions to -inf for max pooling
                h = h.masked_fill(mask == 0, float("-inf"))
            
            pooled = h.max(dim=2).values  # [B, C]
            conv_outputs.append(pooled)
        
        # Concatenate and project
        concat = torch.cat(conv_outputs, dim=1)  # [B, C * num_kernels]
        z_lex = self.proj(concat)
        
        return z_lex


class PragmaticView(nn.Module):
    """
    Pragmatic/Affective View for GenZ Expression Patterns.
    
    Captures:
    - Emoji density and distribution
    - Punctuation intensity (!!!, ???)
    - Character repetition (e.g., "niceeeee")
    - Capitalization patterns
    - Text length and structure
    
    Args:
        feat_dim: Number of input pragmatic features
        hidden_size: Hidden layer size
        proj_size: Output projection dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        feat_dim: int = 8,
        hidden_size: int = 64,
        proj_size: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, proj_size),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Project pragmatic features.
        
        Args:
            features: Pragmatic features [B, feat_dim]
        
        Returns:
            z_prag: Pragmatic representation [B, proj_size]
        """
        return self.net(features)


# ============================================================================
# SOFT LOGIC INFERENCE MODULE
# ============================================================================

class SoftPredicate(nn.Module):
    """
    A differentiable soft predicate function.
    
    Maps a representation to a soft truth value in (0, 1).
    Example: P_pos_sem(z_sem) -> probability of positive semantic content.
    
    Args:
        input_size: Input representation dimension
        hidden_size: Hidden layer size
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute soft predicate value.
        
        Args:
            x: Input representation [B, input_size]
        
        Returns:
            truth_value: Soft truth value [B, 1] in (0, 1)
        """
        return torch.sigmoid(self.net(x))


class SoftLogicModule(nn.Module):
    """
    Soft Logic Inference Module for Sentiment Reasoning.
    
    Implements differentiable fuzzy logic operations:
        AND(a, b) = (a · b) ** p
        OR(a, b)  = a + b − a·b
        NOT(a)    = 1 − a
        IMPLIES(a, b) = 1 − a + a·b
    
    Defines reasoning rules:
        r1 = AND(P_pos_lex, P_neg_sem)          # Sarcasm / contradiction
        r2 = AND(P_neg_lex, P_neg_sem)          # Strong negative (anger/disgust)
        r3 = AND(P_pos_sem, NOT(P_high_int))    # Mild positive (enjoyment)
        r4 = AND(P_high_int, P_neg_sem)         # Intense negative (fear/anger)
        r5 = |P_pos_lex - P_pos_sem|            # Surprise / inconsistency
    
    All operations are continuous and differentiable.
    
    Args:
        proj_size: Size of each view representation
        predicate_hidden: Hidden size for predicate networks
    """
    
    def __init__(self, proj_size: int = 128, predicate_hidden: int = 64):
        super().__init__()
        
        # Soft predicates
        self.pred_pos_sem = SoftPredicate(proj_size, predicate_hidden)
        self.pred_neg_sem = SoftPredicate(proj_size, predicate_hidden)
        self.pred_pos_lex = SoftPredicate(proj_size, predicate_hidden)
        self.pred_neg_lex = SoftPredicate(proj_size, predicate_hidden)
        self.pred_high_int = SoftPredicate(proj_size, predicate_hidden)
        self.p_val = nn.Parameter(torch.tensor(1.0))
    
    def AND(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Parameterized fuzzy AND with learnable sharpness."""
        p = torch.clamp(self.p_val, min=0.5, max=5.0)
        return torch.pow(a * b, p)
    
    @staticmethod
    def OR(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fuzzy OR: probabilistic sum."""
        return a + b - a * b
    
    @staticmethod
    def NOT(a: torch.Tensor) -> torch.Tensor:
        """Fuzzy NOT: complement."""
        return 1.0 - a
    
    @staticmethod
    def IMPLIES(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fuzzy implication: Reichenbach operator."""
        return 1.0 - a + a * b
    
    def forward(
        self,
        z_sem: torch.Tensor,
        z_lex: torch.Tensor,
        z_prag: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute reasoning rule activations.
        
        Args:
            z_sem: Semantic representation [B, proj_size]
            z_lex: Lexical representation [B, proj_size]
            z_prag: Pragmatic representation [B, proj_size]
        
        Returns:
            rules_vector: Concatenated rule activations [B, num_rules]
            rule_details: Dictionary with individual rule and predicate values
        """
        # Compute soft predicates
        p_pos_sem = self.pred_pos_sem(z_sem)      # [B, 1]
        p_neg_sem = self.pred_neg_sem(z_sem)      # [B, 1]
        p_pos_lex = self.pred_pos_lex(z_lex)      # [B, 1]
        p_neg_lex = self.pred_neg_lex(z_lex)      # [B, 1]
        p_high_int = self.pred_high_int(z_prag)   # [B, 1]
        
        # Compute reasoning rules
        r1 = self.AND(p_pos_lex, p_neg_sem)               # Sarcasm/contradiction
        r2 = self.AND(p_neg_lex, p_neg_sem)               # Strong negative
        r3 = self.AND(p_pos_sem, self.NOT(p_high_int))    # Mild positive
        r4 = self.AND(p_high_int, p_neg_sem)              # Intense negative
        r5 = torch.abs(p_pos_lex - p_pos_sem)             # Inconsistency/surprise
        
        # Concatenate rule activations
        rules_vector = torch.cat([r1, r2, r3, r4, r5], dim=1)
        
        # Detailed outputs for interpretability
        rule_details = {
            "r1_sarcasm": r1,
            "r2_strong_neg": r2,
            "r3_mild_pos": r3,
            "r4_intense_neg": r4,
            "r5_inconsistency": r5,
            "p_pos_sem": p_pos_sem,
            "p_neg_sem": p_neg_sem,
            "p_pos_lex": p_pos_lex,
            "p_neg_lex": p_neg_lex,
            "p_high_int": p_high_int,
        }
        
        return rules_vector, rule_details


# ============================================================================
# SENTIMENT INFERENCE SUBNETWORK
# ============================================================================

class SentimentInferenceHead(nn.Module):
    """
    Sentiment Inference Subnetwork.
    
    Combines multi-view representations and reasoning signals
    to produce multi-label classification logits.
    
    Args:
        input_size: Total size of concatenated features
        hidden_size: Hidden layer size
        num_labels: Number of output labels
        num_layers: Number of hidden layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_labels: int = 7,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        layers = []
        current_size = input_size
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_size = hidden_size
        
        layers.append(nn.Linear(hidden_size, num_labels))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute classification logits.
        
        Args:
            x: Concatenated features [B, input_size]
        
        Returns:
            logits: Classification logits [B, num_labels]
        """
        return self.net(x)


# ============================================================================
# MAIN MODEL
# ============================================================================

class SoftLogicViBERT(nn.Module):
    """
    Statistically-Guided ViDeBERTa (SoftLogic ViBERT backbone).
    
    Complete pipeline:
        1. Token-level TF-IDF Gating
        2. ViBERT Encoding
        3. Multi-view Representation (semantic, lexical, pragmatic)
        4. Soft Logic Inference (feature concat)
        5. Sentiment Inference & Classification
    
    Args:
        config: ModelConfig with all hyperparameters
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load ViDeBERTa encoder (force float32 for compatibility with custom layers)
        self.encoder = AutoModel.from_pretrained(config.model_name, torch_dtype=torch.float32)
        self.embeddings = self.encoder.get_input_embeddings()
        hidden_size = self.encoder.config.hidden_size
        
        # Token-level TF-IDF gating
        if config.use_mask:
            cache_path = config.tfidf_cache_path if hasattr(config, "tfidf_cache_path") else "tfidf_cache.pt"
            default_score = config.tfidf_default_score if hasattr(config, "tfidf_default_score") else 0.1
            resolved_cache_path = _resolve_tfidf_cache_path(str(cache_path))
            if resolved_cache_path is None:
                warnings.warn(
                    f"TF-IDF cache not found at '{cache_path}'. Disabling masking layer for this run.",
                    RuntimeWarning,
                )
                self.masking_layer = None
            else:
                self.masking_layer = TFIDFGatingLayer(resolved_cache_path, default_score=default_score)
        else:
            self.masking_layer = None
        
        # Semantic projection (CLS token)
        self.sem_proj = nn.Sequential(
            nn.Linear(hidden_size, config.proj_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        
        # Multi-view modules
        if config.use_multiview:
            self.lex_view = LexicalView(
                hidden_size=hidden_size,
                num_channels=config.lex_channels,
                kernel_sizes=config.lex_kernel_sizes if hasattr(config, 'lex_kernel_sizes') else [1, 2, 3],
                proj_size=config.proj_size,
                dropout=config.dropout,
            )
            self.prag_view = PragmaticView(
                feat_dim=config.prag_feat_dim,
                hidden_size=config.prag_hidden if hasattr(config, 'prag_hidden') else 64,
                proj_size=config.proj_size,
                dropout=config.dropout,
            )
        else:
            self.lex_view = None
            self.prag_view = None
        
        # Soft logic module
        if config.use_logic:
            self.logic_module = SoftLogicModule(
                proj_size=config.proj_size,
                predicate_hidden=config.predicate_hidden if hasattr(config, 'predicate_hidden') else 64,
            )
        else:
            self.logic_module = None
        
        # Calculate inference input size
        infer_input_size = config.proj_size  # z_sem
        if config.use_multiview:
            infer_input_size += config.proj_size * 2  # z_lex, z_prag
        if config.use_logic:
            num_rules = 5 - len(config.drop_rules) if config.drop_rules else 5
            infer_input_size += max(1, num_rules)
        
        # Sentiment inference head
        self.inference_head = SentimentInferenceHead(
            input_size=infer_input_size,
            hidden_size=config.infer_hidden if hasattr(config, 'infer_hidden') else 256,
            num_labels=config.num_labels,
            num_layers=config.infer_layers if hasattr(config, 'infer_layers') else 2,
            dropout=config.dropout,
        )

        # Logic is used only for feature concatenation
        self.gate_sem = None
        self.gate_lex = None
        self.gate_prag = None
    
    def _encode_with_masking(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor],
        special_token_ids: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Get token embeddings, apply masking, and encode with ViBERT.
        
        Returns:
            hidden_states: ViBERT hidden states [B, L, d]
            word_embeddings: Original word embeddings [B, L, d]
            mask_values: Soft mask values [B, L, 1] or None
            mask_stats: Mask statistics dict
        """
        # Get word embeddings
        word_embeddings = self.embeddings(input_ids)
        
        # Apply soft masking
        if self.masking_layer is not None:
            masked_embeddings, mask_values, mask_stats = self.masking_layer(
                embeddings=word_embeddings,
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                special_token_ids=special_token_ids,
            )
        else:
            masked_embeddings = word_embeddings
            mask_values = None
            mask_stats = {}
        
        # Encode with ViDeBERTa (DeBERTa does not use token_type_ids with inputs_embeds)
        outputs = self.encoder(
            inputs_embeds=masked_embeddings,
            attention_mask=attention_mask,
        )
        
        return outputs.last_hidden_state, word_embeddings, mask_values, mask_stats
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        prag_features: Optional[torch.Tensor] = None,
        special_token_ids: Optional[torch.Tensor] = None,
        return_extras: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass through the entire pipeline.
        
        Args:
            input_ids: Input token IDs [B, L]
            attention_mask: Attention mask [B, L]
            token_type_ids: Token type IDs [B, L]
            prag_features: Pragmatic features [B, feat_dim]
            special_token_ids: Special token IDs to preserve [S]
            return_extras: Whether to return interpretability extras
        
        Returns:
            logits: Classification logits [B, num_labels]
            extras: (optional) Dict with interpretability info
        """
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Stage 1: Masking + Encoding
        hidden_states, word_embeddings, mask_values, mask_stats = self._encode_with_masking(
            input_ids, attention_mask, token_type_ids, special_token_ids
        )
        
        # Stage 2: Semantic view (CLS token)
        cls_hidden = hidden_states[:, 0]
        z_sem = self.sem_proj(cls_hidden)
        
        # Stage 3: Multi-view representations
        if self.config.use_multiview:
            # Lexical view
            z_lex = self.lex_view(word_embeddings, attention_mask)
            
            # Pragmatic view
            if prag_features is None:
                prag_features = torch.zeros(
                    batch_size, self.config.prag_feat_dim, device=device
                )
            z_prag = self.prag_view(prag_features)
        else:
            z_lex = None
            z_prag = None
        
        # Stage 4: Soft logic inference
        rules_vector = None
        rule_details = None
        if self.config.use_logic and self.logic_module is not None:
            if z_lex is None:
                z_lex = torch.zeros(batch_size, self.config.proj_size, device=device)
            if z_prag is None:
                z_prag = torch.zeros(batch_size, self.config.proj_size, device=device)
            
            rules_vector, rule_details = self.logic_module(z_sem, z_lex, z_prag)
            
            # Handle rule dropping for ablation
            if self.config.drop_rules:
                rule_names = ["r1_sarcasm", "r2_strong_neg", "r3_mild_pos", "r4_intense_neg", "r5_inconsistency"]
                keep_indices = [
                    i for i, name in enumerate(rule_names)
                    if name.split("_")[0] not in self.config.drop_rules
                ]
                if keep_indices:
                    rules_vector = rules_vector[:, keep_indices]
                else:
                    rules_vector = torch.zeros(batch_size, 1, device=device)
        
        # Stage 5: Feature concatenation and classification
        feature_parts = [z_sem]
        if self.config.use_multiview:
            feature_parts.extend([z_lex, z_prag])
        if self.config.use_logic and rules_vector is not None:
            feature_parts.append(rules_vector)
        
        fused_features = torch.cat(feature_parts, dim=1)
        logits = self.inference_head(fused_features)
        
        if not return_extras:
            return logits
        
        # Compile extras for interpretability
        extras = {
            "z_sem": z_sem,
            "z_lex": z_lex,
            "z_prag": z_prag,
            "hidden_states": hidden_states if (hasattr(self.config, 'return_all_hidden') and self.config.return_all_hidden) else None,
        }

        if mask_values is not None:
            extras["mask_vals"] = mask_values
            extras.update(mask_stats)
        
        if rule_details is not None:
            # Flatten rule details for easier access
            extras["rules"] = {
                "r1": rule_details["r1_sarcasm"],
                "r2": rule_details["r2_strong_neg"],
                "r3": rule_details["r3_mild_pos"],
                "r4": rule_details["r4_intense_neg"],
                "r5": rule_details["r5_inconsistency"],
            }
            extras["predicates"] = {
                "p_pos_sem": rule_details["p_pos_sem"],
                "p_neg_sem": rule_details["p_neg_sem"],
                "p_pos_lex": rule_details["p_pos_lex"],
                "p_neg_lex": rule_details["p_neg_lex"],
                "p_high_int": rule_details["p_high_int"],
            }
        
        return logits, extras
    
    def get_mask_for_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        special_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Get just the mask values for a given input.
        Useful for interpretability without full forward pass.
        """
        if self.masking_layer is None:
            return None
        
        word_embeddings = self.embeddings(input_ids)
        _, mask_values, _ = self.masking_layer(
            embeddings=word_embeddings,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            special_token_ids=special_token_ids,
        )
        return mask_values
    
    def freeze_encoder(self):
        """Freeze ViBERT encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze ViBERT encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    @classmethod
    def from_pretrained(cls, path: str, config=None):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        
        if config is None:
            from .config import ModelConfig
            config = ModelConfig(**checkpoint["model_config"])
        
        model = cls(config)
        
        if "model" in checkpoint:
            # Full model saved
            return checkpoint["model"]
        else:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        
        return model
