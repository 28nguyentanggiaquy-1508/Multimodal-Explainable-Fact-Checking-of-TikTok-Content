"""
Configuration module for SoftLogic ViDeBERTa Sentiment Analysis.

This module defines all hyperparameters and configuration options for:
- Model architecture (masking, multi-view, soft-logic modules)
- Training procedure (optimizer, scheduler, early stopping)
- Ablation studies (selective component disabling)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """
    Configuration for the SoftLogicViDeBERTa model architecture.
    
    Core Philosophy:
    - ViDeBERTa as semantic backbone (DeBERTa-based, Vietnamese)
    - Token-level soft masking for noise suppression
    - Multi-view representations (semantic, lexical, pragmatic)
    - Differentiable fuzzy-logic reasoning
    """
    
    # ===== Backbone Configuration =====
    model_name: str = "Fsoft-AIC/videberta-base"
    max_len: int = 128
    hidden_size: int = 768  # ViDeBERTa hidden dimension
    
    # ===== Classification =====
    num_labels: int = 7  # Default: 7 emotion categories
    
    # ===== Projection Dimensions =====
    proj_size: int = 128  # Output size for each view projection
    
    # ===== Token-level TF-IDF Gating =====
    use_mask: bool = True
    mask_lambda: float = 1e-3  # Sparsity regularization weight
    mask_init_bias: float = 2.0
    tfidf_cache_path: str = "tfidf_cache.pt"
    tfidf_default_score: float = 0.1
    
    # ===== Lexical View (1D CNN) =====
    lex_channels: int = 128
    lex_kernel_sizes: List[int] = field(default_factory=lambda: [1, 2, 3])
    
    # ===== Pragmatic View =====
    prag_feat_dim: int = 8  # Number of pragmatic features
    prag_hidden: int = 64
    
    # ===== Soft Logic Inference =====
    use_multiview: bool = True
    use_logic: bool = True
    num_predicates: int = 5  # P_pos_sem, P_neg_sem, P_pos_lex, P_neg_lex, P_high_int
    num_rules: int = 5  # r1 through r5
    predicate_hidden: int = 64  # Hidden size for predicate networks
    
    # ===== Sentiment Inference MLP =====
    infer_hidden: int = 256
    infer_layers: int = 2
    dropout: float = 0.3
    
    # ===== Ablation Controls =====
    drop_rules: List[str] = field(default_factory=list)
    ablation_mode: str = "full"  # Options: 'full', 'vibert_only', 'mask_only', 'multiview_only'
    
    # ===== Interpretability =====
    return_all_hidden: bool = False
    track_rule_gradients: bool = False
    
    def __post_init__(self):
        """Apply ablation mode shortcuts."""
        if self.ablation_mode == "vibert_only":
            self.use_mask = False
            self.use_multiview = False
            self.use_logic = False
        elif self.ablation_mode == "mask_only":
            self.use_multiview = False
            self.use_logic = False
        elif self.ablation_mode == "multiview_only":
            self.use_logic = False


@dataclass
class TrainConfig:
    """
    Configuration for training procedure.
    
    Designed for:
    - Multi-label classification with class imbalance
    - Early stopping with macro-F1 as primary metric
    - Gradient accumulation for effective larger batch sizes
    """
    
    # ===== Basic Training =====
    batch_size: int = 16
    effective_batch_size: int = 32  # For gradient accumulation
    epochs: int = 20
    seed: int = 42
    
    # ===== Optimizer =====
    lr: float = 2e-5
    encoder_lr: float = 1e-5  # Learning rate for ViDeBERTa encoder
    head_lr: float = 5e-4  # Learning rate for classification heads
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    
    # ===== Scheduler =====
    warmup_ratio: float = 0.1
    scheduler_type: str = "linear"  # Options: 'linear', 'cosine', 'constant'
    
    # ===== Gradient Handling =====
    grad_clip: float = 1.0
    
    # ===== Early Stopping =====
    patience: int = 4
    min_delta: float = 0.001  # Minimum improvement threshold
    
    # ===== Evaluation =====
    threshold: float = 0.5
    eval_steps: Optional[int] = None  # None = evaluate each epoch
    
    # ===== Loss Configuration =====
    use_class_weights: bool = True
    focal_gamma: float = 0.0  # Set > 0 to use focal loss
    label_smoothing: float = 0.0
    deep_supervision_lambda: float = 0.5
    slang_preserve_lambda: float = 0.5
    mask_sparsity_lambda: float = 0.1
    use_polarity_supervision: bool = True
    use_slang_preserve: bool = True
    use_mask_sparsity: bool = True
    pos_labels: str = "enjoyment"
    neg_labels: str = "anger,disgust,fear,sadness"
    slang_lexicon_path: str = "dictionary.json"
    
    # ===== Logging =====
    log_interval: int = 50
    save_all_checkpoints: bool = False
    
    @property
    def accumulation_steps(self) -> int:
        """Calculate gradient accumulation steps."""
        return max(1, self.effective_batch_size // self.batch_size)


@dataclass
class ExperimentConfig:
    """
    Configuration for a complete experiment run.
    Combines model, training, and data configurations.
    """
    
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    # ===== Data Paths =====
    data_path: str = "output_data.json"
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    
    # ===== HuggingFace Dataset =====
    hf_dataset: Optional[str] = None
    hf_split: str = "train"
    hf_val_split: Optional[str] = "validation"
    
    # ===== Output =====
    output_dir: str = "softlogic_outputs"
    experiment_name: str = "softlogic_vibert"
    
    # ===== Reproducibility =====
    save_config: bool = True
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "model": self.model.__dict__,
            "train": self.train.__dict__,
            "data_path": self.data_path,
            "val_path": self.val_path,
            "test_path": self.test_path,
            "hf_dataset": self.hf_dataset,
            "experiment_name": self.experiment_name,
        }


# ===== Predefined Ablation Configurations =====

def get_vibert_only_config() -> ModelConfig:
    """ViBERT baseline without any enhancements."""
    return ModelConfig(
        use_mask=False,
        use_multiview=False,
        use_logic=False,
        ablation_mode="vibert_only"
    )


def get_mask_only_config() -> ModelConfig:
    """ViBERT + soft masking only."""
    return ModelConfig(
        use_mask=True,
        use_multiview=False,
        use_logic=False,
        ablation_mode="mask_only"
    )


def get_multiview_no_logic_config() -> ModelConfig:
    """ViBERT + masking + multi-view, but no logic."""
    return ModelConfig(
        use_mask=True,
        use_multiview=True,
        use_logic=False,
        ablation_mode="multiview_only"
    )


def get_full_config() -> ModelConfig:
    """Full model with all components."""
    return ModelConfig(
        use_mask=True,
        use_multiview=True,
        use_logic=True,
        ablation_mode="full"
    )


def get_ablation_configs() -> dict:
    """Return all ablation configurations as a dictionary."""
    return {
        "vibert_only": get_vibert_only_config(),
        "mask_only": get_mask_only_config(),
        "multiview_no_logic": get_multiview_no_logic_config(),
        "full": get_full_config(),
    }
