from pathlib import Path
import torch


BACKEND_DIR = Path(__file__).resolve().parents[2]

MODEL_NAME = "vinai/phobert-base-v2"

TRAIN_SOURCE = BACKEND_DIR / "data" / "absa" / "processed" / "train.json"
VAL_SOURCE = BACKEND_DIR / "data" / "absa" / "processed" / "validation.json"
TEST_SOURCE = BACKEND_DIR / "data" / "absa" / "processed" / "test.json"

V2_DATA_DIR = BACKEND_DIR / "data" / "absa" / "asc_processed_v2"

TRAIN_FILE = V2_DATA_DIR / "train.json"
VAL_FILE = V2_DATA_DIR / "validation.json"
TEST_FILE = V2_DATA_DIR / "test.json"
PREP_STATS_FILE = V2_DATA_DIR / "prepare_stats.json"

VNCORENLP_DIR = BACKEND_DIR / "assets" / "vncorenlp"

MODEL_OUTPUT_DIR = (
    BACKEND_DIR
    / "model"
    / "absa"
    / "asc_phobert_v2"
)

EVAL_OUTPUT_DIR = (
    BACKEND_DIR
    / "evaluation_outputs"
    / "asc_phobert_v2"
)

LABEL2ID = {
    "NEGATIVE": 0,
    "NEUTRAL": 1,
    "POSITIVE": 2,
}

ID2LABEL = {
    0: "NEGATIVE",
    1: "NEUTRAL",
    2: "POSITIVE",
}

MAX_LENGTH = 256

BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
EPOCHS = 12

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

EARLY_STOPPING_PATIENCE = 3

# Moderate imbalance correction.
# We intentionally avoid full inverse-frequency weights because
# validation has only 4 NEGATIVE examples.
MAX_CLASS_WEIGHT = 2.5

RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
