from pathlib import Path
import torch


BACKEND_DIR = Path(__file__).resolve().parents[2]

MODEL_NAME = "vinai/phobert-base-v2"

TRAIN_SOURCE = BACKEND_DIR / "data" / "absa" / "processed" / "train.json"
VAL_SOURCE = BACKEND_DIR / "data" / "absa" / "processed" / "validation.json"
TEST_SOURCE = BACKEND_DIR / "data" / "absa" / "processed" / "test.json"

V2_DATA_DIR = BACKEND_DIR / "data" / "absa" / "processed_v2"

TRAIN_FILE = V2_DATA_DIR / "train.json"
VAL_FILE = V2_DATA_DIR / "validation.json"
TEST_FILE = V2_DATA_DIR / "test.json"
PREP_STATS_FILE = V2_DATA_DIR / "prepare_stats.json"

MODEL_OUTPUT_DIR = (
    BACKEND_DIR
    / "model"
    / "absa"
    / "ate_phobert_v2"
)

EVAL_OUTPUT_DIR = (
    BACKEND_DIR
    / "evaluation_outputs"
    / "ate_phobert_v2"
)

VNCORENLP_DIR = (
    BACKEND_DIR
    / "assets"
    / "vncorenlp"
)

LABEL2ID = {
    "O": 0,
    "B-ASP": 1,
    "I-ASP": 2,
}

ID2LABEL = {
    0: "O",
    1: "B-ASP",
    2: "I-ASP",
}

MAX_LENGTH = 256

BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
EPOCHS = 10

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

EARLY_STOPPING_PATIENCE = 3
LABEL_SMOOTHING = 0.05

# V2 giảm mức ép B/I so với V1 để hạn chế false positive.
MAX_ASPECT_CLASS_WEIGHT = 4.0

# Tune trên validation, KHÔNG tune trên test.
THRESHOLD_CANDIDATES = [
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
]

RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
