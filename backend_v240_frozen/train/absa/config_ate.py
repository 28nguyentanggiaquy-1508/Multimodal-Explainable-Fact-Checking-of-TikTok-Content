from pathlib import Path
import torch


BACKEND_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = (
    BACKEND_DIR
    / "data"
    / "absa"
    / "processed"
)

TRAIN_FILE = DATA_DIR / "train.json"
VAL_FILE = DATA_DIR / "validation.json"
TEST_FILE = DATA_DIR / "test.json"


MODEL_OUTPUT_DIR = (
    BACKEND_DIR
    / "model"
    / "absa"
    / "ate_phobert_v1"
)


MODEL_NAME = "vinai/phobert-base-v2"


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


MAX_LENGTH = 128

BATCH_SIZE = 8

GRAD_ACCUM_STEPS = 2

EPOCHS = 8

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.1

EARLY_STOPPING_PATIENCE = 3

RANDOM_SEED = 42


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
MODEL_VERSION = "v2"

MODEL_OUTPUT_DIR = (
    BACKEND_DIR
    / "model"
    / "absa"
    / f"ate_phobert_{MODEL_VERSION}"
)