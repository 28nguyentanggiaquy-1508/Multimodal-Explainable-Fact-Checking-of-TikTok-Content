from pathlib import Path
import torch

BACKEND_DIR = Path(__file__).resolve().parents[2]
SOURCE_DATA_DIR = BACKEND_DIR / 'data' / 'absa' / 'processed'
ASC_DATA_DIR = SOURCE_DATA_DIR / 'asc'
TRAIN_FILE = ASC_DATA_DIR / 'train.json'
VAL_FILE = ASC_DATA_DIR / 'validation.json'
TEST_FILE = ASC_DATA_DIR / 'test.json'
MODEL_OUTPUT_DIR = BACKEND_DIR / 'model' / 'absa' / 'asc_phobert_v1'
MODEL_NAME = 'vinai/phobert-base-v2'
LABEL2ID = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
ID2LABEL = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
MAX_LENGTH = 128
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
EPOCHS = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 3
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
