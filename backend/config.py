import os
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")


# ============================================================
# LEGACY EMOTION PIPELINE
# ============================================================
MODEL_PATH = "model/phobert"
DICT_PATH = "assets/dic_final_cleaned.json"
MAX_SEQ_LEN = 256
BATCH_SIZE = 8


# ============================================================
# API / SECURITY
# ============================================================
def _csv_env(name: str, default: str = "") -> list[str]:
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


CORS_ALLOW_ORIGINS = _csv_env(
    "CORS_ALLOW_ORIGINS",
    "http://127.0.0.1:3000,http://localhost:3000",
)


# ============================================================
# GEMINI / VLM
# ============================================================
GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or ""
).strip()

GEMINI_VISION_MODEL = os.getenv(
    "GEMINI_VISION_MODEL",
    "gemini-3.8-flash",
).strip()

GEMINI_ANALYST_MODEL = os.getenv(
    "GEMINI_ANALYST_MODEL",
    "gemini-3.8-flash",
).strip()

GEMINI_SEARCH_MODEL = os.getenv(
    "GEMINI_SEARCH_MODEL",
    GEMINI_ANALYST_MODEL,
).strip()

GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "4"))
GEMINI_RETRY_BASE_SECONDS = float(
    os.getenv("GEMINI_RETRY_BASE_SECONDS", "2")
)
GEMINI_FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL",
    "gemini-3.7-flash",
).strip()

# Model-level circuit breaker. Hard quota errors open the circuit so later
# Phase-3/Phase-4 calls do not repeatedly hit the same exhausted model.
GEMINI_CIRCUIT_DEFAULT_SECONDS = int(
    os.getenv("GEMINI_CIRCUIT_DEFAULT_SECONDS", "900")
)


# ============================================================
# PHASE 3 - SEARCH PROVIDERS
# ============================================================
# brave_then_gemini: Brave primary + Gemini grounding fallback
# brave: Brave only
# gemini: Gemini grounding only
EVIDENCE_SEARCH_PROVIDER = os.getenv(
    "EVIDENCE_SEARCH_PROVIDER",
    "brave_then_gemini",
).strip().lower()

BRAVE_SEARCH_API_KEY = os.getenv(
    "BRAVE_SEARCH_API_KEY",
    "",
).strip()
BRAVE_SEARCH_COUNTRY = os.getenv(
    "BRAVE_SEARCH_COUNTRY",
    "ALL",
).strip().upper()
BRAVE_SEARCH_LANG = os.getenv(
    "BRAVE_SEARCH_LANG",
    "vi",
).strip().lower()

# Kept only so old modules/configs do not crash. The thesis pipeline does not
# rely on Google Custom Search JSON API.
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", "").strip()
GOOGLE_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX", "").strip()


# ============================================================
# RETRIEVAL MODELS - MOCHEG-INSPIRED BASELINE
# ============================================================
TEXT_RETRIEVAL_MODEL = os.getenv(
    "TEXT_RETRIEVAL_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
).strip()

IMAGE_RETRIEVAL_MODEL = os.getenv(
    "IMAGE_RETRIEVAL_MODEL",
    "openai/clip-vit-base-patch32",
).strip()

WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "10"))
TEXT_RETRIEVAL_TOP_N = int(os.getenv("TEXT_RETRIEVAL_TOP_N", "20"))
IMAGE_RETRIEVAL_TOP_N = int(os.getenv("IMAGE_RETRIEVAL_TOP_N", "10"))
TEXT_EVIDENCE_TOP_K = int(os.getenv("TEXT_EVIDENCE_TOP_K", "5"))
IMAGE_EVIDENCE_TOP_K = int(os.getenv("IMAGE_EVIDENCE_TOP_K", "3"))
MAX_CHECK_WORTHY_CLAIMS = int(os.getenv("MAX_CHECK_WORTHY_CLAIMS", "5"))

# Optional second-stage LLM relevance reranker. Keeping it optional prevents
# evidence retrieval from being coupled to Gemini availability.
EVIDENCE_USE_LLM_RERANKER = os.getenv(
    "EVIDENCE_USE_LLM_RERANKER",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}

# Deterministic diversity constraints for the final Top-K evidence.
EVIDENCE_MAX_PER_URL = int(os.getenv("EVIDENCE_MAX_PER_URL", "1"))
EVIDENCE_MAX_PER_DOMAIN = int(os.getenv("EVIDENCE_MAX_PER_DOMAIN", "2"))


# Temporal evidence hardening for time-sensitive/current-state claims.
TEMPORAL_SEARCH_LOOKBACK_DAYS = int(
    os.getenv("TEMPORAL_SEARCH_LOOKBACK_DAYS", "730")
)
TEMPORAL_STRICT_CURRENT_STATE = os.getenv(
    "TEMPORAL_STRICT_CURRENT_STATE",
    "true",
).strip().lower() in {"1", "true", "yes", "on"}


# ============================================================
# NETWORK / SSRF SAFEGUARDS
# ============================================================
EVIDENCE_REQUEST_TIMEOUT = float(
    os.getenv("EVIDENCE_REQUEST_TIMEOUT", "10")
)
EVIDENCE_MAX_PAGE_BYTES = int(
    os.getenv("EVIDENCE_MAX_PAGE_BYTES", "2000000")
)
EVIDENCE_MAX_IMAGE_BYTES = int(
    os.getenv("EVIDENCE_MAX_IMAGE_BYTES", "8000000")
)


# ============================================================
# CoReS-INSPIRED CONCEPT FILTERING
# These are project engineering defaults, not paper hyperparameters.
# ============================================================
CONCEPT_CLUSTER_THRESHOLD = float(
    os.getenv("CONCEPT_CLUSTER_THRESHOLD", "0.86")
)
CONCEPT_CLAIM_THRESHOLD = float(
    os.getenv("CONCEPT_CLAIM_THRESHOLD", "0.40")
)
CONCEPT_MIN_FREQ_RATIO = float(
    os.getenv("CONCEPT_MIN_FREQ_RATIO", "0.20")
)

# Phase 3 is local-first by default: concept extraction + reasoning support
# must remain available even when Gemini quota/service is unavailable.
REASONING_USE_GEMINI = os.getenv(
    "REASONING_USE_GEMINI",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}


# ============================================================
# PHASE 4 VERIFICATION
# ============================================================
VERIFICATION_MAX_IMAGES = int(os.getenv("VERIFICATION_MAX_IMAGES", "1"))
VERIFICATION_MAX_EVIDENCE_CHARS = int(
    os.getenv("VERIFICATION_MAX_EVIDENCE_CHARS", "700")
)

# Batch all evidence-backed claims into one verifier call when possible.
# This materially reduces Gemini calls for videos containing many claims.
VERIFICATION_BATCH_ENABLED = os.getenv(
    "VERIFICATION_BATCH_ENABLED",
    "true",
).strip().lower() in {"1", "true", "yes", "on"}
VERIFICATION_BATCH_MAX_CLAIMS = int(
    os.getenv("VERIFICATION_BATCH_MAX_CLAIMS", "10")
)
VERIFICATION_BATCH_MAX_IMAGES_TOTAL = int(
    os.getenv("VERIFICATION_BATCH_MAX_IMAGES_TOTAL", "6")
)
VERIFICATION_BATCH_EVIDENCE_CHARS = int(
    os.getenv("VERIFICATION_BATCH_EVIDENCE_CHARS", "1200")
)

# ============================================================
# V2.3 MULTI-VLM (OpenAI primary + Gemini fallback)
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_VLM_MODEL = os.getenv("OPENAI_VLM_MODEL", "gpt-6-luna")
OPENAI_VLM_MODELS = os.getenv("OPENAI_VLM_MODELS", OPENAI_VLM_MODEL)
OPENAI_VERIFICATION_MODELS = os.getenv(
    "OPENAI_VERIFICATION_MODELS",
    OPENAI_VLM_MODELS,
)
OPENAI_TRANSCRIBE_MODEL = os.getenv(
    "OPENAI_TRANSCRIBE_MODEL",
    "gpt-4o-mini-transcribe",
)
VLM_PROVIDER_ORDER = os.getenv("VLM_PROVIDER_ORDER", "local,openai,gemini")
VLM_VERIFICATION_PROVIDER_ORDER = os.getenv(
    "VLM_VERIFICATION_PROVIDER_ORDER",
    VLM_PROVIDER_ORDER,
)
VLM_CIRCUIT_STATE_FILE = os.getenv("VLM_CIRCUIT_STATE_FILE", "")
OPENAI_VLM_TIMEOUT = float(os.getenv("OPENAI_VLM_TIMEOUT", "120"))
OPENAI_PHASE2_FRAME_COUNT = int(os.getenv("OPENAI_PHASE2_FRAME_COUNT", "10"))
OPENAI_PHASE2_TRANSCRIBE = os.getenv("OPENAI_PHASE2_TRANSCRIBE", "true")
OPENAI_IMAGE_DETAIL = os.getenv("OPENAI_IMAGE_DETAIL", "low")


# ============================================================
# V2.4 HYBRID QUOTA-INDEPENDENT EXECUTION
# ============================================================
VLM_CACHE_ENABLED = os.getenv("VLM_CACHE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
VLM_CACHE_DIR = os.getenv("VLM_CACHE_DIR", "runtime/vlm_cache")
VLM_QUOTA_STATE_FILE = os.getenv("VLM_QUOTA_STATE_FILE", "runtime/vlm_quota_state.json")
VLM_CHECKPOINT_DIR = os.getenv("VLM_CHECKPOINT_DIR", "runtime/checkpoints")
VLM_QUOTA_TOKEN_RESERVE = int(os.getenv("VLM_QUOTA_TOKEN_RESERVE", "1000"))
VLM_QUOTA_OBSERVATION_TTL_SECONDS = float(os.getenv("VLM_QUOTA_OBSERVATION_TTL_SECONDS", "120"))
VLM_TOKEN_ESTIMATE_CHARS_PER_TOKEN = float(os.getenv("VLM_TOKEN_ESTIMATE_CHARS_PER_TOKEN", "2.5"))

PHASE2_PROMPT_VERSION = os.getenv("PHASE2_PROMPT_VERSION", "v2.4.0-sampled-frames-local-stt")
PHASE4_PROMPT_VERSION = os.getenv("PHASE4_PROMPT_VERSION", "v2.4.0-external-grounding-compact")

VERIFICATION_REFINED_MAX_EVIDENCE = int(os.getenv("VERIFICATION_REFINED_MAX_EVIDENCE", "5"))
VERIFICATION_REFINED_MAX_TEXT = int(os.getenv("VERIFICATION_REFINED_MAX_TEXT", "4"))
VERIFICATION_REASONING_CHARS = int(os.getenv("VERIFICATION_REASONING_CHARS", "1600"))
VERIFICATION_SPEECH_CHARS = int(os.getenv("VERIFICATION_SPEECH_CHARS", "1600"))
VERIFICATION_OCR_CHARS = int(os.getenv("VERIFICATION_OCR_CHARS", "1000"))
VERIFICATION_MAX_VISUAL_OBSERVATIONS = int(os.getenv("VERIFICATION_MAX_VISUAL_OBSERVATIONS", "4"))
VERIFICATION_CONTEXT_PAD_SECONDS = float(os.getenv("VERIFICATION_CONTEXT_PAD_SECONDS", "20"))

GEMINI_VERIFICATION_MODELS = os.getenv(
    "GEMINI_VERIFICATION_MODELS",
    ",".join(x for x in (GEMINI_VISION_MODEL, GEMINI_FALLBACK_MODEL) if x),
)
GEMINI_PHASE4_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_PHASE4_MAX_OUTPUT_TOKENS", "1000"))

LOCAL_VLM_ENABLED = os.getenv("LOCAL_VLM_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
LOCAL_VLM_API_STYLE = os.getenv("LOCAL_VLM_API_STYLE", "ollama").strip().lower()
LOCAL_VLM_BASE_URL = os.getenv("LOCAL_VLM_BASE_URL", "http://127.0.0.1:11434").strip()
LOCAL_VLM_MODEL = os.getenv("LOCAL_VLM_MODEL", "qwen2.5vl:7b").strip()
LOCAL_VLM_API_KEY = os.getenv("LOCAL_VLM_API_KEY", "local").strip()
LOCAL_VLM_TIMEOUT = float(os.getenv("LOCAL_VLM_TIMEOUT", "180"))
LOCAL_PHASE2_MAX_OUTPUT_TOKENS = int(os.getenv("LOCAL_PHASE2_MAX_OUTPUT_TOKENS", "3000"))
LOCAL_PHASE4_MAX_OUTPUT_TOKENS = int(os.getenv("LOCAL_PHASE4_MAX_OUTPUT_TOKENS", "1000"))
