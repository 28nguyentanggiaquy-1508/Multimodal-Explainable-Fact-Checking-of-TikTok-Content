"""Offline configuration/dependency checks for the thesis fact-check pipeline."""
from __future__ import annotations

import importlib.util

from config import (
    BRAVE_SEARCH_API_KEY,
    EVIDENCE_SEARCH_PROVIDER,
    GEMINI_API_KEY,
    IMAGE_RETRIEVAL_MODEL,
    TEXT_RETRIEVAL_MODEL,
)


def _module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    rows = {
        "gemini_api_key_configured": bool(GEMINI_API_KEY),
        "brave_api_key_configured": bool(BRAVE_SEARCH_API_KEY),
        "search_provider": EVIDENCE_SEARCH_PROVIDER,
        "google_genai_installed": _module("google.genai"),
        "transformers_installed": _module("transformers"),
        "torch_installed": _module("torch"),
        "yt_dlp_installed": _module("yt_dlp"),
        "beautifulsoup4_installed": _module("bs4"),
        "pillow_installed": _module("PIL"),
        "text_retrieval_model": TEXT_RETRIEVAL_MODEL,
        "image_retrieval_model": IMAGE_RETRIEVAL_MODEL,
    }

    print("=== Fact-check Thesis Environment Doctor ===")
    for key, value in rows.items():
        print(f"{key}: {value}")

    if not rows["gemini_api_key_configured"]:
        print("WARN: GEMINI_API_KEY missing -> Phase 2/4 will be unavailable/degraded.")
    if EVIDENCE_SEARCH_PROVIDER in {"brave", "brave_then_gemini"} and not rows["brave_api_key_configured"]:
        print("WARN: BRAVE_SEARCH_API_KEY missing -> image retrieval unavailable and web retrieval may rely on Gemini fallback.")


if __name__ == "__main__":
    main()
