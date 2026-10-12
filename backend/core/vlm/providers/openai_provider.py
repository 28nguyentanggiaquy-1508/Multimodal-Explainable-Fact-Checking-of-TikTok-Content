from __future__ import annotations

from pathlib import Path
from typing import Iterable

from core.vlm.openai_http import generate_json as _generate_json
from core.vlm.openai_http import is_configured

from .base import ProviderJSONResult

provider_name = "openai"


def generate_json(
    *,
    model: str,
    prompt: str,
    image_paths: Iterable[str | Path] | None = None,
    image_urls: Iterable[str] | None = None,
    max_output_tokens: int,
    timeout_seconds: float,
    image_detail: str = "low",
) -> ProviderJSONResult:
    data = _generate_json(
        model=model,
        prompt=prompt,
        image_paths=image_paths,
        image_urls=image_urls,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        image_detail=image_detail,
    )
    return ProviderJSONResult(data=data, provider=provider_name, model=model)
