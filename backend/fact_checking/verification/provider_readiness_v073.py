from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from core.vlm import circuit, quota_manager
from core.vlm.providers import gemini_provider, local_provider, openai_provider


@dataclass(frozen=True)
class ModelReadiness:
    provider: str
    model: str
    ready: bool
    reason: str
    circuit_remaining_seconds: float = 0.0
    quota_block_remaining_seconds: float = 0.0


@dataclass(frozen=True)
class ProviderReadiness:
    provider: str
    configured: bool
    ready: bool
    reason: str
    models: tuple[ModelReadiness, ...]
    next_retry_after_seconds: float = 0.0


def _csv_models(name: str, fallback: str) -> list[str]:
    raw = os.getenv(name, fallback)
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    return values or [item.strip() for item in str(fallback or "").split(",") if item.strip()]


def provider_models(provider: str) -> list[str]:
    provider = str(provider or "").strip().lower()
    if provider == "openai":
        return _csv_models(
            "OPENAI_VERIFICATION_MODELS",
            os.getenv("OPENAI_VLM_MODELS", "gpt-6-luna"),
        )
    if provider == "gemini":
        fallback = ",".join(
            item
            for item in [
                os.getenv("GEMINI_VISION_MODEL", "gemini-3.8-flash"),
                os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.7-flash"),
            ]
            if item
        )
        return _csv_models("GEMINI_VERIFICATION_MODELS", fallback)
    if provider == "local":
        model = local_provider.configured_model("phase4")
        return [model] if model else []
    return []


def provider_configured(provider: str) -> bool:
    provider = str(provider or "").strip().lower()
    if provider == "openai":
        return bool(openai_provider.is_configured())
    if provider == "gemini":
        return bool(gemini_provider.is_configured())
    if provider == "local":
        return bool(local_provider.is_configured("phase4"))
    return False


def _quota_block_remaining(provider: str, model: str) -> float:
    snap = quota_manager.snapshot(provider, model)
    try:
        blocked_until = float(snap.get("blocked_until", 0) or 0)
    except Exception:
        return 0.0
    return max(0.0, blocked_until - time.time())


def model_readiness(provider: str, model: str) -> ModelReadiness:
    c_remain = max(0.0, float(circuit.remaining_seconds(provider, model)))
    q_remain = _quota_block_remaining(provider, model)
    if c_remain > 0:
        return ModelReadiness(
            provider=provider,
            model=model,
            ready=False,
            reason="circuit_open",
            circuit_remaining_seconds=c_remain,
            quota_block_remaining_seconds=q_remain,
        )
    if q_remain > 0:
        return ModelReadiness(
            provider=provider,
            model=model,
            ready=False,
            reason="quota_state_blocked",
            circuit_remaining_seconds=c_remain,
            quota_block_remaining_seconds=q_remain,
        )
    return ModelReadiness(
        provider=provider,
        model=model,
        ready=True,
        reason="ready",
        circuit_remaining_seconds=0.0,
        quota_block_remaining_seconds=0.0,
    )


def provider_readiness(provider: str) -> ProviderReadiness:
    provider = str(provider or "").strip().lower()
    if provider not in {"openai", "gemini", "local"}:
        return ProviderReadiness(
            provider=provider,
            configured=False,
            ready=False,
            reason="unsupported_provider",
            models=(),
            next_retry_after_seconds=0.0,
        )

    configured = provider_configured(provider)
    model_names = provider_models(provider)
    if not configured:
        models = tuple(
            ModelReadiness(
                provider=provider,
                model=model,
                ready=False,
                reason="provider_not_configured",
            )
            for model in model_names
        )
        return ProviderReadiness(
            provider=provider,
            configured=False,
            ready=False,
            reason="not_configured",
            models=models,
            next_retry_after_seconds=0.0,
        )
    models = tuple(model_readiness(provider, model) for model in model_names)
    if not models:
        return ProviderReadiness(
            provider=provider,
            configured=True,
            ready=False,
            reason="no_models_configured",
            models=(),
            next_retry_after_seconds=0.0,
        )
    if any(model.ready for model in models):
        return ProviderReadiness(
            provider=provider,
            configured=True,
            ready=True,
            reason="ready",
            models=models,
            next_retry_after_seconds=0.0,
        )

    waits = [
        max(model.circuit_remaining_seconds, model.quota_block_remaining_seconds)
        for model in models
        if max(model.circuit_remaining_seconds, model.quota_block_remaining_seconds) > 0
    ]
    return ProviderReadiness(
        provider=provider,
        configured=True,
        ready=False,
        reason="all_models_blocked",
        models=models,
        next_retry_after_seconds=min(waits) if waits else 0.0,
    )


def readiness_snapshot(order: Iterable[str]) -> list[dict[str, Any]]:
    return [asdict(provider_readiness(provider)) for provider in order]


def any_ready(order: Iterable[str]) -> bool:
    return any(provider_readiness(provider).ready for provider in order)


def min_retry_after_seconds(snapshot: Iterable[dict[str, Any]]) -> float:
    waits: list[float] = []
    for item in snapshot:
        try:
            wait = float(item.get("next_retry_after_seconds", 0) or 0)
        except Exception:
            continue
        if wait > 0:
            waits.append(wait)
    return min(waits) if waits else 0.0
