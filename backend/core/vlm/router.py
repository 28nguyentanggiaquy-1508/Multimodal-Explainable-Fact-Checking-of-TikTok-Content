from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from . import circuit
from .quota_manager import QuotaDecision, can_request


@dataclass(frozen=True)
class RouteCandidate:
    provider: str
    model: str


@dataclass(frozen=True)
class RouteDecision:
    candidate: RouteCandidate
    quota: QuotaDecision


def csv_order(raw: str | None, default: str) -> list[str]:
    values = [item.strip().lower() for item in str(raw or default).split(",") if item.strip()]
    return values or [item.strip().lower() for item in default.split(",") if item.strip()]


def available_candidates(
    candidates: Iterable[RouteCandidate],
    *,
    estimated_input_tokens: int,
    max_output_tokens: int,
) -> list[RouteDecision]:
    decisions: list[RouteDecision] = []
    for candidate in candidates:
        quota = can_request(
            candidate.provider,
            candidate.model,
            estimated_input_tokens=estimated_input_tokens,
            max_output_tokens=max_output_tokens,
        )
        if quota.allowed and not circuit.is_open(candidate.provider, candidate.model):
            decisions.append(RouteDecision(candidate=candidate, quota=quota))
    return decisions
