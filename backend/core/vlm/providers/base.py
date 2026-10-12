from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ProviderJSONResult:
    data: dict[str, Any]
    provider: str
    model: str
    cache_hit: bool = False


class JSONProvider(Protocol):
    provider_name: str

    def is_configured(self) -> bool: ...

    def generate_json(self, **kwargs: Any) -> ProviderJSONResult: ...
