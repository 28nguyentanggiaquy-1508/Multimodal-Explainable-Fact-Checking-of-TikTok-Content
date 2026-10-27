from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


MAX_EXPANDED_QUERIES = 4

_STOPWORDS = {
    "là", "được", "của", "và", "vào", "năm", "mới", "khi", "tại", "thời",
    "điểm", "trong", "quân", "đội", "nhân", "dân", "việt", "nam", "có",
    "không", "qua", "các", "cấp", "mà", "thứ", "vị", "đầu", "tiên",
}

_RELATION_VARIANTS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\bsinh năm\b", re.I), ("tiểu sử", "ngày sinh")),
    (re.compile(r"\b(mất|qua đời) năm\b", re.I), ("qua đời", "tiểu sử")),
    (re.compile(r"\bthụ phong quân hàm\b", re.I), ("phong quân hàm", "được phong hàm")),
    (re.compile(r"\bphong hàm đại tướng\b", re.I), ("phong quân hàm đại tướng", "được phong đại tướng")),
    (re.compile(r"\bphong quân hàm đại tướng\b", re.I), ("phong hàm đại tướng", "được phong đại tướng")),
    (re.compile(r"\btổng tư lệnh duy nhất\b", re.I), ("tổng tư lệnh", "tổng chỉ huy")),
    (re.compile(r"\bthân sinh\b", re.I), ("cha của", "thân phụ")),
)


@dataclass(frozen=True)
class QueryVariant:
    query: str
    strategy: str


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip(" .,:;!?\t\r\n")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", text or "")


def _entity_anchor(claim: str) -> str:
    """Best-effort Vietnamese named-entity anchor without an external NER model."""
    words = _tokens(claim)
    sequences: list[list[str]] = []
    current: list[str] = []
    for word in words:
        first = word[0] if word else ""
        is_name_like = first.isupper() or word.isupper()
        if is_name_like and not word.isdigit():
            current.append(word)
        else:
            if len(current) >= 2:
                sequences.append(current)
            current = []
    if len(current) >= 2:
        sequences.append(current)
    if not sequences:
        return ""
    # Prefer the longest sequence; ties preserve left-most occurrence.
    best = max(sequences, key=len)
    return " ".join(best[:6])


def _keyword_query(claim: str) -> str:
    words = _tokens(claim)
    kept: list[str] = []
    for word in words:
        cf = word.casefold()
        if word.isdigit() or len(word) >= 3 and cf not in _STOPWORDS:
            kept.append(word)
    return _clean(" ".join(kept[:14]))


def _relation_queries(claim: str, entity: str) -> Iterable[str]:
    for pattern, replacements in _RELATION_VARIANTS:
        if not pattern.search(claim):
            continue
        for replacement in replacements:
            transformed = pattern.sub(replacement, claim)
            yield _clean(transformed)
            if entity:
                numbers = " ".join(re.findall(r"\b\d{2,4}\b", claim))
                yield _clean(f"{entity} {replacement} {numbers}")


def generate_query_variants(claim: str, *, max_queries: int = MAX_EXPANDED_QUERIES) -> list[QueryVariant]:
    """Generate deterministic, Gold-blind Vietnamese search queries.

    The first query is always the original claim so the candidate system is a
    strict superset of the current baseline. No Gold URL/domain is ever used.
    """
    claim = _clean(claim)
    if not claim:
        return []

    candidates: list[QueryVariant] = [QueryVariant(claim, "BASE_CLAIM")]
    entity = _entity_anchor(claim)

    keyword = _keyword_query(claim)
    if keyword and keyword.casefold() != claim.casefold():
        candidates.append(QueryVariant(keyword, "KEYWORD_COMPRESSION"))

    for query in _relation_queries(claim, entity):
        if query:
            candidates.append(QueryVariant(query, "RELATION_SYNONYM"))

    if entity:
        numbers = " ".join(re.findall(r"\b\d{2,4}\b", claim))
        suffix = " ".join(
            word for word in _tokens(claim)
            if word.casefold() not in _STOPWORDS and word.casefold() not in entity.casefold()
        )
        entity_query = _clean(f"{entity} {suffix} {numbers}")
        if entity_query:
            candidates.append(QueryVariant(entity_query, "ENTITY_ANCHOR"))

    unique: list[QueryVariant] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.query.casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= max(1, int(max_queries)):
            break
    return unique
