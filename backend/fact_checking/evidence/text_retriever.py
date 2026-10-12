from __future__ import annotations

from datetime import date, datetime, timedelta
import re
from functools import lru_cache
from urllib.parse import urlparse

import torch
import torch.nn.functional as F
from bs4 import BeautifulSoup
try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # Allows lightweight unit-test imports.
    AutoModel = None
    AutoTokenizer = None

from config import (
    EVIDENCE_MAX_PAGE_BYTES,
    EVIDENCE_REQUEST_TIMEOUT,
    TEXT_RETRIEVAL_MODEL,
    TEXT_RETRIEVAL_TOP_N,
    WEB_SEARCH_MAX_RESULTS,
    TEMPORAL_SEARCH_LOOKBACK_DAYS,
    TEMPORAL_STRICT_CURRENT_STATE,
)
from fact_checking.evidence.evidence_schema import (
    EvidenceItem,
    RetrievalChannelResult,
)
from fact_checking.evidence.http_utils import read_limited_bytes, safe_get
from fact_checking.evidence.search_provider import discover_web_sources


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _unit_score(cosine_score: float) -> float:
    return max(0.0, min(1.0, (float(cosine_score) + 1.0) / 2.0))


def _source_quality(domain: str) -> tuple[str, float]:
    value = (domain or "").lower().removeprefix("www.")
    if value.endswith(".gov.vn") or value.endswith(".gov"):
        return "official_government", 1.0
    if value.endswith(".edu.vn") or value.endswith(".edu"):
        return "academic", 0.95
    if value in {"qdnd.vn", "baochinhphu.vn", "vietnamplus.vn"}:
        return "official_or_public_news", 0.92
    if value in {"vietnamnet.vn", "vnexpress.net", "tuoitre.vn", "thanhnien.vn", "plo.vn"}:
        return "major_news", 0.82
    if value in {"thuvienphapluat.vn"}:
        return "legal_reference", 0.82
    if value.endswith("wikipedia.org"):
        return "encyclopedia", 0.68
    return "unclassified", 0.50


class TextEncoder:
    """Multilingual semantic encoder for claim/evidence retrieval."""

    def __init__(self):
        if AutoModel is None or AutoTokenizer is None:
            raise RuntimeError("transformers is not installed")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            "[Evidence/TextEncoder] "
            f"loading={TEXT_RETRIEVAL_MODEL} device={self.device}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_RETRIEVAL_MODEL)
        self.model = AutoModel.from_pretrained(TEXT_RETRIEVAL_MODEL).to(
            self.device
        ).eval()

    def encode(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 0), dtype=torch.float32)

        batches = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.inference_mode():
                outputs = self.model(**encoded)

            hidden = outputs.last_hidden_state
            mask = (
                encoded["attention_mask"]
                .unsqueeze(-1)
                .expand(hidden.size())
                .float()
            )
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = F.normalize(pooled, p=2, dim=1)
            batches.append(pooled.cpu())

        return torch.cat(batches, dim=0)


@lru_cache(maxsize=1)
def get_text_encoder() -> TextEncoder:
    return TextEncoder()


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _fetch_page_paragraphs(url: str) -> list[str]:
    try:
        response = safe_get(
            url,
            timeout=EVIDENCE_REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
            stream=True,
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if content_type and "text/html" not in content_type:
            response.close()
            return []

        raw = read_limited_bytes(response, EVIDENCE_MAX_PAGE_BYTES)
        encoding = response.encoding or "utf-8"
        response.close()

        soup = BeautifulSoup(raw.decode(encoding, errors="replace"), "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside", "noscript"]):
            tag.decompose()

        paragraphs = []
        seen = set()
        for paragraph in soup.find_all("p"):
            text = _clean_text(paragraph.get_text(" "))
            if len(text) < 40:
                continue
            text = text[:1500]
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            paragraphs.append(text)
            if len(paragraphs) >= 60:
                break

        return paragraphs
    except Exception as exc:
        print(
            "[Evidence/Text] page fetch failed "
            f"domain={(urlparse(url).hostname or '')} "
            f"error={type(exc).__name__}"
        )
        return []


def _score_candidates(claim: str, texts: list[str]) -> tuple[list[float], list[float]]:
    if not texts:
        return [], []

    encoder = get_text_encoder()
    embeddings = encoder.encode([claim] + texts)
    claim_embedding = embeddings[0:1]
    evidence_embeddings = embeddings[1:]
    raw_scores = torch.mm(evidence_embeddings, claim_embedding.T).squeeze(1)
    raw = [float(value) for value in raw_scores]
    normalized = [_unit_score(value) for value in raw]
    return raw, normalized


def _parse_page_date(value: str | None) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None



_DIRECTNESS_STOPWORDS = {
    "việt", "nam", "trong", "cùng", "hiện", "nay", "có", "là", "được",
    "theo", "vào", "của", "và", "quân", "đội", "nhân", "dân",
}


def _claim_directness(claim: str, content: str, temporal_mode: str = "none") -> float:
    """Heuristic directness score separate from semantic embedding similarity."""
    claim_cf = _clean_text(claim).casefold()
    content_cf = _clean_text(content).casefold()

    claim_tokens = {
        token.casefold()
        for token in re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", claim)
        if len(token) >= 3 and token.casefold() not in _DIRECTNESS_STOPWORDS
    }
    if claim_tokens:
        overlap = sum(1 for token in claim_tokens if token in content_cf) / len(claim_tokens)
    else:
        overlap = 0.5

    claim_numbers = set(re.findall(r"\b\d+\b", claim_cf))
    content_numbers = set(re.findall(r"\b\d+\b", content_cf))
    number_score = 1.0 if not claim_numbers else (
        1.0 if claim_numbers & content_numbers else 0.0
    )

    phrase_score = 1.0 if "đại tướng" in claim_cf and "đại tướng" in content_cf else 0.0

    if temporal_mode == "current_state":
        state_markers = ("công tác", "hiện có", "hiện nay", "đang", "tính đến", "đến nay")
        state_score = 1.0 if any(marker in content_cf for marker in state_markers) else 0.0
    else:
        state_score = 0.5

    score = (
        0.45 * overlap
        + 0.25 * number_score
        + 0.20 * state_score
        + 0.10 * phrase_score
    )
    return max(0.0, min(1.0, score))


def _temporal_features(
    *,
    page_age: str,
    content: str,
    temporal_anchor: date | None,
    time_sensitive: bool,
    temporal_mode: str = "none",
) -> tuple[float, str, str | None]:
    """Return (score, relation, parsed_date) for temporal evidence alignment."""
    if not time_sensitive or temporal_anchor is None:
        parsed = _parse_page_date(page_age)
        return 1.0, "not_required", parsed.isoformat() if parsed else None

    source_date = _parse_page_date(page_age)
    if source_date is None:
        return 0.78, "unknown", None

    delta = (source_date - temporal_anchor).days
    content_cf = (content or "").casefold()
    anchor_year_present = str(temporal_anchor.year) in content_cf
    present_markers = (
        "hiện nay", "hiện có", "đến nay", "tính đến nay", "currently", "as of today"
    )
    has_present_state = any(marker in content_cf for marker in present_markers)

    if delta <= 0:
        age_days = abs(delta)
        if age_days <= 30:
            score = 1.0
            relation = "near_before_anchor"
        elif age_days <= 180:
            score = 0.90
            relation = "before_anchor_180d"
        elif age_days <= 365:
            score = 0.76
            relation = "before_anchor_365d"
        elif age_days <= 730:
            score = 0.58
            relation = "before_anchor_2y"
        else:
            score = 0.36
            relation = "far_before_anchor"
    elif delta <= 30:
        score = 0.86
        relation = "after_anchor_30d"
    elif delta <= 180:
        score = 0.67
        relation = "after_anchor_180d"
    elif delta <= 365:
        score = 0.52
        relation = "after_anchor_365d"
    else:
        score = 0.35
        relation = "far_after_anchor"

    # Retrospective articles can still be useful, but a later page that speaks
    # in current-state terms must not overwrite the state that existed at the
    # TikTok upload timestamp.
    if temporal_mode == "current_state" and delta > 0:
        if has_present_state:
            score = min(score, 0.12)
        elif anchor_year_present:
            score = min(score, 0.40)
        else:
            score = min(score, 0.25)
    else:
        if anchor_year_present:
            score = min(0.90, score + 0.18)
        if has_present_state and abs(delta) > 180 and not anchor_year_present:
            score = max(0.15, score - 0.25)

    return score, relation, source_date.isoformat()


def retrieve_text_evidence(
    claim: str,
    top_n: int | None = None,
    *,
    temporal_anchor: date | None = None,
    time_sensitive: bool = False,
    temporal_mode: str = "none",
) -> RetrievalChannelResult:
    top_n = top_n or TEXT_RETRIEVAL_TOP_N
    search_query = claim
    if time_sensitive and temporal_anchor is not None:
        # Query enrichment is intentionally soft: later retrospective sources
        # are still allowed and are handled by the temporal score below.
        search_query = f"{claim} {temporal_anchor.year}"

    print(f"[Evidence/Text] query={search_query!r}")
    if time_sensitive and temporal_anchor is not None:
        print(f"[Evidence/Text] temporal_anchor={temporal_anchor.isoformat()}")

    freshness = None
    if (
        temporal_mode == "current_state"
        and temporal_anchor is not None
        and TEMPORAL_STRICT_CURRENT_STATE
    ):
        start = temporal_anchor - timedelta(days=max(30, TEMPORAL_SEARCH_LOOKBACK_DAYS))
        freshness = f"{start.isoformat()}to{temporal_anchor.isoformat()}"
        print(f"[Evidence/Text] brave_freshness={freshness}")

    discovery = discover_web_sources(
        search_query,
        max_results=WEB_SEARCH_MAX_RESULTS,
        freshness=freshness,
    )

    if discovery.status in {"UNAVAILABLE", "FAILED"}:
        return RetrievalChannelResult(
            modality="text",
            status=discovery.status,
            provider_attempts=discovery.attempts,
            errors=[
                attempt.error_message
                for attempt in discovery.attempts
                if attempt.error_message
            ],
        )

    search_results = discovery.results
    print(f"[Evidence/Text] search_sources={len(search_results)}")

    if not search_results:
        return RetrievalChannelResult(
            modality="text",
            status="SUCCESS",
            provider_attempts=discovery.attempts,
            search_source_count=0,
            candidate_count=0,
        )

    candidates = []
    seen_content = set()

    for source_rank, result in enumerate(search_results, start=1):
        source_url = result.url
        title = _clean_text(result.title)
        snippet = _clean_text(result.snippet)
        paragraphs = _fetch_page_paragraphs(source_url)

        if not paragraphs and snippet:
            paragraphs = [snippet]

        domain = (urlparse(source_url).hostname or "").lower()

        for paragraph_index, paragraph in enumerate(paragraphs):
            key = paragraph.casefold()
            if key in seen_content:
                continue
            seen_content.add(key)

            candidates.append(
                {
                    "id": f"text_{source_rank:02d}_{paragraph_index:03d}",
                    "title": title,
                    "source_url": source_url,
                    "source_domain": domain,
                    "content": paragraph,
                    "source_rank": source_rank,
                    "paragraph_index": paragraph_index,
                    "search_provider": result.provider,
                    "page_age": result.page_age,
                }
            )

            if len(candidates) >= 240:
                break
        if len(candidates) >= 240:
            break

    print(f"[Evidence/Text] candidate_passages={len(candidates)}")

    if not candidates:
        return RetrievalChannelResult(
            modality="text",
            status="DEGRADED",
            provider_attempts=discovery.attempts,
            search_source_count=len(search_results),
            candidate_count=0,
            errors=["Search succeeded but no attributable text passage could be extracted."],
        )

    try:
        raw_scores, semantic_scores = _score_candidates(
            claim,
            [candidate["content"] for candidate in candidates],
        )
    except Exception as exc:
        return RetrievalChannelResult(
            modality="text",
            status="FAILED",
            provider_attempts=discovery.attempts,
            search_source_count=len(search_results),
            candidate_count=len(candidates),
            errors=[f"Text encoder error: {type(exc).__name__}: {exc}"],
        )

    evidence = []
    for candidate, raw_score, semantic_score in zip(candidates, raw_scores, semantic_scores):
        quality_label, quality_score = _source_quality(candidate["source_domain"])
        temporal_score, temporal_relation, source_date = _temporal_features(
            page_age=candidate["page_age"],
            content=candidate["content"],
            temporal_anchor=temporal_anchor,
            time_sensitive=time_sensitive,
            temporal_mode=temporal_mode,
        )
        directness_score = _claim_directness(
            claim,
            candidate["content"],
            temporal_mode=temporal_mode,
        )

        # Current-state claims require stronger temporal/directness weighting.
        if temporal_mode == "current_state":
            relevance_score = (
                0.52 * semantic_score
                + 0.24 * directness_score
                + 0.16 * temporal_score
                + 0.08 * quality_score
            )
        else:
            relevance_score = (
                0.62 * semantic_score
                + 0.16 * directness_score
                + 0.12 * temporal_score
                + 0.10 * quality_score
            )

        evidence.append(
            EvidenceItem(
                id=candidate["id"],
                modality="text",
                title=candidate["title"],
                source_url=candidate["source_url"],
                source_domain=candidate["source_domain"],
                content=candidate["content"],
                retrieval_score=semantic_score,
                relevance_score=max(0.0, min(1.0, relevance_score)),
                provider=candidate["search_provider"],
                source_quality=quality_label,
                metadata={
                    "retrieval_method": "multilingual_transformer_cosine",
                    "raw_cosine": round(raw_score, 6),
                    "semantic_score": round(semantic_score, 6),
                    "directness_score": round(directness_score, 4),
                    "source_quality_score": round(quality_score, 4),
                    "source_rank": candidate["source_rank"],
                    "paragraph_index": candidate["paragraph_index"],
                    "page_age": candidate["page_age"],
                    "source_date": source_date,
                    "time_sensitive_claim": bool(time_sensitive),
                    "temporal_mode": temporal_mode,
                    "temporal_anchor": temporal_anchor.isoformat() if temporal_anchor else None,
                    "temporal_relation": temporal_relation,
                    "temporal_score": round(temporal_score, 4),
                    "temporal_eligible": not (
                        temporal_mode == "current_state"
                        and source_date is not None
                        and temporal_anchor is not None
                        and source_date > temporal_anchor.isoformat()
                    ),
                },
            )
        )

    evidence.sort(key=lambda item: item.relevance_score, reverse=True)
    selected = evidence[:top_n]
    print(f"[Evidence/Text] selected={len(selected)}")

    return RetrievalChannelResult(
        modality="text",
        status="SUCCESS",
        evidence=selected,
        provider_attempts=discovery.attempts,
        search_source_count=len(search_results),
        candidate_count=len(candidates),
    )
