from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import requests
try:
    from google import genai
    from google.genai import types
except ImportError:  # Allows offline/unit-test imports.
    genai = None
    types = None

from config import (
    BRAVE_SEARCH_API_KEY,
    BRAVE_SEARCH_COUNTRY,
    BRAVE_SEARCH_LANG,
    EVIDENCE_REQUEST_TIMEOUT,
    EVIDENCE_SEARCH_PROVIDER,
    GEMINI_API_KEY,
    GEMINI_SEARCH_MODEL,
)
from core.gemini_utils import generate_content_with_retry
from fact_checking.evidence.evidence_schema import ProviderAttempt
from fact_checking.schemas import StageStatus


BRAVE_WEB_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_IMAGE_ENDPOINT = "https://api.search.brave.com/res/v1/images/search"


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str = ""
    provider: str = ""
    page_age: str = ""


@dataclass
class ImageSearchResult:
    title: str
    source_url: str
    image_url: str
    thumbnail_url: str = ""
    provider: str = ""


T = TypeVar("T")


@dataclass
class DiscoveryResult(Generic[T]):
    status: StageStatus
    results: list[T] = field(default_factory=list)
    attempts: list[ProviderAttempt] = field(default_factory=list)


_GEMINI_SEARCH_CLIENT = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _safe_error_message(exc: Exception) -> str:
    # Do not serialize request headers or secrets. Brave API keys are sent in
    # headers, and Gemini exceptions do not need request payloads in diagnostics.
    text = str(exc).replace("\n", " ").strip()
    return text[:700]


def _attempt(
    provider: str,
    status: StageStatus,
    *,
    result_count: int = 0,
    exc: Exception | None = None,
) -> ProviderAttempt:
    return ProviderAttempt(
        provider=provider,
        status=status,
        result_count=result_count,
        error_type=type(exc).__name__ if exc else None,
        error_message=_safe_error_message(exc) if exc else None,
    )


def _brave_request(
    endpoint: str,
    *,
    query: str,
    count: int,
    image_search: bool = False,
    freshness: str | None = None,
) -> requests.Response:
    """
    Gọi Brave Search theo kiểu resilient.

    Lần 1:
        dùng country/search_lang cấu hình.

    Nếu Brave trả 422 (request validation):
        retry đúng 1 lần với bộ tham số tối thiểu.
        Việc này tránh toàn bộ retrieval chết chỉ vì một locale code
        không nằm trong enum Brave hiện hỗ trợ.

    Không retry 401/403/429 vì đó là auth/quota/product errors.
    """
    base_params: dict[str, Any] = {
        "q": query,
        "count": count,
    }

    optional_params: dict[str, Any] = {}

    country = (BRAVE_SEARCH_COUNTRY or "").strip().upper()
    search_lang = (BRAVE_SEARCH_LANG or "").strip().lower()

    if country:
        optional_params["country"] = country

    if search_lang:
        optional_params["search_lang"] = search_lang

    if image_search:
        optional_params["safesearch"] = "strict"
    else:
        optional_params["extra_snippets"] = "true"
        if freshness:
            optional_params["freshness"] = freshness

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_SEARCH_API_KEY,
    }

    response = requests.get(
        endpoint,
        headers=headers,
        params={**base_params, **optional_params},
        timeout=EVIDENCE_REQUEST_TIMEOUT,
    )

    # 422 ở Brave thường là validation của optional query params.
    # Retry tối thiểu để provider vẫn hoạt động thay vì fallback sai sang
    # Gemini chỉ vì locale/country không được API hỗ trợ.
    if response.status_code == 422:
        try:
            detail = response.text[:500].replace("\\n", " ")
        except Exception:
            detail = ""

        print(
            "[Brave] 422 validation; retrying with minimal params. "
            f"country={country!r} search_lang={search_lang!r} detail={detail}"
        )

        minimal_params = dict(base_params)

        if image_search:
            minimal_params["safesearch"] = "strict"
        elif freshness:
            # Freshness is a valid Brave Web Search parameter and is retained
            # when retrying without locale parameters.
            minimal_params["freshness"] = freshness

        response = requests.get(
            endpoint,
            headers=headers,
            params=minimal_params,
            timeout=EVIDENCE_REQUEST_TIMEOUT,
        )

    response.raise_for_status()
    return response


def _brave_web_search(query: str, max_results: int, freshness: str | None = None) -> list[WebSearchResult]:
    count = max(1, min(int(max_results), 20))

    response = _brave_request(
        BRAVE_WEB_ENDPOINT,
        query=query,
        count=count,
        image_search=False,
        freshness=freshness,
    )

    raw_results = response.json().get("web", {}).get("results", [])
    results: list[WebSearchResult] = []

    for item in raw_results:
        url = _clean(item.get("url"))
        if not url:
            continue

        snippet_parts = []
        description = _clean(item.get("description"))
        if description:
            snippet_parts.append(description)

        for extra in (item.get("extra_snippets") or [])[:3]:
            value = _clean(extra)
            if value:
                snippet_parts.append(value)

        results.append(
            WebSearchResult(
                title=_clean(item.get("title")),
                url=url,
                snippet=" ".join(snippet_parts).strip(),
                provider="brave_web",
                page_age=_clean(item.get("page_age")),
            )
        )

    return results[:count]


def _brave_image_search(query: str, max_results: int) -> list[ImageSearchResult]:
    count = max(1, min(int(max_results), 50))

    response = _brave_request(
        BRAVE_IMAGE_ENDPOINT,
        query=query,
        count=count,
        image_search=True,
    )

    results: list[ImageSearchResult] = []

    for item in response.json().get("results") or []:
        properties = item.get("properties") or {}
        thumbnail = item.get("thumbnail") or {}

        source_url = _clean(item.get("url"))
        image_url = _clean(properties.get("url"))
        thumbnail_url = _clean(thumbnail.get("src"))
        usable_image_url = image_url or thumbnail_url

        if not usable_image_url:
            continue

        results.append(
            ImageSearchResult(
                title=_clean(item.get("title")),
                source_url=source_url,
                image_url=usable_image_url,
                thumbnail_url=thumbnail_url,
                provider="brave_image",
            )
        )

    return results[:count]


def _grounding_metadata(response):
    for candidate in getattr(response, "candidates", None) or []:
        metadata = getattr(candidate, "grounding_metadata", None)
        if metadata is not None:
            return metadata
    return None


def _gemini_web_search(query: str, max_results: int) -> list[WebSearchResult]:
    if _GEMINI_SEARCH_CLIENT is None:
        return []

    prompt = f"""
Bạn đang thực hiện SOURCE DISCOVERY cho một hệ thống fact-checking dựa trên bằng chứng.

CLAIM:
{query}

Mục tiêu duy nhất:
- tìm nguồn web thực có liên quan trực tiếp đến claim;
- ưu tiên cơ quan chính thức, văn bản, tổ chức uy tín và nguồn báo chí có nội dung kiểm chứng cụ thể.

Không đưa ra verdict. Không tạo nguồn. Chỉ dùng Google Search grounding để tìm nguồn.
""".strip()

    response, model_used = generate_content_with_retry(
        _GEMINI_SEARCH_CLIENT,
        model=GEMINI_SEARCH_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        ),
    )

    metadata = _grounding_metadata(response)
    if metadata is None:
        return []

    chunks = getattr(metadata, "grounding_chunks", None) or []
    supports = getattr(metadata, "grounding_supports", None) or []
    support_text_by_chunk: dict[int, list[str]] = defaultdict(list)

    for support in supports:
        segment = getattr(support, "segment", None)
        segment_text = _clean(getattr(segment, "text", ""))
        indices = getattr(support, "grounding_chunk_indices", None) or []

        if not segment_text:
            continue

        for index in indices:
            try:
                support_text_by_chunk[int(index)].append(segment_text)
            except (TypeError, ValueError):
                continue

    results: list[WebSearchResult] = []
    seen_urls = set()

    for index, chunk in enumerate(chunks):
        web = getattr(chunk, "web", None)
        if web is None:
            continue

        url = _clean(getattr(web, "uri", ""))
        title = _clean(getattr(web, "title", ""))

        if not url or url in seen_urls:
            continue

        seen_urls.add(url)
        results.append(
            WebSearchResult(
                title=title,
                url=url,
                snippet=" ".join(
                    support_text_by_chunk.get(index, [])
                ).strip(),
                provider=f"gemini_google_search:{model_used}",
            )
        )

        if len(results) >= max_results:
            break

    return results


def discover_web_sources(
    query: str,
    max_results: int,
    *,
    freshness: str | None = None,
) -> DiscoveryResult[WebSearchResult]:
    """Provider-independent web discovery with explicit operational status."""
    provider = (EVIDENCE_SEARCH_PROVIDER or "brave_then_gemini").strip().lower()
    attempts: list[ProviderAttempt] = []
    any_provider_configured = False
    any_successful_call = False

    if provider not in {"brave", "gemini", "brave_then_gemini"}:
        return DiscoveryResult(
            status="FAILED",
            attempts=[
                ProviderAttempt(
                    provider=provider,
                    status="FAILED",
                    error_type="ConfigurationError",
                    error_message="Unsupported EVIDENCE_SEARCH_PROVIDER",
                )
            ],
        )

    if provider in {"brave", "brave_then_gemini"}:
        if BRAVE_SEARCH_API_KEY:
            any_provider_configured = True
            try:
                results = _brave_web_search(query, max_results, freshness=freshness)
                any_successful_call = True
                attempts.append(_attempt("brave_web", "SUCCESS", result_count=len(results)))
                if results or provider == "brave":
                    return DiscoveryResult(status="SUCCESS", results=results, attempts=attempts)
            except Exception as exc:
                attempts.append(_attempt("brave_web", "UNAVAILABLE", exc=exc))
        else:
            attempts.append(
                ProviderAttempt(
                    provider="brave_web",
                    status="SKIPPED",
                    error_type="MissingCredential",
                    error_message="BRAVE_SEARCH_API_KEY is not configured",
                )
            )

    if provider in {"gemini", "brave_then_gemini"}:
        if _GEMINI_SEARCH_CLIENT is not None:
            any_provider_configured = True
            try:
                results = _gemini_web_search(query, max_results)
                any_successful_call = True
                attempts.append(
                    _attempt("gemini_google_search", "SUCCESS", result_count=len(results))
                )
                return DiscoveryResult(status="SUCCESS", results=results, attempts=attempts)
            except Exception as exc:
                attempts.append(_attempt("gemini_google_search", "UNAVAILABLE", exc=exc))
        else:
            attempts.append(
                ProviderAttempt(
                    provider="gemini_google_search",
                    status="SKIPPED",
                    error_type="MissingCredential",
                    error_message="GEMINI_API_KEY is not configured",
                )
            )

    if any_successful_call:
        return DiscoveryResult(status="SUCCESS", results=[], attempts=attempts)

    return DiscoveryResult(
        status="UNAVAILABLE" if any_provider_configured else "UNAVAILABLE",
        results=[],
        attempts=attempts,
    )


def discover_image_sources(query: str, max_results: int) -> DiscoveryResult[ImageSearchResult]:
    """
    Image discovery currently uses Brave Image Search. This is an engineering
    provider choice; CLIP-based image relevance remains the research retrieval layer.
    """
    provider = (EVIDENCE_SEARCH_PROVIDER or "brave_then_gemini").strip().lower()

    if provider not in {"brave", "brave_then_gemini"}:
        return DiscoveryResult(
            status="UNAVAILABLE",
            attempts=[
                ProviderAttempt(
                    provider="brave_image",
                    status="SKIPPED",
                    error_type="ProviderDisabled",
                    error_message="Image discovery is not enabled for the selected provider mode",
                )
            ],
        )

    if not BRAVE_SEARCH_API_KEY:
        return DiscoveryResult(
            status="UNAVAILABLE",
            attempts=[
                ProviderAttempt(
                    provider="brave_image",
                    status="SKIPPED",
                    error_type="MissingCredential",
                    error_message="BRAVE_SEARCH_API_KEY is not configured",
                )
            ],
        )

    try:
        results = _brave_image_search(query, max_results)
        return DiscoveryResult(
            status="SUCCESS",
            results=results,
            attempts=[_attempt("brave_image", "SUCCESS", result_count=len(results))],
        )
    except Exception as exc:
        return DiscoveryResult(
            status="UNAVAILABLE",
            attempts=[_attempt("brave_image", "UNAVAILABLE", exc=exc)],
        )
