from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Any, Callable

from PIL import Image, UnidentifiedImageError

from config import EVIDENCE_MAX_IMAGE_BYTES, EVIDENCE_REQUEST_TIMEOUT
from core.vlm import cache, circuit, quota_manager
from core.vlm.providers import gemini_provider, local_provider, openai_provider
from fact_checking.evidence.http_utils import read_limited_bytes, safe_get
from fact_checking.verification.local_component_guard import (
    apply_local_component_guard,
    component_prompt_block,
    decompose_claim,
)

_ALLOWED_VERDICTS = {"SUPPORTED", "REFUTED", "NEI"}
_ALLOWED_STANCES = {"support", "refute", "neutral", "context"}

_LOCAL_PHASE4_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["SUPPORTED", "REFUTED", "NEI"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence_coverage": {"type": "string", "enum": ["COMPLETE", "PARTIAL", "INSUFFICIENT"]},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "component_assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "component_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["SUPPORTED", "REFUTED", "INSUFFICIENT"]},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "rationale": {"type": "string"},
                },
                "required": ["component_id", "status", "evidence_ids", "confidence", "rationale"],
                "additionalProperties": False,
            },
        },
        "evidence_assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "evidence_id": {"type": "string"},
                    "stance": {"type": "string", "enum": ["support", "refute", "neutral", "context"]},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "rationale": {"type": "string"},
                },
                "required": ["evidence_id", "stance", "confidence", "rationale"],
                "additionalProperties": False,
            },
        },
        "rationale": {"type": "string"},
        "uncertainty": {"type": "string"},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "verdict", "confidence", "evidence_coverage", "evidence_ids",
        "component_assessments", "evidence_assessments", "rationale", "uncertainty", "limitations",
    ],
    "additionalProperties": False,
}


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            data = value.model_dump()
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _claim_text(claim: Any) -> str:
    data = _as_dict(claim)
    return str(data.get("text") or getattr(claim, "text", "") or "").strip()


def _evidence_prompt(
    evidence: list[Any],
) -> tuple[str, list[str], list[tuple[str, str]], set[str]]:
    chunks: list[str] = []
    ids: list[str] = []
    image_candidates: list[tuple[str, str]] = []
    image_ids: set[str] = set()
    max_chars = int(os.getenv("VERIFICATION_MAX_EVIDENCE_CHARS", "700"))
    max_images = int(os.getenv("VERIFICATION_MAX_IMAGES", "1"))

    for idx, raw in enumerate(evidence, start=1):
        item = _as_dict(raw)
        evidence_id = str(item.get("id") or item.get("evidence_id") or f"evidence_{idx}")
        modality = str(item.get("modality") or "text").lower()
        title = str(item.get("title") or "")
        url = str(item.get("source_url") or "")
        domain = str(item.get("source_domain") or "")
        content = str(item.get("content") or "")[:max_chars]
        published_at = item.get("published_at")
        source_quality = item.get("source_quality")

        ids.append(evidence_id)
        chunks.append(
            "\n".join(
                [
                    f"EVIDENCE_ID: {evidence_id}",
                    f"MODALITY: {modality}",
                    f"TITLE: {title}",
                    f"SOURCE: {domain}",
                    f"URL: {url}",
                    f"PUBLISHED_AT: {published_at}",
                    f"SOURCE_QUALITY_METADATA: {source_quality}",
                    f"CONTENT: {content}",
                ]
            )
        )

        if modality == "image":
            image_ids.add(evidence_id)
            media_url = str(item.get("media_url") or "").strip()
            if len(image_candidates) < max_images:
                image_candidates.append((evidence_id, media_url))

    return "\n\n---\n\n".join(chunks), ids, image_candidates, image_ids


def _image_to_data_url(image: Image.Image) -> str:
    max_side = max(256, int(os.getenv("OPENAI_PHASE4_IMAGE_MAX_SIDE", "960")))
    quality = max(50, min(95, int(os.getenv("OPENAI_PHASE4_IMAGE_JPEG_QUALITY", "75"))))
    normalized = image.convert("RGB")
    normalized.thumbnail((max_side, max_side))
    buffer = BytesIO()
    normalized.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _fetch_image_as_data_url(url: str) -> str:
    if not url:
        raise ValueError("missing media_url")
    timeout = float(os.getenv("OPENAI_PHASE4_IMAGE_FETCH_TIMEOUT", str(EVIDENCE_REQUEST_TIMEOUT)))
    max_bytes = int(os.getenv("OPENAI_PHASE4_IMAGE_MAX_BYTES", str(EVIDENCE_MAX_IMAGE_BYTES)))
    response = None
    try:
        response = safe_get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
            stream=True,
        )
        response.raise_for_status()
        content_type = str(response.headers.get("content-type") or "").lower()
        if content_type and not content_type.startswith("image/"):
            raise ValueError(f"unexpected content-type: {content_type}")
        raw = read_limited_bytes(response, max_bytes)
        with Image.open(BytesIO(raw)) as image:
            image.load()
            return _image_to_data_url(image)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise RuntimeError(f"image decode/validation failed: {exc}") from exc
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass


def _prepare_image_attachments(
    image_candidates: list[tuple[str, str]],
) -> tuple[list[str], set[str], list[str]]:
    data_urls: list[str] = []
    attached_ids: set[str] = set()
    failures: list[str] = []
    for evidence_id, media_url in image_candidates:
        try:
            data_urls.append(_fetch_image_as_data_url(media_url))
            attached_ids.add(evidence_id)
        except Exception as exc:
            failures.append(f"{evidence_id}: {type(exc).__name__}: {exc}")
    return data_urls, attached_ids, failures


def _multimedia_context(multimedia: Any) -> str:
    data = _as_dict(multimedia)
    visual = data.get("visual_observations") or []
    max_visual = max(0, int(os.getenv("VERIFICATION_MAX_VISUAL_OBSERVATIONS", "4")))
    speech_chars = max(0, int(os.getenv("VERIFICATION_SPEECH_CHARS", "1600")))
    ocr_chars = max(0, int(os.getenv("VERIFICATION_OCR_CHARS", "1000")))
    visual_text = []
    for item in visual[:max_visual]:
        if isinstance(item, dict):
            visual_text.append(f"[{item.get('timestamp')}] {item.get('fact')}")
    return "\n".join(
        [
            f"CAPTION: {data.get('caption') or ''}",
            f"SPEECH_TRANSCRIPT: {str(data.get('speech_transcript') or '')[:speech_chars]}",
            f"OCR_TEXT: {str(data.get('ocr_text') or '')[:ocr_chars]}",
            "VISUAL_OBSERVATIONS:\n" + "\n".join(visual_text),
        ]
    )


def _reasoning_text(reasoning_support: Any) -> str:
    data = _as_dict(reasoning_support)
    max_chars = max(0, int(os.getenv("VERIFICATION_REASONING_CHARS", "1600")))
    return str(data.get("reasoning_support") or "")[:max_chars]


def _normalize_verification(
    data: dict[str, Any],
    *,
    candidate_ids: list[str],
    model: str,
    provider: str,
    attached_image_ids: set[str],
) -> dict[str, Any]:
    verdict = str(data.get("verdict") or "").strip().upper()
    if verdict not in _ALLOWED_VERDICTS:
        raise ValueError(f"Invalid verdict from VLM: {verdict!r}")

    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    allowed = set(candidate_ids)
    evidence_ids = [str(x) for x in (data.get("evidence_ids") or []) if str(x) in allowed]

    assessments: list[dict[str, Any]] = []
    for item in data.get("evidence_assessments") or []:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id not in allowed:
            continue
        stance = str(item.get("stance") or "neutral").lower()
        if stance not in _ALLOWED_STANCES:
            stance = "neutral"
        try:
            item_conf = float(item.get("confidence", 0.0))
        except Exception:
            item_conf = 0.0
        assessments.append(
            {
                "evidence_id": evidence_id,
                "stance": stance,
                "confidence": max(0.0, min(1.0, item_conf)),
                "rationale": str(item.get("rationale") or "").strip(),
            }
        )

    limitations = [str(x).strip() for x in (data.get("limitations") or []) if str(x).strip()]
    coverage = str(data.get("evidence_coverage") or "").strip().upper()
    if coverage not in {"COMPLETE", "PARTIAL", "INSUFFICIENT"}:
        coverage = "INSUFFICIENT"

    # Research-hardening invariant: only external retrieved evidence can support/refute.
    if verdict == "SUPPORTED" and coverage != "COMPLETE":
        verdict = "NEI"
        confidence = min(confidence, 0.65)
        limitations.append(
            "SUPPORTED was downgraded to NEI because external retrieved evidence "
            "did not completely cover every material component of the claim."
        )

    if verdict in {"SUPPORTED", "REFUTED"} and not evidence_ids:
        verdict = "NEI"
        confidence = min(confidence, 0.55)
        coverage = "INSUFFICIENT"
        limitations.append("No external retrieved evidence item was cited for the proposed verdict.")

    multimodal_used = bool(set(evidence_ids) & set(attached_image_ids))

    return {
        "status": "SUCCESS",
        "verdict": verdict,
        "confidence": confidence,
        "evidence_ids": evidence_ids,
        "evidence_assessments": assessments,
        "evidence_coverage": coverage,
        "rationale": str(data.get("rationale") or "").strip(),
        "uncertainty": str(data.get("uncertainty") or "").strip(),
        "limitations": limitations,
        "multimodal_used": multimodal_used,
        "model": model,
        "provider": provider,
    }


def _build_request(
    *,
    claim: Any,
    multimedia: Any,
    evidence: list[Any],
    reasoning_support: Any,
    component_guard: bool = False,
) -> tuple[str, list[str], list[str], set[str], list[str], list[Any]]:
    evidence_text, all_candidate_ids, image_candidates, image_ids = _evidence_prompt(evidence)
    image_data_urls, attached_image_ids, image_transport_failures = _prepare_image_attachments(image_candidates)
    candidate_ids = [
        evidence_id
        for evidence_id in all_candidate_ids
        if evidence_id not in image_ids or evidence_id in attached_image_ids
    ]
    if not candidate_ids:
        raise RuntimeError("No usable evidence remained after image transport preparation")

    image_transport_note = ""
    if image_candidates:
        failed_ids = [item.split(":", 1)[0] for item in image_transport_failures]
        image_transport_note = "\n".join(
            [
                "# IMAGE ATTACHMENT STATUS",
                f"Attached image evidence IDs: {sorted(attached_image_ids) or '(none)'}",
                f"Unavailable image evidence IDs: {failed_ids or '(none)'}",
                "Do not infer visual facts from unavailable image evidence. Continue verification from remaining usable evidence.",
            ]
        )

    claim_text = _claim_text(claim)
    components = decompose_claim(claim_text) if component_guard else []
    reasoning_text = _reasoning_text(reasoning_support)
    context_text = _multimedia_context(multimedia)
    component_section = ""
    if component_guard:
        component_section = (
            "# MATERIAL CLAIM COMPONENTS\n"
            + component_prompt_block(components)
            + "\n\n# LOCAL COMPONENT COVERAGE RULES\n"
            + "- Evaluate EVERY listed component independently using external retrieved evidence only.\n"
            + "- For each component return SUPPORTED, REFUTED, or INSUFFICIENT.\n"
            + "- SUPPORTED/REFUTED component assessments MUST cite one or more valid EVIDENCE_ID values.\n"
            + "- Do not infer missing qualifiers such as only/unique, exact date, family relation, title, quantity, or causality unless external evidence directly establishes them.\n"
            + "- The backend will recompute final coverage/verdict deterministically; never assume partial evidence means COMPLETE.\n"
            + "- For qualifiers such as duy nhất/only, family relation, exact year/age/quantity, and phong thẳng/no intermediate rank, cite evidence that DIRECTLY states that qualifier/value.\n"
            + "- Keep each component rationale short (prefer <= 180 characters) and do not repeat full evidence text."
        )
    prompt = f"""
# ROLE
You are the Phase 4 evidence-grounded verifier for a multimodal fact-checking system.

# HARD RULES
- Decide truthfulness ONLY from EXTERNAL RETRIEVED EVIDENCE below.
- SOURCE MULTIMEDIA CONTEXT is provided only to interpret what the TikTok claim means, resolve references, and recover its time anchor. It is NOT evidence that the claim is true or false.
- Never support/refute a claim because the source video, caption, OCR, speech transcript, frame, uploader, or on-screen graphic repeats the same claim.
- SUPPORTED: external retrieved evidence completely confirms every material component of the atomic claim.
- REFUTED: external retrieved evidence contradicts the core proposition.
- NEI: evidence collection succeeded, but external retrieved evidence is incomplete or insufficient to support or refute.
- For conjunctions, lists, quantities, or multi-person claims, SUPPORTED requires external evidence covering ALL material components. Partial corroboration => NEI.
- A technical/API/retrieval failure is NEVER NEI. Technical failure is handled outside this prompt.
- Do not treat source_quality metadata as truth by itself.
- Respect dates/temporal anchors and distinguish historical vs current-state claims.
- evidence_ids MUST be selected only from the provided EVIDENCE_ID values.
- evidence_coverage MUST be COMPLETE, PARTIAL, or INSUFFICIENT and refers only to external retrieved evidence.
- Do not reveal private chain-of-thought. Return concise evidence-grounded rationales only.

# CLAIM
{claim_text}

# SOURCE MULTIMEDIA CONTEXT — INTERPRETATION ONLY, NOT TRUTH EVIDENCE
{context_text}

# CONCEPTUAL REASONING SUPPORT
{reasoning_text or '(none)'}

# RETRIEVED EVIDENCE
{evidence_text}

{image_transport_note}

{component_section}

# OUTPUT
Return ONLY a JSON object:
{{
  "verdict": "SUPPORTED | REFUTED | NEI",
  "confidence": 0.0,
  "evidence_coverage": "COMPLETE | PARTIAL | INSUFFICIENT",
  "evidence_ids": ["..."],
  "component_assessments": [
    {{
      "component_id": "C1",
      "status": "SUPPORTED | REFUTED | INSUFFICIENT",
      "evidence_ids": ["..."],
      "confidence": 0.0,
      "rationale": "short evidence-grounded rationale"
    }}
  ],
  "evidence_assessments": [
    {{
      "evidence_id": "...",
      "stance": "support | refute | neutral | context",
      "confidence": 0.0,
      "rationale": "short evidence-grounded rationale"
    }}
  ],
  "rationale": "concise explanation of why the evidence implies the verdict",
  "uncertainty": "remaining uncertainty, if any",
  "limitations": ["..."]
}}
""".strip()
    return prompt, candidate_ids, image_data_urls, attached_image_ids, image_transport_failures, components


def _cache_payload(claim: Any, multimedia: Any, evidence: list[Any], reasoning_support: Any) -> dict[str, Any]:
    mm = _as_dict(multimedia)
    return {
        "claim": _as_dict(claim),
        "evidence": [
            {
                "id": _as_dict(item).get("id"),
                "modality": _as_dict(item).get("modality"),
                "source_url": _as_dict(item).get("source_url"),
                "content": _as_dict(item).get("content"),
                "media_url": _as_dict(item).get("media_url"),
            }
            for item in evidence
        ],
        "multimedia": {
            "caption": mm.get("caption"),
            "speech_transcript": mm.get("speech_transcript"),
            "ocr_text": mm.get("ocr_text"),
            "visual_observations": mm.get("visual_observations") or [],
            "upload_date": (mm.get("metadata") or {}).get("upload_date") if isinstance(mm.get("metadata"), dict) else None,
        },
        "reasoning_support": _reasoning_text(reasoning_support),
    }


def _cached_result(
    *, provider: str, model: str, claim: Any, multimedia: Any, evidence: list[Any], reasoning_support: Any
) -> tuple[str, dict[str, Any] | None]:
    if provider == "local":
        prompt_version = os.getenv(
            "LOCAL_PHASE4_PROMPT_VERSION",
            "v2.4.5-local-evidence-qualifier-guard-v2",
        )
    else:
        prompt_version = os.getenv("PHASE4_PROMPT_VERSION", "v2.4.0-external-grounding-compact")
    key = cache.build_key(
        task="phase4_verification",
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        payload=_cache_payload(claim, multimedia, evidence, reasoning_support),
    )
    return key, cache.get_json(task="phase4_verification", key=key)


def _store_cache(key: str, result: dict[str, Any], *, provider: str, model: str) -> None:
    cache.set_json(
        task="phase4_verification",
        key=key,
        value=result,
        metadata={"provider": provider, "model": model},
    )


def _verify_provider_models(
    *,
    provider: str,
    models: list[str],
    call: Callable[..., Any],
    claim: Any,
    multimedia: Any,
    evidence: list[Any],
    reasoning_support: Any,
) -> dict[str, Any]:
    if not evidence:
        return {
            "status": "UNAVAILABLE",
            "verdict": None,
            "confidence": None,
            "evidence_ids": [],
            "evidence_assessments": [],
            "evidence_coverage": None,
            "rationale": "Không thể xác minh vì Evidence Retrieval không cung cấp evidence khả dụng.",
            "uncertainty": "INSUFFICIENT_COLLECTION",
            "limitations": ["No usable retrieved evidence."],
            "multimodal_used": False,
            "model": "",
            "provider": provider,
        }

    errors: list[str] = []
    # Cache is checked before circuit/preflight so a previous successful result can be reused offline.
    for model in models:
        key, cached = _cached_result(
            provider=provider,
            model=model,
            claim=claim,
            multimedia=multimedia,
            evidence=evidence,
            reasoning_support=reasoning_support,
        )
        if cached and str(cached.get("status") or "").upper() == "SUCCESS":
            cached = dict(cached)
            cached.setdefault("limitations", [])
            cached["limitations"] = list(cached.get("limitations") or []) + ["Phase 4 persistent cache hit."]
            return cached

    (
        prompt,
        candidate_ids,
        image_data_urls,
        attached_image_ids,
        image_transport_failures,
        components,
    ) = _build_request(
        claim=claim,
        multimedia=multimedia,
        evidence=evidence,
        reasoning_support=reasoning_support,
        component_guard=(provider == "local"),
    )

    for model in models:
        key, _ = _cached_result(
            provider=provider,
            model=model,
            claim=claim,
            multimedia=multimedia,
            evidence=evidence,
            reasoning_support=reasoning_support,
        )
        if circuit.is_open(provider, model):
            errors.append(f"{model}: circuit_open remaining~{circuit.remaining_seconds(provider, model):.0f}s")
            continue
        try:
            max_output = int(os.getenv("OPENAI_PHASE4_MAX_OUTPUT_TOKENS", "1000"))
            if provider == "gemini":
                # Gemini structured verification needs a little more completion
                # headroom than the OpenAI quota-safe profile. Keep provider
                # budgets independent so an OpenAI-specific 1000-token cap does
                # not truncate Gemini JSON.
                max_output = int(os.getenv("GEMINI_PHASE4_MAX_OUTPUT_TOKENS", "1600"))
            elif provider == "local":
                max_output = int(os.getenv("LOCAL_PHASE4_MAX_OUTPUT_TOKENS", "2600"))

            estimate = quota_manager.estimate_input_tokens(prompt, image_count=len(image_data_urls))
            decision = quota_manager.can_request(
                provider,
                model,
                estimated_input_tokens=estimate,
                max_output_tokens=max_output,
            )
            if not decision.allowed:
                errors.append(
                    f"{model}: quota_preflight_blocked reason={decision.reason} "
                    f"remaining={decision.remaining_tokens} retry~{decision.retry_after_seconds:.0f}s"
                )
                continue

            if provider == "openai":
                response = call(
                    model=model,
                    prompt=prompt,
                    image_urls=image_data_urls,
                    max_output_tokens=max_output,
                    timeout_seconds=float(os.getenv("OPENAI_VLM_TIMEOUT", "120")),
                    image_detail=os.getenv("OPENAI_IMAGE_DETAIL", "low"),
                )
            elif provider == "gemini":
                response = call(
                    model=model,
                    prompt=prompt,
                    image_urls=image_data_urls,
                    max_output_tokens=max_output,
                    timeout_seconds=float(os.getenv("GEMINI_VLM_TIMEOUT", "120")),
                )
            else:
                response = call(
                    model=model,
                    prompt=prompt,
                    image_urls=image_data_urls,
                    max_output_tokens=max_output,
                    timeout_seconds=float(os.getenv("LOCAL_VLM_TIMEOUT", "180")),
                    image_detail=os.getenv("LOCAL_VLM_IMAGE_DETAIL", "low"),
                    response_schema=_LOCAL_PHASE4_RESPONSE_SCHEMA,
                )

            raw_data = response.data if hasattr(response, "data") else response
            result = _normalize_verification(
                raw_data,
                candidate_ids=candidate_ids,
                model=model,
                provider=provider,
                attached_image_ids=attached_image_ids,
            )
            if provider == "local":
                result = apply_local_component_guard(
                    raw_data=raw_data,
                    base_result=result,
                    components=components,
                    candidate_ids=candidate_ids,
                    attached_image_ids=attached_image_ids,
                    evidence=evidence,
                )
            if image_transport_failures:
                result["limitations"] = list(result.get("limitations") or []) + [
                    f"{len(image_transport_failures)} image evidence item(s) could not be attached; verification continued with remaining evidence."
                ]
            _store_cache(key, result, provider=provider, model=model)
            return result
        except Exception as exc:
            errors.append(f"{model}: {type(exc).__name__}: {exc}")

    raise RuntimeError(f"{provider} Phase 4 failed: " + " | ".join(errors))


def _csv_models(name: str, fallback: str) -> list[str]:
    raw = os.getenv(name, fallback)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or [item.strip() for item in fallback.split(",") if item.strip()]


def verify_openai(*, claim: Any, multimedia: Any, evidence: list[Any], reasoning_support: Any) -> dict[str, Any]:
    if not openai_provider.is_configured():
        raise RuntimeError("OPENAI_API_KEY is not configured")
    models = _csv_models("OPENAI_VERIFICATION_MODELS", os.getenv("OPENAI_VLM_MODELS", "gpt-6-luna"))
    return _verify_provider_models(
        provider="openai",
        models=models,
        call=openai_provider.generate_json,
        claim=claim,
        multimedia=multimedia,
        evidence=evidence,
        reasoning_support=reasoning_support,
    )


def verify_gemini(*, claim: Any, multimedia: Any, evidence: list[Any], reasoning_support: Any) -> dict[str, Any]:
    if not gemini_provider.is_configured():
        raise RuntimeError("Gemini provider is not configured")
    fallback = ",".join(
        x for x in [
            os.getenv("GEMINI_VISION_MODEL", "gemini-3.8-flash"),
            os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.7-flash"),
        ] if x
    )
    models = _csv_models("GEMINI_VERIFICATION_MODELS", fallback)
    return _verify_provider_models(
        provider="gemini",
        models=models,
        call=gemini_provider.generate_json,
        claim=claim,
        multimedia=multimedia,
        evidence=evidence,
        reasoning_support=reasoning_support,
    )


def verify_local(*, claim: Any, multimedia: Any, evidence: list[Any], reasoning_support: Any) -> dict[str, Any]:
    if not local_provider.is_configured("phase4"):
        raise RuntimeError("Local VLM provider is not configured")
    model = local_provider.configured_model("phase4")
    return _verify_provider_models(
        provider="local",
        models=[model],
        call=local_provider.generate_json,
        claim=claim,
        multimedia=multimedia,
        evidence=evidence,
        reasoning_support=reasoning_support,
    )
