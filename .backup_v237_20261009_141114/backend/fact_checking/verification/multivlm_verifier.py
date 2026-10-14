from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Any

from PIL import Image, UnidentifiedImageError

from config import EVIDENCE_MAX_IMAGE_BYTES, EVIDENCE_REQUEST_TIMEOUT
from fact_checking.evidence.http_utils import read_limited_bytes, safe_get
from core.vlm import circuit
from core.vlm.openai_http import generate_json, is_configured


_ALLOWED_VERDICTS = {"SUPPORTED", "REFUTED", "NEI"}
_ALLOWED_STANCES = {"support", "refute", "neutral", "context"}


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
    """
    Build the text prompt and collect image evidence references.

    Returns:
        prompt_text,
        all_candidate_ids,
        image_candidates[(evidence_id, media_url)],
        image_evidence_ids
    """
    chunks: list[str] = []
    ids: list[str] = []
    image_candidates: list[tuple[str, str]] = []
    image_ids: set[str] = set()
    max_chars = int(os.getenv("VERIFICATION_MAX_EVIDENCE_CHARS", "1800"))
    max_images = int(os.getenv("VERIFICATION_MAX_IMAGES", "3"))

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
    """Normalize a remotely fetched image to a compact JPEG data URL."""
    max_side = max(256, int(os.getenv("OPENAI_PHASE4_IMAGE_MAX_SIDE", "1280")))
    quality = max(50, min(95, int(os.getenv("OPENAI_PHASE4_IMAGE_JPEG_QUALITY", "82"))))

    normalized = image.convert("RGB")
    normalized.thumbnail((max_side, max_side))

    buffer = BytesIO()
    normalized.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _fetch_image_as_data_url(url: str) -> str:
    """
    Fetch image server-side using the project's guarded HTTP helper.

    This avoids asking the OpenAI API to download third-party CDN URLs itself,
    which is unreliable for slow/blocked/redirecting image hosts.
    """
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
    """
    Materialize image evidence to data URLs.

    One failed image must not fail the whole claim. The verifier continues with
    successfully attached images and/or text evidence.
    """
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
    visual_text = []
    for item in visual[:8]:
        if isinstance(item, dict):
            visual_text.append(f"[{item.get('timestamp')}] {item.get('fact')}")
    return "\n".join(
        [
            f"CAPTION: {data.get('caption') or ''}",
            f"SPEECH_TRANSCRIPT: {str(data.get('speech_transcript') or '')[:5000]}",
            f"OCR_TEXT: {str(data.get('ocr_text') or '')[:3500]}",
            "VISUAL_OBSERVATIONS:\n" + "\n".join(visual_text),
        ]
    )


def _reasoning_text(reasoning_support: Any) -> str:
    data = _as_dict(reasoning_support)
    return str(data.get("reasoning_support") or "")[:6000]


def _normalize_verification(
    data: dict[str, Any],
    *,
    candidate_ids: list[str],
    model: str,
    multimodal_used: bool,
) -> dict[str, Any]:
    verdict = str(data.get("verdict") or "").strip().upper()
    if verdict not in _ALLOWED_VERDICTS:
        raise ValueError(f"Invalid verdict from VLM: {verdict!r}")

    confidence_raw = data.get("confidence")
    try:
        confidence = float(confidence_raw)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    allowed = set(candidate_ids)
    evidence_ids = [
        str(x)
        for x in (data.get("evidence_ids") or [])
        if str(x) in allowed
    ]

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

    limitations = [
        str(x).strip()
        for x in (data.get("limitations") or [])
        if str(x).strip()
    ]

    return {
        "status": "SUCCESS",
        "verdict": verdict,
        "confidence": confidence,
        "evidence_ids": evidence_ids,
        "evidence_assessments": assessments,
        "rationale": str(data.get("rationale") or "").strip(),
        "uncertainty": str(data.get("uncertainty") or "").strip(),
        "limitations": limitations,
        "multimodal_used": bool(multimodal_used),
        "model": model,
        "provider": "openai",
    }


def verify_openai(
    *,
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
            "rationale": "Không thể xác minh vì Evidence Retrieval không cung cấp evidence khả dụng.",
            "uncertainty": "INSUFFICIENT_COLLECTION",
            "limitations": ["No usable retrieved evidence."],
            "multimodal_used": False,
            "model": "",
            "provider": "openai",
        }

    if not is_configured():
        raise RuntimeError("OPENAI_API_KEY is not configured")

    evidence_text, all_candidate_ids, image_candidates, image_ids = _evidence_prompt(evidence)
    image_data_urls, attached_image_ids, image_transport_failures = _prepare_image_attachments(
        image_candidates
    )

    # Image evidence that could not be materialized must not be cited as visual
    # evidence. Text evidence remains usable, so verification can continue.
    candidate_ids = [
        evidence_id
        for evidence_id in all_candidate_ids
        if evidence_id not in image_ids or evidence_id in attached_image_ids
    ]

    if not candidate_ids:
        raise RuntimeError(
            "No usable evidence remained after image transport preparation"
        )

    image_transport_note = ""
    if image_candidates:
        failed_ids = [
            item.split(":", 1)[0]
            for item in image_transport_failures
        ]
        image_transport_note = "\n".join(
            [
                "# IMAGE ATTACHMENT STATUS",
                f"Attached image evidence IDs: {sorted(attached_image_ids) or '(none)'}",
                f"Unavailable image evidence IDs: {failed_ids or '(none)'}",
                (
                    "Do not infer visual facts from unavailable image evidence. "
                    "Continue verification from the remaining usable evidence."
                ),
            ]
        )

    claim_text = _claim_text(claim)
    reasoning_text = _reasoning_text(reasoning_support)
    context_text = _multimedia_context(multimedia)

    prompt = f"""
# ROLE
You are the Phase 4 evidence-grounded verifier for a multimodal fact-checking system.

# HARD RULES
- Decide ONLY from the supplied claim, retrieved evidence, source context, and reasoning support.
- SUPPORTED: evidence sufficiently confirms the atomic claim.
- REFUTED: evidence sufficiently contradicts the core proposition.
- NEI: evidence collection succeeded, but the supplied evidence is insufficient to support or refute.
- A technical/API/retrieval failure is NEVER NEI. (Technical failure is handled outside this prompt.)
- Do not treat source_quality metadata as truth by itself.
- Respect dates/temporal anchors and distinguish historical vs current-state claims.
- evidence_ids MUST be selected only from the provided EVIDENCE_ID values.
- Do not reveal private chain-of-thought. Return concise evidence-grounded rationales only.

# CLAIM
{claim_text}

# SOURCE MULTIMEDIA CONTEXT
{context_text}

# CONCEPTUAL REASONING SUPPORT
{reasoning_text or '(none)'}

# RETRIEVED EVIDENCE
{evidence_text}

{image_transport_note}

# OUTPUT
Return ONLY a JSON object:
{{
  "verdict": "SUPPORTED | REFUTED | NEI",
  "confidence": 0.0,
  "evidence_ids": ["..."],
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

    models_raw = os.getenv("OPENAI_VERIFICATION_MODELS", os.getenv("OPENAI_VLM_MODELS", "gpt-6-luna"))
    models = [x.strip() for x in models_raw.split(",") if x.strip()] or ["gpt-6-luna"]
    errors: list[str] = []

    for model in models:
        if circuit.is_open("openai", model):
            remain = circuit.remaining_seconds("openai", model)
            errors.append(f"{model}: circuit_open remaining~{remain:.0f}s")
            continue
        try:
            data = generate_json(
                model=model,
                prompt=prompt,
                image_urls=image_data_urls,
                max_output_tokens=int(os.getenv("OPENAI_PHASE4_MAX_OUTPUT_TOKENS", "3000")),
                timeout_seconds=float(os.getenv("OPENAI_VLM_TIMEOUT", "120")),
                image_detail=os.getenv("OPENAI_IMAGE_DETAIL", "low"),
            )
            result = _normalize_verification(
                data,
                candidate_ids=candidate_ids,
                model=model,
                multimodal_used=bool(image_data_urls),
            )
            if image_transport_failures:
                result["limitations"] = list(result.get("limitations") or []) + [
                    (
                        f"{len(image_transport_failures)} image evidence item(s) could not "
                        "be attached to the VLM; verification continued with remaining evidence."
                    )
                ]
            return result
        except Exception as exc:
            errors.append(f"{model}: {type(exc).__name__}: {exc}")

    raise RuntimeError("OpenAI Phase 4 failed: " + " | ".join(errors))
