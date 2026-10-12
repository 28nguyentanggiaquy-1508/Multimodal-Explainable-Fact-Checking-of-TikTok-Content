from __future__ import annotations

from datetime import date
from functools import lru_cache
from io import BytesIO
from urllib.parse import urlparse
import re

import requests
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:  # Allows lightweight unit-test imports.
    CLIPModel = None
    CLIPProcessor = None

from config import (
    EVIDENCE_MAX_IMAGE_BYTES,
    EVIDENCE_REQUEST_TIMEOUT,
    IMAGE_RETRIEVAL_MODEL,
    IMAGE_RETRIEVAL_TOP_N,
)
from fact_checking.evidence.evidence_schema import (
    EvidenceItem,
    RetrievalChannelResult,
)
from fact_checking.evidence.http_utils import read_limited_bytes, safe_get
from fact_checking.evidence.search_provider import discover_image_sources


Image.MAX_IMAGE_PIXELS = 30_000_000


def _unit_score(cosine_score: float) -> float:
    return max(0.0, min(1.0, (float(cosine_score) + 1.0) / 2.0))


def _source_quality(domain: str) -> str:
    value = (domain or "").lower()
    if value.endswith(".gov.vn") or value.endswith(".gov"):
        return "official_government"
    if value.endswith(".edu.vn") or value.endswith(".edu"):
        return "academic"
    return "unclassified"


class ImageEncoder:
    """CLIP baseline for claim-to-image retrieval."""

    def __init__(self):
        if CLIPModel is None or CLIPProcessor is None:
            raise RuntimeError("transformers is not installed")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            "[Evidence/ImageEncoder] "
            f"loading={IMAGE_RETRIEVAL_MODEL} device={self.device}"
        )
        self.processor = CLIPProcessor.from_pretrained(IMAGE_RETRIEVAL_MODEL)
        self.model = CLIPModel.from_pretrained(IMAGE_RETRIEVAL_MODEL).to(
            self.device
        ).eval()

    @staticmethod
    def _pooled(output) -> torch.Tensor:
        pooled = getattr(output, "pooler_output", None)
        if pooled is not None:
            return pooled
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError("CLIP encoder returned no pooler_output/last_hidden_state")
        return hidden[:, 0, :]

    def encode_text(self, text: str) -> torch.Tensor:
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            output = self.model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            pooled = self._pooled(output)
            features = self.model.text_projection(pooled)

        return F.normalize(features, p=2, dim=-1).cpu()

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.inference_mode():
            output = self.model.vision_model(pixel_values=pixel_values)
            pooled = self._pooled(output)
            features = self.model.visual_projection(pooled)

        return F.normalize(features, p=2, dim=-1).cpu()




def _image_temporal_features(
    *,
    title: str,
    source_url: str,
    image_url: str,
    temporal_anchor: date | None,
    temporal_mode: str,
) -> tuple[float, str, bool]:
    if temporal_mode != "current_state" or temporal_anchor is None:
        return 1.0, "not_required", True

    blob = " ".join([title or "", source_url or "", image_url or ""])
    years = {int(value) for value in re.findall(r"\b(20\d{2})\b", blob)}
    if not years:
        return 0.75, "unknown", True

    if any(year > temporal_anchor.year for year in years):
        return 0.10, "future_year_detected", False
    if temporal_anchor.year in years:
        return 1.0, "anchor_year", True
    return 0.82, "before_anchor_year", True


@lru_cache(maxsize=1)
def get_image_encoder() -> ImageEncoder:
    return ImageEncoder()


def load_remote_image(url: str) -> Image.Image | None:
    try:
        response = safe_get(
            url,
            timeout=EVIDENCE_REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
            stream=True,
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if content_type and not content_type.startswith("image/"):
            response.close()
            return None

        raw = read_limited_bytes(response, EVIDENCE_MAX_IMAGE_BYTES)
        response.close()

        image = Image.open(BytesIO(raw))
        image.load()
        image = image.convert("RGB")
        image.thumbnail((2048, 2048))
        return image

    except (UnidentifiedImageError, OSError, ValueError, requests.RequestException):
        return None


def retrieve_image_evidence(
    claim: str,
    top_n: int | None = None,
    *,
    temporal_anchor: date | None = None,
    time_sensitive: bool = False,
    temporal_mode: str = "none",
) -> RetrievalChannelResult:
    top_n = top_n or IMAGE_RETRIEVAL_TOP_N
    search_query = claim
    if time_sensitive and temporal_anchor is not None:
        search_query = f"{claim} {temporal_anchor.year}"
    print(f"[Evidence/Image] query={search_query!r}")

    discovery = discover_image_sources(
        search_query,
        max_results=min(max(top_n, 1), 20),
    )

    if discovery.status in {"UNAVAILABLE", "FAILED"}:
        return RetrievalChannelResult(
            modality="image",
            status=discovery.status,
            provider_attempts=discovery.attempts,
            errors=[
                attempt.error_message
                for attempt in discovery.attempts
                if attempt.error_message
            ],
        )

    search_results = discovery.results
    print(f"[Evidence/Image] search_sources={len(search_results)}")

    if not search_results:
        return RetrievalChannelResult(
            modality="image",
            status="SUCCESS",
            provider_attempts=discovery.attempts,
            search_source_count=0,
            candidate_count=0,
        )

    try:
        encoder = get_image_encoder()
        text_embedding = encoder.encode_text(claim)
    except Exception as exc:
        return RetrievalChannelResult(
            modality="image",
            status="FAILED",
            provider_attempts=discovery.attempts,
            search_source_count=len(search_results),
            errors=[f"CLIP load/encode error: {type(exc).__name__}: {exc}"],
        )

    evidence = []
    attempted_images = 0

    for index, result in enumerate(search_results, start=1):
        attempted_images += 1
        image = load_remote_image(result.image_url)

        if (
            image is None
            and result.thumbnail_url
            and result.thumbnail_url != result.image_url
        ):
            image = load_remote_image(result.thumbnail_url)

        if image is None:
            continue

        try:
            image_embedding = encoder.encode_image(image)
            raw_cosine = torch.mm(image_embedding, text_embedding.T).item()
        except Exception:
            continue

        clip_score = _unit_score(raw_cosine)
        source_url = result.source_url or result.image_url
        domain = (urlparse(source_url).hostname or "").lower()
        temporal_score, temporal_relation, temporal_eligible = _image_temporal_features(
            title=result.title,
            source_url=source_url,
            image_url=result.image_url,
            temporal_anchor=temporal_anchor,
            temporal_mode=temporal_mode,
        )
        score = (
            0.82 * clip_score + 0.18 * temporal_score
            if temporal_mode == "current_state"
            else clip_score
        )

        evidence.append(
            EvidenceItem(
                id=f"image_{index:02d}",
                modality="image",
                title=result.title,
                source_url=source_url,
                source_domain=domain,
                content=result.title,
                media_url=result.image_url,
                retrieval_score=clip_score,
                relevance_score=max(0.0, min(1.0, score)),
                provider=result.provider,
                source_quality=_source_quality(domain),
                metadata={
                    "retrieval_method": "clip_cosine",
                    "raw_cosine": round(float(raw_cosine), 6),
                    "clip_score": round(float(clip_score), 6),
                    "time_sensitive_claim": bool(time_sensitive),
                    "temporal_mode": temporal_mode,
                    "temporal_anchor": temporal_anchor.isoformat() if temporal_anchor else None,
                    "temporal_relation": temporal_relation,
                    "temporal_score": round(float(temporal_score), 4),
                    "temporal_eligible": bool(temporal_eligible),
                },
            )
        )

    evidence.sort(
        key=lambda item: (
            bool(item.metadata.get("temporal_eligible", True)),
            item.relevance_score,
        ),
        reverse=True,
    )
    eligible = [item for item in evidence if item.metadata.get("temporal_eligible", True)]
    selected = eligible[:top_n]
    if len(selected) < top_n:
        selected_ids = {item.id for item in selected}
        for item in evidence:
            if item.id in selected_ids:
                continue
            selected.append(item)
            if len(selected) >= top_n:
                break
    print(f"[Evidence/Image] selected={len(selected)}")

    # Search worked but every image URL failed to load: degraded, not provider
    # unavailable. This distinction matters for NEI vs operational failure.
    status = "SUCCESS" if selected or attempted_images == 0 else "DEGRADED"
    errors = []
    if status == "DEGRADED":
        errors.append("Image search succeeded but candidate images could not be fetched/decoded.")

    return RetrievalChannelResult(
        modality="image",
        status=status,
        evidence=selected,
        provider_attempts=discovery.attempts,
        search_source_count=len(search_results),
        candidate_count=attempted_images,
        errors=errors,
    )
