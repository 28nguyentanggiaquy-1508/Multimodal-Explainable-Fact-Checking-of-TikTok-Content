from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

_ALLOWED_COMPONENT_STATUSES = {"SUPPORTED", "REFUTED", "INSUFFICIENT"}


@dataclass(frozen=True)
class ClaimComponent:
    component_id: str
    text: str


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


def _normalize(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join(value.split())


def _ascii_fold(text: Any) -> str:
    value = unicodedata.normalize("NFKD", _normalize(text))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return value.replace("đ", "d")


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    buffer: list[str] = []
    depth = 0
    for ch in str(text or ""):
        if ch in "([{":
            depth += 1
        elif ch in ")]}" and depth > 0:
            depth -= 1
        if depth == 0 and ch in {",", ";"}:
            piece = "".join(buffer).strip(" .")
            if piece:
                parts.append(piece)
            buffer = []
            continue
        buffer.append(ch)
    tail = "".join(buffer).strip(" .")
    if tail:
        parts.append(tail)
    return parts


def _split_predicate_conjunctions(text: str) -> list[str]:
    # Split only where the right-hand side clearly starts a new factual predicate.
    pattern = re.compile(
        r"\s+(?:và|đồng thời)\s+(?=(?:là|được|giữ|có|đã|từng|sẽ|trở\s+thành|làm)\b)",
        flags=re.IGNORECASE,
    )
    pieces = [part.strip(" .") for part in pattern.split(text) if part.strip(" .")]
    return pieces or [text.strip()]


def decompose_claim(claim_text: str) -> list[ClaimComponent]:
    """Conservative deterministic decomposition for Vietnamese factual claims."""
    text = " ".join(str(claim_text or "").split()).strip()
    if not text:
        return []

    raw_parts: list[str] = []
    for comma_part in _split_top_level_commas(text):
        raw_parts.extend(_split_predicate_conjunctions(comma_part))

    cleaned: list[str] = []
    seen: set[str] = set()
    for part in raw_parts:
        part = part.strip(" .")
        if len(part) < 4:
            continue
        key = part.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(part)

    if not cleaned:
        cleaned = [text]

    max_components = max(1, int(os.getenv("LOCAL_PHASE4_MAX_COMPONENTS", "8")))
    cleaned = cleaned[:max_components]
    return [ClaimComponent(component_id=f"C{idx}", text=part) for idx, part in enumerate(cleaned, start=1)]


def component_prompt_block(components: list[ClaimComponent]) -> str:
    if not components:
        return "C1: (claim could not be decomposed; evaluate the full claim conservatively)"
    return "\n".join(f"{item.component_id}: {item.text}" for item in components)


def _clamp(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        numeric = default
    return max(0.0, min(1.0, numeric))


# Critical qualifiers are intentionally conservative. They only *downgrade* a
# local SUPPORTED component when the cited external evidence does not directly
# carry the material qualifier/value. They never upgrade a component.
_QUALIFIER_MARKERS: dict[str, tuple[str, ...]] = {
    "unique": (
        "duy nhất", "độc nhất", "only", "sole", "unique", "first and only",
    ),
    "kinship_parent": (
        "thân sinh", "cha của", "mẹ của", "father of", "mother of", "parent of",
    ),
    "real_name": (
        "tên thật", "tên khai sinh", "real name", "birth name", "born as",
    ),
    "ordinal_second": (
        "thứ hai", "người thứ 2", "vị thứ 2", "second", "2nd",
    ),
    "direct_promotion": (
        "phong thẳng", "được phong trực tiếp", "không qua", "không trải qua",
        "directly conferred", "directly promoted", "without intermediate",
        "without prior ranks", "skipping intermediate",
    ),
}

_STOP_TOKENS = {
    "dai", "tuong", "thuong", "trung", "chu", "tich", "quan", "doi", "nhan", "dan",
    "viet", "nam", "la", "duoc", "cua", "va", "nam", "nguoi", "vi", "giu", "chuc",
    "cap", "cac", "khong", "qua", "phong", "thang", "quan", "ham", "tong", "tu", "lenh",
    "the", "of", "and", "was", "is", "in", "to", "by", "general", "army", "people",
}


def _distinctive_tokens(text: str) -> set[str]:
    folded = _ascii_fold(text)
    tokens = set(re.findall(r"[a-z0-9]+", folded))
    return {token for token in tokens if len(token) >= 3 and token not in _STOP_TOKENS}


def _extract_requirements(component_text: str) -> list[dict[str, Any]]:
    norm = _normalize(component_text)
    folded = _ascii_fold(component_text)
    requirements: list[dict[str, Any]] = []

    def add(kind: str, label: str, markers: tuple[str, ...] = (), values: tuple[str, ...] = ()) -> None:
        requirements.append(
            {
                "kind": kind,
                "label": label,
                "markers": tuple(_ascii_fold(item) for item in markers),
                "values": values,
            }
        )

    if any(_normalize(marker) in norm for marker in _QUALIFIER_MARKERS["unique"]):
        add("unique", "uniqueness/only qualifier", _QUALIFIER_MARKERS["unique"])

    if any(_normalize(marker) in norm for marker in _QUALIFIER_MARKERS["kinship_parent"]):
        add("kinship_parent", "parent/kinship relation", _QUALIFIER_MARKERS["kinship_parent"])

    if any(_normalize(marker) in norm for marker in _QUALIFIER_MARKERS["real_name"]):
        add("real_name", "real-name identity", _QUALIFIER_MARKERS["real_name"])

    if any(_normalize(marker) in norm for marker in _QUALIFIER_MARKERS["ordinal_second"]):
        add("ordinal_second", "second/ordinal qualifier", _QUALIFIER_MARKERS["ordinal_second"])

    if (
        "phong thang" in folded
        or "khong qua" in folded
        or "khong trai qua" in folded
        or "without intermediate" in folded
    ):
        add("direct_promotion", "direct-promotion/no-intermediate-rank qualifier", _QUALIFIER_MARKERS["direct_promotion"])

    years = tuple(dict.fromkeys(re.findall(r"\b(?:18|19|20)\d{2}\b", folded)))
    for year in years:
        add("year", f"exact year {year}", values=(year,))

    ages = tuple(dict.fromkeys(re.findall(r"\b(\d{1,3})\s*(?:tuoi|years?\s+old|age)\b", folded)))
    for age in ages:
        add("age", f"exact age {age}", markers=("tuổi", "years old", "age", "aged"), values=(age,))

    # Exact cardinal quantities attached to a factual noun phrase are material.
    quantities = tuple(
        dict.fromkeys(
            match.group(1)
            for match in re.finditer(
                r"\b(\d{1,4})\s+(?:vi\s+)?(?:dai\s+tuong|tuong|general|people|nguoi)\b",
                folded,
            )
        )
    )
    for value in quantities:
        add("quantity", f"exact quantity {value}", values=(value,))

    return requirements


def _evidence_text_map(evidence: list[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, raw in enumerate(evidence, start=1):
        item = _as_dict(raw)
        evidence_id = str(item.get("id") or item.get("evidence_id") or f"evidence_{index}")
        text = "\n".join(
            str(value or "")
            for value in (
                item.get("title"),
                item.get("content"),
                item.get("source_domain"),
            )
        )
        result[evidence_id] = text
    return result


def _requirement_matches(
    requirement: dict[str, Any],
    *,
    component_text: str,
    cited_ids: list[str],
    evidence_map: dict[str, str],
) -> tuple[bool, list[str]]:
    component_tokens = _distinctive_tokens(component_text)
    matches: list[str] = []

    for evidence_id in cited_ids:
        text = evidence_map.get(evidence_id, "")
        if not text:
            continue
        folded = _ascii_fold(text)
        evidence_tokens = _distinctive_tokens(text)

        markers = tuple(requirement.get("markers") or ())
        values = tuple(requirement.get("values") or ())
        marker_ok = True if not markers else any(marker in folded for marker in markers)
        values_ok = all(value in folded for value in values)

        # Require some lexical/entity anchoring so a qualifier elsewhere in a
        # long article cannot accidentally validate the current component.
        shared = component_tokens & evidence_tokens
        overlap_needed = 1 if len(component_tokens) <= 2 else 2
        anchor_ok = len(shared) >= overlap_needed

        if marker_ok and values_ok and anchor_ok:
            matches.append(evidence_id)

    return bool(matches), matches


def _apply_evidence_qualifier_checks(
    normalized: list[dict[str, Any]],
    *,
    evidence: list[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence_map = _evidence_text_map(evidence)
    all_checks: list[dict[str, Any]] = []

    for item in normalized:
        checks_for_component: list[dict[str, Any]] = []
        if item["status"] == "SUPPORTED":
            requirements = _extract_requirements(item["text"])
            for requirement in requirements:
                passed, matched_ids = _requirement_matches(
                    requirement,
                    component_text=item["text"],
                    cited_ids=list(item.get("evidence_ids") or []),
                    evidence_map=evidence_map,
                )
                check = {
                    "component_id": item["component_id"],
                    "kind": requirement["kind"],
                    "label": requirement["label"],
                    "passed": passed,
                    "matched_evidence_ids": matched_ids,
                }
                checks_for_component.append(check)
                all_checks.append(check)

            failed = [check for check in checks_for_component if not check["passed"]]
            if failed:
                item["status"] = "INSUFFICIENT"
                item["confidence"] = 0.0
                item["rationale"] = (
                    "Evidence qualifier guard downgraded this component because cited external evidence "
                    "did not directly establish: " + ", ".join(check["label"] for check in failed)
                )

        item["qualifier_checks"] = checks_for_component

    return normalized, all_checks


def _normalize_component_assessments(
    raw_items: Any,
    *,
    components: list[ClaimComponent],
    allowed_evidence_ids: set[str],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    valid_component_ids = {c.component_id for c in components}
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            component_id = str(item.get("component_id") or "").strip().upper()
            if component_id not in valid_component_ids:
                continue
            status = str(item.get("status") or "INSUFFICIENT").strip().upper()
            if status not in _ALLOWED_COMPONENT_STATUSES:
                status = "INSUFFICIENT"
            evidence_ids = [
                str(value)
                for value in (item.get("evidence_ids") or [])
                if str(value) in allowed_evidence_ids
            ]
            if status in {"SUPPORTED", "REFUTED"} and not evidence_ids:
                status = "INSUFFICIENT"
            by_id[component_id] = {
                "component_id": component_id,
                "status": status,
                "evidence_ids": evidence_ids,
                "confidence": _clamp(item.get("confidence"), 0.0),
                "rationale": str(item.get("rationale") or "").strip(),
            }

    normalized: list[dict[str, Any]] = []
    for component in components:
        item = by_id.get(component.component_id)
        if item is None:
            item = {
                "component_id": component.component_id,
                "status": "INSUFFICIENT",
                "evidence_ids": [],
                "confidence": 0.0,
                "rationale": "No grounded component assessment was returned.",
            }
        item["text"] = component.text
        normalized.append(item)
    return normalized


def apply_local_component_guard(
    *,
    raw_data: dict[str, Any],
    base_result: dict[str, Any],
    components: list[ClaimComponent],
    candidate_ids: list[str],
    attached_image_ids: set[str],
    evidence: list[Any] | None = None,
) -> dict[str, Any]:
    """Recompute local coverage/verdict and conservatively validate qualifiers.

    The local VLM assesses evidence, but it cannot self-upgrade a compound
    claim to COMPLETE/SUPPORTED. For high-risk qualifiers (e.g. "duy nhất",
    kinship, exact year/age, "phong thẳng"), cited evidence must directly carry
    the qualifier/value or that component is downgraded to INSUFFICIENT.
    """
    result = dict(base_result)
    if not components:
        return result

    allowed = set(candidate_ids)
    normalized = _normalize_component_assessments(
        raw_data.get("component_assessments"),
        components=components,
        allowed_evidence_ids=allowed,
    )

    normalized, qualifier_checks = _apply_evidence_qualifier_checks(
        normalized,
        evidence=list(evidence or []),
    )

    supported = [item for item in normalized if item["status"] == "SUPPORTED"]
    refuted = [item for item in normalized if item["status"] == "REFUTED"]
    insufficient = [item for item in normalized if item["status"] == "INSUFFICIENT"]

    cited_ids: list[str] = []
    for item in normalized:
        for evidence_id in item["evidence_ids"]:
            if evidence_id not in cited_ids:
                cited_ids.append(evidence_id)

    total = len(normalized)
    resolved = len(supported) + len(refuted)
    coverage_score = resolved / total if total else 0.0

    if refuted:
        final_verdict = "REFUTED"
        final_coverage = "COMPLETE" if resolved == total else "PARTIAL"
        confidence_basis = max((item["confidence"] for item in refuted), default=0.0)
    elif len(supported) == total:
        final_verdict = "SUPPORTED"
        final_coverage = "COMPLETE"
        confidence_basis = min((item["confidence"] for item in supported), default=0.0)
    else:
        final_verdict = "NEI"
        final_coverage = "PARTIAL" if resolved > 0 else "INSUFFICIENT"
        confidence_basis = _clamp(raw_data.get("confidence"), 0.5)

    model_confidence = _clamp(raw_data.get("confidence"), 0.0)
    local_cap = _clamp(os.getenv("LOCAL_PHASE4_MAX_CONFIDENCE", "0.95"), 0.95)
    nei_cap = _clamp(os.getenv("LOCAL_PHASE4_MAX_NEI_CONFIDENCE", "0.85"), 0.85)

    if final_verdict in {"SUPPORTED", "REFUTED"}:
        confidence = min(model_confidence, confidence_basis, local_cap)
    else:
        # Coverage score prevents a partially-covered claim from retaining 1.0.
        confidence = min(model_confidence, nei_cap, max(0.5, coverage_score))

    limitations = list(result.get("limitations") or [])
    if insufficient:
        missing_ids = ", ".join(item["component_id"] for item in insufficient)
        limitations.append(
            f"Local component/evidence guard: insufficient external evidence for {missing_ids}."
        )
    failed_qualifiers = [check for check in qualifier_checks if not check["passed"]]
    if failed_qualifiers:
        limitations.append(
            "Evidence qualifier guard rejected unsupported material qualifier(s): "
            + ", ".join(f"{check['component_id']}:{check['kind']}" for check in failed_qualifiers)
            + "."
        )
    if len(components) > 1:
        limitations.append(
            "Final local verdict/coverage was recomputed deterministically from material claim components."
        )

    status_summary = "; ".join(
        f"{item['component_id']}={item['status']}" for item in normalized
    )
    if final_verdict == "SUPPORTED":
        rationale = (
            f"Đánh giá theo thành phần: {status_summary}. "
            "Evidence ngoài bao phủ đầy đủ mọi thành phần trọng yếu của claim."
        )
    elif final_verdict == "REFUTED":
        rationale = (
            f"Đánh giá theo thành phần: {status_summary}. "
            "Ít nhất một thành phần trọng yếu bị evidence ngoài phản bác trực tiếp."
        )
    else:
        rationale = (
            f"Đánh giá theo thành phần: {status_summary}. "
            "Evidence ngoài chưa bao phủ trực tiếp đầy đủ mọi thành phần/định ngữ trọng yếu nên verdict được chuẩn hóa thành NEI."
        )

    result.update(
        {
            "verdict": final_verdict,
            "confidence": confidence,
            "evidence_coverage": final_coverage,
            "evidence_ids": cited_ids,
            "rationale": rationale,
            "limitations": limitations,
            "multimodal_used": bool(set(cited_ids) & set(attached_image_ids)),
            "_guard_metadata": {
                "mode": "local_component_evidence_guard_v2",
                "component_count": total,
                "resolved_components": resolved,
                "coverage_score": round(coverage_score, 4),
                "components": normalized,
                "qualifier_checks": qualifier_checks,
                "raw_model_verdict": str(raw_data.get("verdict") or ""),
                "raw_model_coverage": str(raw_data.get("evidence_coverage") or ""),
                "raw_model_confidence": model_confidence,
                "final_verdict": final_verdict,
                "final_coverage": final_coverage,
                "final_confidence": confidence,
            },
        }
    )
    return result
