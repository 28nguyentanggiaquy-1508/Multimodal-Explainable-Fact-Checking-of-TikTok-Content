from __future__ import annotations

from copy import deepcopy
from typing import Any

TECHNICAL_PHASE2_STATUSES = {"FAILED", "UNAVAILABLE"}
TECHNICAL_SOURCES = {
    "multimedia_error",
    "gemini_error",
    "vlm_error",
    "multivlm_unavailable",
    "openai_error",
    "unavailable",
}
TECHNICAL_MARKERS = (
    "resource_exhausted",
    "quota",
    "429",
    "timeout",
    "timed out",
    "unavailable",
    "servererror",
    "clienterror",
    "api error",
    "connection",
    "attributeerror",
    "circuit open",
    "persistent circuit open",
)


def _contains_technical_marker(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        s = value.casefold()
        return any(marker.casefold() in s for marker in TECHNICAL_MARKERS)
    if isinstance(value, list):
        return any(_contains_technical_marker(v) for v in value)
    if isinstance(value, dict):
        return any(_contains_technical_marker(v) for v in value.values())
    return False


def _phase2_info(payload: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    phase2 = ((payload.get("phase_status") or {}).get("phase_2") or {})
    details = phase2.get("details") or {}
    multimedia = payload.get("multimedia") or payload.get("multimodal") or {}
    status = str(phase2.get("status") or multimedia.get("status") or "").upper()
    return status, details, multimedia


def normalize_factcheck_response_semantics(response: Any) -> Any:
    """Normalize response semantics without inventing claims or verdicts.

    Critical invariant:
    - technical failure during claim extraction != "no claim detected".
    - a true no-claim result is only asserted when Phase 2 completed successfully.

    The function accepts a dict or a Pydantic model and returns the same logical
    response as a plain dict when normalization is needed.
    """
    if isinstance(response, dict):
        payload = deepcopy(response)
    else:
        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            payload = model_dump(mode="json")
        else:
            dict_method = getattr(response, "dict", None)
            if callable(dict_method):
                payload = dict_method()
            else:
                return response

    claims = payload.get("claims") or []
    claim_count = len(claims)
    phase2_status, phase2_details, multimedia = _phase2_info(payload)
    selected_count = phase2_details.get("selected_claim_count")
    if selected_count is None:
        selected_count = claim_count

    source = str(multimedia.get("source") or phase2_details.get("source") or "").lower()
    phase2_message = ((payload.get("phase_status") or {}).get("phase_2") or {}).get("message")
    technical_reason = (
        multimedia.get("error")
        or multimedia.get("degraded_reason")
        or phase2_message
    )

    technical_failure = (
        phase2_status in TECHNICAL_PHASE2_STATUSES
        or source in TECHNICAL_SOURCES
        or (
            phase2_status == "DEGRADED"
            and int(selected_count or 0) == 0
            and _contains_technical_marker(technical_reason)
        )
    )

    # Add explicit machine-readable semantics for the dataset/tooling layer.
    if technical_failure and claim_count == 0:
        payload["claim_extraction_status"] = "UNAVAILABLE"
        payload["claim_extraction_completed"] = False
        payload["no_claim_detected"] = None
        payload["skip_reason"] = (
            "Phase 2 không hoàn tất claim extraction do multimedia/VLM hoặc dependency "
            "không khả dụng; không thể kết luận video không có checkable/check-worthy claim."
        )
    elif phase2_status == "SUCCESS" and claim_count == 0:
        payload["claim_extraction_status"] = "SUCCESS"
        payload["claim_extraction_completed"] = True
        payload["no_claim_detected"] = True
        payload["skip_reason"] = (
            "Phase 2 đã hoàn tất nhưng không phát hiện checkable/check-worthy claim để tiếp tục fact-check."
        )
    else:
        payload["claim_extraction_status"] = phase2_status or "UNKNOWN"
        payload["claim_extraction_completed"] = bool(claim_count) or phase2_status == "SUCCESS"
        payload["no_claim_detected"] = False if claim_count else None

    return payload
