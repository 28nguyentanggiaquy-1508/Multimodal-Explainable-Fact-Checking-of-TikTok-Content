from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from evaluation.temporary_gold_v071 import (
    ALLOWED_VERDICTS,
    DATASET_ROLE,
    TemporaryGoldRecord,
    evaluate_predictions,
    normalize_domain,
    write_json,
    write_jsonl,
)

FULL_PIPELINE_VERSION = "v0.8.1"
FINAL_VERDICTS = {"SUPPORTED", "REFUTED", "NEI"}
TECHNICAL_STATUSES = {"FAILED", "UNAVAILABLE"}
PENDING_GOLD_REAUDIT_IDS = {
    "TK_7687526963484822792_AI_e38abc30b751",
    "TK_7687526963484822792_AI_ef5fcd646aed",
    "TK_7689007311947435265_AI_f3046dfb5a80",
    "TK_7699809677638520084_AI_17e772981959",
    # v0.8.0 full-pipeline surfaced conflicting temporal evidence for this row.
    "TK_7687526963484822792_AI_9a9af3394e3f",
}


@dataclass(frozen=True)
class AlignmentCandidate:
    gold_record_id: str
    system_index: int
    score: float
    text_similarity: float
    token_jaccard: float
    timestamp_score: float
    modality_score: float
    exact_normalized: bool


def _norm_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    chars: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            chars.append(" ")
        else:
            chars.append(ch)
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def _tokens(value: Any) -> set[str]:
    return {x for x in _norm_text(value).split() if x}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _parse_hms(value: str) -> float | None:
    parts = [p for p in str(value or "").strip().split(":") if p != ""]
    if not parts or len(parts) > 3:
        return None
    try:
        nums = [float(x) for x in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    return nums[0]


def _parse_interval(value: Any) -> tuple[float, float] | None:
    text = str(value or "").strip()
    if not text:
        return None
    parts = re.split(r"\s*[-–—]\s*", text, maxsplit=1)
    if len(parts) == 1:
        one = _parse_hms(parts[0])
        return (one, one) if one is not None else None
    left = _parse_hms(parts[0])
    right = _parse_hms(parts[1])
    if left is None or right is None:
        return None
    return (min(left, right), max(left, right))


def _timestamp_similarity(gold_ts: Any, system_ts: Any) -> float:
    g = _parse_interval(gold_ts)
    s = _parse_interval(system_ts)
    if g is None or s is None:
        return 0.5
    if g == s:
        return 1.0
    lo = max(g[0], s[0])
    hi = min(g[1], s[1])
    if hi >= lo:
        union = max(g[1], s[1]) - min(g[0], s[0])
        overlap = hi - lo
        if union <= 0:
            return 1.0
        return max(0.6, min(1.0, overlap / union))
    distance = min(abs(g[0] - s[1]), abs(s[0] - g[1]))
    if distance <= 2:
        return 0.6
    if distance <= 5:
        return 0.3
    return 0.0


def _gold_modality(record: TemporaryGoldRecord) -> str:
    values = {str(x).casefold().replace("-", "_") for x in record.source_modalities}
    if "crossmodal" in values or "cross_modal" in values:
        return "cross_modal"
    if "ocr" in values:
        return "ocr"
    if "visual" in values or "image" in values:
        return "visual"
    if "caption" in values:
        return "caption"
    return "speech"


def _modality_similarity(record: TemporaryGoldRecord, system_claim: dict[str, Any]) -> float:
    g = _gold_modality(record)
    s = str(system_claim.get("source_modality") or "").casefold().replace("-", "_")
    if not s:
        return 0.5
    if g == s:
        return 1.0
    # Cross-modal is intentionally treated as partially compatible with a
    # single observed modality because Phase 2 may collapse source provenance.
    if "cross_modal" in {g, s}:
        return 0.7
    return 0.0


def alignment_score(record: TemporaryGoldRecord, system_claim: dict[str, Any]) -> AlignmentCandidate:
    gold_norm = _norm_text(record.claim_text)
    sys_norm = _norm_text(system_claim.get("text"))
    exact = bool(gold_norm and gold_norm == sys_norm)
    text_sim = SequenceMatcher(None, gold_norm, sys_norm).ratio() if gold_norm or sys_norm else 0.0
    jac = _jaccard(_tokens(gold_norm), _tokens(sys_norm))
    ts = _timestamp_similarity(record.original_timestamp, system_claim.get("timestamp"))
    mod = _modality_similarity(record, system_claim)
    # Text dominates. Timestamp/modality are tie-breakers and should never
    # rescue semantically unrelated text.
    score = 0.62 * text_sim + 0.25 * jac + 0.08 * ts + 0.05 * mod
    if exact:
        score = max(score, 0.995)
    return AlignmentCandidate(
        gold_record_id=record.record_id,
        system_index=-1,
        score=round(score, 6),
        text_similarity=round(text_sim, 6),
        token_jaccard=round(jac, 6),
        timestamp_score=round(ts, 6),
        modality_score=round(mod, 6),
        exact_normalized=exact,
    )


def align_video_claims(
    gold_records: list[TemporaryGoldRecord],
    system_claim_rows: list[dict[str, Any]],
    *,
    min_score: float = 0.68,
    min_text_similarity: float = 0.62,
) -> dict[str, Any]:
    """Greedy one-to-one claim alignment with deterministic score ordering.

    The Temporary Gold is not assumed exhaustive, so unmatched system claims are
    reported but are not automatically counted as false positives.
    """
    candidates: list[AlignmentCandidate] = []
    for gold in gold_records:
        for idx, row in enumerate(system_claim_rows):
            claim = row.get("claim") if isinstance(row.get("claim"), dict) else row
            cand = alignment_score(gold, claim if isinstance(claim, dict) else {})
            candidates.append(AlignmentCandidate(**{**cand.__dict__, "system_index": idx}))
    candidates.sort(key=lambda x: (x.exact_normalized, x.score, x.text_similarity, x.token_jaccard), reverse=True)

    used_gold: set[str] = set()
    used_system: set[int] = set()
    matches: list[dict[str, Any]] = []
    by_gold = {x.record_id: x for x in gold_records}
    for c in candidates:
        if c.gold_record_id in used_gold or c.system_index in used_system:
            continue
        if c.score < min_score or c.text_similarity < min_text_similarity:
            continue
        gold = by_gold[c.gold_record_id]
        row = system_claim_rows[c.system_index]
        claim = row.get("claim") if isinstance(row.get("claim"), dict) else row
        used_gold.add(c.gold_record_id)
        used_system.add(c.system_index)
        matches.append({
            "record_id": c.gold_record_id,
            "system_index": c.system_index,
            "score": c.score,
            "text_similarity": c.text_similarity,
            "token_jaccard": c.token_jaccard,
            "timestamp_score": c.timestamp_score,
            "modality_score": c.modality_score,
            "exact_normalized": c.exact_normalized,
            "gold_claim_text": gold.claim_text,
            "system_claim_text": str(claim.get("text") or "") if isinstance(claim, dict) else "",
            "gold_timestamp": gold.original_timestamp,
            "system_timestamp": claim.get("timestamp") if isinstance(claim, dict) else None,
        })

    unmatched_gold = [x.record_id for x in gold_records if x.record_id not in used_gold]
    unmatched_system = [idx for idx in range(len(system_claim_rows)) if idx not in used_system]
    return {
        "matches": matches,
        "unmatched_gold_record_ids": unmatched_gold,
        "unmatched_system_indices": unmatched_system,
        "gold_claim_count": len(gold_records),
        "system_claim_count": len(system_claim_rows),
        "matched_count": len(matches),
        "gold_claim_recall": len(matches) / len(gold_records) if gold_records else 0.0,
        "system_alignment_rate": len(matches) / len(system_claim_rows) if system_claim_rows else 0.0,
        "exact_normalized_match_count": sum(1 for x in matches if x["exact_normalized"]),
        "mean_alignment_score": round(sum(x["score"] for x in matches) / len(matches), 6) if matches else 0.0,
        "thresholds": {"min_score": min_score, "min_text_similarity": min_text_similarity},
    }


def _sanitized_provider_attempts(verification: dict[str, Any]) -> list[dict[str, Any]]:
    meta = verification.get("operational_metadata") if isinstance(verification.get("operational_metadata"), dict) else {}
    attempts = meta.get("provider_attempts") if isinstance(meta.get("provider_attempts"), list) else []
    out: list[dict[str, Any]] = []
    for item in attempts:
        if not isinstance(item, dict):
            continue
        out.append({
            "provider": item.get("provider"),
            "status": item.get("status"),
            "reason": item.get("reason") if str(item.get("status") or "").upper() == "SKIPPED" else None,
            "routing_stage": item.get("routing_stage"),
        })
    return out


def prediction_from_matched_row(record: TemporaryGoldRecord, system_row: dict[str, Any]) -> dict[str, Any]:
    verification = system_row.get("verification") if isinstance(system_row.get("verification"), dict) else {}
    evidence = [x for x in (system_row.get("evidence") or []) if isinstance(x, dict)]
    runtime_decision = system_row.get("runtime_decision")
    runtime_reason = system_row.get("runtime_limit_reason")
    status = str(verification.get("status") or "UNAVAILABLE").upper()
    raw_verdict = str(verification.get("verdict") or "").upper()
    verdict = raw_verdict if raw_verdict in FINAL_VERDICTS else None
    if runtime_decision != "VERDICT_ELIGIBLE_FOR_DISPLAY_NOT_GOLD":
        verdict = None
    return {
        "record_id": record.record_id,
        "video_id": record.video_id,
        "operational_status": status,
        "verdict": verdict,
        "raw_verdict_before_runtime_eligibility": raw_verdict if raw_verdict in FINAL_VERDICTS else None,
        "runtime_decision": runtime_decision,
        "runtime_limit_reason": runtime_reason,
        "evidence_ids": list(verification.get("evidence_ids") or []),
        "retrieved_evidence": evidence,
        "provider": verification.get("provider"),
        "model": verification.get("model"),
        "confidence": verification.get("confidence"),
        "rationale": verification.get("rationale"),
        "provider_attempts": _sanitized_provider_attempts(verification),
    }


def _root_cause(record: TemporaryGoldRecord, system_row: dict[str, Any] | None, prediction: dict[str, Any] | None) -> str:
    if system_row is None:
        return "PHASE2_CLAIM_MISS"
    retrieval = system_row.get("retrieval") if isinstance(system_row.get("retrieval"), dict) else {}
    verification = system_row.get("verification") if isinstance(system_row.get("verification"), dict) else {}
    rstatus = str(retrieval.get("status") or "").upper()
    vstatus = str(verification.get("status") or "").upper()
    runtime_decision = str(system_row.get("runtime_decision") or "")
    runtime_reason = str(system_row.get("runtime_limit_reason") or "")
    if rstatus in TECHNICAL_STATUSES:
        return "PHASE3_TECHNICAL"
    if vstatus in TECHNICAL_STATUSES and runtime_decision != "WITHHELD":
        return "PHASE4_TECHNICAL"
    if runtime_decision == "WITHHELD":
        return f"GUARD_WITHHELD:{runtime_reason or 'UNSPECIFIED'}"
    if prediction is not None and prediction.get("verdict") == record.verdict:
        return "CORRECT"
    if not system_row.get("evidence"):
        return "PHASE3_NO_EVIDENCE"
    if prediction is None or prediction.get("verdict") is None:
        return "NO_ELIGIBLE_VERDICT"
    if record.record_id in PENDING_GOLD_REAUDIT_IDS:
        return "PENDING_GOLD_REAUDIT"
    return "SEMANTIC_DISAGREEMENT"


def evaluate_full_pipeline(
    records: list[TemporaryGoldRecord],
    video_runs: list[dict[str, Any]],
    *,
    min_alignment_score: float = 0.68,
    min_text_similarity: float = 0.62,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold_by_video: dict[str, list[TemporaryGoldRecord]] = defaultdict(list)
    for r in records:
        gold_by_video[r.video_id].append(r)
    run_by_video = {str(x.get("video_id") or ""): x for x in video_runs if str(x.get("video_id") or "")}

    predictions: list[dict[str, Any]] = []
    alignments: dict[str, Any] = {}
    root_causes: Counter[str] = Counter()
    claim_details: list[dict[str, Any]] = []
    system_extra_claim_count = 0
    video_statuses: Counter[str] = Counter()
    phase_statuses: dict[str, Counter[str]] = {f"phase_{i}": Counter() for i in range(1, 5)}

    for video_id, gold_records in gold_by_video.items():
        run = run_by_video.get(video_id)
        response = run.get("response") if isinstance(run, dict) and isinstance(run.get("response"), dict) else None
        if response is None:
            alignments[video_id] = {
                "matches": [],
                "unmatched_gold_record_ids": [x.record_id for x in gold_records],
                "unmatched_system_indices": [],
                "gold_claim_count": len(gold_records),
                "system_claim_count": 0,
                "matched_count": 0,
                "gold_claim_recall": 0.0,
                "system_alignment_rate": 0.0,
                "exact_normalized_match_count": 0,
                "mean_alignment_score": 0.0,
                "run_missing": True,
            }
            for gold in gold_records:
                root_causes["VIDEO_RUN_MISSING"] += 1
                claim_details.append({"record_id": gold.record_id, "video_id": video_id, "root_cause": "VIDEO_RUN_MISSING"})
            continue

        video_statuses[str(response.get("status") or "UNKNOWN").upper()] += 1
        phases = response.get("phase_status") if isinstance(response.get("phase_status"), dict) else {}
        for phase in phase_statuses:
            info = phases.get(phase) if isinstance(phases.get(phase), dict) else {}
            phase_statuses[phase][str(info.get("status") or "MISSING").upper()] += 1

        system_rows = [x for x in (response.get("claims") or []) if isinstance(x, dict)]
        alignment = align_video_claims(
            gold_records,
            system_rows,
            min_score=min_alignment_score,
            min_text_similarity=min_text_similarity,
        )
        alignments[video_id] = alignment
        system_extra_claim_count += len(alignment["unmatched_system_indices"])
        match_by_record = {x["record_id"]: x for x in alignment["matches"]}
        gold_map = {x.record_id: x for x in gold_records}
        for record_id, gold in gold_map.items():
            match = match_by_record.get(record_id)
            if match is None:
                root_causes["PHASE2_CLAIM_MISS"] += 1
                claim_details.append({
                    "record_id": record_id,
                    "video_id": video_id,
                    "root_cause": "PHASE2_CLAIM_MISS",
                    "gold_verdict": gold.verdict,
                    "system_verdict": None,
                    "matched": False,
                })
                continue
            system_row = system_rows[int(match["system_index"])]
            pred = prediction_from_matched_row(gold, system_row)
            predictions.append(pred)
            cause = _root_cause(gold, system_row, pred)
            root_causes[cause] += 1
            claim_details.append({
                "record_id": record_id,
                "video_id": video_id,
                "root_cause": cause,
                "gold_verdict": gold.verdict,
                "system_verdict": pred.get("verdict"),
                "raw_verdict": pred.get("raw_verdict_before_runtime_eligibility"),
                "runtime_decision": pred.get("runtime_decision"),
                "runtime_limit_reason": pred.get("runtime_limit_reason"),
                "alignment": match,
                "matched": True,
            })

    base = evaluate_predictions(records, predictions)
    matched_count = sum(x.get("matched_count", 0) for x in alignments.values())
    total_gold = len(records)
    gold_claim_recall = matched_count / total_gold if total_gold else 0.0
    exact_matches = sum(x.get("exact_normalized_match_count", 0) for x in alignments.values())
    pending_ids = [x.record_id for x in records if x.record_id in PENDING_GOLD_REAUDIT_IDS]
    clean_records = [x for x in records if x.record_id not in PENDING_GOLD_REAUDIT_IDS]
    clean_predictions = [x for x in predictions if x.get("record_id") not in PENDING_GOLD_REAUDIT_IDS]
    clean_metrics = evaluate_predictions(clean_records, clean_predictions) if clean_records else None

    report = {
        "benchmark_version": FULL_PIPELINE_VERSION,
        "dataset_role": DATASET_ROLE,
        "mode": "FULL_PIPELINE_TEMPORARY_GOLD_ENDPOINT_EVALUATION",
        "gold_claim_count": total_gold,
        "video_count": len(gold_by_video),
        "video_run_count": len(video_runs),
        "alignment": {
            "matched_gold_claim_count": matched_count,
            "unmatched_gold_claim_count": total_gold - matched_count,
            "gold_claim_recall": round(gold_claim_recall, 6),
            "exact_normalized_match_count": exact_matches,
            "exact_normalized_match_rate_over_matched": round(exact_matches / matched_count, 6) if matched_count else 0.0,
            "unmatched_system_claim_count": system_extra_claim_count,
            "note": "Temporary Gold is not assumed exhaustive; unmatched system claims are reported but not treated as false positives.",
            "by_video": alignments,
        },
        "full_pipeline_metrics": base,
        "clean_semantic_subset": {
            "excluded_pending_gold_reaudit_ids": pending_ids,
            "metrics": clean_metrics,
            "note": "Development-only diagnostic subset; exclusion is audit-driven and must not be used as final thesis metric.",
        },
        "root_cause_distribution": dict(sorted(root_causes.items())),
        "claim_diagnostics": claim_details,
        "run_health": {
            "video_status_distribution": dict(video_statuses),
            "phase_status_distribution": {k: dict(v) for k, v in phase_statuses.items()},
        },
        "safety": {
            "violation_count": base.get("safety_violation_count", 0),
            "violations": base.get("safety_violations", []),
        },
        "gold_reaudit": {
            "pending_count": len(pending_ids),
            "record_ids": pending_ids,
            "automatic_relabel_allowed": False,
        },
        "interpretation": {
            "temporary_gold_only": True,
            "final_thesis_metric_allowed": False,
            "gold_used_as_runtime_input": False,
            "gold_evidence_supplied_to_runtime": False,
            "technical_failure_is_nei": False,
            "unmatched_system_claim_is_false_positive": False,
        },
    }
    return predictions, report


def save_evaluation_bundle(
    *,
    predictions_path: Path,
    predictions: Iterable[dict[str, Any]],
    report_path: Path,
    report: dict[str, Any],
) -> None:
    write_jsonl(predictions_path, predictions)
    write_json(report_path, report)
