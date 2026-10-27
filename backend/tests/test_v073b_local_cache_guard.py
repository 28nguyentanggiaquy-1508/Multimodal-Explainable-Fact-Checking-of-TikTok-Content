from __future__ import annotations

from fact_checking.verification import multivlm_verifier as verifier


def test_local_cache_v2_is_rejected() -> None:
    cached = {
        "status": "SUCCESS",
        "operational_metadata": {
            "local_component_guard": {"mode": "local_component_evidence_guard_v2"}
        },
    }
    assert verifier._cached_result_compatible("local", cached) is False


def test_local_cache_v3_is_accepted() -> None:
    cached = {
        "status": "SUCCESS",
        "operational_metadata": {
            "local_component_guard": {"mode": "local_component_evidence_guard_v3"}
        },
    }
    assert verifier._cached_result_compatible("local", cached) is True


def test_local_cache_bypass_env(monkeypatch) -> None:
    monkeypatch.delenv("PHASE4_CACHE_BYPASS", raising=False)
    monkeypatch.setenv("LOCAL_PHASE4_CACHE_BYPASS", "1")
    assert verifier._cache_bypass("local") is True
    assert verifier._cache_bypass("gemini") is False
