from __future__ import annotations

import asyncio

from fact_checking.verification import provider_readiness_v073 as readiness
from fact_checking.verification import verification_service_multivlm as service


def test_provider_readiness_skips_unconfigured_local(monkeypatch):
    monkeypatch.setattr(readiness.local_provider, "is_configured", lambda task=None: False)
    result = readiness.provider_readiness("local")
    assert result.configured is False
    assert result.ready is False
    assert result.reason == "not_configured"


def test_provider_readiness_reports_circuit_block(monkeypatch):
    monkeypatch.setattr(readiness.gemini_provider, "is_configured", lambda: True)
    monkeypatch.setattr(readiness, "provider_models", lambda provider: ["gemini-x"])
    monkeypatch.setattr(readiness.circuit, "remaining_seconds", lambda provider, model: 123.0)
    monkeypatch.setattr(readiness.quota_manager, "snapshot", lambda provider, model: {})
    result = readiness.provider_readiness("gemini")
    assert result.ready is False
    assert result.reason == "all_models_blocked"
    assert 122 <= result.next_retry_after_seconds <= 123


def test_verifier_preflight_skips_blocked_and_uses_ready_provider(monkeypatch):
    monkeypatch.setattr(service, "prepare_verification_inputs", lambda **kwargs: (kwargs["evidence"], kwargs["multimedia"], kwargs["reasoning_support"], {"ok": True}))
    monkeypatch.setattr(service, "_provider_order", lambda: ["gemini", "local", "openai"])

    class R:
        def __init__(self, provider, ready, reason, wait=0.0):
            self.provider = provider
            self.ready = ready
            self.reason = reason
            self.next_retry_after_seconds = wait

    states = {
        "gemini": R("gemini", False, "all_models_blocked", 100.0),
        "local": R("local", False, "not_configured", 0.0),
        "openai": R("openai", True, "ready", 0.0),
    }
    monkeypatch.setattr(service, "provider_readiness", lambda provider: states[provider])
    monkeypatch.setattr(service, "readiness_snapshot", lambda order: [{"provider": p, "ready": states[p].ready, "next_retry_after_seconds": states[p].next_retry_after_seconds} for p in order])

    called = []
    monkeypatch.setattr(service, "verify_gemini", lambda **kwargs: (_ for _ in ()).throw(AssertionError("blocked Gemini must not be called")))
    monkeypatch.setattr(service, "verify_local", lambda **kwargs: (_ for _ in ()).throw(AssertionError("unconfigured local must not be called")))

    def openai(**kwargs):
        called.append("openai")
        return {
            "status": "SUCCESS",
            "verdict": "SUPPORTED",
            "confidence": 0.9,
            "evidence_ids": ["e1"],
            "evidence_assessments": [],
            "evidence_coverage": "COMPLETE",
            "rationale": "ok",
            "uncertainty": "",
            "limitations": [],
            "multimodal_used": False,
            "model": "gpt-test",
            "provider": "openai",
        }

    monkeypatch.setattr(service, "verify_openai", openai)
    result = asyncio.run(service.verify_claim(claim={}, multimedia={}, evidence=[{"id": "e1"}], reasoning_support={}))
    assert result.status == "SUCCESS"
    assert result.verdict == "SUPPORTED"
    assert called == ["openai"]
    attempts = result.operational_metadata["provider_attempts"]
    assert attempts[0]["status"] == "SKIPPED"
    assert attempts[0]["provider"] == "gemini"
    assert attempts[1]["status"] == "SKIPPED"
    assert attempts[2]["status"] == "SUCCESS"
    assert result.operational_metadata["provider_routing_version"] == "v0.7.3"
