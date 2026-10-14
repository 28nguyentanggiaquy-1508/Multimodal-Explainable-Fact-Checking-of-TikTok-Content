from __future__ import annotations

import json
import os

from core.vlm.providers import local_provider


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_task_specific_models(monkeypatch):
    monkeypatch.setenv("LOCAL_VLM_MODEL", "fallback")
    monkeypatch.setenv("LOCAL_PHASE2_MODEL", "phase2-model")
    monkeypatch.setenv("LOCAL_PHASE4_MODEL", "phase4-model")
    assert local_provider.configured_model("phase2") == "phase2-model"
    assert local_provider.configured_model("phase4") == "phase4-model"
    assert local_provider.configured_model() == "fallback"


def test_health_check(monkeypatch):
    monkeypatch.setenv("LOCAL_VLM_ENABLED", "true")
    monkeypatch.setenv("LOCAL_PHASE4_MODEL", "qwen3-vl:8b-instruct-q4_K_M")
    monkeypatch.setattr(local_provider.requests, "get", lambda *a, **k: _Response({"models": [{"name": "qwen3-vl:8b-instruct-q4_K_M"}]}))
    result = local_provider.health_check("phase4")
    assert result["reachable"] is True
    assert result["model_installed"] is True


def test_generate_json_ollama(monkeypatch):
    monkeypatch.setenv("LOCAL_VLM_ENABLED", "true")
    monkeypatch.setenv("LOCAL_VLM_API_STYLE", "ollama")
    monkeypatch.setenv("LOCAL_VLM_MODEL", "qwen3-vl:4b-instruct-q4_K_M")
    monkeypatch.setenv("LOCAL_VLM_NUM_CTX", "8192")
    monkeypatch.setattr(local_provider.circuit, "is_open", lambda *a, **k: False)
    monkeypatch.setattr(local_provider.circuit, "close_circuit", lambda *a, **k: None)

    captured = {}
    def _post(url, json=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["body"] = json
        return _Response({"message": {"content": '{"ok": true}'}})
    monkeypatch.setattr(local_provider.requests, "post", _post)

    result = local_provider.generate_json(prompt="return json", max_output_tokens=256)
    assert result.data == {"ok": True}
    assert result.provider == "local"
    assert captured["body"]["format"] == "json"
    assert captured["body"]["options"]["num_ctx"] == 8192


def test_generate_json_ollama_structured_schema(monkeypatch):
    monkeypatch.setenv("LOCAL_VLM_ENABLED", "true")
    monkeypatch.setenv("LOCAL_VLM_API_STYLE", "ollama")
    monkeypatch.setenv("LOCAL_VLM_MODEL", "qwen3-vl:8b-instruct-q4_K_M")
    monkeypatch.setattr(local_provider.circuit, "is_open", lambda *a, **k: False)
    monkeypatch.setattr(local_provider.circuit, "close_circuit", lambda *a, **k: None)

    captured = {}
    schema = {
        "type": "object",
        "properties": {"verdict": {"type": "string"}},
        "required": ["verdict"],
        "additionalProperties": False,
    }

    def _post(url, json=None, timeout=None, **kwargs):
        captured["body"] = json
        return _Response({"message": {"content": '{"verdict":"NEI"}'}, "done_reason": "stop"})

    monkeypatch.setattr(local_provider.requests, "post", _post)
    result = local_provider.generate_json(
        prompt="return json",
        max_output_tokens=256,
        response_schema=schema,
    )
    assert result.data == {"verdict": "NEI"}
    assert captured["body"]["format"] == schema
