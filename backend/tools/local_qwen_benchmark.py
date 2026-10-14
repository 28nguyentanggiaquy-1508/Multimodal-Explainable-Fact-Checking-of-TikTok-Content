from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _gpu_snapshot() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        raw = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
        parts = [x.strip() for x in raw.splitlines()[0].split(",")]
        return {
            "name": parts[0],
            "memory_total_mib": int(parts[1]),
            "memory_used_mib": int(parts[2]),
            "memory_free_mib": int(parts[3]),
            "gpu_util_percent": int(parts[4]),
            "temperature_c": int(parts[5]),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _run_once(base_url: str, model: str, *, image: Path | None, num_ctx: int, timeout: float) -> dict[str, Any]:
    prompt = (
        'Return ONLY JSON with this exact shape: '
        '{"status":"ok","model_test":true,"summary":"short sentence"}. '
        'If an image is attached, briefly identify its visible subject in summary.'
    )
    message: dict[str, Any] = {"role": "user", "content": prompt}
    if image is not None:
        message["images"] = [_image_b64(image)]
    body = {
        "model": model,
        "stream": False,
        "format": "json",
        "keep_alive": "10m",
        "options": {"temperature": 0, "num_predict": 256, "num_ctx": num_ctx},
        "messages": [message],
    }
    gpu_before = _gpu_snapshot()
    started = time.perf_counter()
    response = requests.post(f"{base_url.rstrip('/')}/api/chat", json=body, timeout=timeout)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    payload = response.json()
    gpu_after = _gpu_snapshot()
    text = ((payload.get("message") or {}).get("content") or "")
    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        pass
    eval_count = int(payload.get("eval_count") or 0)
    eval_duration_ns = int(payload.get("eval_duration") or 0)
    prompt_count = int(payload.get("prompt_eval_count") or 0)
    prompt_duration_ns = int(payload.get("prompt_eval_duration") or 0)
    return {
        "model": model,
        "elapsed_seconds": round(elapsed, 3),
        "json_valid": isinstance(parsed, dict),
        "response": parsed if isinstance(parsed, dict) else text[:500],
        "load_duration_seconds": round(int(payload.get("load_duration") or 0) / 1e9, 3),
        "prompt_eval_count": prompt_count,
        "prompt_tokens_per_second": round(prompt_count / (prompt_duration_ns / 1e9), 2) if prompt_duration_ns else None,
        "eval_count": eval_count,
        "generation_tokens_per_second": round(eval_count / (eval_duration_ns / 1e9), 2) if eval_duration_ns else None,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark local Qwen3-VL through Ollama without touching dataset state.")
    ap.add_argument("--models", default="qwen3-vl:4b-instruct-q4_K_M,qwen3-vl:8b-instruct-q4_K_M")
    ap.add_argument("--base-url", default=os.getenv("LOCAL_VLM_BASE_URL", "http://127.0.0.1:11434"))
    ap.add_argument("--image", type=Path)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--runs", type=int, default=2, help="Per model; first run includes cold-load when applicable.")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.image and not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    try:
        tags = requests.get(f"{args.base_url.rstrip('/')}/api/tags", timeout=5)
        tags.raise_for_status()
        installed = [str(x.get("name") or x.get("model") or "") for x in tags.json().get("models", [])]
    except Exception as exc:
        print(f"[FAIL] Ollama endpoint unavailable: {type(exc).__name__}: {exc}")
        return 2

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "num_ctx": args.num_ctx,
        "image": str(args.image) if args.image else None,
        "installed_models": installed,
        "models": {},
    }

    for model in models:
        print(f"\n=== {model} ===")
        base = model.split(":", 1)[0]
        if not any(name == model or name.split(":", 1)[0] == base for name in installed):
            print(f"SKIP: model not installed. Run: ollama pull {model}")
            report["models"][model] = {"status": "NOT_INSTALLED", "runs": []}
            continue
        rows = []
        for idx in range(1, max(1, args.runs) + 1):
            try:
                row = _run_once(args.base_url, model, image=args.image, num_ctx=args.num_ctx, timeout=args.timeout)
                rows.append(row)
                print(
                    f"run={idx} elapsed={row['elapsed_seconds']}s "
                    f"gen_tps={row['generation_tokens_per_second']} json={row['json_valid']} "
                    f"gpu_after={row['gpu_after'].get('memory_used_mib')}MiB"
                )
            except Exception as exc:
                rows.append({"error": f"{type(exc).__name__}: {exc}"})
                print(f"run={idx} ERROR {type(exc).__name__}: {exc}")
                break
        report["models"][model] = {"status": "DONE", "runs": rows}

    output = args.output or (BACKEND / "runtime" / f"local_qwen_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
