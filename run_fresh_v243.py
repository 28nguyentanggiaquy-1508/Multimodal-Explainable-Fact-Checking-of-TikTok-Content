import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, "backend")

from fact_checking.pipeline import run_fact_check_pipeline

VIDEO_URL = "https://www.tiktok.com/@thuvienphapluat.vn/video/7571730219149561096?q=vi%E1%BB%87t%20nam%20c%C3%B3%20bao%20nhi%C3%AAu%20t%C6%B0%E1%BB%9Bng&t=1791389591280"

async def main():
    result = await run_fact_check_pipeline(VIDEO_URL)

    output = Path("tiktok_mfc_d3/inbox/phase14/FRESH_V240_VALIDATION_RUN2.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== FRESH V2.4.3 VALIDATION ===")
    print("pipeline_version =", result.get("pipeline_version"))
    print("status =", result.get("status"))
    print("claim_count =", result.get("claim_count"))
    print("verdict_ready =", result.get("verdict_ready"))
    print("explanation_ready =", result.get("explanation_ready"))
    print("output =", output)

    for row in result.get("claims", []):
        claim = row.get("claim") or {}
        ver = row.get("verification") or {}
        print(
            claim.get("id"),
            "|", ver.get("status"),
            "|", ver.get("verdict"),
            "|", ver.get("provider"),
            "|", ver.get("model"),
        )

asyncio.run(main())
