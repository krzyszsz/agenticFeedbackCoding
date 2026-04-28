from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .agent import FeedbackLoopAgent
from .config import load_config


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    steps = []
    for item in summary.get("steps", []):
        attempts = item.get("attempts", [])
        steps.append({
            "step_id": item.get("step_id"),
            "status": item.get("status"),
            "attempts": len(attempts),
            "last_review_status": (attempts[-1].get("review", {}) if attempts else {}).get("status"),
        })
    return {
        "final_status": summary.get("final_status"),
        "workspace": summary.get("workspace"),
        "steps": steps,
        "final_review_status": (summary.get("final_review") or {}).get("status"),
        "transcript_jsonl": summary.get("transcript_jsonl"),
        "transcript_markdown": summary.get("transcript_markdown"),
        "active_transcript_jsonl": summary.get("active_transcript_jsonl"),
        "active_transcript_markdown": summary.get("active_transcript_markdown"),
        "summary_json": ".agent_state/summary.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the configurable feedback-loop coding agent.")
    parser.add_argument("--config", default="config.example.json")
    args = parser.parse_args()

    repo_root = Path(os.getenv("REPO_ROOT", Path.cwd())).resolve()
    config = load_config(args.config, repo_root=repo_root)
    summary = FeedbackLoopAgent(config).run()
    mode = config.runtime.final_summary.lower()
    if mode == "full":
        print(json.dumps(summary, indent=2))
    elif mode != "none":
        print(json.dumps(_compact_summary(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
