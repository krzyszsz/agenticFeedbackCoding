from __future__ import annotations

import argparse
from dataclasses import replace
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
    parser.add_argument("--workspace", help="Override runtime.workspace from the config.")
    parser.add_argument("--title", help="Override project_design.title from the config.")
    parser.add_argument("--prompt", help="Override project_design.prompt from the config.")
    parser.add_argument("--prompt-file", help="Read project_design.prompt from a text file.")
    parser.add_argument("--offline", action="store_true", help="Disable web research/scraping for this run.")
    args = parser.parse_args()

    repo_root = Path(os.getenv("REPO_ROOT", Path.cwd())).resolve()
    config = load_config(args.config, repo_root=repo_root)
    if args.workspace:
        workspace = Path(args.workspace)
        if not workspace.is_absolute():
            workspace = (repo_root / workspace).resolve()
        config = replace(config, runtime=replace(config.runtime, workspace=workspace))
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    if prompt is not None or args.title:
        config = replace(
            config,
            project_design=replace(
                config.project_design,
                title=args.title or config.project_design.title,
                prompt=prompt if prompt is not None else config.project_design.prompt,
            ),
        )
    if args.offline:
        config = replace(
            config,
            mcp_tools=replace(config.mcp_tools, web_scraping=False),
            web_research=replace(config.web_research, enabled=False),
        )
    summary = FeedbackLoopAgent(config).run()
    mode = config.runtime.final_summary.lower()
    if mode == "full":
        print(json.dumps(summary, indent=2))
    elif mode != "none":
        print(json.dumps(_compact_summary(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
