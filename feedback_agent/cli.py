from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .agent import FeedbackLoopAgent
from .config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the configurable feedback-loop coding agent.")
    parser.add_argument("--config", default="config.example.json")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock model responses.")
    args = parser.parse_args()

    repo_root = Path(os.getenv("REPO_ROOT", Path.cwd())).resolve()
    config = load_config(args.config, repo_root=repo_root)
    summary = FeedbackLoopAgent(config, mock=args.mock).run()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
