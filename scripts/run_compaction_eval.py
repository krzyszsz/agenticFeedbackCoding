#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feedback_agent.compaction_eval import (
    build_stress_corpus,
    extract_corpus,
    iter_corpus,
    judge_results,
    render_summary_markdown,
    run_corpus,
    summarize_results,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract and evaluate local-model context compaction in isolation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="freeze exact first-compaction transcript states")
    extract.add_argument("--source-root", type=Path, default=Path("workspaces"))
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--cases", type=int, default=240)
    extract.add_argument("--development-cases", type=int, default=32)
    extract.add_argument("--seed", type=int, default=20260822)
    extract.add_argument(
        "--exclude-corpus",
        type=Path,
        action="append",
        default=[],
        help="exclude stable case IDs present in an earlier corpus; may be repeated",
    )

    stress = subparsers.add_parser(
        "stress",
        help="derive long-request, provenance, noise, and repeated-compaction stresses",
    )
    stress.add_argument("--corpus", type=Path, required=True)
    stress.add_argument("--output", type=Path, required=True)
    stress.add_argument("--cases", type=int, default=300)
    stress.add_argument("--seed", type=int, default=20260823)

    run = subparsers.add_parser("run", help="run one local model over a frozen corpus")
    run.add_argument("--corpus", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--profile", required=True)
    run.add_argument("--base-url")
    run.add_argument("--split", choices=("all", "development", "heldout"), default="all")
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--summary-max-tokens", type=int, default=2048)
    run.add_argument("--reasoning-budget-tokens", type=int, default=1024)
    run.add_argument("--critical-reasoning-budget-tokens", type=int, default=4096)
    run.add_argument("--model-repair-attempts", type=int, default=1)
    run.add_argument("--request-timeout-seconds", type=int, default=21600)
    run.add_argument(
        "--production-flow",
        action="store_true",
        help="exercise maybe_compact staging and active-context assembly, not only the summarizer",
    )

    judge = subparsers.add_parser("judge", help="semantically review compaction results with a local model")
    judge.add_argument("--corpus", type=Path, required=True)
    judge.add_argument("--results", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--profile", required=True)
    judge.add_argument("--base-url")
    judge.add_argument("--limit", type=int, default=0)
    judge.add_argument("--reasoning-budget-tokens", type=int, default=1024)
    judge.add_argument("--request-timeout-seconds", type=int, default=21600)

    report = subparsers.add_parser("report", help="rebuild summary files from a result JSONL")
    report.add_argument("--results", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "extract":
        excluded_ids = {
            item["id"]
            for corpus_path in args.exclude_corpus
            for item in iter_corpus(corpus_path)
        }
        result = extract_corpus(
            args.source_root,
            args.output,
            case_count=args.cases,
            development_count=args.development_cases,
            seed=args.seed,
            excluded_ids=excluded_ids,
        )
    elif args.command == "stress":
        result = build_stress_corpus(
            args.corpus,
            args.output,
            case_count=args.cases,
            seed=args.seed,
        )
    elif args.command == "run":
        result = run_corpus(
            args.corpus,
            args.output,
            profile_name=args.profile,
            base_url=args.base_url,
            split=args.split,
            limit=args.limit,
            summary_max_tokens=args.summary_max_tokens,
            reasoning_budget_tokens=args.reasoning_budget_tokens,
            critical_reasoning_budget_tokens=args.critical_reasoning_budget_tokens,
            model_repair_attempts=args.model_repair_attempts,
            request_timeout_seconds=args.request_timeout_seconds,
            production_flow=args.production_flow,
        )
    elif args.command == "judge":
        result = judge_results(
            args.corpus,
            args.results,
            args.output,
            profile_name=args.profile,
            base_url=args.base_url,
            limit=args.limit,
            reasoning_budget_tokens=args.reasoning_budget_tokens,
            request_timeout_seconds=args.request_timeout_seconds,
        )
    else:
        result = summarize_results(args.results)
        args.results.with_suffix(".summary.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        args.results.with_suffix(".summary.md").write_text(
            render_summary_markdown(result),
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
