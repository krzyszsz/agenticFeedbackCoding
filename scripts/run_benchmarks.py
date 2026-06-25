from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feedback_agent.config import DEFAULT_CONFIG, _deep_merge
from feedback_agent.model_profiles import resolve_profile
from feedback_agent.workspace import run_commands, write_files


def load_tasks(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["tasks"])


def select_tasks(tasks: list[dict[str, Any]], ids: list[str], limit: int | None) -> list[dict[str, Any]]:
    if ids:
        wanted = set(ids)
        tasks = [task for task in tasks if task["id"] in wanted]
        missing = wanted - {task["id"] for task in tasks}
        if missing:
            raise SystemExit(f"Unknown benchmark task id(s): {', '.join(sorted(missing))}")
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def benchmark_config(
    task: dict[str, Any],
    *,
    repo_root: Path,
    workspace: Path,
    implementation_profile: str,
    feedback_profile: str | None,
    docker_isolation: bool,
    reasoning_budget_tokens: int | None,
) -> dict[str, Any]:
    impl = resolve_profile(implementation_profile)
    feedback = resolve_profile(feedback_profile) if feedback_profile else None
    model_cfg = {
        "name": impl.name,
        "base_url": f"http://127.0.0.1:{impl.port}/v1",
        "api_key": "not-needed",
        "model": "local-gguf",
        "context_window": impl.context_window,
        "max_tokens": 32768,
        "temperature": 0.2,
        "request_timeout_seconds": 21600,
        "retry_attempts": 20,
        "retry_sleep_seconds": 30,
        "request_heartbeat_seconds": 30,
        "preserve_reasoning": True,
        "reasoning_budget_tokens": reasoning_budget_tokens or impl.reasoning_budget_tokens,
    }
    data: dict[str, Any] = {
        "implementation_model": model_cfg,
        "feedback_model": None,
        "runtime": {
            "docker_isolation": docker_isolation,
            "workspace": str(workspace.relative_to(repo_root) if workspace.is_relative_to(repo_root) else workspace),
            "command_timeout_seconds": 120,
            "max_command_timeout_seconds": 21600,
            "print_transcript": True,
            "live_turn_max_chars": 20000,
            "final_summary": "compact",
        },
        "mcp_tools": {
            "terminal": True,
            "web_scraping": task.get("web_research", False),
            "web_interaction": True,
        },
        "loop": {
            "max_approach_reattempts": 5,
        },
        "project_design": {
            "title": task["title"],
            "prompt": task["prompt"],
        },
    }
    if feedback:
        data["feedback_model"] = {
            **model_cfg,
            "name": feedback.name,
            "base_url": f"http://127.0.0.1:{feedback.port}/v1",
            "context_window": feedback.context_window,
            "reasoning_budget_tokens": reasoning_budget_tokens or feedback.reasoning_budget_tokens,
        }
    return _deep_merge(DEFAULT_CONFIG, data)


def write_config(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def seed_workspace(workspace: Path, task: dict[str, Any]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    setup_files = task.get("setup_files") or []
    if setup_files:
        write_files(workspace, setup_files)


def run_harness(repo_root: Path, config_path: Path, *, implementation_profile: str, feedback_profile: str | None) -> tuple[int, float, str]:
    env = {
        **os.environ,
        "MODEL_PROFILE": implementation_profile,
    }
    if feedback_profile:
        env["FEEDBACK_MODEL_PROFILE"] = feedback_profile
    start = time.monotonic()
    proc = subprocess.run(
        ["bash", "scripts/run_agent.sh", "--config", str(config_path)],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return proc.returncode, time.monotonic() - start, proc.stdout


def grade_task(workspace: Path, task: dict[str, Any]) -> dict[str, Any]:
    grading = task.get("grading", "manual")
    commands = task.get("post_validation_commands") or []
    validation_results = []
    if commands:
        validation_results = run_commands(workspace, commands, 120, 21600, output_limit_chars=8000)
    if grading == "manual":
        status = "manual_review"
    elif validation_results:
        status = "pass" if all(result["returncode_matches_expected"] and not result["timed_out"] for result in validation_results) else "fail"
    else:
        summary_path = workspace / ".agent_state" / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            status = "pass" if summary.get("final_status") == "resolved" else "fail"
        else:
            status = "fail"
    return {"grade": status, "validation_results": validation_results}


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    summary_path = Path(result["workspace"]) / ".agent_state" / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {
        "final_status": summary.get("final_status"),
        "steps": len(summary.get("steps", [])),
        "approach_attempts": len(summary.get("approach_history", [])),
        "final_review_status": (summary.get("final_review") or {}).get("status"),
        "approach_review_status": (summary.get("approach_review") or {}).get("status"),
    }


def markdown_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Task | Category | Model | Verifier | Budget | Grade | Final | Seconds | Approach Attempts |",
        "|---|---|---|---|---:|---|---:|---:|---:|",
    ]
    for result in results:
        summary = result.get("summary") or {}
        lines.append(
            "| {task} | {category} | {model} | {verifier} | {budget} | {grade} | {final} | {seconds:.1f} | {attempts} |".format(
                task=result["task_id"],
                category=result["category"],
                model=result["implementation_profile"],
                verifier=result.get("feedback_profile") or "same",
                budget=result.get("reasoning_budget_tokens") or "profile",
                grade=result["grade"],
                final=summary.get("final_status", "n/a"),
                seconds=float(result.get("elapsed_seconds") or 0),
                attempts=summary.get("approach_attempts", "n/a"),
            )
        )
    return "\n".join(lines) + "\n"


def summary_table(results: list[dict[str, Any]]) -> str:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        key = (result["implementation_profile"], result.get("feedback_profile") or "same")
        groups.setdefault(key, []).append(result)
    lines = [
        "| Model | Verifier | Budget | Tasks | Pass | Fail | Manual | Avg Seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (model, verifier), items in sorted(groups.items()):
        passed = sum(1 for item in items if item["grade"] == "pass")
        failed = sum(1 for item in items if item["grade"] == "fail")
        manual = sum(1 for item in items if item["grade"] == "manual_review")
        avg = sum(float(item.get("elapsed_seconds") or 0) for item in items) / max(1, len(items))
        budgets = sorted({str(item.get("reasoning_budget_tokens") or "profile") for item in items})
        lines.append(f"| {model} | {verifier} | {', '.join(budgets)} | {len(items)} | {passed} | {failed} | {manual} | {avg:.1f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agenticFeedbackCoding benchmark corpus.")
    parser.add_argument("--tasks", default="benchmarks/tasks.json")
    parser.add_argument("--implementation-profile", default="gemma4-26b-a4b-qat-mtp")
    parser.add_argument("--feedback-profile")
    parser.add_argument("--reasoning-budget-tokens", type=int)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--docker-isolation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    tasks = select_tasks(load_tasks(repo_root / args.tasks), args.task_id, args.limit)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "runs" / f"benchmarks-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    if args.dry_run:
        for task in tasks:
            print(f"{task['id']}\t{task['category']}\t{task.get('grading', 'manual')}\t{task['title']}")
        return 0

    for task in tasks:
        workspace = repo_root / "workspaces" / "benchmarks" / stamp / task["id"]
        seed_workspace(workspace, task)
        config_path = output_dir / f"{task['id']}.json"
        cfg = benchmark_config(
            task,
            repo_root=repo_root,
            workspace=workspace,
            implementation_profile=args.implementation_profile,
            feedback_profile=args.feedback_profile,
            docker_isolation=args.docker_isolation,
            reasoning_budget_tokens=args.reasoning_budget_tokens,
        )
        write_config(config_path, cfg)
        returncode, elapsed, output = run_harness(
            repo_root,
            config_path,
            implementation_profile=args.implementation_profile,
            feedback_profile=args.feedback_profile,
        )
        (output_dir / f"{task['id']}.log").write_text(output, encoding="utf-8")
        grade = grade_task(workspace, task)
        result = {
            "task_id": task["id"],
            "title": task["title"],
            "category": task["category"],
            "grading": task.get("grading", "manual"),
            "implementation_profile": args.implementation_profile,
            "feedback_profile": args.feedback_profile,
            "reasoning_budget_tokens": args.reasoning_budget_tokens,
            "workspace": str(workspace),
            "returncode": returncode,
            "elapsed_seconds": elapsed,
            "grade": grade["grade"] if returncode == 0 else "fail",
            "post_validation": grade["validation_results"],
        }
        result["summary"] = summarize_result(result)
        results.append(result)
        (output_dir / "results.json").write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
        (output_dir / "results.md").write_text(
            "# Benchmark Results\n\n## Summary\n\n"
            + summary_table(results)
            + "\n## Details\n\n"
            + markdown_table(results),
            encoding="utf-8",
        )
        print(f"{task['id']}: {result['grade']} in {elapsed:.1f}s")

    print(f"Results: {output_dir / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
