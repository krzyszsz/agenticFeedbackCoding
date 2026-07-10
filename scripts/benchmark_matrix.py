from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_benchmarks import (
    display_grade,
    load_suite_ids,
    load_tasks,
    resolve_selection_ids,
    select_tasks,
)


def load_result_file(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    return [item for item in results if isinstance(item, dict)]


def parse_result_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--result must be LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("--result label must not be empty")
    return label, Path(path)


def latest_by_task(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_task: dict[str, dict[str, Any]] = {}
    for result in results:
        task_id = result.get("task_id")
        if isinstance(task_id, str):
            by_task[task_id] = result
    return by_task


def grade_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "pass": 0,
        "fail": 0,
        "manual_pass": 0,
        "manual_fail": 0,
        "manual_review": 0,
        "timeouts": 0,
    }
    for result in results:
        grade = result.get("grade")
        if grade == "pass":
            counts["pass"] += 1
        elif grade == "fail":
            counts["fail"] += 1
        elif grade == "manual_pass":
            counts["manual_pass"] += 1
        elif grade == "manual_fail":
            counts["manual_fail"] += 1
        elif grade == "manual_review":
            counts["manual_review"] += 1
        if result.get("returncode") == 124:
            counts["timeouts"] += 1
    return counts


def format_cell(result: dict[str, Any] | None) -> str:
    if result is None:
        return "-"
    grade = display_grade(str(result.get("grade", "unknown")))
    minutes = int(round(float(result.get("elapsed_seconds") or 0) / 60.0))
    return f"{grade} {minutes}m"


def summary_table(label_results: list[tuple[str, Path, list[dict[str, Any]]]]) -> str:
    lines = [
        "| Run | Tasks | Pass | Fail | Manual pass | Manual fail | Manual review | Timeouts | Avg s | Total h | Evidence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label, path, results in label_results:
        counts = grade_counts(results)
        total_seconds = sum(float(result.get("elapsed_seconds") or 0) for result in results)
        avg_seconds = total_seconds / max(1, len(results))
        pass_total = counts["pass"] + counts["manual_pass"]
        fail_total = counts["fail"] + counts["manual_fail"]
        lines.append(
            f"| {label} | {len(results)} | {pass_total} | {fail_total} | "
            f"{counts['manual_pass']} | {counts['manual_fail']} | {counts['manual_review']} | "
            f"{counts['timeouts']} | {avg_seconds:.1f} | {total_seconds / 3600:.2f} | `{path}` |"
        )
    return "\n".join(lines) + "\n"


def matrix_table(tasks: list[dict[str, Any]], label_results: list[tuple[str, Path, list[dict[str, Any]]]]) -> str:
    task_maps = [(label, latest_by_task(results)) for label, _path, results in label_results]
    header = "| Task | " + " | ".join(label for label, _task_map in task_maps) + " |"
    separator = "|---|" + "|".join("---:" for _label, _task_map in task_maps) + "|"
    lines = [header, separator]
    for task in tasks:
        task_id = task["id"]
        cells = [format_cell(task_map.get(task_id)) for _label, task_map in task_maps]
        lines.append(f"| `{task_id}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Markdown comparison matrix from benchmark result files.")
    parser.add_argument("--tasks", default="benchmarks/tasks.json")
    parser.add_argument("--suites", default="benchmarks/suites.json")
    parser.add_argument("--suite", default="publication-30")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--result", action="append", type=parse_result_arg, required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    suite_ids = load_suite_ids(REPO_ROOT / args.suites, args.suite)
    tasks = select_tasks(load_tasks(REPO_ROOT / args.tasks), resolve_selection_ids(suite_ids, args.task_id), args.limit)
    label_results = [(label, path, load_result_file(path)) for label, path in args.result]
    output = (
        "## Summary\n\n"
        + summary_table(label_results)
        + "\n## Per-task Matrix\n\n"
        + matrix_table(tasks, label_results)
    )
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
