from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feedback_agent.config import DEFAULT_CONFIG, ModelConfig, _deep_merge
from feedback_agent.llm import OpenAICompatClient
from feedback_agent.model_profiles import ModelProfile, resolve_profile
from feedback_agent.workspace import collect_workspace_files, extract_json_object, run_commands, write_files


def load_tasks(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["tasks"])


def load_suite_ids(path: Path, suite: str | None) -> list[str]:
    if not suite:
        return []
    if not path.exists():
        raise SystemExit(f"Benchmark suite file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    suites = data.get("suites", {})
    if suite not in suites:
        known = ", ".join(sorted(suites))
        raise SystemExit(f"Unknown benchmark suite '{suite}'. Known suites: {known}")
    entry = suites[suite]
    task_ids = entry.get("task_ids") if isinstance(entry, dict) else entry
    if not isinstance(task_ids, list) or not all(isinstance(item, str) for item in task_ids):
        raise SystemExit(f"Benchmark suite '{suite}' must define a list of task_ids.")
    return task_ids


def select_tasks(tasks: list[dict[str, Any]], ids: list[str], limit: int | None) -> list[dict[str, Any]]:
    if ids:
        by_id = {task["id"]: task for task in tasks}
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        missing: set[str] = set()
        for task_id in ids:
            if task_id in seen:
                continue
            seen.add(task_id)
            task = by_id.get(task_id)
            if task is None:
                missing.add(task_id)
            else:
                selected.append(task)
        if missing:
            raise SystemExit(f"Unknown benchmark task id(s): {', '.join(sorted(missing))}")
        tasks = selected
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def resolve_selection_ids(suite_ids: list[str], explicit_task_ids: list[str]) -> list[str]:
    """Choose benchmark task ids from the most specific CLI selector."""
    return explicit_task_ids if explicit_task_ids else suite_ids


def benchmark_config(
    task: dict[str, Any],
    *,
    repo_root: Path,
    workspace: Path,
    implementation_profile: str,
    feedback_profile: str | None,
    docker_isolation: bool,
    reasoning_budget_tokens: int | None,
    max_tokens: int,
    feedback_response_max_tokens: int,
) -> dict[str, Any]:
    impl = resolve_profile(implementation_profile)
    feedback = resolve_profile(feedback_profile) if feedback_profile else None
    model_cfg = {
        "name": impl.name,
        "base_url": f"http://127.0.0.1:{impl.port}/v1",
        "api_key": "not-needed",
        "model": "local-gguf",
        "context_window": impl.context_window,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "request_timeout_seconds": 21600,
        "retry_attempts": 20,
        "retry_sleep_seconds": 30,
        "request_heartbeat_seconds": 30,
        "preserve_reasoning": True,
        "reasoning_budget_tokens": reasoning_budget_tokens or impl.reasoning_budget_tokens,
        "send_reasoning_budget": True,
    }
    data: dict[str, Any] = {
        "implementation_model": model_cfg,
        "feedback_model": None,
        "runtime": {
            "docker_isolation": docker_isolation,
            "workspace": str(workspace.relative_to(repo_root) if workspace.is_relative_to(repo_root) else workspace),
            "command_timeout_seconds": 120,
            "max_command_timeout_seconds": 21600,
            "command_progress_review_interval_seconds": 300,
            "command_progress_review_min_interval_seconds": 30,
            "print_transcript": True,
            "live_turn_max_chars": 20000,
            "final_summary": "compact",
            "feedback_response_max_tokens": feedback_response_max_tokens,
        },
        "mcp_tools": {
            "terminal": True,
            "web_scraping": task.get("web_research", False),
            "web_interaction": True,
        },
        "loop": {
            "max_approach_reattempts": 5,
        },
        "phases": {
            "analysis": {"max_iterations": 2},
            "requirements_refinement": {"max_iterations": 4},
            "plan_validation": {"max_iterations": 4},
            "implementation": {"max_iterations": 7},
        },
        "project_design": {
            "title": task["title"],
            "prompt": task["prompt"],
        },
    }
    if task.get("web_research", False):
        data["web_research"] = {"enabled": True}
    if feedback:
        data["feedback_model"] = {
            **model_cfg,
            "name": feedback.name,
            "base_url": f"http://127.0.0.1:{feedback.port}/v1",
            "context_window": feedback.context_window,
            "reasoning_budget_tokens": reasoning_budget_tokens or feedback.reasoning_budget_tokens,
        }
    if task.get("config_overrides"):
        data = _deep_merge(data, task["config_overrides"])
    return _deep_merge(DEFAULT_CONFIG, data)


def write_config(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def seed_workspace(workspace: Path, task: dict[str, Any]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    setup_files = task.get("setup_files") or []
    if setup_files:
        write_files(workspace, setup_files)


def direct_model_config(
    profile_name: str,
    *,
    reasoning_budget_tokens: int | None,
    max_tokens: int,
) -> ModelConfig:
    profile = resolve_profile(profile_name)
    return ModelConfig(
        name=profile.name,
        base_url=f"http://127.0.0.1:{profile.port}/v1",
        api_key="not-needed",
        model="local-gguf",
        context_window=profile.context_window,
        max_tokens=max_tokens,
        temperature=0.2,
        request_timeout_seconds=21600,
        retry_attempts=20,
        retry_sleep_seconds=30,
        request_heartbeat_seconds=30,
        preserve_reasoning=True,
        reasoning_budget_tokens=reasoning_budget_tokens or profile.reasoning_budget_tokens,
        send_reasoning_budget=True,
    )


def single_shot_prompt(task: dict[str, Any], workspace: Path) -> str:
    workspace_files = collect_workspace_files(workspace, max_file_bytes=12000)
    return (
        "You are running without the agentic feedback harness. This is a single-shot benchmark.\n"
        "Complete the requested project in one response. You cannot ask follow-up questions, run tools, "
        "inspect the filesystem after this response, or receive reviewer repair feedback.\n"
        "Return strict JSON only, with this shape:\n"
        "{\n"
        '  "files": [{"path": "relative/path", "content": "complete file content"}],\n'
        '  "notes": "brief implementation note",\n'
        '  "self_check": ["short check you performed mentally"]\n'
        "}\n"
        "Use only relative paths inside the workspace. Do not include markdown fences or extra prose. "
        "Do not create harness state files such as PLAN.md, REQUIREMENTS.md, RESEARCH.md, or .agent_state files "
        "unless the user explicitly asks for those as project deliverables.\n\n"
        f"Task title: {task['title']}\n"
        f"Task prompt:\n{task['prompt']}\n\n"
        "Existing workspace files, if any, are provided as bounded context. Preserve user files unless the task requires changing them:\n"
        f"{json.dumps(workspace_files, indent=2)}"
    )


def run_single_shot(
    workspace: Path,
    task: dict[str, Any],
    *,
    implementation_profile: str,
    reasoning_budget_tokens: int | None,
    max_tokens: int,
) -> tuple[int, float, str, dict[str, Any]]:
    start = time.monotonic()
    prompt = single_shot_prompt(task, workspace)
    client = OpenAICompatClient(
        direct_model_config(
            implementation_profile,
            reasoning_budget_tokens=reasoning_budget_tokens,
            max_tokens=max_tokens,
        )
    )
    raw = ""
    try:
        raw = client.chat(
            [
                {
                    "role": "system",
                    "content": "You produce one strict JSON object for a coding benchmark. No tools and no follow-up.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        payload = extract_json_object(raw)
        files = payload.get("files", [])
        if not isinstance(files, list):
            raise ValueError("single-shot response field 'files' is not a list")
        written = write_files(workspace, files)
        elapsed = time.monotonic() - start
        metadata = {
            "files_written": written,
            "notes": payload.get("notes", ""),
            "self_check": payload.get("self_check", []),
        }
        log = "===== SINGLE SHOT PROMPT =====\n" + prompt + "\n\n===== SINGLE SHOT RESPONSE =====\n" + raw + "\n"
        return 0, elapsed, log, metadata
    except Exception as exc:
        elapsed = time.monotonic() - start
        metadata = {"error": str(exc), "raw_tail": raw[-4000:]}
        log = (
            "===== SINGLE SHOT PROMPT =====\n"
            + prompt
            + "\n\n===== SINGLE SHOT ERROR =====\n"
            + repr(exc)
            + "\n\n===== SINGLE SHOT RESPONSE TAIL =====\n"
            + raw[-4000:]
            + "\n"
        )
        return 2, elapsed, log, metadata


def _docker_model_url(profile: ModelProfile) -> str:
    return f"http://{profile.container_name}:{profile.port}/v1"


def _timeout_output(exc: subprocess.TimeoutExpired) -> str:
    output = exc.output or ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return str(output)


def _stream_process_output(
    proc: subprocess.Popen[str],
    *,
    timeout_seconds: int | None,
    start: float,
) -> str:
    """Stream real benchmark subprocess output instead of hiding long runs.

    Tests use small fake ``Popen`` objects without a real stdout file descriptor;
    those continue through ``communicate``. Real Docker harness runs get live
    transcript output, so a long model call or tool command is visible while the
    result log still captures the same text.
    """
    pipe = getattr(proc, "stdout", None)
    if pipe is None or not hasattr(pipe, "fileno"):
        stdout, _stderr = proc.communicate(timeout=timeout_seconds)
        return stdout or ""

    selector = selectors.DefaultSelector()
    selector.register(pipe, selectors.EVENT_READ)
    chunks: list[str] = []
    deadline = start + timeout_seconds if timeout_seconds else None
    last_heartbeat = start
    pipe_closed = False
    while not pipe_closed:
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            raise subprocess.TimeoutExpired(
                cmd=getattr(proc, "args", "scripts/run_agent.sh"),
                timeout=timeout_seconds,
                output="".join(chunks),
            )
        events = selector.select(timeout=1.0)
        if not events and proc.poll() is not None:
            events = selector.select(timeout=0)
        for key, _event in events:
            data = os.read(key.fileobj.fileno(), 8192)
            if data:
                text = data.decode("utf-8", errors="replace")
                chunks.append(text)
                print(text, end="", flush=True)
            else:
                selector.unregister(key.fileobj)
                key.fileobj.close()
                pipe_closed = True
        if now - last_heartbeat >= 60 and proc.poll() is None:
            elapsed = int(now - start)
            print(f"[benchmark-runner] harness subprocess still running: {elapsed}s elapsed.", flush=True)
            last_heartbeat = now
    proc.wait(timeout=5)
    return "".join(chunks)


def run_harness(
    repo_root: Path,
    config_path: Path,
    *,
    implementation_profile: str,
    feedback_profile: str | None,
    timeout_seconds: int | None,
) -> tuple[int, float, str]:
    safe_stem = re.sub(r"[^a-zA-Z0-9_.-]+", "-", config_path.stem).strip("-")[:40] or "task"
    container_name = f"agentic-bench-{safe_stem}-{os.getpid()}-{int(time.time() * 1000) % 1000000}"
    env = {
        **os.environ,
        "MODEL_PROFILE": implementation_profile,
        "AGENT_CONTAINER_NAME": container_name,
        "AGENT_CONTAINER_LABEL": "agentic-feedback-benchmark=1",
    }
    if feedback_profile:
        feedback = resolve_profile(feedback_profile)
        env["FEEDBACK_MODEL_PROFILE"] = feedback_profile
        env.setdefault("AGENT_FEEDBACK_BASE_URL", _docker_model_url(feedback))
    start = time.monotonic()
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            ["bash", "scripts/run_agent.sh", "--config", str(config_path)],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        stdout = _stream_process_output(proc, timeout_seconds=timeout_seconds, start=start)
        return proc.returncode, time.monotonic() - start, stdout or ""
    except subprocess.TimeoutExpired as exc:
        output = _timeout_output(exc)
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError:
                proc.terminate()
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
            )
            try:
                more, _stderr = proc.communicate(timeout=5)
                if isinstance(more, bytes):
                    more = more.decode("utf-8", errors="replace")
                output += more or ""
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    proc.kill()
                more, _stderr = proc.communicate(timeout=5)
                if isinstance(more, bytes):
                    more = more.decode("utf-8", errors="replace")
                output += more or ""
        output += f"\n[BENCHMARK_TIMEOUT] harness task exceeded {timeout_seconds} seconds and was stopped.\n"
        return 124, time.monotonic() - start, output


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


def result_matches_run(
    result: dict[str, Any],
    *,
    run_mode: str,
    task_id: str,
    implementation_profile: str,
    feedback_profile: str | None,
    reasoning_budget_tokens: int | None,
    max_tokens: int | None = None,
    feedback_response_max_tokens: int | None = None,
) -> bool:
    matches = (
        result.get("run_mode", "harness") == run_mode
        and result.get("task_id") == task_id
        and result.get("implementation_profile") == implementation_profile
        and result.get("feedback_profile") == feedback_profile
        and result.get("reasoning_budget_tokens") == reasoning_budget_tokens
    )
    if max_tokens is not None:
        matches = matches and result.get("max_tokens") == max_tokens
    if feedback_response_max_tokens is not None:
        matches = matches and result.get("feedback_response_max_tokens") == feedback_response_max_tokens
    return matches


def load_resume_results(
    output_dir: Path,
    *,
    run_mode: str,
    selected_tasks: list[dict[str, Any]],
    implementation_profile: str,
    feedback_profile: str | None,
    reasoning_budget_tokens: int | None,
    max_tokens: int,
    feedback_response_max_tokens: int,
) -> list[dict[str, Any]]:
    results_path = output_dir / "results.json"
    if not results_path.exists():
        return []
    try:
        existing = json.loads(results_path.read_text(encoding="utf-8")).get("results", [])
    except json.JSONDecodeError:
        return []
    return [result for result in existing if isinstance(result, dict)]


def should_skip_existing_result(result: dict[str, Any] | None) -> bool:
    """Resume should preserve completed passes but rerun failures/timeouts."""
    return bool(result is not None and result.get("grade") == "pass")


def markdown_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Task | Category | Mode | Model | Verifier | Budget | Grade | Final | Seconds | Approach Attempts |",
        "|---|---|---|---|---|---:|---|---:|---:|---:|",
    ]
    for result in results:
        summary = result.get("summary") or {}
        lines.append(
            "| {task} | {category} | {mode} | {model} | {verifier} | {budget} | {grade} | {final} | {seconds:.1f} | {attempts} |".format(
                task=result["task_id"],
                category=result["category"],
                mode=result.get("run_mode", "harness"),
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
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for result in results:
        key = (result.get("run_mode", "harness"), result["implementation_profile"], result.get("feedback_profile") or "same")
        groups.setdefault(key, []).append(result)
    lines = [
        "| Mode | Model | Verifier | Budget | Tasks | Pass | Fail | Manual | Avg Seconds |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (mode, model, verifier), items in sorted(groups.items()):
        passed = sum(1 for item in items if item["grade"] == "pass")
        failed = sum(1 for item in items if item["grade"] == "fail")
        manual = sum(1 for item in items if item["grade"] == "manual_review")
        avg = sum(float(item.get("elapsed_seconds") or 0) for item in items) / max(1, len(items))
        budgets = sorted({str(item.get("reasoning_budget_tokens") or "profile") for item in items})
        lines.append(f"| {mode} | {model} | {verifier} | {', '.join(budgets)} | {len(items)} | {passed} | {failed} | {manual} | {avg:.1f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agenticFeedbackCoding benchmark corpus.")
    parser.add_argument("--tasks", default="benchmarks/tasks.json")
    parser.add_argument("--suites", default="benchmarks/suites.json")
    parser.add_argument("--suite")
    parser.add_argument("--mode", choices=["harness", "single-shot"], default="harness")
    parser.add_argument("--implementation-profile", default="gemma4-26b-a4b-qat-mtp")
    parser.add_argument("--feedback-profile")
    parser.add_argument("--reasoning-budget-tokens", type=int)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--feedback-response-max-tokens", type=int, default=4096)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task-timeout-seconds", type=int, default=0)
    parser.add_argument("--docker-isolation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    suite_ids = load_suite_ids(repo_root / args.suites, args.suite)
    tasks = select_tasks(load_tasks(repo_root / args.tasks), resolve_selection_ids(suite_ids, args.task_id), args.limit)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "runs" / f"benchmarks-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = load_resume_results(
        output_dir,
        run_mode=args.mode,
        selected_tasks=tasks,
        implementation_profile=args.implementation_profile,
        feedback_profile=args.feedback_profile,
        reasoning_budget_tokens=args.reasoning_budget_tokens,
        max_tokens=args.max_tokens,
        feedback_response_max_tokens=args.feedback_response_max_tokens,
    ) if args.resume else []
    if args.dry_run:
        for task in tasks:
            print(f"{task['id']}\t{task['category']}\t{task.get('grading', 'manual')}\t{task['title']}")
        return 0

    for task in tasks:
        existing_result = next(
            (
                result
                for result in results
                if result_matches_run(
                    result,
                    run_mode=args.mode,
                    task_id=task["id"],
                    implementation_profile=args.implementation_profile,
                    feedback_profile=args.feedback_profile,
                    reasoning_budget_tokens=args.reasoning_budget_tokens,
                    max_tokens=args.max_tokens,
                    feedback_response_max_tokens=args.feedback_response_max_tokens,
                )
            ),
            None,
        )
        if should_skip_existing_result(existing_result):
            print(f"{task['id']}: skipped existing {existing_result['grade']} in {existing_result.get('elapsed_seconds', 0):.1f}s")
            continue
        if existing_result is not None:
            print(
                f"{task['id']}: rerunning existing {existing_result['grade']} "
                f"from {existing_result.get('elapsed_seconds', 0):.1f}s"
            )
            results = [
                result
                for result in results
                if not result_matches_run(
                    result,
                    run_mode=args.mode,
                    task_id=task["id"],
                    implementation_profile=args.implementation_profile,
                    feedback_profile=args.feedback_profile,
                    reasoning_budget_tokens=args.reasoning_budget_tokens,
                    max_tokens=args.max_tokens,
                    feedback_response_max_tokens=args.feedback_response_max_tokens,
                )
            ]
        workspace = repo_root / "workspaces" / "benchmarks" / stamp / args.mode / task["id"]
        seed_workspace(workspace, task)
        config_path = output_dir / f"{task['id']}.json"
        single_shot_metadata: dict[str, Any] = {}
        if args.mode == "harness":
            cfg = benchmark_config(
                task,
                repo_root=repo_root,
                workspace=workspace,
                implementation_profile=args.implementation_profile,
                feedback_profile=args.feedback_profile,
                docker_isolation=args.docker_isolation,
                reasoning_budget_tokens=args.reasoning_budget_tokens,
                max_tokens=args.max_tokens,
                feedback_response_max_tokens=args.feedback_response_max_tokens,
            )
            write_config(config_path, cfg)
            returncode, elapsed, output = run_harness(
                repo_root,
                config_path,
                implementation_profile=args.implementation_profile,
                feedback_profile=args.feedback_profile,
                timeout_seconds=args.task_timeout_seconds or None,
            )
        else:
            returncode, elapsed, output, single_shot_metadata = run_single_shot(
                workspace,
                task,
                implementation_profile=args.implementation_profile,
                reasoning_budget_tokens=args.reasoning_budget_tokens,
                max_tokens=args.max_tokens,
            )
        (output_dir / f"{task['id']}.log").write_text(output, encoding="utf-8")
        grade = grade_task(workspace, task)
        result = {
            "run_mode": args.mode,
            "task_id": task["id"],
            "title": task["title"],
            "category": task["category"],
            "grading": task.get("grading", "manual"),
            "implementation_profile": args.implementation_profile,
            "feedback_profile": args.feedback_profile,
            "reasoning_budget_tokens": args.reasoning_budget_tokens,
            "max_tokens": args.max_tokens,
            "feedback_response_max_tokens": args.feedback_response_max_tokens,
            "workspace": str(workspace),
            "returncode": returncode,
            "elapsed_seconds": elapsed,
            "grade": grade["grade"] if returncode == 0 else "fail",
            "post_validation": grade["validation_results"],
        }
        if args.mode == "harness":
            result["summary"] = summarize_result(result)
        else:
            result["summary"] = {
                "final_status": "single_shot_written" if returncode == 0 else "single_shot_failed",
                "files_written": len(single_shot_metadata.get("files_written", [])),
            }
            result["single_shot"] = single_shot_metadata
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
