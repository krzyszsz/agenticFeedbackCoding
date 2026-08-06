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
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feedback_agent.config import (
    DEFAULT_CONFIG,
    ModelConfig,
    _deep_merge,
    derive_critical_reasoning_budget,
)
from feedback_agent.bounds import clamp_text, run_bounded_process
from feedback_agent.llm import OpenAICompatClient
from feedback_agent.model_profiles import ModelProfile, resolve_profile
from feedback_agent.workspace import (
    _command_parts_and_timeout,
    collect_workspace_files,
    extract_json_object,
    run_commands,
    write_files,
)


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
    print_transcript: bool,
    live_turn_max_chars: int,
    critical_reasoning_budget_tokens: int | None = None,
) -> dict[str, Any]:
    impl = resolve_profile(implementation_profile)
    feedback = resolve_profile(feedback_profile) if feedback_profile else None
    implementation_max_tokens = profile_safe_max_tokens(impl, max_tokens)
    implementation_reasoning_budget = (
        reasoning_budget_tokens
        if reasoning_budget_tokens is not None
        else impl.reasoning_budget_tokens
    )
    implementation_critical_reasoning_budget = derive_critical_reasoning_budget(
        implementation_reasoning_budget,
        implementation_max_tokens,
        critical_reasoning_budget_tokens,
    )
    _validate_direct_model_budget(
        implementation_profile,
        max_tokens=implementation_max_tokens,
        reasoning_budget_tokens=implementation_reasoning_budget,
        critical_reasoning_budget_tokens=implementation_critical_reasoning_budget,
    )
    model_cfg = {
        "name": impl.name,
        "base_url": f"http://127.0.0.1:{impl.port}/v1",
        "api_key": "not-needed",
        "model": "local-gguf",
        "context_window": impl.context_window,
        "max_tokens": implementation_max_tokens,
        "temperature": impl.temperature,
        "top_p": impl.top_p,
        "top_k": impl.top_k,
        "min_p": impl.min_p,
        "presence_penalty": impl.presence_penalty,
        "repeat_penalty": impl.repeat_penalty,
        "request_timeout_seconds": 21600,
        "retry_attempts": 20,
        "retry_sleep_seconds": 30,
        "request_heartbeat_seconds": 30,
        "preserve_reasoning": True,
        "reasoning_budget_tokens": implementation_reasoning_budget,
        "critical_reasoning_budget_tokens": critical_reasoning_budget_tokens,
        "send_reasoning_budget": True,
        "system_prompt_as_user": impl.system_prompt_as_user,
    }
    data: dict[str, Any] = {
        "implementation_model": model_cfg,
        "feedback_model": None,
        "runtime": {
            "docker_isolation": docker_isolation,
            "workspace": str(workspace.relative_to(repo_root) if workspace.is_relative_to(repo_root) else workspace),
            "command_timeout_seconds": 0,
            "max_command_timeout_seconds": 21600,
            "command_progress_review_interval_seconds": 300,
            "command_progress_review_min_interval_seconds": 30,
            "command_progress_review_max_interval_seconds": 3600,
            "command_progress_review_request_timeout_seconds": 120,
            "print_transcript": print_transcript,
            "live_turn_max_chars": live_turn_max_chars,
            "final_summary": "compact",
            "feedback_response_max_tokens": feedback_response_max_tokens,
        },
        "mcp_tools": {
            "terminal": True,
            "web_scraping": task.get("web_research", False),
            "web_interaction": True,
        },
        "loop": {
            "max_approach_reattempts": 1,
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
        feedback_max_tokens = profile_safe_max_tokens(feedback, max_tokens)
        feedback_reasoning_budget = (
            reasoning_budget_tokens
            if reasoning_budget_tokens is not None
            else feedback.reasoning_budget_tokens
        )
        feedback_critical_reasoning_budget = derive_critical_reasoning_budget(
            feedback_reasoning_budget,
            feedback_max_tokens,
            critical_reasoning_budget_tokens,
        )
        _validate_direct_model_budget(
            feedback.name,
            max_tokens=feedback_max_tokens,
            reasoning_budget_tokens=feedback_reasoning_budget,
            critical_reasoning_budget_tokens=feedback_critical_reasoning_budget,
        )
        data["feedback_model"] = {
            **model_cfg,
            "name": feedback.name,
            "base_url": f"http://127.0.0.1:{feedback.port}/v1",
            "context_window": feedback.context_window,
            "max_tokens": feedback_max_tokens,
            "temperature": feedback.temperature,
            "top_p": feedback.top_p,
            "top_k": feedback.top_k,
            "min_p": feedback.min_p,
            "presence_penalty": feedback.presence_penalty,
            "repeat_penalty": feedback.repeat_penalty,
            "reasoning_budget_tokens": feedback_reasoning_budget,
            "critical_reasoning_budget_tokens": critical_reasoning_budget_tokens,
            "system_prompt_as_user": feedback.system_prompt_as_user,
        }
    if task.get("config_overrides"):
        data = _deep_merge(data, task["config_overrides"])
    return _deep_merge(DEFAULT_CONFIG, data)


def write_config(path: Path, data: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(data, indent=2) + "\n")


def write_text_atomic(path: Path, content: str) -> None:
    """Replace benchmark metadata atomically so interruption cannot corrupt resume state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


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
    critical_reasoning_budget_tokens: int | None = None,
) -> ModelConfig:
    profile = resolve_profile(profile_name)
    effective_max_tokens = profile_safe_max_tokens(profile, max_tokens)
    effective_reasoning_budget = (
        reasoning_budget_tokens
        if reasoning_budget_tokens is not None
        else profile.reasoning_budget_tokens
    )
    effective_critical_reasoning_budget = derive_critical_reasoning_budget(
        effective_reasoning_budget,
        effective_max_tokens,
        critical_reasoning_budget_tokens,
    )
    _validate_direct_model_budget(
        profile_name,
        max_tokens=effective_max_tokens,
        reasoning_budget_tokens=effective_reasoning_budget,
        critical_reasoning_budget_tokens=effective_critical_reasoning_budget,
    )
    return ModelConfig(
        name=profile.name,
        base_url=f"http://127.0.0.1:{profile.port}/v1",
        api_key="not-needed",
        model="local-gguf",
        context_window=profile.context_window,
        max_tokens=effective_max_tokens,
        temperature=profile.temperature,
        top_p=profile.top_p,
        top_k=profile.top_k,
        min_p=profile.min_p,
        presence_penalty=profile.presence_penalty,
        repeat_penalty=profile.repeat_penalty,
        request_timeout_seconds=21600,
        retry_attempts=20,
        retry_sleep_seconds=30,
        request_heartbeat_seconds=30,
        preserve_reasoning=True,
        reasoning_budget_tokens=effective_reasoning_budget,
        critical_reasoning_budget_tokens=effective_critical_reasoning_budget,
        send_reasoning_budget=True,
        request_json_object=True,
        system_prompt_as_user=profile.system_prompt_as_user,
    )


def profile_safe_max_tokens(profile: ModelProfile, requested_max_tokens: int) -> int:
    if requested_max_tokens <= 0:
        raise ValueError(f"{profile.name}: max_tokens must be greater than zero")
    if profile.context_window <= 0 or requested_max_tokens < profile.context_window:
        return requested_max_tokens
    return max(1, profile.context_window - 1)


def _validate_direct_model_budget(
    profile_name: str,
    *,
    max_tokens: int,
    reasoning_budget_tokens: int | None,
    critical_reasoning_budget_tokens: int | None = None,
) -> None:
    if max_tokens <= 0:
        raise ValueError(f"{profile_name}: max_tokens must be greater than zero")
    if reasoning_budget_tokens is not None:
        if reasoning_budget_tokens < 0:
            raise ValueError(f"{profile_name}: reasoning budget must be zero or greater")
        if reasoning_budget_tokens >= max_tokens:
            raise ValueError(
                f"{profile_name}: reasoning budget ({reasoning_budget_tokens}) must be smaller than "
                f"max_tokens ({max_tokens}) so the model can return an answer"
            )
    if critical_reasoning_budget_tokens is None:
        return
    if critical_reasoning_budget_tokens < 0:
        raise ValueError(f"{profile_name}: critical reasoning budget must be zero or greater")
    if critical_reasoning_budget_tokens >= max_tokens:
        raise ValueError(
            f"{profile_name}: critical reasoning budget ({critical_reasoning_budget_tokens}) must be smaller than "
            f"max_tokens ({max_tokens}) so the model can return an answer"
        )
    if (
        reasoning_budget_tokens is not None
        and critical_reasoning_budget_tokens < reasoning_budget_tokens
    ):
        raise ValueError(
            f"{profile_name}: critical reasoning budget ({critical_reasoning_budget_tokens}) must be at least "
            f"the normal reasoning budget ({reasoning_budget_tokens})"
        )


def single_shot_prompt(task: dict[str, Any], workspace: Path) -> str:
    workspace_files = collect_workspace_files(
        workspace,
        max_file_bytes=12000,
        max_files=100,
        max_total_chars=80_000,
    )
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
        "Do not write .agent_state or .git control state. Do not create harness document names such as PLAN.md, "
        "REQUIREMENTS.md, or RESEARCH.md unless the user explicitly asks for those as project deliverables.\n\n"
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


def _normalize_manual_grade(value: Any) -> str | None:
    grade = str(value or "").strip()
    return grade if grade in {"manual_pass", "manual_fail"} else None


def manual_grade_task(
    workspace: Path,
    task: dict[str, Any],
    *,
    grader_profile: str,
    reasoning_budget_tokens: int | None,
    max_tokens: int,
    validation_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Judge a manual benchmark task as pass/fail from bounded artifacts.

    This is benchmark measurement code only. It does not feed back into the
    harness run and deliberately asks for a methodology marker in the grade so
    README tables can distinguish automatic checks from model-judged checks.
    """
    criteria = task.get("manual_pass_criteria") or []
    workspace_files = collect_workspace_files(
        workspace,
        max_file_bytes=10000,
        max_files=100,
        max_total_chars=100_000,
    )
    validation_evidence = clamp_text(
        json.dumps(validation_results, indent=2),
        40_000,
        marker="validation evidence truncated",
    )
    prompt = (
        "You are grading one completed benchmark task. Decide whether the produced workspace satisfies the task.\n"
        "Return strict JSON only with this shape:\n"
        '{ "grade": "manual_fail", "evidence": [], "concerns": ["missing or failed requirement"] }\n'
        "Set grade to exactly manual_pass or manual_fail; the example is the conservative failure shape, "
        "not a predetermined verdict.\n"
        "Use manual_pass only when the artifacts satisfy the user prompt and all pass criteria. "
        "Use manual_fail when required evidence is missing, invalid, or ambiguous.\n\n"
        f"Task title: {task['title']}\n"
        f"Task prompt:\n{task['prompt']}\n\n"
        f"Manual pass criteria:\n{json.dumps(criteria, indent=2)}\n\n"
        f"Validation results, if any:\n{validation_evidence}\n\n"
        "Workspace files:\n"
        f"{json.dumps(workspace_files, indent=2)}"
    )
    client = OpenAICompatClient(
        direct_model_config(
            grader_profile,
            reasoning_budget_tokens=reasoning_budget_tokens,
            max_tokens=max_tokens,
        )
    )
    messages = [
        {
            "role": "system",
            "content": "You are a strict benchmark grader. Return one JSON object and no prose.",
        },
        {"role": "user", "content": prompt},
    ]
    raw = ""
    last_error = ""
    for attempt in range(2):
        try:
            raw = client.chat(messages, max_tokens=max_tokens, temperature=0.0)
            payload = extract_json_object(raw)
            grade = _normalize_manual_grade(payload.get("grade"))
            if grade is not None:
                return {
                    "grade": grade,
                    "evidence": payload.get("evidence", []),
                    "concerns": payload.get("concerns", []),
                    "raw_tail": raw[-4000:],
                    "grader_profile": grader_profile,
                }
            messages.extend([
                {"role": "assistant", "content": raw[-4000:]},
                {
                    "role": "user",
                    "content": (
                        "Your response did not include grade as exactly manual_pass or manual_fail. "
                        "Re-answer the same grading question as strict JSON only with that grade field."
                    ),
                },
            ])
        except Exception as exc:
            last_error = repr(exc)
            if attempt == 0:
                if raw:
                    messages.append({"role": "assistant", "content": raw[-4000:]})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your response did not match the requested grading JSON protocol. "
                        f"The parser reported: {exc}. Re-answer the same grading question as exactly one JSON "
                        "object with grade equal to manual_pass or manual_fail, plus evidence and concerns lists."
                    ),
                })
                continue
            break
    return {
        "grade": "manual_fail",
        "evidence": [],
        "concerns": [
            "manual grader did not return a valid manual_pass or manual_fail decision"
            + (f": {last_error}" if last_error else "")
        ],
        "raw_tail": raw[-4000:],
        "grader_profile": grader_profile,
    }


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
    stream_output: bool,
    log_stream: TextIO | None = None,
    capture_limit_chars: int = 200_000,
) -> str:
    """Stream real benchmark subprocess output instead of hiding long runs.

    Tests use small fake ``Popen`` objects without a real stdout file descriptor;
    those continue through ``communicate``. Real Docker harness runs get live
    transcript output, so a long model call or tool command is visible while the
    result log still captures the same text.
    """
    capture_limit_chars = max(1000, capture_limit_chars)
    head_limit = capture_limit_chars // 2
    tail_limit = capture_limit_chars - head_limit
    captured_head = ""
    captured_tail = ""
    captured_total = 0

    def record(text: str) -> None:
        nonlocal captured_head, captured_tail, captured_total
        if not text:
            return
        if log_stream is not None:
            log_stream.write(text)
            log_stream.flush()
        captured_total += len(text)
        if len(captured_head) < head_limit:
            take = min(head_limit - len(captured_head), len(text))
            captured_head += text[:take]
        captured_tail = (captured_tail + text)[-tail_limit:]

    def captured_output() -> str:
        if captured_total <= capture_limit_chars:
            overlap = max(0, len(captured_head) + len(captured_tail) - captured_total)
            return captured_head + captured_tail[overlap:]
        return (
            captured_head
            + f"\n[BENCHMARK_LOG_CAPTURE_TRUNCATED: kept first {len(captured_head)} and last "
            f"{len(captured_tail)} of {captured_total} chars; full output is in the task log]\n"
            + captured_tail
        )

    pipe = getattr(proc, "stdout", None)
    if pipe is None or not hasattr(pipe, "fileno"):
        stdout, _stderr = proc.communicate(timeout=timeout_seconds)
        record(stdout or "")
        return captured_output()

    selector = selectors.DefaultSelector()
    selector.register(pipe, selectors.EVENT_READ)
    deadline = start + timeout_seconds if timeout_seconds else None
    last_heartbeat = start
    pipe_closed = False
    while not pipe_closed:
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            raise subprocess.TimeoutExpired(
                cmd=getattr(proc, "args", "scripts/run_agent.sh"),
                timeout=timeout_seconds,
                output=captured_output(),
            )
        events = selector.select(timeout=1.0)
        if not events and proc.poll() is not None:
            events = selector.select(timeout=0)
        for key, _event in events:
            data = os.read(key.fileobj.fileno(), 8192)
            if data:
                text = data.decode("utf-8", errors="replace")
                record(text)
                if stream_output:
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
    return captured_output()


def run_harness(
    repo_root: Path,
    config_path: Path,
    *,
    implementation_profile: str,
    feedback_profile: str | None,
    timeout_seconds: int | None,
    stream_output: bool,
    log_path: Path | None = None,
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
    log_stream: TextIO | None = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("w", encoding="utf-8")
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
        stdout = _stream_process_output(
            proc,
            timeout_seconds=timeout_seconds,
            start=start,
            stream_output=stream_output,
            log_stream=log_stream,
        )
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
                if more and log_stream is not None:
                    log_stream.write(more)
                    log_stream.flush()
                output = clamp_text(output + (more or ""), 200_000, marker="benchmark capture truncated")
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
                if more and log_stream is not None:
                    log_stream.write(more)
                    log_stream.flush()
                output = clamp_text(output + (more or ""), 200_000, marker="benchmark capture truncated")
        timeout_marker = f"\n[BENCHMARK_TIMEOUT] harness task exceeded {timeout_seconds} seconds and was stopped.\n"
        output += timeout_marker
        if log_stream is not None:
            log_stream.write(timeout_marker)
            log_stream.flush()
        return 124, time.monotonic() - start, output
    finally:
        if log_stream is not None:
            log_stream.close()


def build_benchmark_agent_image(repo_root: Path, image: str) -> None:
    """Build the current harness source once before using it in a benchmark."""
    print(f"[benchmark-runner] building current agent image: {image}", flush=True)
    subprocess.run(
        ["docker", "build", "-t", image, "."],
        cwd=repo_root,
        check=True,
    )


def run_docker_post_validation_commands(
    repo_root: Path,
    workspace: Path,
    commands: list[list[str] | dict[str, Any]],
    *,
    image: str = "agentic-feedback-coding:local",
    timeout_seconds: int = 120,
    max_timeout_seconds: int = 21600,
    output_limit_chars: int = 8000,
) -> list[dict[str, Any]]:
    """Run benchmark post-validation inside the same Docker image as the harness.

    Harness tasks run with Docker isolation by default. Grading them on the host
    can create false failures when the task correctly used container-provided
    tools such as Python Playwright.
    """
    results: list[dict[str, Any]] = []
    uid_gid = f"{os.getuid()}:{os.getgid()}"
    for index, command in enumerate(commands):
        parts, command_timeout, expected_returncode = _command_parts_and_timeout(
            command,
            timeout_seconds,
            max_timeout_seconds,
        )
        if not parts:
            continue
        docker_user = command.get("docker_user", "host") if isinstance(command, dict) else "host"
        if docker_user not in {"host", "root"}:
            raise ValueError("post-validation docker_user must be 'host' or 'root'")
        container_user = "0:0" if docker_user == "root" else uid_gid
        container_name = f"agentic-bench-post-{os.getpid()}-{index}-{int(time.time() * 1000) % 1000000}"
        docker_command = [
            "docker",
            "run",
            "--rm",
            "--init",
            "--name",
            container_name,
            "--label",
            "agentic-feedback-benchmark-post=1",
            "--security-opt",
            "label=disable",
            "--user",
            container_user,
            "-e",
            "AGENT_IN_CONTAINER=1",
            "-e",
            "HOME=/tmp",
            "-e",
            "DOTNET_ROOT=/tmp/.dotnet",
            "-e",
            "PATH=/tmp/.dotnet:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "-e",
            "PYTHONPATH=/workspace/project:/app",
            "-v",
            f"{workspace}:/workspace/project",
            "-w",
            "/workspace/project",
            "--entrypoint",
            "/usr/bin/env",
            image,
            *parts,
        ]
        result = run_bounded_process(
            docker_command,
            cwd=repo_root,
            timeout_seconds=command_timeout,
            output_limit_chars=output_limit_chars,
        )
        if result.get("timed_out") or result.get("stopped_by_progress_review"):
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
            )
        result["docker_command"] = result.get("command", docker_command)
        result["command"] = parts
        result["ran_in_docker"] = True
        result["expected_returncode"] = expected_returncode
        result["returncode_matches_expected"] = result.get("returncode") == expected_returncode
        results.append(result)
    return results


def grade_task(
    workspace: Path,
    task: dict[str, Any],
    *,
    repo_root: Path | None = None,
    docker_post_validation: bool = False,
    docker_image: str = "agentic-feedback-coding:local",
    manual_grader_profile: str | None = None,
    manual_grader_reasoning_budget_tokens: int | None = None,
    manual_grader_max_tokens: int = 8192,
) -> dict[str, Any]:
    grading = task.get("grading", "manual")
    commands = task.get("post_validation_commands") or []
    validation_results = []
    manual_review = None
    if commands:
        if docker_post_validation:
            validation_results = run_docker_post_validation_commands(
                repo_root or REPO_ROOT,
                workspace,
                commands,
                image=docker_image,
                timeout_seconds=120,
                max_timeout_seconds=21600,
                output_limit_chars=8000,
            )
        else:
            validation_results = run_commands(workspace, commands, 120, 21600, output_limit_chars=8000)
    if grading == "manual":
        manual_review = None
        if manual_grader_profile:
            manual_review = manual_grade_task(
                workspace,
                task,
                grader_profile=manual_grader_profile,
                reasoning_budget_tokens=manual_grader_reasoning_budget_tokens,
                max_tokens=manual_grader_max_tokens,
                validation_results=validation_results,
            )
            status = manual_review["grade"]
        else:
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
    result = {"grade": status, "validation_results": validation_results}
    if manual_review is not None:
        result["manual_review"] = manual_review
    return result


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
    critical_reasoning_budget_tokens: int | None = None,
    max_tokens: int | None = None,
    feedback_response_max_tokens: int | None = None,
) -> bool:
    matches = (
        result.get("run_mode", "harness") == run_mode
        and result.get("task_id") == task_id
        and result.get("implementation_profile") == implementation_profile
        and result.get("feedback_profile") == feedback_profile
        and result.get("reasoning_budget_tokens") == reasoning_budget_tokens
        and result.get("critical_reasoning_budget_tokens") == critical_reasoning_budget_tokens
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
    critical_reasoning_budget_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Load the shared result set; per-run matching happens at task selection."""
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
    return bool(result is not None and result.get("grade") in {"pass", "manual_pass"})


def display_grade(grade: str) -> str:
    return {
        "manual_pass": "manual pass",
        "manual_fail": "manual fail",
        "manual_review": "manual review",
    }.get(grade, grade)


def display_reasoning_budget(value: Any) -> str:
    return "profile" if value is None else str(value)


def display_critical_reasoning_budget(value: Any, *, run_mode: str) -> str:
    if run_mode == "single-shot":
        return "n/a"
    return "auto" if value is None else str(value)


def final_benchmark_grade(task: dict[str, Any], returncode: int, measured_grade: str) -> str:
    if returncode == 0:
        return measured_grade
    if task.get("grading", "manual") == "manual":
        return "manual_fail"
    return "fail"


def markdown_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Task | Category | Mode | Model | Verifier | Base Budget | Critical Budget | Grade | Final | Seconds | Approach Attempts |",
        "|---|---|---|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for result in results:
        summary = result.get("summary") or {}
        lines.append(
            "| {task} | {category} | {mode} | {model} | {verifier} | {budget} | {critical_budget} | {grade} | {final} | {seconds:.1f} | {attempts} |".format(
                task=result["task_id"],
                category=result["category"],
                mode=result.get("run_mode", "harness"),
                model=result["implementation_profile"],
                verifier=result.get("feedback_profile") or "same",
                budget=display_reasoning_budget(result.get("reasoning_budget_tokens")),
                critical_budget=display_critical_reasoning_budget(
                    result.get("critical_reasoning_budget_tokens"),
                    run_mode=result.get("run_mode", "harness"),
                ),
                grade=display_grade(result["grade"]),
                final=summary.get("final_status", "n/a"),
                seconds=float(result.get("elapsed_seconds") or 0),
                attempts=summary.get("approach_attempts", "n/a"),
            )
        )
    return "\n".join(lines) + "\n"


def summary_table(results: list[dict[str, Any]]) -> str:
    groups: dict[tuple[str, str, str, Any, Any, Any, Any], list[dict[str, Any]]] = {}
    for result in results:
        key = (
            result.get("run_mode", "harness"),
            result["implementation_profile"],
            result.get("feedback_profile") or "same",
            result.get("reasoning_budget_tokens"),
            result.get("critical_reasoning_budget_tokens"),
            result.get("max_tokens"),
            result.get("feedback_response_max_tokens"),
        )
        groups.setdefault(key, []).append(result)
    lines = [
        "| Mode | Model | Verifier | Base Budget | Critical Budget | Max Out | Review Out | Tasks | Pass | Fail | Manual | Avg Seconds |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (mode, model, verifier, budget, critical_budget, max_tokens, feedback_tokens), items in sorted(
        groups.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    ):
        passed = sum(1 for item in items if item["grade"] in {"pass", "manual_pass"})
        failed = sum(1 for item in items if item["grade"] in {"fail", "manual_fail"})
        manual = sum(1 for item in items if str(item["grade"]).startswith("manual_"))
        avg = sum(float(item.get("elapsed_seconds") or 0) for item in items) / max(1, len(items))
        lines.append(
            f"| {mode} | {model} | {verifier} | {display_reasoning_budget(budget)} | "
            f"{display_critical_reasoning_budget(critical_budget, run_mode=mode)} | "
            f"{max_tokens} | {feedback_tokens} | {len(items)} | {passed} | {failed} | {manual} | {avg:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agenticFeedbackCoding benchmark corpus.")
    parser.add_argument("--tasks", default="benchmarks/tasks.json")
    parser.add_argument("--suites", default="benchmarks/suites.json")
    parser.add_argument("--suite")
    parser.add_argument("--mode", choices=["harness", "single-shot"], default="harness")
    parser.add_argument("--implementation-profile", default="gemma4-26b-a4b-qat-mtp")
    parser.add_argument("--feedback-profile")
    parser.add_argument(
        "--reasoning-budget-tokens",
        type=int,
        help="Normal model reasoning allowance; omit to use the selected profile.",
    )
    parser.add_argument(
        "--critical-reasoning-budget-tokens",
        type=int,
        help="Escalated harness reasoning allowance; omit for bounded 4x automatic sizing.",
    )
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--feedback-response-max-tokens", type=int, default=2048)
    parser.add_argument("--manual-grader-profile", default="")
    parser.add_argument("--manual-grader-reasoning-budget-tokens", type=int)
    parser.add_argument("--manual-grader-max-tokens", type=int, default=8192)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task-timeout-seconds", type=int, default=0)
    parser.add_argument("--docker-isolation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-transcript", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--live-turn-max-chars", type=int, default=20000)
    parser.add_argument("--stream-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-agent-image", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_critical_reasoning_budget = (
        args.critical_reasoning_budget_tokens if args.mode == "harness" else None
    )

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
        critical_reasoning_budget_tokens=run_critical_reasoning_budget,
        max_tokens=args.max_tokens,
        feedback_response_max_tokens=args.feedback_response_max_tokens,
    ) if args.resume else []
    if args.dry_run:
        for task in tasks:
            print(f"{task['id']}\t{task['category']}\t{task.get('grading', 'manual')}\t{task['title']}")
        return 0

    prepared_images: set[str] = set()
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
                    critical_reasoning_budget_tokens=run_critical_reasoning_budget,
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
                    critical_reasoning_budget_tokens=run_critical_reasoning_budget,
                    max_tokens=args.max_tokens,
                    feedback_response_max_tokens=args.feedback_response_max_tokens,
                )
            ]
        workspace = repo_root / "workspaces" / "benchmarks" / stamp / args.mode / task["id"]
        seed_workspace(workspace, task)
        config_path = output_dir / f"{task['id']}.json"
        single_shot_metadata: dict[str, Any] = {}
        docker_image = "agentic-feedback-coding:local"
        if args.mode == "harness":
            cfg = benchmark_config(
                task,
                repo_root=repo_root,
                workspace=workspace,
                implementation_profile=args.implementation_profile,
                feedback_profile=args.feedback_profile,
                docker_isolation=args.docker_isolation,
                reasoning_budget_tokens=args.reasoning_budget_tokens,
                critical_reasoning_budget_tokens=run_critical_reasoning_budget,
                max_tokens=args.max_tokens,
                feedback_response_max_tokens=args.feedback_response_max_tokens,
                print_transcript=args.print_transcript,
                live_turn_max_chars=args.live_turn_max_chars,
            )
            write_config(config_path, cfg)
            docker_image = str(cfg["runtime"]["docker_image"])
        if args.docker_isolation and args.rebuild_agent_image and docker_image not in prepared_images:
            build_benchmark_agent_image(repo_root, docker_image)
            prepared_images.add(docker_image)
        if args.mode == "harness":
            task_log_path = output_dir / f"{task['id']}.log"
            returncode, elapsed, output = run_harness(
                repo_root,
                config_path,
                implementation_profile=args.implementation_profile,
                feedback_profile=args.feedback_profile,
                timeout_seconds=args.task_timeout_seconds or None,
                stream_output=args.stream_output,
                log_path=task_log_path,
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
        grade = grade_task(
            workspace,
            task,
            repo_root=repo_root,
            docker_post_validation=args.docker_isolation,
            docker_image=docker_image,
            manual_grader_profile=(
                (args.manual_grader_profile or args.implementation_profile)
                if returncode == 0
                else None
            ),
            manual_grader_reasoning_budget_tokens=args.manual_grader_reasoning_budget_tokens,
            manual_grader_max_tokens=args.manual_grader_max_tokens,
        )
        result = {
            "run_mode": args.mode,
            "task_id": task["id"],
            "title": task["title"],
            "category": task["category"],
            "grading": task.get("grading", "manual"),
            "implementation_profile": args.implementation_profile,
            "feedback_profile": args.feedback_profile,
            "reasoning_budget_tokens": args.reasoning_budget_tokens,
            "critical_reasoning_budget_tokens": run_critical_reasoning_budget,
            "max_tokens": args.max_tokens,
            "feedback_response_max_tokens": args.feedback_response_max_tokens,
            "workspace": str(workspace),
            "returncode": returncode,
            "elapsed_seconds": elapsed,
            "grade": final_benchmark_grade(task, returncode, grade["grade"]),
            "post_validation": grade["validation_results"],
        }
        if grade.get("manual_review"):
            result["manual_review"] = grade["manual_review"]
        if args.mode == "harness":
            result["summary"] = summarize_result(result)
        else:
            result["summary"] = {
                "final_status": "single_shot_written" if returncode == 0 else "single_shot_failed",
                "files_written": len(single_shot_metadata.get("files_written", [])),
            }
            result["single_shot"] = single_shot_metadata
        results.append(result)
        write_text_atomic(
            output_dir / "results.json",
            json.dumps({"results": results}, indent=2) + "\n",
        )
        write_text_atomic(
            output_dir / "results.md",
            "# Benchmark Results\n\n## Summary\n\n"
            + summary_table(results)
            + "\n## Details\n\n"
            + markdown_table(results),
        )
        print(f"{task['id']}: {result['grade']} in {elapsed:.1f}s")

    print(f"Results: {output_dir / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
