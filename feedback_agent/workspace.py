from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any, Callable

from .bounds import run_bounded_process


PLAN_TEMPLATE = """# Agent Plan

## Background Notes

- Workspace initialized.

## Refined Requirements

- Pending requirements refinement.

## Assumptions and Resolutions

- Pending requirements refinement.

## Ordered Tasks

- [ ] Pending plan validation.

## Current Status

- Phase: initialized
"""


BINARY_FILE_SUFFIXES = {
    ".apng",
    ".avif",
    ".bmp",
    ".gif",
    ".ico",
    ".jpg",
    ".jpeg",
    ".mp3",
    ".mp4",
    ".ogg",
    ".pdf",
    ".png",
    ".tar",
    ".webm",
    ".webp",
    ".zip",
}


SKIPPED_WORKSPACE_DIRS = {
    "$HOME",
    ".agent_state",
    ".dotnet",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "bin",
    "build",
    "dist",
    "node_modules",
    "obj",
    "out",
    "target",
    "venv",
}


def ensure_plan(workspace: Path, plan_filename: str = "PLAN.md") -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / plan_filename
    if not path.exists():
        path.write_text(PLAN_TEMPLATE, encoding="utf-8")
    return path


def append_plan_note(workspace: Path, note: str, plan_filename: str = "PLAN.md") -> None:
    plan = ensure_plan(workspace, plan_filename)
    with plan.open("a", encoding="utf-8") as f:
        f.write(f"\n- {note.strip()}\n")


def _json_object_candidates(text: str) -> list[tuple[int, int, str]]:
    """Return balanced JSON-object-looking substrings from noisy model output."""
    candidates: list[tuple[int, int, str]] = []
    in_string = False
    escaped = False
    starts: list[int] = []
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            starts.append(index)
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                depth += 1
                continue
            if char == "}" and depth:
                depth -= 1
                if depth == 0:
                    candidates.append((start, index + 1, text[start:index + 1]))
                    break
    return candidates


def _strip_common_model_wrappers(text: str) -> str:
    """Remove chat-template debris without changing the model's JSON content."""
    stripped = text.strip()
    stripped = re.sub(r"^\s*<\|channel\>[^<]*<channel\|>\s*", "", stripped)
    stripped = re.sub(r"^\s*<\|[^>]+?\|>\s*", "", stripped)
    stripped = _strip_complete_think_blocks(stripped)
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    return stripped


def _strip_complete_think_blocks(text: str) -> str:
    """Remove completed thinking blocks before structured-output parsing.

    Thinking is intentionally preserved in transcripts for later model context,
    but JSON extraction should not let an unfinished `{` inside `<think>` poison
    the parse of the final structured response.
    """
    return re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)


def extract_json_object(text: str) -> dict:
    stripped = _strip_common_model_wrappers(text)
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    # Qwen-family models often prepend reasoning even when asked for JSON.
    # We do not treat that as success, but the parser can still recover the
    # first complete object so the orchestration loop can keep moving.
    candidates = _json_object_candidates(stripped)
    parsed: list[tuple[int, int, dict]] = []
    for start, end, candidate in candidates:
        for variant in (candidate, _repair_common_model_json_escapes(candidate)):
            try:
                value = json.loads(variant)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                parsed.append((start, end, value))
                break
    if parsed:
        # Local models sometimes emit an initial JSON object, then say "wait"
        # and emit the corrected object. The last parseable object is usually
        # the one the model intended us to use. Ignore parseable nested objects
        # even when the containing object failed to parse: accepting a nested
        # planning_confirmation or command fragment as the phase payload silently
        # corrupts the workflow state and hides the real malformed response.
        top_level = [
            item for item in parsed
            if not any(
                other_start < item[0] and item[1] < other_end
                for other_start, other_end, _candidate in candidates
            )
        ]
        if top_level:
            return top_level[-1][2]
        raise ValueError("Only nested JSON objects were parseable inside a malformed larger object.")
    raise ValueError(f"No JSON object found in model output: {text[:200]}")


def _repair_common_model_json_escapes(candidate: str) -> str:
    r"""Repair common invalid escapes in otherwise JSON-looking model output.

    Qwen/Gemma-style local models sometimes write strings like `</div\>` while
    discussing HTML. JSON only permits a small escape alphabet, so that one
    character can make a useful review impossible to parse. Removing the stray
    backslash preserves the model's intended text without accepting arbitrary
    non-JSON syntax.
    """
    return re.sub(r'\\([^"\\/bfnrtu])', r"\1", candidate)


def _safe_relpath(path_text: str) -> Path:
    rel = Path(path_text)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe file path from model: {rel}")
    return rel


def write_files(workspace: Path, files: list[dict]) -> list[str]:
    """Apply complete-file edits requested by the implementation model.

    The harness deliberately accepts whole-file content only. That keeps each
    agent turn easy to replay from JSON logs and avoids patch-format ambiguity
    when a local model is tired, verbose, or creatively wrong.
    """
    written: list[str] = []
    for item in files:
        rel = _safe_relpath(str(item["path"]))
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        content = item.get("content", "")
        if isinstance(content, str):
            text = content
        else:
            # Models sometimes return structured JSON content instead of a
            # string. Writing Python's repr would corrupt those files with
            # single quotes, so serialize non-string content as real JSON.
            text = json.dumps(content, ensure_ascii=False, indent=2) + "\n"
        target.write_text(text, encoding="utf-8")
        if text.startswith("#!"):
            target.chmod(target.stat().st_mode | 0o111)
        written.append(str(rel))
    return written


def _command_parts_and_timeout(
    command: list[Any] | str | dict[str, Any],
    default_timeout_seconds: int,
    max_timeout_seconds: int,
) -> tuple[list[str], int, int]:
    """Normalize command formats used by implementation and validation plans.

    Most commands are simple argv lists. For long-running tools, a model may use
    {"cmd": [...], "timeout_seconds": 7200}; for expected negative-path tests,
    it may also use {"cmd": [...], "expected_returncode": 2}. Positive requested
    timeouts are clamped by a positive max timeout. A requested/default timeout
    of 0 disables the hard command deadline so progress review, not elapsed time
    alone, decides whether the process should keep running. If a local model
    returns a plain string despite the schema, split it as a shell-like command
    but still execute it without a shell.
    """
    if isinstance(command, dict):
        parts = command.get("cmd") or command.get("command") or []
        requested_timeout = int(command.get("timeout_seconds", default_timeout_seconds))
        expected_returncode = int(command.get("expected_returncode", 0))
    else:
        parts = command
        requested_timeout = default_timeout_seconds
        expected_returncode = 0
    if isinstance(parts, str):
        parts = shlex.split(parts)
    if requested_timeout <= 0:
        timeout = 0
    elif max_timeout_seconds <= 0:
        timeout = requested_timeout
    else:
        timeout = max(1, min(requested_timeout, max_timeout_seconds))
    return [str(part) for part in parts], timeout, expected_returncode


def _server_only_command_reason(parts: list[str]) -> str:
    joined = " ".join(parts).lower()
    if len(parts) >= 3 and parts[0].endswith("python") and parts[1] == "-m" and parts[2] == "http.server":
        return "python -m http.server starts a long-running server but does not assert behavior"
    if "python -m http.server" in joined:
        return "python -m http.server starts a long-running server but does not assert behavior"
    if parts and parts[0] in {"http-server", "live-server", "vite"} and not any(
        marker in joined for marker in ("test", "validate", "check", "playwright", "selenium")
    ):
        return f"{parts[0]} starts a long-running server but does not assert behavior"
    return ""


def _git_mutation_reason(parts: list[str]) -> str:
    """Return a reason when an agent command would mutate repository history.

    The harness owns git staging and commits. Implementation agents may inspect
    git state (`git status`, `git diff`, `git log`) as evidence, but allowing
    them to commit during an implementation attempt hides the diff from the
    feedback agent and breaks the accepted-step commit protocol.
    """
    if not parts or parts[0] != "git":
        return ""
    readonly = {
        "status",
        "diff",
        "log",
        "show",
        "rev-parse",
        "ls-files",
        "branch",
        "config",
    }
    subcommand = ""
    for part in parts[1:]:
        if not part.startswith("-"):
            subcommand = part
            break
    if not subcommand or subcommand in readonly:
        return ""
    return (
        f"git {subcommand} mutates repository state. The harness owns staging, "
        "commits, and final reset policy after feedback accepts a step"
    )


def run_commands(
    workspace: Path,
    commands: list[list[str] | dict[str, Any]],
    timeout_seconds: int,
    max_timeout_seconds: int | None = None,
    allow_git_mutation: bool = False,
    output_limit_chars: int = 4000,
    progress_callback: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    progress_interval_seconds: int = 0,
    progress_min_interval_seconds: int = 1,
) -> list[dict]:
    """Run bounded validation commands inside the workspace.

    Command results are fed directly to the feedback agent, so stdout/stderr are
    truncated to keep long sessions compact while preserving enough failure text
    for useful critique.
    """
    results: list[dict] = []
    max_timeout = timeout_seconds if max_timeout_seconds is None else max_timeout_seconds
    for index, command in enumerate(commands):
        if not command:
            continue
        parts, command_timeout, expected_returncode = _command_parts_and_timeout(command, timeout_seconds, max_timeout)
        if not parts:
            continue
        server_reason = _server_only_command_reason(parts)
        if server_reason:
            results.append({
                "command": parts,
                "timeout_seconds": command_timeout,
                "returncode": 125,
                "expected_returncode": expected_returncode,
                "returncode_matches_expected": 125 == expected_returncode,
                "stdout": "",
                "stderr": (
                    f"{server_reason}. Put server startup inside a validation script "
                    "that starts the server, performs assertions, then exits."
                ),
                "timed_out": False,
                "skipped_as_non_verifying_server": True,
            })
            continue
        if not allow_git_mutation:
            git_reason = _git_mutation_reason(parts)
            if git_reason:
                results.append({
                    "command": parts,
                    "timeout_seconds": command_timeout,
                    "returncode": 126,
                    "expected_returncode": expected_returncode,
                    "returncode_matches_expected": 126 == expected_returncode,
                    "stdout": "",
                    "stderr": git_reason,
                    "timed_out": False,
                    "blocked_git_mutation": True,
                })
                continue
        command_progress_callback = None
        if progress_callback is not None:
            def command_progress_callback(snapshot: dict[str, Any], *, command_index: int = index) -> dict[str, Any] | None:
                snapshot = dict(snapshot)
                snapshot["command_index"] = command_index
                return progress_callback(snapshot)

        result = run_bounded_process(
            parts,
            cwd=workspace,
            timeout_seconds=command_timeout,
            output_limit_chars=output_limit_chars,
            progress_callback=command_progress_callback,
            progress_interval_seconds=progress_interval_seconds,
            progress_min_interval_seconds=progress_min_interval_seconds,
        )
        result["expected_returncode"] = expected_returncode
        result["returncode_matches_expected"] = result["returncode"] == expected_returncode
        results.append(result)
    return results


def normalize_step(step: dict[str, Any], index: int) -> dict[str, Any]:
    step_id = str(step.get("id") or f"S{index}")
    return {
        "id": step_id,
        "title": str(step.get("title") or f"Step {index}"),
        "description": str(step.get("description") or ""),
        "depends_on": [str(x) for x in step.get("depends_on", [])],
        "acceptance_criteria": [str(x) for x in step.get("acceptance_criteria", [])],
        "validation_commands": step.get("validation_commands", []),
        "status": str(step.get("status") or "pending"),
    }


def normalize_plan_steps(raw_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_step(step, index) for index, step in enumerate(raw_steps, start=1)]


def write_requirements_doc(
    workspace: Path,
    requirements: dict[str, Any],
    review: dict[str, Any] | None = None,
    requirements_filename: str = "REQUIREMENTS.md",
) -> None:
    lines = ["# Refined Requirements", ""]
    lines.append("## Project")
    lines.append("")
    lines.append(str(requirements.get("project_summary") or requirements.get("summary") or "Pending."))
    lines.append("")
    lines.append("## Requirements")
    for item in requirements.get("refined_requirements", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Assumptions and Gap Resolutions")
    assumptions = requirements.get("assumptions", []) or requirements.get("open_questions", [])
    if assumptions:
        for item in assumptions:
            if isinstance(item, dict):
                question = item.get("question") or item.get("gap") or "gap"
                decision = item.get("decision") or item.get("resolution") or item.get("resolution_strategy") or "noted"
                lines.append(f"- {question}: {decision}")
            else:
                lines.append(f"- {item}")
    else:
        lines.append("- None recorded.")
    confirmation = requirements.get("planning_confirmation")
    if isinstance(confirmation, dict):
        lines.append("")
        lines.append("## Planning Confirmation")
        lines.append(f"- Feasible: {confirmation.get('is_feasible')}")
        lines.append(f"- Clear: {confirmation.get('is_clear')}")
        lines.append(f"- Verifiable: {confirmation.get('is_verifiable')}")
        if confirmation.get("verification_strategy"):
            lines.append(f"- Verification strategy: {confirmation['verification_strategy']}")
        risks = confirmation.get("remaining_risks") or []
        if risks:
            lines.append("- Remaining risks:")
            for risk in risks:
                lines.append(f"  - {risk}")
    if review:
        lines.append("")
        lines.append("## Last Requirements Review")
        lines.append(f"- Status: {review.get('status') or ('needs_rework' if review.get('needs_rework') else 'resolved')}")
        lines.append(f"- Summary: {review.get('summary', 'no summary')}")
    (workspace / requirements_filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plan_doc(
    workspace: Path,
    requirements: dict[str, Any],
    steps: list[dict[str, Any]],
    notes: list[str] | None = None,
    plan_filename: str = "PLAN.md",
) -> None:
    notes = notes or []
    lines = ["# Agent Plan", ""]
    lines.append("## Background Notes")
    lines.append("")
    lines.append(f"- Project: {requirements.get('project_summary') or requirements.get('summary') or 'configured project'}")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Refined Requirements")
    lines.append("")
    for item in requirements.get("refined_requirements", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Assumptions and Resolutions")
    lines.append("")
    assumptions = requirements.get("assumptions", []) or requirements.get("open_questions", [])
    if assumptions:
        for item in assumptions:
            if isinstance(item, dict):
                question = item.get("question") or item.get("gap") or "gap"
                decision = item.get("decision") or item.get("resolution") or item.get("resolution_strategy") or "noted"
                lines.append(f"- {question}: {decision}")
            else:
                lines.append(f"- {item}")
    else:
        lines.append("- None recorded.")
    confirmation = requirements.get("planning_confirmation")
    if isinstance(confirmation, dict):
        lines.append("")
        lines.append("## Planning Confirmation")
        lines.append("")
        lines.append(f"- Feasible: {confirmation.get('is_feasible')}")
        lines.append(f"- Clear: {confirmation.get('is_clear')}")
        lines.append(f"- Verifiable: {confirmation.get('is_verifiable')}")
        if confirmation.get("verification_strategy"):
            lines.append(f"- Verification strategy: {confirmation['verification_strategy']}")
        risks = confirmation.get("remaining_risks") or []
        if risks:
            lines.append("- Remaining risks:")
            for risk in risks:
                lines.append(f"  - {risk}")
    lines.append("")
    lines.append("## Ordered Tasks")
    lines.append("")
    for step in steps:
        mark = "x" if step.get("status") == "resolved" else " "
        lines.append(f"- [{mark}] {step['id']}: {step['title']} (`{step.get('status', 'pending')}`)")
        if step.get("description"):
            lines.append(f"  - Description: {step['description']}")
        if step.get("depends_on"):
            lines.append(f"  - Depends on: {', '.join(step['depends_on'])}")
        if step.get("acceptance_criteria"):
            lines.append("  - Acceptance criteria:")
            for criterion in step["acceptance_criteria"]:
                lines.append(f"    - {criterion}")
        if step.get("validation_commands"):
            lines.append("  - Validation commands:")
            for command in step["validation_commands"]:
                if isinstance(command, list):
                    lines.append(f"    - `{' '.join(str(part) for part in command)}`")
                else:
                    lines.append(f"    - `{command}`")
    lines.append("")
    lines.append("## Current Status")
    lines.append("")
    unresolved = [s for s in steps if s.get("status") != "resolved"]
    lines.append(f"- Resolved steps: {len(steps) - len(unresolved)} / {len(steps)}")
    (workspace / plan_filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_workspace_files(workspace: Path, max_file_bytes: int = 20000) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or any(part in SKIPPED_WORKSPACE_DIRS for part in path.relative_to(workspace).parts):
            continue
        rel = path.relative_to(workspace)
        size = path.stat().st_size
        with path.open("rb") as f:
            sample = f.read(min(size, 4096))
        if _looks_like_binary_file(path, sample):
            files.append({
                "path": str(rel),
                "content": f"[binary artifact omitted from prompt; size={size} bytes]",
                "size": size,
                "truncated": False,
                "binary": True,
            })
            continue
        if size <= max_file_bytes:
            files.append({
                "path": str(rel),
                "content": path.read_text(encoding="utf-8", errors="replace"),
                "size": size,
                "truncated": False,
            })
            continue
        with path.open("rb") as f:
            head = f.read(max_file_bytes // 2)
            f.seek(max(size - max_file_bytes // 2, 0))
            tail = f.read(max_file_bytes // 2)
        content = (
            head.decode("utf-8", errors="replace")
            + f"\n\n[workspace file truncated: kept first and last {max_file_bytes // 2} bytes of {size}]\n\n"
            + tail.decode("utf-8", errors="replace")
        )
        files.append({"path": str(rel), "content": content, "size": size, "truncated": True})
    return files


def _looks_like_binary_file(path: Path, sample: bytes) -> bool:
    """Return True when a workspace artifact should be summarized, not pasted.

    Screenshots and other binary evidence are useful as files, but shoving raw
    bytes into the reviewer prompt consumes context and can derail long local
    model runs. Keep the path/size metadata so the feedback agent knows the
    artifact exists, while leaving visual inspection to explicit tooling later.
    """
    if path.suffix.lower() in BINARY_FILE_SUFFIXES:
        return True
    if b"\0" in sample:
        return True
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return True
    if not sample:
        return False
    control_bytes = sum(1 for byte in sample if byte < 32 and byte not in (9, 10, 13))
    return control_bytes > max(8, len(sample) // 4)
