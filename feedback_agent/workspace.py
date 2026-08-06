from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterator

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
    "node_modules",
    "venv",
}


LOW_PRIORITY_WORKSPACE_DIRS = {
    "bin",
    "build",
    "dist",
    "obj",
    "out",
    "target",
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


def _strip_common_model_wrappers(text: str) -> str:
    """Remove server transport debris without accepting model-authored prose."""
    stripped = text.strip()
    stripped = re.sub(r"^\s*<\|channel\>[^<]*<channel\|>\s*", "", stripped)
    stripped = re.sub(r"^\s*<\|[^>]+?\|>\s*", "", stripped)
    stripped = _strip_complete_think_blocks(stripped)
    return stripped


def _strip_complete_think_blocks(text: str) -> str:
    """Remove completed thinking blocks before structured-output parsing.

    The current phase can inspect a complete reasoning block before parsing, but
    durable active history omits it. JSON extraction must not let an unfinished
    `{` inside `<think>` poison the parse of the final structured response.
    """
    return re.sub(
        r"^\s*(?:<think\b[^>]*>.*?</think>\s*)+",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )


def extract_json_object(text: str) -> dict:
    if re.search(r"^\s*<think\b", text, flags=re.IGNORECASE) and not re.search(
        r"</think>", text, flags=re.IGNORECASE
    ):
        raise ValueError("Model output contains an unclosed reasoning block; request a clean JSON response.")
    stripped = _strip_common_model_wrappers(text)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Model output was not exactly one valid JSON object; request the same answer again in the protocol. "
            f"JSON error: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("Model output JSON must be an object.")
    return value


def _safe_relpath(path_text: str) -> Path:
    rel = Path(path_text)
    if not path_text.strip() or rel == Path(".") or rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe file path from model: {rel}")
    if rel.parts[0] in {".agent_state", ".git"}:
        raise ValueError(f"Model file path targets harness or repository control state: {rel}")
    return rel


def _workspace_target(workspace: Path, rel: Path) -> Path:
    """Resolve existing symlinks and reject paths that escape the workspace."""
    root = workspace.resolve()
    target = (root / rel).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Unsafe file path from model resolves outside workspace: {rel}") from exc
    return target


def write_files(workspace: Path, files: list[dict]) -> list[str]:
    """Apply complete-file edits requested by the implementation model.

    The harness deliberately accepts whole-file content only. That keeps each
    agent turn easy to replay from JSON logs and avoids patch-format ambiguity
    when a local model is tired, verbose, or creatively wrong.
    """
    written: list[str] = []
    for item in files:
        rel = _safe_relpath(str(item["path"]))
        target = _workspace_target(workspace, rel)
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
    command: list[Any] | dict[str, Any],
    default_timeout_seconds: int,
    max_timeout_seconds: int,
) -> tuple[list[str], int, int]:
    """Normalize command formats used by implementation and validation plans.

    Most commands are simple argv lists. For long-running tools, a model may use
    {"cmd": [...], "timeout_seconds": 7200}; for expected negative-path tests,
    it may also use {"cmd": [...], "expected_returncode": 2}. Positive requested
    timeouts are clamped by a positive max timeout. A requested/default timeout
    of 0 disables the hard command deadline so progress review, not elapsed time
    alone, decides whether the process should keep running. The execution
    boundary accepts the same explicit argv protocol that model turns are asked
    to use; malformed command shapes must be repaired conversationally before
    execution rather than being guessed here.
    """
    if isinstance(command, dict):
        if "cmd" not in command:
            raise ValueError("command object must contain a list-valued cmd")
        parts = command["cmd"]
        timeout_value = command.get("timeout_seconds", default_timeout_seconds)
        returncode_value = command.get("expected_returncode", 0)
        if isinstance(timeout_value, bool) or not isinstance(timeout_value, int):
            raise ValueError("timeout_seconds must be an integer")
        if isinstance(returncode_value, bool) or not isinstance(returncode_value, int):
            raise ValueError("expected_returncode must be an integer")
        requested_timeout = timeout_value
        expected_returncode = returncode_value
    else:
        parts = command
        requested_timeout = default_timeout_seconds
        expected_returncode = 0
    if not isinstance(parts, list):
        raise ValueError("command must be an argv list or an object with a list-valued cmd")
    if not parts:
        raise ValueError("command argv must not be empty")
    if not all(isinstance(part, str) and part for part in parts):
        raise ValueError("command argv must contain only non-empty strings")
    if any("\x00" in part for part in parts):
        raise ValueError("command argv must not contain NUL bytes")
    if requested_timeout <= 0:
        timeout = 0
    elif max_timeout_seconds <= 0:
        timeout = requested_timeout
    else:
        timeout = max(1, min(requested_timeout, max_timeout_seconds))
    return list(parts), timeout, expected_returncode


def _git_mutation_reason(parts: list[str]) -> str:
    """Return a reason when an agent command would mutate repository history.

    The harness owns git staging and commits. Implementation agents may inspect
    git state (`git status`, `git diff`, `git log`) as evidence, but allowing
    them to commit during an implementation attempt hides the diff from the
    feedback agent and breaks the accepted-step commit protocol.
    """
    if not parts or Path(parts[0]).name != "git":
        return ""
    readonly = {
        "status",
        "diff",
        "log",
        "show",
        "rev-parse",
        "ls-files",
    }
    subcommand = ""
    for part in parts[1:]:
        if not part.startswith("-"):
            subcommand = part
            break
    if not subcommand or subcommand in readonly:
        return ""
    subcommand_index = parts.index(subcommand)
    subcommand_args = parts[subcommand_index + 1:]
    if subcommand == "branch" and all(
        part in {"--list", "-l", "--show-current", "--contains", "--no-contains", "--merged", "--no-merged"}
        or part.startswith("--format=")
        for part in subcommand_args
    ):
        return ""
    if subcommand == "config":
        read_options = {"--get", "--get-all", "--get-regexp", "--get-urlmatch", "--list", "-l"}
        mutating_options = {"--add", "--replace-all", "--unset", "--unset-all", "--remove-section", "--rename-section"}
        if any(part in mutating_options for part in subcommand_args):
            return (
                "git config mutation is not allowed in model-requested commands. "
                "The harness owns repository configuration"
            )
        if subcommand_args and subcommand_args[0] in read_options:
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
    progress_max_interval_seconds: int = 0,
) -> list[dict]:
    """Run bounded validation commands inside the workspace.

    Command results are fed directly to the feedback agent, so stdout/stderr are
    truncated to keep long sessions compact while preserving enough failure text
    for useful critique.
    """
    results: list[dict] = []
    max_timeout = timeout_seconds if max_timeout_seconds is None else max_timeout_seconds
    for index, command in enumerate(commands):
        command_metadata = {
            "timeout_explicit": isinstance(command, dict) and "timeout_seconds" in command,
        }
        declared_validation = isinstance(command, dict) and command.get("validation") is True
        final_state = not (isinstance(command, dict) and command.get("final_state") is False)
        try:
            parts, command_timeout, expected_returncode = _command_parts_and_timeout(
                command,
                timeout_seconds,
                max_timeout,
            )
        except (TypeError, ValueError) as exc:
            results.append({
                "command": command,
                "command_index": index,
                "timeout_seconds": timeout_seconds,
                "returncode": 125,
                "expected_returncode": 0,
                "returncode_matches_expected": False,
                "stdout": "",
                "stderr": f"Invalid command payload: {exc}",
                "timed_out": False,
                "invalid_command": True,
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
                    "declared_validation": declared_validation,
                    "final_state": final_state,
                    "command_metadata": command_metadata,
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
            progress_max_interval_seconds=progress_max_interval_seconds,
        )
        result["expected_returncode"] = expected_returncode
        result["returncode_matches_expected"] = result["returncode"] == expected_returncode
        result["declared_validation"] = declared_validation
        result["final_state"] = final_state
        result["command_metadata"] = command_metadata
        results.append(result)
    return results


def normalize_step(step: dict[str, Any], index: int) -> dict[str, Any]:
    step_id = str(step.get("id") or f"S{index}")
    return {
        "id": step_id,
        "title": str(step.get("title") or f"Step {index}"),
        "description": str(step.get("description") or ""),
        "depends_on": [str(x) for x in step.get("depends_on", [])],
        "persistent_paths": [str(x) for x in step.get("persistent_paths", [])],
        "acceptance_criteria": [str(x) for x in step.get("acceptance_criteria", [])],
        "validation_method": str(step.get("validation_method") or ""),
        "validation_commands": step.get("validation_commands", []),
        # Execution state belongs to the harness, never to model-authored plan JSON.
        "status": "pending",
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
    final_state = requirements.get("final_state", {})
    if isinstance(final_state, dict):
        lines.append("")
        lines.append("## Final State")
        paths = final_state.get("required_project_paths", [])
        lines.append(f"- Unrequested new paths allowed: {final_state.get('allow_unrequested_new_paths')}")
        if final_state.get("path_policy_basis"):
            lines.append(f"- Path policy basis: {final_state['path_policy_basis']}")
        for path in paths:
            lines.append(f"- Required project path: {path}")
        for item in final_state.get("other_constraints", []):
            lines.append(f"- Constraint: {item}")
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
        lines.append(f"- Status: {review.get('status') or 'unknown'}")
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
    final_state = requirements.get("final_state", {})
    if isinstance(final_state, dict):
        lines.append("")
        lines.append("## Final State")
        lines.append("")
        paths = final_state.get("required_project_paths", [])
        lines.append(f"- Unrequested new paths allowed: {final_state.get('allow_unrequested_new_paths')}")
        if final_state.get("path_policy_basis"):
            lines.append(f"- Path policy basis: {final_state['path_policy_basis']}")
        for path in paths:
            lines.append(f"- Required project path: {path}")
        for item in final_state.get("other_constraints", []):
            lines.append(f"- Constraint: {item}")
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
        if step.get("persistent_paths"):
            lines.append(f"  - Persistent paths: {', '.join(step['persistent_paths'])}")
        if step.get("acceptance_criteria"):
            lines.append("  - Acceptance criteria:")
            for criterion in step["acceptance_criteria"]:
                lines.append(f"    - {criterion}")
        if step.get("validation_method"):
            lines.append(f"  - Validation method: {step['validation_method']}")
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


def collect_workspace_files(
    workspace: Path,
    max_file_bytes: int = 20000,
    *,
    max_files: int = 1000,
    max_total_chars: int = 2_000_000,
) -> list[dict[str, Any]]:
    """Collect a bounded project snapshot for model evidence.

    Per-file clipping is insufficient when a tool creates thousands of small
    files. This boundary also caps the number and aggregate represented text of
    files before the snapshot is retained in run state. Common build-output
    directories are considered after ordinary source paths rather than hidden,
    because they may be the requested deliverable.
    """
    files: list[dict[str, Any]] = []
    root = workspace.resolve()
    represented_chars = 0

    def path_priority(path: Path) -> tuple[int, str]:
        rel_parts = path.relative_to(root).parts
        low_priority = int(any(part in LOW_PRIORITY_WORKSPACE_DIRS for part in rel_parts))
        return low_priority, path.as_posix()

    def candidate_paths() -> Iterator[Path]:
        deferred_roots: list[Path] = []
        for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
            current_path = Path(current)
            ordinary_directories: list[str] = []
            for name in sorted(directory_names):
                if name in SKIPPED_WORKSPACE_DIRS:
                    continue
                child = current_path / name
                if name in LOW_PRIORITY_WORKSPACE_DIRS and not child.is_symlink():
                    deferred_roots.append(child)
                else:
                    ordinary_directories.append(name)
            directory_names[:] = ordinary_directories
            for name in sorted(file_names):
                yield current_path / name

        for deferred_root in sorted(set(deferred_roots), key=path_priority):
            for current, directory_names, file_names in os.walk(
                deferred_root,
                topdown=True,
                followlinks=False,
            ):
                directory_names[:] = sorted(
                    name for name in directory_names
                    if name not in SKIPPED_WORKSPACE_DIRS
                )
                current_path = Path(current)
                for name in sorted(file_names):
                    yield current_path / name

    def append_boundary(reason: str, first_omitted_path: Path) -> None:
        nonlocal represented_chars
        omitted_path = str(first_omitted_path.relative_to(root))
        content = (
            f"[workspace snapshot truncated: {reason}; first omitted path: "
            f"{omitted_path}; additional files may exist]"
        )
        if max_total_chars > 0:
            remaining = max(0, max_total_chars - represented_chars)
            content = content[:remaining]
        files.append({
            "path": "[workspace snapshot boundary]",
            "content": content,
            "size": 0,
            "truncated": True,
            "snapshot_boundary": True,
            "snapshot_boundary_reason": reason,
            "first_omitted_path": omitted_path,
        })
        represented_chars += len(content)

    def exceeds_total_limit(item_chars: int) -> bool:
        return max_total_chars > 0 and represented_chars + item_chars > max_total_chars

    for path in candidate_paths():
        if not path.is_file():
            continue
        if max_files > 0 and len(files) >= max_files:
            append_boundary(f"kept at most {max_files} files", path)
            break
        rel = path.relative_to(root)
        try:
            path.resolve(strict=True).relative_to(root)
        except (OSError, ValueError):
            item = {
                "path": str(rel),
                "content": "[workspace file omitted: path resolves outside workspace or became unavailable]",
                "size": 0,
                "truncated": False,
                "unsafe_path": True,
            }
            item_chars = len(item["content"])
            if exceeds_total_limit(item_chars):
                append_boundary(f"kept at most {max_total_chars} represented characters", path)
                break
            files.append(item)
            represented_chars += item_chars
            continue
        size = path.stat().st_size
        with path.open("rb") as f:
            sample = f.read(min(size, 4096))
        if _looks_like_binary_file(path, sample):
            item = {
                "path": str(rel),
                "content": f"[binary artifact omitted from prompt; size={size} bytes]",
                "size": size,
                "truncated": False,
                "binary": True,
            }
            item_chars = len(item["content"])
            if exceeds_total_limit(item_chars):
                append_boundary(f"kept at most {max_total_chars} represented characters", path)
                break
            files.append(item)
            represented_chars += item_chars
            continue
        if size <= max_file_bytes:
            item = {
                "path": str(rel),
                "content": path.read_text(encoding="utf-8", errors="replace"),
                "size": size,
                "truncated": False,
            }
            item_chars = len(item["content"])
            if exceeds_total_limit(item_chars):
                append_boundary(f"kept at most {max_total_chars} represented characters", path)
                break
            files.append(item)
            represented_chars += item_chars
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
        if exceeds_total_limit(len(content)):
            append_boundary(f"kept at most {max_total_chars} represented characters", path)
            break
        files.append({"path": str(rel), "content": content, "size": size, "truncated": True})
        represented_chars += len(content)
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
