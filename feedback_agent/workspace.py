from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any


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


def ensure_plan(workspace: Path) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / "PLAN.md"
    if not path.exists():
        path.write_text(PLAN_TEMPLATE, encoding="utf-8")
    return path


def append_plan_note(workspace: Path, note: str) -> None:
    plan = ensure_plan(workspace)
    with plan.open("a", encoding="utf-8") as f:
        f.write(f"\n- {note.strip()}\n")


def extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output: {text[:200]}")
    return json.loads(stripped[start:end + 1])


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
        target.write_text(str(item.get("content", "")), encoding="utf-8")
        written.append(str(rel))
    return written


def _command_parts_and_timeout(
    command: list[Any] | dict[str, Any],
    default_timeout_seconds: int,
    max_timeout_seconds: int,
) -> tuple[list[str], int]:
    """Normalize command formats used by implementation and validation plans.

    Most commands are simple argv lists. For long-running tools, a model may use
    {"cmd": [...], "timeout_seconds": 7200}; the harness clamps that request to
    runtime.max_command_timeout_seconds so a typo cannot create an accidental
    infinite process.
    """
    if isinstance(command, dict):
        parts = command.get("cmd") or command.get("command") or []
        requested_timeout = int(command.get("timeout_seconds", default_timeout_seconds))
    else:
        parts = command
        requested_timeout = default_timeout_seconds
    timeout = max(1, min(requested_timeout, max_timeout_seconds))
    return [str(part) for part in parts], timeout


def run_commands(
    workspace: Path,
    commands: list[list[str] | dict[str, Any]],
    timeout_seconds: int,
    max_timeout_seconds: int | None = None,
) -> list[dict]:
    """Run bounded validation commands inside the workspace.

    Command results are fed directly to the feedback agent, so stdout/stderr are
    truncated to keep long sessions compact while preserving enough failure text
    for useful critique.
    """
    results: list[dict] = []
    max_timeout = max_timeout_seconds or timeout_seconds
    for command in commands:
        if not command:
            continue
        parts, command_timeout = _command_parts_and_timeout(command, timeout_seconds, max_timeout)
        if not parts:
            continue
        try:
            proc = subprocess.run(
                parts,
                cwd=workspace,
                text=True,
                capture_output=True,
                timeout=command_timeout,
                check=False,
            )
            results.append({
                "command": parts,
                "timeout_seconds": command_timeout,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
                "timed_out": False,
            })
        except subprocess.TimeoutExpired as exc:
            results.append({
                "command": parts,
                "returncode": 124,
                "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
                "timed_out": True,
                "timeout_seconds": command_timeout,
            })
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


def write_requirements_doc(workspace: Path, requirements: dict[str, Any], review: dict[str, Any] | None = None) -> None:
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
    (workspace / "REQUIREMENTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plan_doc(
    workspace: Path,
    requirements: dict[str, Any],
    steps: list[dict[str, Any]],
    notes: list[str] | None = None,
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
    (workspace / "PLAN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_workspace_files(workspace: Path, max_file_bytes: int = 20000) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    skipped_dirs = {".agent_state", ".git", "__pycache__", ".pytest_cache"}
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or any(part in skipped_dirs for part in path.parts):
            continue
        rel = path.relative_to(workspace)
        if path.stat().st_size <= max_file_bytes:
            files.append({"path": str(rel), "content": path.read_text(encoding="utf-8", errors="replace")})
    return files
