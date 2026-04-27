from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any


DEFAULT_GITIGNORE = """# Harness/runtime state
.agent_state/
__pycache__/
*.pyc
.pytest_cache/
out/
"""


HARNESS_ONLY_PATHS = {
    ".agent_state",
    ".gitignore",
    "PLAN.md",
    "REQUIREMENTS.md",
    "RESEARCH.md",
}


def run_git(workspace: Path, args: list[str], timeout_seconds: int = 30) -> dict[str, Any]:
    """Run a bounded git command and return JSON-friendly evidence."""
    command = ["git", *args]
    try:
        proc = subprocess.run(
            command,
            cwd=workspace,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-20000:],
            "stderr": proc.stderr[-20000:],
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "stdout": (exc.stdout or "")[-20000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-20000:] if isinstance(exc.stderr, str) else "",
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
        }


def ensure_git_repo(workspace: Path, *, user_name: str, user_email: str) -> dict[str, Any]:
    """Initialize a workspace-local repository and configure deterministic identity."""
    workspace.mkdir(parents=True, exist_ok=True)
    gitignore = workspace / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(DEFAULT_GITIGNORE, encoding="utf-8")
    results = [
        run_git(workspace, ["init"]),
        run_git(workspace, ["config", "user.name", user_name]),
        run_git(workspace, ["config", "user.email", user_email]),
        run_git(workspace, ["config", "core.autocrlf", "false"]),
        run_git(workspace, ["config", "safe.directory", str(workspace)]),
    ]
    return {"enabled": True, "results": results}


def git_head(workspace: Path) -> str:
    result = run_git(workspace, ["rev-parse", "--verify", "HEAD"])
    if result["returncode"] == 0:
        return str(result["stdout"]).strip()
    return ""


def git_status_short(workspace: Path) -> str:
    # Preserve leading porcelain status columns, e.g. " M file.py".
    return str(run_git(workspace, ["status", "--short"])["stdout"]).rstrip()


def _status_path(line: str) -> str:
    # Handles ordinary porcelain lines and simple rename lines: "R  old -> new".
    path = line[3:].strip() if len(line) > 3 else line.strip()
    if " -> " in path:
        path = path.rsplit(" -> ", 1)[-1]
    return path.strip()


def meaningful_changed_paths(status_short: str) -> list[str]:
    """Return changed paths that are not just harness bookkeeping files."""
    changed: list[str] = []
    for line in status_short.splitlines():
        path = _status_path(line)
        if not path:
            continue
        root = path.split("/", 1)[0]
        if path in HARNESS_ONLY_PATHS or root in HARNESS_ONLY_PATHS:
            continue
        changed.append(path)
    return changed


def git_evidence(workspace: Path) -> dict[str, Any]:
    """Capture the current diff/status for the feedback agent."""
    status = git_status_short(workspace)
    return {
        "enabled": True,
        "head": git_head(workspace),
        "status_short": status,
        "meaningful_changed_paths": meaningful_changed_paths(status),
        "diff_stat": run_git(workspace, ["diff", "--stat"])["stdout"].strip(),
        "diff": run_git(workspace, ["diff", "--", "."])["stdout"][-20000:],
    }


def commit_all(
    workspace: Path,
    message: str,
    *,
    allow_empty: bool = False,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Commit all workspace changes if present, optionally creating an empty anchor commit."""
    before = git_head(workspace)
    status_before = git_status_short(workspace)
    add = run_git(workspace, ["add", "-A"], timeout_seconds)
    if not allow_empty and not git_status_short(workspace):
        return {
            "committed": False,
            "reason": "no changes",
            "head_before": before,
            "head_after": before,
            "status_before": status_before,
            "add": add,
        }
    args = ["commit", "-m", message]
    if allow_empty:
        args.insert(1, "--allow-empty")
    commit = run_git(workspace, args, timeout_seconds)
    after = git_head(workspace)
    return {
        "committed": commit["returncode"] == 0,
        "head_before": before,
        "head_after": after,
        "status_before": status_before,
        "status_after": git_status_short(workspace),
        "add": add,
        "commit": commit,
    }


def reset_to_ref(workspace: Path, ref: str, *, mode: str = "soft") -> dict[str, Any]:
    """Expose final changes as uncommitted work when the user asks for that mode."""
    if not ref:
        return {"reset": False, "reason": "missing ref"}
    if mode not in {"soft", "mixed"}:
        mode = "soft"
    result = run_git(workspace, ["reset", f"--{mode}", ref])
    return {
        "reset": result["returncode"] == 0,
        "mode": mode,
        "target": ref,
        "result": result,
        "status_after": git_status_short(workspace),
    }
