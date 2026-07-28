from __future__ import annotations

from pathlib import Path
from typing import Any

from .bounds import run_bounded_process


LOCAL_GIT_EXCLUDES = """# agenticFeedbackCoding local runtime state
.agent_state/
__pycache__/
*.pyc
.pytest_cache/
$HOME/
.dotnet/
node_modules/
.venv/
venv/
"""


HARNESS_ONLY_PATHS = {
    ".agent_state",
}


def run_git(
    workspace: Path,
    args: list[str],
    timeout_seconds: int = 30,
    output_limit_chars: int = 20000,
) -> dict[str, Any]:
    """Run a bounded git command and return JSON-friendly evidence."""
    command = [
        "git",
        "-c",
        f"safe.directory={workspace.resolve()}",
        "-c",
        "core.hooksPath=/dev/null",
        *args,
    ]
    return run_bounded_process(
        command,
        cwd=workspace,
        timeout_seconds=timeout_seconds,
        output_limit_chars=output_limit_chars,
    )


def _normalized_ignored_paths(paths: set[str] | None) -> list[str]:
    normalized: list[str] = []
    for value in sorted(paths or set()):
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            continue
        text = path.as_posix()
        while text.startswith("./"):
            text = text[2:]
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def ensure_git_repo(
    workspace: Path,
    *,
    user_name: str,
    user_email: str,
    ignored_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Initialize a workspace-local repository and configure deterministic identity."""
    workspace.mkdir(parents=True, exist_ok=True)
    init = run_git(workspace, ["init"])
    if init.get("returncode") == 0:
        exclude_path = workspace / ".git" / "info" / "exclude"
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude_path.read_text(encoding="utf-8") if exclude_path.exists() else ""
        desired = [
            *[line for line in LOCAL_GIT_EXCLUDES.splitlines() if line],
            *_normalized_ignored_paths(ignored_paths),
        ]
        missing = [line for line in desired if line not in existing.splitlines()]
        if missing:
            prefix = "" if not existing or existing.endswith("\n") else "\n"
            exclude_path.write_text(existing + prefix + "\n".join(missing) + "\n", encoding="utf-8")
    results = [
        init,
        run_git(workspace, ["config", "user.name", user_name]),
        run_git(workspace, ["config", "user.email", user_email]),
        run_git(workspace, ["config", "core.autocrlf", "false"]),
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


def meaningful_changed_paths(status_short: str, *, ignored_paths: set[str] | None = None) -> list[str]:
    """Return changed paths that are not just harness bookkeeping files."""
    ignored = HARNESS_ONLY_PATHS | {str(Path(path)) for path in (ignored_paths or set())}
    changed: list[str] = []
    for line in status_short.splitlines():
        if line.lstrip().startswith("[stdout truncated"):
            continue
        path = _status_path(line)
        if not path:
            continue
        root = path.split("/", 1)[0]
        if path in ignored or root in ignored:
            continue
        changed.append(path)
    return changed


def git_evidence(
    workspace: Path,
    *,
    max_diff_chars: int = 20000,
    ignored_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Capture the current diff/status for the feedback agent."""
    status_result = run_git(
        workspace,
        ["status", "--short"],
        output_limit_chars=max_diff_chars,
    )
    # Preserve leading porcelain status columns, e.g. " M file.py".
    status = str(status_result.get("stdout") or "").rstrip()
    ignored = HARNESS_ONLY_PATHS | {str(Path(path)) for path in (ignored_paths or set())}
    diff_pathspec = [".", *[f":(exclude){path}" for path in sorted(ignored)]]
    head = git_head(workspace)
    diff_prefix = ["diff", "HEAD"] if head else ["diff"]
    return {
        "enabled": True,
        "head": head,
        "status_short": status,
        "status_truncated": bool(status_result.get("stdout_truncated")),
        "status_bytes": status_result.get("stdout_bytes"),
        "meaningful_changed_paths": meaningful_changed_paths(status, ignored_paths=ignored),
        "diff_stat": run_git(
            workspace,
            [*diff_prefix, "--stat", "--", *diff_pathspec],
            output_limit_chars=max_diff_chars,
        )["stdout"].strip(),
        "diff": run_git(
            workspace,
            [*diff_prefix, "--", *diff_pathspec],
            output_limit_chars=max_diff_chars,
        )["stdout"],
    }


def commit_all(
    workspace: Path,
    message: str,
    *,
    allow_empty: bool = False,
    timeout_seconds: int = 30,
    ignored_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Commit project changes while leaving harness control files untracked."""
    before = git_head(workspace)
    status_before = git_status_short(workspace)
    excluded = _normalized_ignored_paths(HARNESS_ONLY_PATHS | (ignored_paths or set()))
    add = run_git(workspace, ["add", "-A"], timeout_seconds)
    if add.get("returncode") != 0:
        return {
            "committed": False,
            "reason": "git add failed",
            "head_before": before,
            "head_after": git_head(workspace),
            "status_before": status_before,
            "status_after": git_status_short(workspace),
            "add": add,
        }
    unstage_control_paths: dict[str, Any] | None = None
    if before and excluded:
        unstage_control_paths = run_git(
            workspace,
            ["reset", "-q", before, "--", *excluded],
            timeout_seconds,
        )
        if unstage_control_paths.get("returncode") != 0:
            return {
                "committed": False,
                "reason": "could not exclude tracked harness control files from the commit",
                "head_before": before,
                "head_after": git_head(workspace),
                "status_before": status_before,
                "status_after": git_status_short(workspace),
                "add": add,
                "unstage_control_paths": unstage_control_paths,
                "ignored_paths": excluded,
            }
    staged = run_git(workspace, ["diff", "--cached", "--quiet", "--exit-code"], timeout_seconds)
    if not allow_empty and staged.get("returncode") == 0:
        return {
            "committed": False,
            "reason": "no project changes",
            "head_before": before,
            "head_after": before,
            "status_before": status_before,
            "add": add,
            "staged_check": staged,
            "unstage_control_paths": unstage_control_paths,
            "ignored_paths": excluded,
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
        "staged_check": staged,
        "unstage_control_paths": unstage_control_paths,
        "ignored_paths": excluded,
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
