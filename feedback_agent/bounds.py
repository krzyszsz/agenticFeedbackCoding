from __future__ import annotations

from pathlib import Path
import os
import selectors
import signal
import subprocess
import time
from typing import Any, Callable


def estimate_tokens(text: str) -> int:
    """Cheap conservative token estimate used for context guardrails."""
    return max(1, len(text) // 4)


def clamp_text(text: str, max_chars: int, *, marker: str = "truncated") -> str:
    """Keep the tail of long text with an explicit truncation marker.

    Tool output is usually most useful at the end because that is where stack
    traces, assertion summaries, and final status lines appear. Keeping the tail
    also prevents one runaway command or generated diff from entering the model
    context as a huge blob.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    kept = text[-max_chars:]
    return f"[{marker}: kept last {max_chars} of {len(text)} chars]\n{kept}"


def run_bounded_process(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    output_limit_chars: int,
    progress_callback: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    progress_interval_seconds: int = 0,
    progress_min_interval_seconds: int = 1,
) -> dict[str, Any]:
    """Run a subprocess while keeping only bounded stdout/stderr tails.

    `subprocess.run(capture_output=True)` stores the whole output in memory.
    That is dangerous for an agent harness because one accidental verbose tool
    call can produce enough text to break the next model request or pressure the
    host. This runner continuously drains both pipes, counts total bytes, and
    retains only the configured tail for evidence.
    """
    output_limit_bytes = max(1024, output_limit_chars)
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    byte_counts = {"stdout": 0, "stderr": 0}
    timed_out = False
    stopped_by_progress_review = False
    progress_reviews: list[dict[str, Any]] = []
    started = time.monotonic()
    hard_timeout_seconds = timeout_seconds if timeout_seconds > 0 else None

    try:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        return {
            "command": command,
            "returncode": 127,
            "stdout": "",
            "stderr": f"command not found: {exc.filename or command[0]}",
            "timed_out": False,
            "timeout_seconds": hard_timeout_seconds,
            "hard_timeout_disabled": hard_timeout_seconds is None,
            "stdout_truncated": False,
            "stderr_truncated": False,
            "stdout_bytes": 0,
            "stderr_bytes": 0,
        }
    except PermissionError as exc:
        return {
            "command": command,
            "returncode": 126,
            "stdout": "",
            "stderr": f"command not executable: {exc.filename or command[0]}",
            "timed_out": False,
            "timeout_seconds": hard_timeout_seconds,
            "hard_timeout_disabled": hard_timeout_seconds is None,
            "stdout_truncated": False,
            "stderr_truncated": False,
            "stdout_bytes": 0,
            "stderr_bytes": 0,
        }
    selector = selectors.DefaultSelector()
    assert proc.stdout is not None
    assert proc.stderr is not None
    selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
    selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
    deadline = started + timeout_seconds if hard_timeout_seconds is not None else None
    progress_interval = max(0, progress_interval_seconds)
    min_progress_interval = max(1, progress_min_interval_seconds)
    next_progress_review = started + progress_interval if progress_callback and progress_interval else None
    killed_process_group = False
    force_close_deadline: float | None = None

    def absorb(stream_name: str, data: bytes) -> None:
        byte_counts[stream_name] += len(data)
        buffer = buffers[stream_name]
        buffer.extend(data)
        if len(buffer) > output_limit_bytes:
            del buffer[: len(buffer) - output_limit_bytes]

    def kill_process_group(sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        except OSError:
            if proc.poll() is None:
                proc.kill()

    def decoded_tail(stream_name: str) -> str:
        text = buffers[stream_name].decode("utf-8", errors="replace")
        if byte_counts[stream_name] > output_limit_bytes:
            return f"[{stream_name} truncated for progress review: kept last {output_limit_bytes} of {byte_counts[stream_name]} bytes]\n{text}"
        return text

    def progress_snapshot(now: float, review_count: int) -> dict[str, Any]:
        return {
            "command": command,
            "cwd": str(cwd),
            "elapsed_seconds": round(now - started, 3),
            "timeout_seconds": hard_timeout_seconds,
            "hard_timeout_disabled": hard_timeout_seconds is None,
            "review_count": review_count,
            "returncode": proc.poll(),
            "stdout": decoded_tail("stdout"),
            "stderr": decoded_tail("stderr"),
            "stdout_bytes": byte_counts["stdout"],
            "stderr_bytes": byte_counts["stderr"],
            "stdout_truncated": byte_counts["stdout"] > output_limit_bytes,
            "stderr_truncated": byte_counts["stderr"] > output_limit_bytes,
        }

    def normalize_progress_decision(decision: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(decision, dict):
            decision = {}
        normalized = dict(decision)
        raw_decision = str(normalized.get("decision") or normalized.get("status") or "continue").strip().lower()
        if raw_decision in {"stop", "stopped", "terminate", "terminated", "kill", "cancel"}:
            raw_decision = "terminate"
        elif raw_decision != "continue":
            raw_decision = "continue"
        normalized["decision"] = raw_decision
        try:
            next_check = int(normalized.get("next_check_seconds", progress_interval or min_progress_interval))
        except (TypeError, ValueError):
            next_check = progress_interval or min_progress_interval
        normalized["next_check_seconds"] = max(min_progress_interval, next_check)
        if not normalized.get("summary"):
            normalized["summary"] = (
                "Progress reviewer requested termination."
                if raw_decision == "terminate"
                else "Progress reviewer allowed the command to continue."
            )
        return normalized

    def maybe_run_progress_review(now: float) -> None:
        nonlocal killed_process_group, force_close_deadline, next_progress_review, stopped_by_progress_review
        if progress_callback is None or next_progress_review is None:
            return
        if killed_process_group or proc.poll() is not None or now < next_progress_review:
            return
        snapshot = progress_snapshot(now, len(progress_reviews) + 1)
        try:
            decision = normalize_progress_decision(progress_callback(snapshot))
        except Exception as exc:  # pragma: no cover - defensive boundary around optional reviewer code.
            decision = normalize_progress_decision({
                "decision": "continue",
                "summary": f"Progress review failed, so the harness kept draining the running command: {exc}",
                "review_error": repr(exc),
            })
        progress_reviews.append(decision)
        if decision["decision"] == "terminate":
            stopped_by_progress_review = True
            killed_process_group = True
            force_close_deadline = time.monotonic() + 1.0
            kill_process_group(signal.SIGTERM)
            return
        next_progress_review = time.monotonic() + int(decision["next_check_seconds"])

    def cleanup_process_group() -> None:
        """Clean up descendants left behind by shells or validation scripts.

        Agent-generated commands often start background processes for tests.
        Even when the direct child exits, those descendants can keep running in
        the same process group. Validation commands are expected to be
        self-contained, so the harness cleans up the group after collecting the
        parent result.
        """
        kill_process_group(signal.SIGTERM)
        time.sleep(0.05)
        kill_process_group(signal.SIGKILL)

    while selector.get_map():
        now = time.monotonic()
        if deadline is not None and not killed_process_group and now >= deadline:
            timed_out = True
            killed_process_group = True
            force_close_deadline = now + 1.0
            kill_process_group(signal.SIGKILL)
        maybe_run_progress_review(now)
        if force_close_deadline is not None and now >= force_close_deadline:
            for key in list(selector.get_map().values()):
                stream = key.fileobj
                selector.unregister(stream)
                stream.close()
            break
        events = selector.select(timeout=0.05)
        if not events and proc.poll() is not None:
            # Give EOF notifications a chance to drain both pipes.
            events = selector.select(timeout=0)
        for key, _ in events:
            stream = key.fileobj
            data = os.read(stream.fileno(), 8192)
            if data:
                absorb(str(key.data), data)
            else:
                selector.unregister(stream)
                stream.close()

    try:
        returncode = proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_process_group(signal.SIGKILL)
        returncode = proc.wait(timeout=1)
    cleanup_process_group()
    stdout = buffers["stdout"].decode("utf-8", errors="replace")
    stderr = buffers["stderr"].decode("utf-8", errors="replace")
    stdout_truncated = byte_counts["stdout"] > output_limit_bytes
    stderr_truncated = byte_counts["stderr"] > output_limit_bytes
    if stdout_truncated:
        stdout = f"[stdout truncated: kept last {output_limit_bytes} of {byte_counts['stdout']} bytes]\n{stdout}"
    if stderr_truncated:
        stderr = f"[stderr truncated: kept last {output_limit_bytes} of {byte_counts['stderr']} bytes]\n{stderr}"
    if stopped_by_progress_review:
        summaries = "; ".join(str(item.get("summary", "")) for item in progress_reviews[-3:] if item.get("summary"))
        marker = "[command stopped by progress review]"
        if summaries:
            marker += f" {summaries}"
        stderr = (stderr + "\n" + marker).strip()
    return {
        "command": command,
        "returncode": 125 if stopped_by_progress_review else 124 if timed_out else returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "timeout_seconds": hard_timeout_seconds,
        "hard_timeout_disabled": hard_timeout_seconds is None,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "stdout_bytes": byte_counts["stdout"],
        "stderr_bytes": byte_counts["stderr"],
        "stopped_by_progress_review": stopped_by_progress_review,
        "progress_reviews": progress_reviews,
    }
