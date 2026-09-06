from __future__ import annotations

from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor
import os
import selectors
import signal
import subprocess
import time
from typing import Any, Callable


MAX_RETAINED_PROGRESS_REVIEWS = 20


def estimate_tokens(text: str) -> int:
    """Cheap conservative token estimate used for context guardrails."""
    return max(1, len(text) // 4)


def clamp_text(text: str, max_chars: int, *, marker: str = "truncated") -> str:
    """Keep bounded tail text with an explicit truncation marker.

    Tool output is usually most useful at the end because that is where stack
    traces, assertion summaries, and final status lines appear. Keeping the tail
    also prevents one runaway command or generated diff from entering the model
    context as a huge blob. A positive ``max_chars`` is a hard bound including
    the marker. A non-positive value preserves the existing convention of
    disabling clipping.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    prefix = f"[{marker}: original length {len(text)} chars]\n"
    if len(prefix) >= max_chars:
        return prefix[:max_chars]
    kept_chars = max_chars - len(prefix)
    return prefix + text[-kept_chars:]


def run_bounded_process(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    output_limit_chars: int,
    progress_callback: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    progress_interval_seconds: int = 0,
    progress_min_interval_seconds: int = 1,
    progress_max_interval_seconds: int = 0,
) -> dict[str, Any]:
    """Run a subprocess while keeping bounded stdout/stderr head and tail evidence.

    `subprocess.run(capture_output=True)` stores the whole output in memory.
    That is dangerous for an agent harness because one accidental verbose tool
    call can produce enough text to break the next model request or pressure the
    host. This runner continuously drains both pipes, counts total bytes, and
    retains bounded head and tail evidence with an explicit omission marker.
    """
    output_limit_bytes = max(1, output_limit_chars)
    head_limit_bytes = output_limit_bytes // 2
    tail_limit_bytes = output_limit_bytes - head_limit_bytes
    head_buffers = {"stdout": bytearray(), "stderr": bytearray()}
    tail_buffers = {"stdout": bytearray(), "stderr": bytearray()}
    byte_counts = {"stdout": 0, "stderr": 0}
    timed_out = False
    stopped_by_progress_review = False
    satisfied_by_progress_review = False
    progress_reviews: list[dict[str, Any]] = []
    progress_review_count = 0
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
            "spawn_error": True,
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
            "spawn_error": True,
        }
    except (OSError, ValueError) as exc:
        return {
            "command": command,
            "returncode": 126,
            "stdout": "",
            "stderr": f"command could not be started: {exc}",
            "timed_out": False,
            "timeout_seconds": hard_timeout_seconds,
            "hard_timeout_disabled": hard_timeout_seconds is None,
            "stdout_truncated": False,
            "stderr_truncated": False,
            "stdout_bytes": 0,
            "stderr_bytes": 0,
            "spawn_error": True,
        }
    selector = selectors.DefaultSelector()
    assert proc.stdout is not None
    assert proc.stderr is not None
    selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
    selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
    deadline = started + timeout_seconds if hard_timeout_seconds is not None else None
    progress_interval = max(0, progress_interval_seconds)
    min_progress_interval = max(1, progress_min_interval_seconds)
    max_progress_interval = max(0, progress_max_interval_seconds)
    if progress_interval and max_progress_interval:
        progress_interval = min(progress_interval, max_progress_interval)
    next_progress_review = started + progress_interval if progress_callback and progress_interval else None
    killed_process_group = False
    force_close_deadline: float | None = None
    progress_executor: ThreadPoolExecutor | None = None
    progress_future: Future[dict[str, Any] | None] | None = None
    progress_snapshot_in_flight: dict[str, Any] | None = None
    parent_exit_seen_at: float | None = None
    if progress_callback is not None and progress_interval:
        progress_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tool-progress-review")

    def absorb(stream_name: str, data: bytes) -> None:
        byte_counts[stream_name] += len(data)
        head = head_buffers[stream_name]
        if len(head) < head_limit_bytes:
            head.extend(data[: head_limit_bytes - len(head)])
        tail = tail_buffers[stream_name]
        tail.extend(data)
        if len(tail) > tail_limit_bytes:
            del tail[: len(tail) - tail_limit_bytes]

    def kill_process_group(sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        except OSError:
            if proc.poll() is None:
                proc.kill()

    def decoded_output(stream_name: str, *, progress: bool = False) -> str:
        count = byte_counts[stream_name]
        head = bytes(head_buffers[stream_name])
        tail = bytes(tail_buffers[stream_name])
        if count <= output_limit_bytes:
            overlap = max(0, len(head) + len(tail) - count)
            return (head + tail[overlap:]).decode("utf-8", errors="replace")
        scope = " for progress review" if progress else ""
        marker = (
            f"\n[{stream_name} truncated{scope}: kept first {len(head)} and last {len(tail)} "
            f"of {count} bytes]\n"
        ).encode("utf-8")
        if len(marker) >= output_limit_bytes:
            rendered = marker[:output_limit_bytes]
        else:
            available = output_limit_bytes - len(marker)
            kept_head = head[: available // 2]
            kept_tail = tail[-(available - len(kept_head)):] if available > len(kept_head) else b""
            rendered = kept_head + marker + kept_tail
        return rendered.decode("utf-8", errors="replace")

    def progress_snapshot(now: float, review_count: int) -> dict[str, Any]:
        return {
            "command": command,
            "cwd": str(cwd),
            "elapsed_seconds": round(now - started, 3),
            "timeout_seconds": hard_timeout_seconds,
            "hard_timeout_disabled": hard_timeout_seconds is None,
            "review_count": review_count,
            "returncode": proc.poll(),
            "stdout": decoded_output("stdout", progress=True),
            "stderr": decoded_output("stderr", progress=True),
            "stdout_bytes": byte_counts["stdout"],
            "stderr_bytes": byte_counts["stderr"],
            "stdout_truncated": byte_counts["stdout"] > output_limit_bytes,
            "stderr_truncated": byte_counts["stderr"] > output_limit_bytes,
        }

    def normalize_progress_decision(decision: dict[str, Any] | None) -> dict[str, Any]:
        protocol_error = not isinstance(decision, dict)
        if not isinstance(decision, dict):
            decision = {}
        normalized = dict(decision)
        supplied_decision = normalized.get("decision")
        raw_decision = str(supplied_decision or "").strip()
        if raw_decision not in {"continue", "stop_satisfied", "terminate"}:
            raw_decision = "continue"
            protocol_error = True
        normalized["decision"] = raw_decision
        if protocol_error:
            normalized["protocol_error"] = True
        try:
            next_check = int(normalized.get("next_check_seconds", progress_interval or min_progress_interval))
        except (TypeError, ValueError):
            next_check = progress_interval or min_progress_interval
        next_check = max(min_progress_interval, next_check)
        if max_progress_interval > 0:
            next_check = min(next_check, max_progress_interval)
        normalized["next_check_seconds"] = next_check
        if not normalized.get("summary"):
            summary_by_decision = {
                "continue": "Progress reviewer allowed the command to continue.",
                "stop_satisfied": "Progress reviewer found sufficient evidence and ended the command.",
                "terminate": "Progress reviewer requested unsuccessful termination.",
            }
            normalized["summary"] = summary_by_decision[raw_decision]
        return normalized

    def apply_progress_decision(decision: dict[str, Any], now: float) -> None:
        nonlocal killed_process_group, force_close_deadline, next_progress_review
        nonlocal progress_review_count, stopped_by_progress_review, satisfied_by_progress_review
        progress_review_count += 1
        progress_reviews.append(decision)
        if len(progress_reviews) > MAX_RETAINED_PROGRESS_REVIEWS:
            del progress_reviews[: len(progress_reviews) - MAX_RETAINED_PROGRESS_REVIEWS]
        if (
            decision["decision"] in {"stop_satisfied", "terminate"}
            and not timed_out
            and proc.poll() is None
        ):
            stopped_by_progress_review = decision["decision"] == "terminate"
            satisfied_by_progress_review = decision["decision"] == "stop_satisfied"
            killed_process_group = True
            force_close_deadline = now + 1.0
            kill_process_group(signal.SIGTERM)
            return
        next_progress_review = now + int(decision["next_check_seconds"])

    def collect_progress_review(now: float, *, wait: bool = False) -> None:
        nonlocal progress_future, progress_snapshot_in_flight
        if progress_future is None or (not wait and not progress_future.done()):
            return
        try:
            decision = normalize_progress_decision(progress_future.result())
        except Exception as exc:  # pragma: no cover - defensive boundary around optional reviewer code.
            decision = normalize_progress_decision({
                "decision": "continue",
                "summary": f"Progress review failed, so the harness kept draining the running command: {exc}",
                "review_error": repr(exc),
                "protocol_error": True,
            })
        if progress_snapshot_in_flight is not None:
            decision.setdefault(
                "reviewed_elapsed_seconds",
                progress_snapshot_in_flight.get("elapsed_seconds"),
            )
        progress_future = None
        progress_snapshot_in_flight = None
        apply_progress_decision(decision, now)

    def maybe_run_progress_review(now: float) -> None:
        nonlocal next_progress_review, progress_future, progress_snapshot_in_flight
        collect_progress_review(now)
        if progress_callback is None or next_progress_review is None:
            return
        if progress_future is not None or killed_process_group or proc.poll() is not None or now < next_progress_review:
            return
        snapshot = progress_snapshot(now, progress_review_count + 1)
        assert progress_executor is not None
        progress_snapshot_in_flight = snapshot
        progress_future = progress_executor.submit(progress_callback, snapshot)
        next_progress_review = None

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

    while selector.get_map() or proc.poll() is None:
        now = time.monotonic()
        if deadline is not None and not killed_process_group and now >= deadline:
            timed_out = True
            killed_process_group = True
            force_close_deadline = now + 1.0
            kill_process_group(signal.SIGKILL)
        maybe_run_progress_review(now)
        if proc.poll() is not None and parent_exit_seen_at is None:
            parent_exit_seen_at = now
        if (
            parent_exit_seen_at is not None
            and selector.get_map()
            and force_close_deadline is None
            and now - parent_exit_seen_at >= 0.1
        ):
            # The direct command has finished, but a background descendant is
            # still holding a pipe open. Validation calls are self-contained;
            # terminate the leftover process group after a short final drain.
            force_close_deadline = now + 1.0
            kill_process_group(signal.SIGTERM)
        if force_close_deadline is not None and now >= force_close_deadline:
            for key in list(selector.get_map().values()):
                stream = key.fileobj
                selector.unregister(stream)
                stream.close()
            break
        if selector.get_map():
            events = selector.select(timeout=0.05)
        else:
            time.sleep(0.05)
            events = []
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
    if progress_future is not None:
        collect_progress_review(time.monotonic(), wait=True)
    if progress_executor is not None:
        progress_executor.shutdown(wait=True)
    cleanup_process_group()
    stdout = decoded_output("stdout")
    stderr = decoded_output("stderr")
    stdout_truncated = byte_counts["stdout"] > output_limit_bytes
    stderr_truncated = byte_counts["stderr"] > output_limit_bytes
    ended_by_progress_review = stopped_by_progress_review or satisfied_by_progress_review
    if ended_by_progress_review:
        summaries = "; ".join(str(item.get("summary", "")) for item in progress_reviews[-3:] if item.get("summary"))
        marker = (
            "[command stopped after progress review found sufficient evidence]"
            if satisfied_by_progress_review
            else "[command stopped by progress review]"
        )
        if summaries:
            marker += f" {summaries}"
        stderr = clamp_text(
            (stderr + "\n" + marker).strip(),
            output_limit_bytes,
            marker="stderr truncated after progress review",
        )
    return {
        "command": command,
        "returncode": 125 if ended_by_progress_review else 124 if timed_out else returncode,
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
        "ended_by_progress_review": ended_by_progress_review,
        "satisfied_by_progress_review": satisfied_by_progress_review,
        "stopped_by_progress_review": stopped_by_progress_review,
        "progress_reviews": progress_reviews,
        "progress_review_count": progress_review_count,
        "progress_reviews_truncated": progress_review_count > len(progress_reviews),
    }
