from __future__ import annotations

from pathlib import Path
import os
import selectors
import signal
import subprocess
import time
from typing import Any


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
            "timeout_seconds": timeout_seconds,
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
            "timeout_seconds": timeout_seconds,
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
    deadline = time.monotonic() + timeout_seconds
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
        if not killed_process_group and now >= deadline:
            timed_out = True
            killed_process_group = True
            force_close_deadline = now + 1.0
            kill_process_group(signal.SIGKILL)
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
    return {
        "command": command,
        "returncode": 124 if timed_out else returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "stdout_bytes": byte_counts["stdout"],
        "stderr_bytes": byte_counts["stderr"],
    }
