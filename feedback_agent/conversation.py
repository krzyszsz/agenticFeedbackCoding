from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import re
import sys
from collections.abc import Iterable, Iterator

from .bounds import clamp_text, estimate_tokens
from .protocol import (
    HARNESS_EFFECTIVE_REVIEW_MARKER,
    HARNESS_RESPONSE_OMISSION_MARKER,
    SHARED_SYSTEM_CONTEXT_MARKER,
    VALIDATED_FEEDBACK_DECISION_MARKER,
)


@dataclass
class Turn:
    role: str
    content: str


class Conversation:
    """Durable chat transcript shared by the implementation and feedback agents.

    The harness stores every prompt and response instead of rebuilding isolated
    one-shot prompts. That is intentional: local models are much better at long
    agentic work when each new request is appended to a visible conversation.
    """

    def __init__(
        self,
        path: Path,
        *,
        echo: bool = False,
        full_path: Path | None = None,
        echo_limit_chars: int = 0,
        color: bool = True,
    ):
        self.path = path
        self.full_path = full_path
        self.echo = echo
        self.echo_limit_chars = echo_limit_chars
        self.color = color and sys.stdout.isatty()
        self._repair_final_record(path)
        if self.full_path and self.full_path != path:
            self._repair_final_record(self.full_path)
        self.turns: list[Turn] = self._load_turns(path)
        if self.full_path and not self.full_path.exists() and self.turns:
            self._write_turns(self.full_path, self.turns)

    def _load_turns(self, path: Path) -> list[Turn]:
        return list(self._iter_turns(path))

    @staticmethod
    def _repair_final_record(path: Path) -> None:
        """Remove only a torn final JSONL record before the next append.

        A process interruption can leave the final append incomplete. Ignoring
        it while loading is insufficient: a later append would join new JSON to
        those bytes and turn recoverable tail damage into a malformed middle
        record. This check reads only the final physical line, truncates it when
        invalid, and preserves every earlier byte.
        """
        if not path.exists() or path.stat().st_size == 0:
            return
        whitespace = b" \t\r\n"
        with path.open("r+b") as stream:
            size = stream.seek(0, 2)
            position = size
            record_end = 0
            while position > 0 and record_end == 0:
                start = max(0, position - 8192)
                stream.seek(start)
                chunk = stream.read(position - start)
                for index in range(len(chunk) - 1, -1, -1):
                    if chunk[index] not in whitespace:
                        record_end = start + index + 1
                        break
                position = start
            if record_end == 0:
                return

            position = record_end
            record_start = 0
            while position > 0:
                start = max(0, position - 8192)
                stream.seek(start)
                chunk = stream.read(position - start)
                newline = chunk.rfind(b"\n")
                if newline >= 0:
                    record_start = start + newline + 1
                    break
                position = start

            stream.seek(record_start)
            raw = stream.read(record_end - record_start)
            try:
                item = json.loads(raw.decode("utf-8"))
                if not isinstance(item, dict):
                    raise TypeError("record must be an object")
                if not isinstance(item.get("role"), str) or not isinstance(item.get("content"), str):
                    raise TypeError("role and content must be strings")
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
                stream.truncate(record_start)
                print(
                    f"[conversation] discarded incomplete final JSONL record at {path}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                return

            stream.seek(0, 2)
            if size > 0:
                stream.seek(-1, 2)
                if stream.read(1) != b"\n":
                    stream.seek(0, 2)
                    stream.write(b"\n")

    @staticmethod
    def _iter_turns(path: Path) -> Iterator[Turn]:
        if not path.exists():
            return
        pending: tuple[int, str] | None = None
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                if pending is not None:
                    yield Conversation._decode_turn(path, *pending, allow_incomplete=False)
                pending = (line_number, line)
        if pending is not None:
            turn = Conversation._decode_turn(path, *pending, allow_incomplete=True)
            if turn is not None:
                yield turn

    @staticmethod
    def _decode_turn(
        path: Path,
        line_number: int,
        line: str,
        *,
        allow_incomplete: bool,
    ) -> Turn | None:
        try:
            item = json.loads(line)
            role = item["role"]
            content = item["content"]
            if not isinstance(role, str) or not isinstance(content, str):
                raise TypeError("role and content must be strings")
            return Turn(role=role, content=content)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            if allow_incomplete:
                print(
                    f"[conversation] ignored incomplete final JSONL record at {path}:{line_number}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                return None
            raise ValueError(f"Malformed conversation JSONL at {path}:{line_number}: {exc}") from exc

    def _append_turn_to_path(self, path: Path, turn: Turn) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")

    def _write_turns(self, path: Path, turns: list[Turn]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            "".join(json.dumps(asdict(t), ensure_ascii=False) + "\n" for t in turns),
            encoding="utf-8",
        )
        temporary.replace(path)

    def append(self, role: str, content: str, *, full_content: str | None = None) -> None:
        """Append active context and, when supplied, a raw audit representation."""
        turn = Turn(role=role, content=content)
        self.turns.append(turn)
        self._append_turn_to_path(self.path, turn)
        if self.full_path and self.full_path != self.path:
            self._append_turn_to_path(
                self.full_path,
                Turn(role=role, content=full_content if full_content is not None else content),
            )
        if self.echo:
            self._print_turn(turn)

    def replace_last_turn(
        self,
        *,
        role: str,
        content_prefix: str,
        new_content: str,
        replacement_role: str | None = None,
    ) -> bool:
        """Replace the active-context copy of the latest turn when it is unsafe.

        The append-only full transcript remains the audit log. The compact active
        transcript is allowed to replace pathological content with a bounded,
        explicitly harness-owned turn so later model calls do not inherit
        malformed output or mistake the replacement for model speech.
        """
        if not self.turns:
            return False
        latest = self.turns[-1]
        if latest.role != role or not latest.content.startswith(content_prefix):
            return False
        self.turns[-1] = Turn(role=replacement_role or role, content=new_content)
        self._write_turns(self.path, self.turns)
        if self.full_path and self.full_path != self.path:
            self._append_turn_to_path(
                self.full_path,
                Turn(
                    role="system",
                    content=(
                        "ACTIVE_CONTEXT_TURN_REPLACED: the latest active-context turn was "
                        "rewritten with a bounded recovery note. The original turn remains "
                        "earlier in this append-only full transcript."
                    ),
                ),
            )
        if self.echo:
            self._print_turn(self.turns[-1])
        return True

    def messages(
        self,
        *,
        recipient: str | None = None,
        system_as_user: bool = False,
        reviewer_view: bool = False,
    ) -> list[dict[str, str]]:
        """Return chat messages with roles appropriate to the receiving model.

        Audit labels record which participant produced a turn, but they are not
        model protocol. Replaying those labels taught smaller models to emit the
        same wrappers even though the current phase requires bare JSON. Each
        participant therefore sees its own requests as user turns, its own
        responses as assistant turns, and the other participant's output as
        external user evidence. Requests addressed only to the other participant
        stay in the audit transcript but are omitted from this model-facing view.
        """
        if recipient is None:
            recipient = "reviewer" if reviewer_view else "implementation"
        if recipient not in {"implementation", "reviewer"}:
            raise ValueError("recipient must be 'implementation' or 'reviewer'")

        messages: list[dict[str, str]] = []
        for turn in self.turns:
            if system_as_user and turn.role == "system":
                messages.append({"role": "user", "content": "Shared harness context:\n" + turn.content})
                continue
            if turn.role == "system":
                messages.append(asdict(turn))
                continue

            marker, separator, body = turn.content.partition("\n")
            if not separator:
                messages.append(asdict(turn))
                continue

            if marker == "IMPLEMENTATION_AGENT_REQUEST:":
                if recipient == "implementation":
                    messages.append({"role": "user", "content": body})
                continue
            if marker == "IMPLEMENTATION_AGENT_RESPONSE:":
                body = self._strip_repeated_audit_wrapper(
                    body,
                    "IMPLEMENTATION_AGENT_RESPONSE:",
                )
                if recipient == "implementation":
                    messages.append({"role": "assistant", "content": body})
                else:
                    messages.append({"role": "user", "content": "Implementation output to review:\n" + body})
                continue
            if marker == "FEEDBACK_AGENT_REQUEST:":
                if recipient == "reviewer":
                    messages.append({"role": "user", "content": body})
                continue
            if marker == "FEEDBACK_AGENT_RESPONSE:":
                body = self._strip_repeated_audit_wrapper(
                    body,
                    "FEEDBACK_AGENT_RESPONSE:",
                )
                if recipient == "reviewer":
                    messages.append({"role": "assistant", "content": body})
                else:
                    messages.append({"role": "user", "content": "External review of prior work:\n" + body})
                continue
            messages.append(asdict(turn))
        return messages

    @staticmethod
    def _strip_repeated_audit_wrapper(content: str, marker: str) -> str:
        """Remove only duplicated harness transport labels from a model view."""
        prefix = marker + "\n"
        while content.startswith(prefix):
            content = content[len(prefix):]
        return content

    def estimated_tokens(self) -> int:
        return max(1, sum(estimate_tokens(t.content) for t in self.turns))

    def replace_with_memory(self, memory: str, keep_recent_turns: int) -> None:
        base_system = next(
            (
                turn
                for turn in self.turns
                if turn.role == "system" and turn.content.startswith(SHARED_SYSTEM_CONTEXT_MARKER)
            ),
            None,
        )
        raw_recent = self.turns[-keep_recent_turns:] if keep_recent_turns > 0 else []
        recent = [turn for turn in raw_recent if turn.role != "system"]
        memory_turn = Turn(
            role="system",
            content=(
                "Compacted durable memory from earlier turns. Preserve these decisions, "
                "constraints, and unresolved risks:\n\n" + memory
            ),
        )
        self.turns = [turn for turn in (base_system, memory_turn) if turn is not None] + recent
        self._write_turns(self.path, self.turns)
        if self.full_path and self.full_path != self.path:
            self._append_turn_to_path(
                self.full_path,
                Turn(
                    role="system",
                    content=(
                        "ACTIVE_CONTEXT_COMPACTED: conversation.jsonl was rewritten "
                        "with compacted memory plus recent turns. Full prior turns "
                        "remain above in this append-only transcript."
                    ),
                ),
            )
        if self.echo:
            self._print_turn(self.turns[0])

    def write_markdown(self, path: Path, *, full: bool = False) -> None:
        """Export the JSONL transcript in a form people can read after the run."""
        turns: Iterable[Turn]
        turns = self._iter_turns(self.full_path) if full and self.full_path else self.turns
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            stream.write("# Agent Transcript\n\n")
            for index, turn in enumerate(turns, start=1):
                longest_backtick_run = max(
                    (len(match.group(0)) for match in re.finditer(r"`+", turn.content)),
                    default=0,
                )
                fence = "`" * max(3, longest_backtick_run + 1)
                stream.write(f"## {index}. {turn.role}\n\n{fence}text\n")
                stream.write(turn.content)
                if not turn.content.endswith("\n"):
                    stream.write("\n")
                stream.write(f"{fence}\n\n")

    def _print_turn(self, turn: Turn) -> None:
        label = self._turn_label(turn)
        color = self._turn_color(turn)
        reset = "\033[0m" if color else ""
        print(f"\n{color}===== {label} ====={reset}", flush=True)
        if self.echo_limit_chars > 0:
            print(
                clamp_text(turn.content, self.echo_limit_chars, marker="live transcript turn truncated"),
                flush=True,
            )
        else:
            print(turn.content, flush=True)
        print(f"{color}===== END TURN ====={reset}\n", flush=True)
        sys.stdout.flush()

    def _turn_label(self, turn: Turn) -> str:
        content = turn.content
        if content.startswith("IMPLEMENTATION_AGENT_REQUEST"):
            return "IMPLEMENTATION REQUEST"
        if content.startswith("IMPLEMENTATION_AGENT_RESPONSE"):
            return "IMPLEMENTATION RESPONSE"
        if content.startswith("FEEDBACK_AGENT_REQUEST"):
            return "FEEDBACK REQUEST"
        if content.startswith("FEEDBACK_AGENT_RESPONSE"):
            return "FEEDBACK RESPONSE"
        if content.startswith(HARNESS_EFFECTIVE_REVIEW_MARKER):
            return "HARNESS EFFECTIVE REVIEW"
        if content.startswith(VALIDATED_FEEDBACK_DECISION_MARKER):
            return "VALIDATED FEEDBACK DECISION"
        if content.startswith(HARNESS_RESPONSE_OMISSION_MARKER):
            return "HARNESS RESPONSE OMISSION"
        return turn.role.upper()

    def _turn_color(self, turn: Turn) -> str:
        if not self.color:
            return ""
        content = turn.content
        if content.startswith("IMPLEMENTATION_AGENT"):
            return "\033[36m"
        if content.startswith("FEEDBACK_AGENT"):
            return "\033[33m"
        if content.startswith(HARNESS_EFFECTIVE_REVIEW_MARKER):
            return "\033[2m"
        if content.startswith(VALIDATED_FEEDBACK_DECISION_MARKER):
            return "\033[2m"
        if content.startswith(HARNESS_RESPONSE_OMISSION_MARKER):
            return "\033[2m"
        if turn.role == "system":
            return "\033[2m"
        return ""
