from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import sys

from .bounds import clamp_text, estimate_tokens


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
        self.turns: list[Turn] = self._load_turns(path)
        if self.full_path and not self.full_path.exists() and self.turns:
            self._write_turns(self.full_path, self.turns)

    def _load_turns(self, path: Path) -> list[Turn]:
        turns: list[Turn] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    turns.append(Turn(role=item["role"], content=item["content"]))
        return turns

    def _append_turn_to_path(self, path: Path, turn: Turn) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")

    def _write_turns(self, path: Path, turns: list[Turn]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(asdict(t), ensure_ascii=False) + "\n" for t in turns),
            encoding="utf-8",
        )

    def append(self, role: str, content: str) -> None:
        turn = Turn(role=role, content=content)
        self.turns.append(turn)
        self._append_turn_to_path(self.path, turn)
        if self.full_path and self.full_path != self.path:
            self._append_turn_to_path(self.full_path, turn)
        if self.echo:
            self._print_turn(turn)

    def replace_last_turn(self, *, role: str, content_prefix: str, new_content: str) -> bool:
        """Replace the active-context copy of the latest turn when it is unsafe.

        The append-only full transcript remains the audit log. The compact active
        transcript is allowed to replace pathological content with a bounded note
        so later model calls do not inherit malformed or repetitive output.
        """
        if not self.turns:
            return False
        latest = self.turns[-1]
        if latest.role != role or not latest.content.startswith(content_prefix):
            return False
        self.turns[-1] = Turn(role=role, content=new_content)
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

    def messages(self, *, system_as_user: bool = False) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for turn in self.turns:
            if system_as_user and turn.role == "system":
                messages.append({"role": "user", "content": "TRANSCRIPT_SYSTEM_NOTE:\n" + turn.content})
            else:
                messages.append(asdict(turn))
        return messages

    def estimated_tokens(self) -> int:
        return max(1, sum(estimate_tokens(t.content) for t in self.turns))

    def replace_with_memory(self, memory: str, keep_recent_turns: int) -> None:
        raw_recent = self.turns[-keep_recent_turns:] if keep_recent_turns > 0 else []
        recent = [turn for turn in raw_recent if turn.role != "system"]
        memory_turn = Turn(
            role="system",
            content=(
                "Compacted durable memory from earlier turns. Preserve these decisions, "
                "constraints, and unresolved risks:\n\n" + memory
            ),
        )
        self.turns = [memory_turn, *recent]
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
        turns = self._load_turns(self.full_path) if full and self.full_path else self.turns
        lines: list[str] = ["# Agent Transcript", ""]
        for index, turn in enumerate(turns, start=1):
            lines.append(f"## {index}. {turn.role}")
            lines.append("")
            lines.append("```text")
            lines.append(turn.content)
            lines.append("```")
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

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
        return turn.role.upper()

    def _turn_color(self, turn: Turn) -> str:
        if not self.color:
            return ""
        content = turn.content
        if content.startswith("IMPLEMENTATION_AGENT"):
            return "\033[36m"
        if content.startswith("FEEDBACK_AGENT"):
            return "\033[33m"
        if turn.role == "system":
            return "\033[2m"
        return ""
