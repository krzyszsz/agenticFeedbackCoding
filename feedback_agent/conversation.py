from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import sys


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

    def __init__(self, path: Path, *, echo: bool = False):
        self.path = path
        self.echo = echo
        self.turns: list[Turn] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    self.turns.append(Turn(role=item["role"], content=item["content"]))

    def append(self, role: str, content: str) -> None:
        turn = Turn(role=role, content=content)
        self.turns.append(turn)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")
        if self.echo:
            self._print_turn(turn)

    def messages(self, *, system_as_user: bool = False) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for turn in self.turns:
            if system_as_user and turn.role == "system":
                messages.append({"role": "user", "content": "TRANSCRIPT_SYSTEM_NOTE:\n" + turn.content})
            else:
                messages.append(asdict(turn))
        return messages

    def estimated_tokens(self) -> int:
        return max(1, sum(len(t.content) for t in self.turns) // 4)

    def replace_with_memory(self, memory: str, keep_recent_turns: int) -> None:
        recent = self.turns[-keep_recent_turns:] if keep_recent_turns > 0 else []
        self.turns = [
            Turn(
                role="system",
                content=(
                    "Compacted durable memory from earlier turns. Preserve these decisions, "
                    "constraints, and unresolved risks:\n\n" + memory
                ),
            ),
            *recent,
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "".join(json.dumps(asdict(t), ensure_ascii=False) + "\n" for t in self.turns),
            encoding="utf-8",
        )
        if self.echo:
            self._print_turn(self.turns[0])

    def write_markdown(self, path: Path) -> None:
        """Export the JSONL transcript in a form people can read after the run."""
        lines: list[str] = ["# Agent Transcript", ""]
        for index, turn in enumerate(self.turns, start=1):
            lines.append(f"## {index}. {turn.role}")
            lines.append("")
            lines.append("```text")
            lines.append(turn.content)
            lines.append("```")
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    def _print_turn(self, turn: Turn) -> None:
        print(f"\n===== {turn.role.upper()} =====", flush=True)
        print(turn.content, flush=True)
        print("===== END TURN =====\n", flush=True)
        sys.stdout.flush()
