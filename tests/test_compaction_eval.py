from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from feedback_agent.compaction import (
    build_compaction_prompt,
    compaction_source_from_turns,
)
from feedback_agent.compaction_eval import (
    _run_production_case,
    _recover_exact_fenced_judgment,
    _semantic_judge_prompt,
    build_stress_corpus,
    extract_corpus,
    grade_memory,
    iter_corpus,
)
from feedback_agent.conversation import Turn


class ContextCompactionEvaluationTests(unittest.TestCase):
    def test_production_runner_exercises_real_staging_and_assembly(self) -> None:
        class CompactionClient:
            def __init__(self) -> None:
                self.cfg = SimpleNamespace(
                    name="test-compactor",
                    context_window=131072,
                    send_reasoning_budget=False,
                )
                self.last_response_finish_reason = "stop"
                self.last_response_usage = {}

            def chat_for_compaction(
                self,
                messages: list[dict[str, str]],
                *,
                max_tokens: int,
                reasoning_budget_tokens: int,
            ) -> str:
                return (
                    "PIVOTAL HISTORY\n"
                    "- A validated review left empty-record handling unresolved and required another check."
                )

        initial = "user: PROJECT DESIGN: Parser\n\nPreserve empty records."
        case = {
            "id": "production-case",
            "split": "heldout",
            "category": "test",
            "task_id": "parser",
            "initial_context": initial,
            "source_chars": 1000,
            "facts": [],
            "turns": [
                {"role": "user", "content": initial.removeprefix("user: ")},
                {"role": "assistant", "content": "Earlier unvalidated work."},
                {"role": "user", "content": "Latest review request."},
                {"role": "assistant", "content": "Latest unvalidated response."},
            ],
        }

        result = _run_production_case(
            case,
            CompactionClient(),
            profile_name="test",
            summary_max_tokens=2048,
            reasoning_budget_tokens=1024,
            critical_reasoning_budget_tokens=4096,
            model_repair_attempts=1,
        )

        self.assertEqual(result["error"], "")
        self.assertEqual(result["compaction_stage"], "conservative")
        self.assertTrue(result["effective_grade"]["initial_request_preserved"])
        self.assertTrue(result["post_compaction_fits_reserved_request"])

    def test_source_shaping_removes_repeated_request_and_bulky_file_content(self) -> None:
        turns = [
            Turn("user", "PROJECT DESIGN: Example\n\nBuild a checked artifact."),
            Turn(
                "assistant",
                "IMPLEMENTATION_AGENT_RESPONSE:\n"
                + json.dumps({
                    "plan_note": "Created the requested parser and retained one unresolved edge case.",
                    "files": [{"path": "parser.py", "content": "x" * 20_000}],
                    "commands": [],
                    "resolution_request": "none",
                }),
            ),
            Turn(
                "user",
                "VALIDATED_FEEDBACK_DECISION:\n"
                + json.dumps({
                    "phase": "STEP_REVIEW_PHASE",
                    "status": "needs_rework",
                    "needs_rework": True,
                    "summary": "The parser still mishandles empty records.",
                    "required_changes": ["Preserve empty records and rerun the parser test."],
                }),
            ),
        ]

        source = compaction_source_from_turns(turns)

        self.assertIn("initial project request omitted", source)
        self.assertIn("Unvalidated model response (claim only", source)
        self.assertIn("Created the requested parser", source)
        self.assertIn("parser.py", source)
        self.assertIn("mishandles empty records", source)
        self.assertNotIn("x" * 100, source)

    def test_prompt_assigns_clear_priority_and_states_deterministic_boundaries(self) -> None:
        prompt = build_compaction_prompt(
            initial_context="user: PROJECT DESIGN: Example\n\nBuild it.",
            source="user: validated failure evidence",
        )

        self.assertIn("1. Pivotal", prompt)
        self.assertIn("2. Contributory", prompt)
        self.assertIn("3. Noise", prompt)
        self.assertIn("separately preserves the original request", prompt)
        self.assertIn("reference only; it is pinned separately", prompt)
        self.assertIn("does not prove that the command ran or succeeded", prompt)
        self.assertIn("do not invent repairs, commands, facts, or decisions", prompt)
        self.assertIn("Do not infer a current phase or next action", prompt)
        self.assertNotIn("Preserve the newest structured workflow status exactly", prompt)

    def test_extract_freezes_first_event_and_strips_visible_reasoning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "benchmarks" / "run" / "harness" / "task-one" / ".agent_state"
            state.mkdir(parents=True)
            transcript = state / "conversation.full.jsonl"
            turns = [
                {"role": "system", "content": "HARNESS_SHARED_CONTEXT:\nRules"},
                {"role": "user", "content": "PROJECT DESIGN: Exact task\n\nPreserve alpha."},
                {"role": "user", "content": "IMPLEMENTATION_AGENT_REQUEST:\nPROBLEM_ANALYSIS_PHASE"},
                {
                    "role": "assistant",
                    "content": "IMPLEMENTATION_AGENT_RESPONSE:\n<think>scratch only</think>\n"
                    + json.dumps({"plan_note": "alpha remains pivotal", "files": [], "commands": []}),
                },
                {
                    "role": "system",
                    "content": "ACTIVE_CONTEXT_COMPACTED: conversation.jsonl was rewritten.",
                },
            ]
            transcript.write_text(
                "".join(json.dumps(turn) + "\n" for turn in turns),
                encoding="utf-8",
            )
            output = root / "corpus.jsonl.gz"

            summary = extract_corpus(
                root,
                output,
                case_count=10,
                development_count=1,
                seed=7,
            )
            cases = list(iter_corpus(output))

            self.assertEqual(summary["eligible_snapshots"], 1)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0]["snapshot_kind"], "exact-first-compaction-active-state")
            active_response = cases[0]["turns"][-1]["content"]
            self.assertIn("alpha remains pivotal", active_response)
            self.assertNotIn("scratch only", active_response)

    def test_extract_rejects_unreconstructable_pre_event_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "task-two" / ".agent_state"
            state.mkdir(parents=True)
            records = [
                {"role": "system", "content": "HARNESS_SHARED_CONTEXT:\nRules"},
                {"role": "user", "content": "PROJECT DESIGN: Exact task\n\nPreserve alpha."},
                {"role": "assistant", "content": "bad response"},
                {"role": "system", "content": "ACTIVE_CONTEXT_TURN_REPLACED: rewritten"},
                {"role": "user", "content": "retry"},
                {"role": "system", "content": "ACTIVE_CONTEXT_COMPACTED: rewritten"},
            ]
            (state / "conversation.full.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8",
            )
            output = root / "corpus.jsonl.gz"

            summary = extract_corpus(
                root,
                output,
                case_count=10,
                development_count=1,
                seed=7,
            )

            self.assertEqual(summary["selected_cases"], 0)
            self.assertEqual(summary["skipped"]["pre-event-active-replacement"], 1)
            with gzip.open(output, "rt", encoding="utf-8") as stream:
                self.assertEqual(stream.read(), "")

    def test_extract_can_exclude_stable_ids_from_prior_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "task-three" / ".agent_state"
            state.mkdir(parents=True)
            records = [
                {"role": "system", "content": "HARNESS_SHARED_CONTEXT:\nRules"},
                {"role": "user", "content": "PROJECT DESIGN: Exact task\n\nPreserve alpha."},
                {
                    "role": "user",
                    "content": "IMPLEMENTATION_AGENT_REQUEST:\nPROBLEM_ANALYSIS_PHASE",
                },
                {
                    "role": "assistant",
                    "content": "IMPLEMENTATION_AGENT_RESPONSE:\n"
                    + json.dumps({"plan_note": "Useful accepted history.", "files": [], "commands": []}),
                },
                {"role": "system", "content": "ACTIVE_CONTEXT_COMPACTED: rewritten"},
            ]
            (state / "conversation.full.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8",
            )
            first = root / "first.jsonl.gz"
            second = root / "second.jsonl.gz"
            first_summary = extract_corpus(root, first, case_count=1, development_count=0, seed=7)
            excluded_id = next(iter_corpus(first))["id"]

            self.assertEqual(first_summary["development_cases"], 0)
            self.assertEqual(first_summary["heldout_cases"], 1)

            summary = extract_corpus(
                root,
                second,
                case_count=1,
                development_count=0,
                seed=8,
                excluded_ids={excluded_id},
            )

            self.assertEqual(summary["selected_cases"], 0)
            self.assertEqual(summary["excluded_ids"], 1)
            self.assertEqual(summary["skipped"]["excluded-case"], 1)

    def test_stress_corpus_derives_all_boundary_kinds_from_real_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl.gz"
            base = {
                "version": 1,
                "id": "base-case",
                "split": "heldout",
                "snapshot_kind": "exact-first-compaction-active-state",
                "source_path": "/tmp/base",
                "event_line": 10,
                "task_id": "base-task",
                "category": "base",
                "origin_model": "test",
                "transcript_sha256": "abc",
                "turn_count": 4,
                "estimated_tokens": 100,
                "source_chars": 400,
                "initial_context": "user: PROJECT DESIGN: Base\n\nPreserve alpha.",
                "facts": [],
                "turns": [
                    {"role": "user", "content": "PROJECT DESIGN: Base\n\nPreserve alpha."},
                    {"role": "user", "content": "IMPLEMENTATION_AGENT_REQUEST:\nPROBLEM_ANALYSIS_PHASE"},
                    {"role": "assistant", "content": "IMPLEMENTATION_AGENT_RESPONSE:\n{}"},
                    {"role": "user", "content": "latest evidence"},
                ],
            }
            with gzip.open(source, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(base) + "\n")
            output = root / "stress.jsonl.gz"

            summary = build_stress_corpus(source, output, case_count=12, seed=7)
            cases = list(iter_corpus(output))

            self.assertEqual(summary["selected_cases"], 12)
            self.assertEqual(set(summary["stress_kinds"].values()), {2})
            self.assertEqual(len({item["id"] for item in cases}), 12)
            longest = max(
                (item for item in cases if item["stress_kind"].startswith("long-request")),
                key=lambda item: len(item["initial_context"]),
            )
            self.assertGreater(len(longest["initial_context"]), 140000)
            repeated = next(item for item in cases if item["stress_kind"] == "repeated-compaction")
            self.assertTrue(any(
                turn["content"].startswith("Compacted context from earlier turns")
                for turn in repeated["turns"]
            ))

    def test_grade_uses_weighted_fact_tokens_without_exact_sentence_matching(self) -> None:
        case = {
            "initial_context": "user: PROJECT DESIGN: Parser\n\nPreserve empty records.",
            "source_chars": 5000,
            "facts": [
                {
                    "id": "pivotal-1",
                    "priority": "pivotal",
                    "text": "Validation failed because empty records were discarded by parser.py.",
                },
                {
                    "id": "medium-1",
                    "priority": "contributory",
                    "text": "The JSONL approach remains dependency free.",
                },
            ],
        }
        candidate = (
            "PIVOTAL HISTORY\n- parser.py discarded empty records, causing validation failure.\n"
            "CONTRIBUTORY HISTORY\n- Keep the dependency-free JSONL approach."
        )
        assembled = case["initial_context"] + "\n" + candidate

        grade = grade_memory(candidate, assembled, case, summary_max_tokens=1024)

        self.assertTrue(grade["pass"])
        self.assertGreater(grade["pivotal_recall"], 0.5)
        self.assertGreater(grade["contributory_recall"], 0.5)

    def test_judge_recovery_accepts_only_one_exact_valid_json_fence(self) -> None:
        payload = {
            "decision": "pass",
            "score": 80,
            "pivotal_retention": 4,
            "contributory_compression": 3,
            "provenance_correctness": 4,
            "noise_control": 3,
            "material_omissions": [],
            "contradictions": [],
            "unnecessary_content": [],
            "summary": "Material history is preserved.",
        }

        self.assertEqual(
            _recover_exact_fenced_judgment("```json\n" + json.dumps(payload) + "\n```"),
            payload,
        )
        self.assertEqual(
            _recover_exact_fenced_judgment("narration\n```json\n" + json.dumps(payload) + "\n```"),
            {},
        )
        self.assertEqual(_recover_exact_fenced_judgment("```json\n{}\n```"), {})

    def test_semantic_judge_does_not_demand_separately_preserved_state(self) -> None:
        prompt = _semantic_judge_prompt(
            source="assistant: unvalidated work remains",
            control_state="step S1 remains pending",
            candidate="OPEN RISKS / NEXT ACTIONS\n- Validate S1 before accepting it.",
        )

        self.assertIn("complete labelled active context", prompt)
        self.assertIn("do not require their facts to be duplicated", prompt)
        self.assertIn("proof absent from the evicted history", prompt)
        self.assertIn("not evidence that a command ran or succeeded", prompt)


if __name__ == "__main__":
    unittest.main()
