from __future__ import annotations

import contextlib
import functools
import http.server
import io
import json
from pathlib import Path
import socketserver
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
from typing import Any

from feedback_agent.agent import FeedbackLoopAgent
from feedback_agent.config import load_config
from feedback_agent.conversation import Conversation
from feedback_agent.llm import ModelRequestRetrier
from feedback_agent.workspace import extract_json_object, run_commands, write_plan_doc


class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


class ScriptedClient:
    """Tiny test double for model calls.

    Production code always talks to an OpenAI-compatible endpoint. Unit tests use
    this in-memory client only to make phase/review behavior deterministic enough
    to test without requiring a local 27B model for every assertion.
    """

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "temperature": temperature})
        if self.responses:
            return self.responses.pop(0)
        return json.dumps({
            "status": "resolved",
            "needs_rework": False,
            "summary": "Scripted review accepted the evidence.",
            "required_changes": [],
            "verification_evidence": ["reviewer-owned validation evidence inspected"],
        })


@contextlib.contextmanager
def local_http_server(root: Path):
    handler = functools.partial(QuietHTTPRequestHandler, directory=str(root))
    with ReusableTCPServer(("127.0.0.1", 0), handler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{httpd.server_address[1]}"
        finally:
            httpd.shutdown()
            thread.join(timeout=5)


def write_config(root: Path, workspace: Path, title: str, prompt: str) -> Path:
    config_path = root / "config.json"
    config_path.write_text(json.dumps({
        "implementation_model": {
            "name": "scripted-test-client",
            "base_url": "http://127.0.0.1:1/v1",
            "api_key": "not-needed",
            "model": "local-gguf",
            "context_window": 100000,
            "max_tokens": 512,
            "temperature": 0.1,
            "request_timeout_seconds": 21600,
        },
        "feedback_model": None,
        "mcp_tools": {"terminal": True, "web_scraping": True, "web_interaction": True},
        "runtime": {
            "docker_isolation": False,
            "docker_image": "agentic-feedback-coding:local",
            "workspace": str(workspace),
            "command_timeout_seconds": 30,
            "max_command_timeout_seconds": 300,
            "print_transcript": False,
        },
        "context_compaction": {
            "enabled": True,
            "threshold_ratio": 0.8,
            "keep_recent_turns": 4,
            "summary_max_tokens": 256,
        },
        "loop": {"max_iterations": 3},
        "phases": {
            "requirements_refinement": {"max_iterations": 2},
            "plan_validation": {"max_iterations": 2},
            "implementation": {"max_iterations": 3},
        },
        "resolution_policy": {
            "max_same_error_repeats": 2,
            "allow_requirement_dilution": True,
            "allow_skip_with_note": True,
            "stop_on_cannot_resolve": False,
        },
        "quality_policy": {
            "assume_code_quality_when_unspecified": True,
            "require_research_and_structure_step": True,
        },
        "review_policy": {
            "hard_pushback_iterations": 3,
            "compromise_iterations": 4,
            "final_review_iterations": 1,
        },
        "web_research": {
            "enabled": True,
            "max_search_results": 2,
            "max_pages": 2,
            "timeout_seconds": 5,
            "max_page_bytes": 200000,
            "excerpt_chars": 2000,
            "user_agent": "agenticFeedbackCoding-tests/0.1",
        },
        "git_policy": {
            "enabled": True,
            "commit_completed_steps": True,
            "require_step_diff": True,
            "leave_final_changes_uncommitted": False,
            "final_reset_mode": "soft",
            "commit_user_name": "agenticFeedbackCoding-tests",
            "commit_user_email": "agentic-feedback-tests@example.local",
        },
        "project_design": {"title": title, "prompt": prompt},
    }), encoding="utf-8")
    return config_path


def load_test_agent(
    root: Path,
    workspace: Path,
    *,
    title: str = "checked artifact",
    prompt: str = "Build a small checked artifact.",
    feedback_responses: list[str] | None = None,
    implementation_responses: list[str] | None = None,
) -> FeedbackLoopAgent:
    cfg = load_config(write_config(root, workspace, title, prompt), repo_root=root)
    return FeedbackLoopAgent(
        cfg,
        implementation_client=ScriptedClient(implementation_responses),
        feedback_client=ScriptedClient(feedback_responses),
    )


def base_requirements(summary: str = "Checked artifact") -> dict[str, Any]:
    return {
        "project_summary": summary,
        "refined_requirements": ["Feedback must verify evidence independently."],
        "assumptions": [],
        "planning_confirmation": {
            "is_feasible": True,
            "is_clear": True,
            "is_verifiable": True,
            "verification_strategy": "Run reviewer-owned validation commands for each plan step.",
            "remaining_risks": [],
        },
    }


class FeedbackLoopAgentTests(unittest.TestCase):
    def test_model_request_retrier_returns_without_retry_on_success(self) -> None:
        calls: list[int] = []
        output = io.StringIO()
        retrier = ModelRequestRetrier(attempts=20, sleep_seconds=30, sleep=lambda _seconds: None, stream=output)

        result = retrier.run(lambda: calls.append(1) or "ok")

        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 1)
        self.assertEqual(output.getvalue(), "")

    def test_model_request_retrier_retries_transient_failure(self) -> None:
        calls: list[int] = []
        sleeps: list[float] = []
        output = io.StringIO()
        retrier = ModelRequestRetrier(attempts=3, sleep_seconds=30, sleep=sleeps.append, stream=output)

        def flaky() -> str:
            calls.append(1)
            if len(calls) < 3:
                raise urllib.error.URLError("temporary model server hiccup")
            return "ok"

        result = retrier.run(flaky)

        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [30, 30])
        self.assertIn("attempt 1/3", output.getvalue())
        self.assertIn("1 attempts left", output.getvalue())

    def test_model_request_retrier_reports_exhaustion(self) -> None:
        output = io.StringIO()
        retrier = ModelRequestRetrier(attempts=2, sleep_seconds=0, sleep=lambda _seconds: None, stream=output)

        with self.assertRaisesRegex(RuntimeError, "failed after 2 attempts"):
            retrier.run(lambda: (_ for _ in ()).throw(TimeoutError("slow model server")))

        self.assertIn("attempt 1/2", output.getvalue())

    def test_json_extractor_recovers_first_balanced_object_from_noisy_output(self) -> None:
        payload = extract_json_object(
            "<think>not json { nope }</think>\n"
            "{\"status\":\"resolved\",\"needs_rework\":false}\n"
            "trailing duplicate-ish text {broken"
        )

        self.assertEqual(payload["status"], "resolved")
        self.assertFalse(payload["needs_rework"])

    def test_command_timeout_can_be_overridden_per_command_and_clamped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = run_commands(
                root,
                [{"cmd": ["python", "-c", "print('long command shape accepted')"], "timeout_seconds": 999}],
                timeout_seconds=1,
                max_timeout_seconds=7,
            )

            self.assertEqual(results[0]["returncode"], 0)
            self.assertEqual(results[0]["timeout_seconds"], 7)
            self.assertIn("long command shape accepted", results[0]["stdout"])

    def test_command_output_is_bounded_at_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = run_commands(
                root,
                [[
                    "python",
                    "-c",
                    "import sys; print('A' * 5000); print('B' * 5000, file=sys.stderr)",
                ]],
                timeout_seconds=30,
                max_timeout_seconds=300,
                output_limit_chars=512,
            )

            self.assertEqual(results[0]["returncode"], 0)
            self.assertTrue(results[0]["stdout_truncated"])
            self.assertTrue(results[0]["stderr_truncated"])
            self.assertLess(len(results[0]["stdout"]), 1300)
            self.assertLess(len(results[0]["stderr"]), 1300)
            self.assertIn("truncated", results[0]["stdout"])

    def test_command_timeout_kills_child_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = run_commands(
                root,
                [[
                    "bash",
                    "-c",
                    "python -c 'import time; time.sleep(60)' & echo $! > child.pid; wait",
                ]],
                timeout_seconds=1,
                max_timeout_seconds=1,
            )

            self.assertTrue(results[0]["timed_out"])
            self.assertEqual(results[0]["returncode"], 124)
            child_pid = (root / "child.pid").read_text(encoding="utf-8").strip()
            time.sleep(0.2)
            ps = subprocess.run(["ps", "-p", child_pid], capture_output=True, text=True)
            self.assertNotEqual(ps.returncode, 0)

    def test_compaction_keeps_append_only_full_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "conversation.jsonl"
            full = root / "conversation.full.jsonl"
            conversation = Conversation(active, full_path=full)

            conversation.append("user", "first request")
            conversation.append("assistant", "first response")
            conversation.append("user", "second request")
            conversation.replace_with_memory("compressed first exchange", keep_recent_turns=1)

            active_text = active.read_text(encoding="utf-8")
            full_text = full.read_text(encoding="utf-8")
            self.assertNotIn("first request", active_text)
            self.assertIn("compressed first exchange", active_text)
            self.assertIn("first request", full_text)
            self.assertIn("ACTIVE_CONTEXT_COMPACTED", full_text)

    def test_live_echo_can_be_bounded_without_truncating_saved_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "conversation.jsonl"
            conversation = Conversation(active, echo=False, echo_limit_chars=32)

            long_turn = "x" * 200
            conversation.append("user", long_turn)

            saved = active.read_text(encoding="utf-8")
            self.assertIn(long_turn, saved)

    def test_load_config_without_repo_root_uses_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_dir = root / "nested"
            workspace = cfg_dir / "work"
            cfg_dir.mkdir()
            cfg_path = write_config(cfg_dir, Path("work"), "relative workspace", "Build anything.")

            cfg = load_config(cfg_path)

            self.assertEqual(cfg.runtime.workspace, workspace.resolve())

    def test_existing_project_can_use_agent_owned_state_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "existing-project"
            workspace.mkdir()
            (workspace / "PLAN.md").write_text("# Project-owned plan\n\nDo not overwrite.\n", encoding="utf-8")
            config_path = write_config(root, workspace, "existing project", "Fix a bug in this existing project.")
            data = json.loads(config_path.read_text(encoding="utf-8"))
            data["runtime"]["plan_file"] = "AGENT_PLAN.md"
            data["runtime"]["requirements_file"] = "AGENT_REQUIREMENTS.md"
            data["runtime"]["research_file"] = "AGENT_RESEARCH.md"
            config_path.write_text(json.dumps(data), encoding="utf-8")
            cfg = load_config(config_path, repo_root=root)
            agent = FeedbackLoopAgent(
                cfg,
                implementation_client=ScriptedClient(),
                feedback_client=ScriptedClient(),
            )

            agent.initialize()
            agent._web_research_phase()
            agent.requirements = base_requirements("Existing project")
            agent._write_requirements_doc()
            agent._write_plan_doc()

            self.assertEqual((workspace / "PLAN.md").read_text(encoding="utf-8"), "# Project-owned plan\n\nDo not overwrite.\n")
            self.assertTrue((workspace / "AGENT_PLAN.md").exists())
            self.assertTrue((workspace / "AGENT_REQUIREMENTS.md").exists())
            self.assertTrue((workspace / "AGENT_RESEARCH.md").exists())

    def test_expected_nonzero_returncode_is_successful_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = run_commands(
                root,
                [{"cmd": ["python", "-c", "import sys; print('usage: app text', file=sys.stderr); sys.exit(2)"], "expected_returncode": 2}],
                timeout_seconds=30,
                max_timeout_seconds=300,
            )

            self.assertEqual(results[0]["returncode"], 2)
            self.assertEqual(results[0]["expected_returncode"], 2)
            self.assertTrue(results[0]["returncode_matches_expected"])
            self.assertIn("usage", results[0]["stderr"])

    def test_server_only_validation_command_is_rejected_without_hanging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = run_commands(root, [["python", "-m", "http.server", "8080"]], 30, 300)

            self.assertEqual(results[0]["returncode"], 125)
            self.assertFalse(results[0]["timed_out"])
            self.assertTrue(results[0]["skipped_as_non_verifying_server"])
            self.assertIn("does not assert behavior", results[0]["stderr"])

    def test_agent_commands_cannot_mutate_git_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            results = run_commands(
                root,
                [
                    ["git", "status", "--short"],
                    ["git", "add", "anything.txt"],
                    ["git", "commit", "-m", "agent-owned commit"],
                ],
                30,
                300,
            )

            self.assertEqual(results[0]["returncode"], 0)
            self.assertEqual(results[1]["returncode"], 126)
            self.assertEqual(results[2]["returncode"], 126)
            self.assertTrue(results[1]["blocked_git_mutation"])
            self.assertIn("harness owns", results[1]["stderr"])

    def test_shared_transcript_keeps_implementation_and_feedback_context_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            impl = ScriptedClient([json.dumps({"note": "implementation turn"})])
            reviewer = ScriptedClient()
            cfg = load_config(write_config(root, workspace, "chat continuity", "Build anything."), repo_root=root)
            agent = FeedbackLoopAgent(cfg, implementation_client=impl, feedback_client=reviewer)
            agent.initialize()

            agent._implementation_chat("Create the first file.")
            agent._feedback_chat("Review the first file.")

            feedback_context = "\n".join(message["content"] for message in reviewer.calls[0]["messages"])
            transcript = (workspace / ".agent_state" / "conversation.jsonl").read_text(encoding="utf-8")
            self.assertIn("IMPLEMENTATION_AGENT_REQUEST", feedback_context)
            self.assertIn("IMPLEMENTATION_AGENT_RESPONSE", feedback_context)
            self.assertIn("FEEDBACK_AGENT_REQUEST", transcript)
            self.assertIn("FEEDBACK_AGENT_RESPONSE", transcript)

    def test_plan_validation_enforces_step_limit_and_server_only_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                title="strict plan",
                prompt=(
                    "Build a small browser app. Hard limit: group the implementation plan into "
                    "at most 1 independently verifiable steps."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Strict plan")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Research required patterns and plan project structure",
                    "description": "Research and choose structure.",
                    "depends_on": [],
                    "acceptance_criteria": ["ARCHITECTURE.md records Structure and Plan order"],
                    "validation_commands": [["test", "-f", "ARCHITECTURE.md"]],
                    "status": "pending",
                },
                {
                    "id": "S2",
                    "title": "Browser UI",
                    "description": "Render browser UI.",
                    "depends_on": ["S1"],
                    "acceptance_criteria": ["UI renders"],
                    "validation_commands": [["python", "-m", "http.server", "8080"]],
                    "status": "pending",
                },
            ]

            review = agent._plan_validation_review(1)

            self.assertEqual(review["status"], "needs_plan_change")
            changes = "\n".join(review["required_changes"])
            self.assertIn("at most 1", changes)
            self.assertIn("HTTP server", changes)

    def test_soft_step_limit_does_not_block_verifiable_quality_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                title="soft plan",
                prompt="Build a small checked app. Prefer at most 1 independently verifiable steps.",
            )
            agent.initialize()
            agent.requirements = base_requirements("Soft plan")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Research required patterns and plan project structure",
                    "description": "Research patterns, choose structure, and record plan order.",
                    "depends_on": [],
                    "acceptance_criteria": ["Structure is recorded"],
                    "validation_commands": [["python", "-c", "print('structure ok')"]],
                    "status": "pending",
                },
                {
                    "id": "S2",
                    "title": "Implement checked artifact",
                    "description": "Create the artifact.",
                    "depends_on": ["S1"],
                    "acceptance_criteria": ["Artifact is checked"],
                    "validation_commands": [["python", "-c", "print('artifact ok')"]],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            self.assertNotIn("at most 1", "\n".join(findings))

    def test_feedback_review_runs_reviewer_owned_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements()
            step = {
                "id": "T1",
                "title": "Create checked artifact",
                "description": "Write ok.txt and validate it.",
                "depends_on": [],
                "acceptance_criteria": ["ok.txt exists and contains pass"],
                "validation_commands": [[
                    "python",
                    "-c",
                    "from pathlib import Path; assert Path('ok.txt').read_text().strip() == 'pass'; print('review evidence ok')",
                ]],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])
            (workspace / "ok.txt").write_text("pass\n", encoding="utf-8")

            review = agent._step_review_pass(
                step,
                1,
                {"written": ["ok.txt"], "commands": [], "raw": {"test_evidence": ["ok.txt validation"]}},
                "hard_pushback",
            )

            self.assertEqual(review["status"], "resolved")
            self.assertEqual(review["deterministic_evidence_findings"], [])
            evidence = review["feedback_tool_evidence"]
            self.assertEqual(evidence["validation_results"][0]["returncode"], 0)
            self.assertIn("ok.txt", {item["path"] for item in evidence["workspace_files"]})

    def test_feedback_review_rejects_failing_reviewer_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Failing reviewer validation")
            step = {
                "id": "T1",
                "title": "Create checked artifact",
                "description": "Validate a missing artifact.",
                "depends_on": [],
                "acceptance_criteria": ["missing.txt exists"],
                "validation_commands": [[
                    "python",
                    "-c",
                    "from pathlib import Path; assert Path('missing.txt').exists(), 'missing expected artifact'",
                ]],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])

            review = agent._step_review_pass(
                step,
                1,
                {"written": [], "commands": [], "raw": {"test_evidence": []}},
                "hard_pushback",
            )

            self.assertEqual(review["status"], "needs_rework")
            self.assertIn("Feedback validation command returned 1 but expected 0", "\n".join(review["required_changes"]))

    def test_feedback_review_accepts_expected_negative_path_returncode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Expected negative path")
            step = {
                "id": "T1",
                "title": "Validate missing argument behavior",
                "description": "Command should exit 2 and print usage.",
                "depends_on": [],
                "acceptance_criteria": ["CLI reports usage for missing required argument"],
                "validation_commands": [{
                    "cmd": ["python", "-c", "import sys; print('usage: app text', file=sys.stderr); sys.exit(2)"],
                    "expected_returncode": 2,
                }],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])
            (workspace / "cli.py").write_text("print('placeholder')\n", encoding="utf-8")

            review = agent._step_review_pass(
                step,
                1,
                {"written": ["cli.py"], "commands": [], "raw": {"test_evidence": ["negative path checked"]}},
                "hard_pushback",
            )

            self.assertEqual(review["status"], "resolved")
            self.assertEqual(review["deterministic_evidence_findings"], [])
            result = review["feedback_tool_evidence"]["validation_results"][0]
            self.assertEqual(result["returncode"], 2)
            self.assertTrue(result["returncode_matches_expected"])

    def test_plan_validation_rejects_unwrapped_expected_failure_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Expected exception plan")
            agent.plan_steps = [{
                "id": "T1",
                "title": "Implement empty-input behavior",
                "description": "mean raises ValueError on empty input.",
                "depends_on": [],
                "acceptance_criteria": ["mean raises ValueError on empty iterable"],
                "validation_commands": [["python", "-c", "from arithmetic_box import mean; mean([])"]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            self.assertIn("expected failure path", "\n".join(findings))

    def test_feedback_review_routes_unwrapped_expected_failure_to_plan_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Expected exception review")
            step = {
                "id": "T1",
                "title": "Implement empty-input behavior",
                "description": "mean raises ValueError on empty input.",
                "depends_on": [],
                "acceptance_criteria": ["mean raises ValueError on empty iterable"],
                "validation_commands": [["python", "-c", "from arithmetic_box import mean; mean([])"]],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])
            (workspace / "arithmetic_box.py").write_text(
                "def mean(values):\n"
                "    values = list(values)\n"
                "    if not values:\n"
                "        raise ValueError('empty')\n"
                "    return sum(values) / len(values)\n",
                encoding="utf-8",
            )

            review = agent._step_review_pass(
                step,
                1,
                {"written": ["arithmetic_box.py"], "commands": [], "raw": {"test_evidence": ["negative path checked"]}},
                "hard_pushback",
            )

            self.assertEqual(review["status"], "needs_plan_change")
            self.assertIn("expected failure path", "\n".join(review["required_changes"]))

    def test_review_transcript_payload_drops_full_tool_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            review = {
                "status": "needs_rework",
                "needs_rework": True,
                "summary": "Large evidence review.",
                "required_changes": ["Please fix the generated artifact."],
                "feedback_tool_evidence": {
                    "kind": "step_feedback_tools",
                    "step_id": "T1",
                    "workspace_files": [{"path": "huge.txt", "content": "x" * 100000, "size": 100000, "truncated": True}],
                    "validation_results": [{
                        "command": ["python", "check.py"],
                        "returncode": 1,
                        "expected_returncode": 0,
                        "returncode_matches_expected": False,
                        "timed_out": False,
                        "stdout": "y" * 100000,
                        "stderr": "z" * 100000,
                    }],
                    "git": {"enabled": True, "diff": "d" * 100000, "status_short": " M huge.txt"},
                },
            }

            compact = agent._compact_review_for_transcript(review)
            encoded = json.dumps(compact)

            self.assertNotIn("feedback_tool_evidence", compact)
            self.assertIn("feedback_tool_evidence_summary", compact)
            self.assertLess(len(encoded), agent.config.context_compaction.transcript_review_max_chars)
            self.assertNotIn("x" * 5000, encoded)

    def test_feedback_review_routes_malformed_validation_command_to_plan_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Malformed validation")
            step = {
                "id": "T1",
                "title": "Create checked artifact",
                "description": "Validate with a malformed python -c command.",
                "depends_on": [],
                "acceptance_criteria": ["artifact exists"],
                "validation_commands": [["python", "-c", "assert True]"]],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])
            (workspace / "artifact.txt").write_text("present\n", encoding="utf-8")

            review = agent._step_review_pass(
                step,
                1,
                {"written": ["artifact.txt"], "commands": [], "raw": {"test_evidence": []}},
                "hard_pushback",
            )

            self.assertEqual(review["status"], "needs_plan_change")
            self.assertIn("Plan validation command appears malformed", "\n".join(review["required_changes"]))

    def test_browser_guidance_prefers_python_playwright_without_node_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            guidance = agent._browser_validation_guidance()

            self.assertIn("sync_playwright", guidance)
            self.assertIn("assume there is no", guidance)
            self.assertIn("npx", guidance)

    def test_malformed_feedback_repair_becomes_actionable_rework_instead_of_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=["<|channel>thought\n```json\n{\"status\":\"needs_rework\",\"required_changes\":["],
            )
            agent.initialize()

            review = agent._extract_json_or_retry(
                "not valid json",
                phase="STEP_REVIEW_PHASE",
                contract='{"status":"resolved|needs_rework","required_changes":["specific change"]}',
                feedback=True,
            )

            self.assertEqual(review["status"], "needs_rework")
            self.assertIn("smaller directly verifiable change", "\n".join(review["required_changes"]))
            self.assertIn("parse_error", review)

    def test_malformed_implementation_repair_becomes_noop_payload_instead_of_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                implementation_responses=["```json\n{\"files\":["],
            )
            agent.initialize()

            payload = agent._extract_json_or_retry(
                "not valid json",
                phase="IMPLEMENTATION_PHASE",
                contract='{"files":[],"commands":[]}',
            )

            self.assertEqual(payload["files"], [])
            self.assertEqual(payload["commands"], [])
            self.assertIn("parse_error", payload)

    def test_git_diff_gate_rejects_no_change_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("No-change attempt")
            step = {
                "id": "T1",
                "title": "Create marker file",
                "description": "Write marker.txt.",
                "depends_on": [],
                "acceptance_criteria": ["marker.txt exists"],
                "validation_commands": [],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])

            review = agent._step_review_pass(
                step,
                1,
                {"written": [], "commands": [], "raw": {"test_evidence": []}},
                "hard_pushback",
            )

            self.assertEqual(review["status"], "needs_rework")
            self.assertIn("Git working tree has no implementation changes", "\n".join(review["required_changes"]))

    def test_git_diff_gate_allows_validation_only_step_with_passing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Validation-only checkpoint")
            step = {
                "id": "T2",
                "title": "Full integration validation",
                "description": "Run all tests together and verify the project.",
                "depends_on": [],
                "acceptance_criteria": ["All tests pass in a single run", "CLI works end-to-end"],
                "validation_commands": [["python", "-c", "print('integration ok')"]],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])

            review = agent._step_review_pass(
                step,
                1,
                {"written": [], "commands": [], "raw": {"test_evidence": ["integration validation"]}},
                "hard_pushback",
            )

            self.assertEqual(review["status"], "resolved")
            self.assertEqual(review["deterministic_evidence_findings"], [])
            self.assertEqual(review["feedback_tool_evidence"]["validation_results"][0]["returncode"], 0)

    def test_git_diff_gate_allows_explicit_already_implemented_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Previously implemented artifact")
            (workspace / "marker.txt").write_text("done\n", encoding="utf-8")
            subprocess.run(["git", "add", "marker.txt"], cwd=workspace, check=True)
            subprocess.run(["git", "commit", "-m", "Add marker before current step"], cwd=workspace, check=True, stdout=subprocess.DEVNULL)
            step = {
                "id": "T3",
                "title": "Add marker file",
                "description": "Create marker.txt.",
                "depends_on": [],
                "acceptance_criteria": ["marker.txt exists"],
                "validation_commands": [["test", "-f", "marker.txt"]],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])

            review = agent._step_review_pass(
                step,
                1,
                {
                    "written": [],
                    "commands": [],
                    "raw": {
                        "plan_note": "marker.txt was already implemented in an earlier accepted step; validating it now.",
                        "test_evidence": ["test -f marker.txt passed"],
                    },
                },
                "hard_pushback",
            )

            self.assertEqual(review["status"], "resolved")
            self.assertEqual(review["deterministic_evidence_findings"], [])

    def test_final_review_reruns_plan_validation_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Final review evidence")
            step = {
                "id": "T1",
                "title": "Create checked artifact",
                "description": "Write final.txt and validate it.",
                "depends_on": [],
                "acceptance_criteria": ["final.txt exists"],
                "validation_commands": [[
                    "python",
                    "-c",
                    "from pathlib import Path; assert Path('final.txt').read_text().strip() == 'done'; print('final evidence ok')",
                ]],
                "status": "resolved",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])
            (workspace / "final.txt").write_text("done\n", encoding="utf-8")

            review = agent._final_project_review(
                1,
                [{"step_id": "T1", "status": "resolved", "attempts": [{"implementation": {"commands": []}}]}],
            )

            self.assertEqual(review["status"], "resolved")
            self.assertEqual(review["deterministic_evidence_findings"], [])
            validations = review["feedback_tool_evidence"]["step_validations"]
            self.assertEqual(validations[0]["validation_results"][0]["returncode"], 0)

    def test_web_research_phase_records_local_source_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = root / "site"
            site.mkdir()
            (site / "research.html").write_text(
                "<html><head><title>Harness Research Page</title></head>"
                "<body><main>CITATION_MARKER_ALPHA: use an adapter boundary, "
                "write deterministic tests, and cite this source in architecture notes.</main></body></html>",
                encoding="utf-8",
            )
            with local_http_server(site) as base_url:
                workspace = root / "workspace"
                url = f"{base_url}/research.html"
                agent = load_test_agent(
                    root,
                    workspace,
                    title="researched artifact",
                    prompt=f"Research this source before planning and building a small artifact: {url}",
                )
                agent.initialize()
                result = agent._web_research_phase()

            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["requested"])
            self.assertEqual(result["targets"][0]["status"], "ok")
            research_doc = (workspace / "RESEARCH.md").read_text(encoding="utf-8")
            transcript = (workspace / ".agent_state" / "conversation.jsonl").read_text(encoding="utf-8")
            self.assertIn(url, research_doc)
            self.assertIn("CITATION_MARKER_ALPHA", research_doc)
            self.assertIn("WEB_RESEARCH_TOOL_RESULT", transcript)

    def test_research_usage_gate_rejects_ignored_source_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, prompt="Research this source: http://example.test/research")
            agent.initialize()
            agent.web_research_result = {
                "status": "completed",
                "requested": True,
                "targets": [{
                    "url": "http://example.test/research",
                    "status": "ok",
                    "title": "Ignored Research",
                    "excerpt": "Use this evidence in architecture notes.",
                }],
            }
            agent.requirements = base_requirements("Research must be used")
            step = {
                "id": "S1",
                "title": "Research required patterns and plan project structure",
                "description": "Record structure and source usage.",
                "depends_on": [],
                "acceptance_criteria": ["ARCHITECTURE.md has Structure and Plan order"],
                "validation_commands": [[
                    "python",
                    "-c",
                    "from pathlib import Path; text=Path('ARCHITECTURE.md').read_text(); assert 'Structure' in text and 'Plan order' in text; print('architecture plan ok')",
                ]],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])
            (workspace / "ARCHITECTURE.md").write_text(
                "# Architecture Notes\n\n## Structure\n\n- Small module.\n\n## Plan order\n\nThe order is unchanged.\n",
                encoding="utf-8",
            )

            review = agent._step_review_pass(
                step,
                1,
                {"written": ["ARCHITECTURE.md"], "commands": [], "raw": {}},
                "hard_pushback",
            )

            self.assertEqual(review["status"], "needs_rework")
            self.assertIn("researched source URL", "\n".join(review["required_changes"]))

    def test_git_commits_accepted_step_and_can_leave_final_changes_uncommitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            config_path = write_config(root, workspace, "git policy", "Build a small checked artifact.")
            data = json.loads(config_path.read_text(encoding="utf-8"))
            data["git_policy"]["leave_final_changes_uncommitted"] = True
            config_path.write_text(json.dumps(data), encoding="utf-8")
            cfg = load_config(config_path, repo_root=root)
            agent = FeedbackLoopAgent(
                cfg,
                implementation_client=ScriptedClient(),
                feedback_client=ScriptedClient(),
            )
            agent.initialize()
            agent.requirements = base_requirements("Git policy")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Create checked artifact",
                "description": "Write file.txt.",
                "depends_on": [],
                "acceptance_criteria": ["file.txt exists"],
                "validation_commands": [],
                "status": "resolved",
            }]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])
            agent._git_baseline_commit()
            (workspace / "file.txt").write_text("done\n", encoding="utf-8")
            step_commit = agent._git_commit_completed_step(agent.plan_steps[0])
            finalize = agent._git_finalize_policy()

            self.assertTrue(step_commit["committed"])
            self.assertTrue(finalize["left_uncommitted"])
            status = subprocess.run(
                ["git", "-C", str(workspace), "status", "--short"],
                text=True,
                capture_output=True,
                check=True,
            ).stdout
            self.assertIn("file.txt", status)


if __name__ == "__main__":
    unittest.main()
