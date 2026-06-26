from __future__ import annotations

import contextlib
import functools
import http.server
import io
import json
import os
from pathlib import Path
import socketserver
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
from typing import Any

from feedback_agent.agent import ANALYSIS_CONTRACT, FeedbackLoopAgent
from feedback_agent.compaction import (
    _clean_compaction_memory,
    _compaction_memory_is_too_weak,
    deterministic_compact,
    initial_request_context,
    latest_control_state,
    maybe_compact,
)
from feedback_agent.config import load_config
from feedback_agent.conversation import Conversation, Turn
from feedback_agent.git_tools import meaningful_changed_paths
from feedback_agent.llm import ModelRequestHeartbeat, ModelRequestRetrier, format_assistant_message
from feedback_agent.model_profiles import resolve_profile
from feedback_agent.web_research import compact_research_for_prompt, fetch_page, search_queries_for_prompt
from feedback_agent.workspace import collect_workspace_files, extract_json_object, run_commands, write_files, write_plan_doc


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
    def test_workspace_collection_summarizes_binary_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "README.md").write_text("# Notes\n", encoding="utf-8")
            (workspace / "screenshot.png").write_bytes(
                b"\x89PNG\r\n\x1a\n" + bytes(range(32)) + b"\x00\x00IEND"
            )

            files = {item["path"]: item for item in collect_workspace_files(workspace)}

            self.assertEqual(files["README.md"]["content"], "# Notes\n")
            self.assertTrue(files["screenshot.png"]["binary"])
            self.assertIn("binary artifact omitted", files["screenshot.png"]["content"])
            self.assertNotIn("\x89PNG", files["screenshot.png"]["content"])

    def test_workspace_collection_skips_dependency_install_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "README.md").write_text("# Project\n", encoding="utf-8")
            sdk_dir = workspace / "$HOME" / ".dotnet" / "shared"
            sdk_dir.mkdir(parents=True)
            (sdk_dir / "System.Private.CoreLib.dll").write_bytes(b"\0" * 1024)
            node_dir = workspace / "node_modules" / "package"
            node_dir.mkdir(parents=True)
            (node_dir / "index.js").write_text("console.log('skip me')\n", encoding="utf-8")

            paths = {item["path"] for item in collect_workspace_files(workspace)}

            self.assertEqual(paths, {"README.md"})

    def test_git_meaningful_changes_ignore_dependency_install_dirs(self) -> None:
        status = "\n".join([
            "?? $HOME/",
            "?? node_modules/",
            "?? ARCHITECTURE.md",
            " M PLAN.md",
        ])

        self.assertEqual(meaningful_changed_paths(status), ["ARCHITECTURE.md"])

    def test_reviewer_prompt_keeps_small_docs_intact_and_skips_harness_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            readme = "# Demo\n" + ("a" * 1800) + "\npython -m http.server 8000\n" + ("b" * 1800)
            evidence = {
                "kind": "step_feedback_tools",
                "workspace_files": [
                    {"path": "PLAN.md", "content": "harness plan", "size": 12, "truncated": False},
                    {"path": "README.md", "content": readme, "size": len(readme), "truncated": False},
                ],
                "validation_results": [],
                "git": {},
            }

            compact = agent._compact_step_evidence_for_prompt(evidence)
            files = {item["path"]: item for item in compact["workspace_files"]}

            self.assertNotIn("PLAN.md", files)
            self.assertIn("README.md", files)
            self.assertFalse(files["README.md"]["prompt_truncated"])
            self.assertIn("python -m http.server 8000", files["README.md"]["content"])

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

    def test_model_request_heartbeat_reports_long_inflight_call(self) -> None:
        output = io.StringIO()
        heartbeat = ModelRequestHeartbeat(interval_seconds=0.05, stream=output, clock=time.monotonic)

        result = heartbeat.run("test-model", lambda: time.sleep(0.12) or "ok")

        self.assertEqual(result, "ok")
        self.assertIn("still waiting for test-model", output.getvalue())
        self.assertIn("health=not-configured", output.getvalue())

    def test_model_request_heartbeat_reports_health_without_touching_transcript(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "conversation.jsonl"
            conversation = Conversation(transcript)
            conversation.append("user", "real model-facing prompt")

            heartbeat = ModelRequestHeartbeat(
                interval_seconds=0.05,
                stream=output,
                clock=time.monotonic,
                health_check=lambda: "health=ok http=200",
            )

            result = heartbeat.run("test-model", lambda: time.sleep(0.12) or "ok")

            self.assertEqual(result, "ok")
            self.assertIn("health=ok http=200", output.getvalue())
            saved = transcript.read_text(encoding="utf-8")
            self.assertIn("real model-facing prompt", saved)
            self.assertNotIn("model-call", saved)
            self.assertNotIn("health=ok", saved)

    def test_latest_control_state_prefers_newer_feedback_over_stale_directive(self) -> None:
        turns = [
            Turn("user", "IMPLEMENTATION_AGENT_REQUEST:\nIMPLEMENT_PLAN_STEP_PHASE step_id=S6 attempt=2"),
            Turn("assistant", "NEXT_IMPLEMENTATION_DIRECTIVE:\n{\"status\":\"needs_rework\",\"needs_rework\":true,\"summary\":\"old failure\"}"),
            Turn("assistant", "FEEDBACK_AGENT_RESPONSE:\n{\"status\":\"resolved\",\"needs_rework\":false,\"summary\":\"new pass\"}"),
        ]

        state = latest_control_state(turns)

        self.assertIn("step_id=S6 attempt=2", state)
        self.assertIn("Last reviewer response: status=resolved, needs_rework=false", state)
        self.assertNotIn("old failure", state)

    def test_latest_control_state_final_review_overrides_implementation_request(self) -> None:
        turns = [
            Turn("user", "IMPLEMENTATION_AGENT_REQUEST:\nIMPLEMENT_PLAN_STEP_PHASE step_id=S6 attempt=2"),
            Turn("user", "FEEDBACK_AGENT_REQUEST:\nFINAL_PROJECT_REVIEW_PHASE"),
            Turn(
                "assistant",
                "FEEDBACK_AGENT_RESPONSE:\nSome prose first.\n```json\n"
                "{\"status\":\"resolved\",\"needs_rework\":false,\"summary\":\"final pass\"}\n```",
            ),
        ]

        state = latest_control_state(turns)

        self.assertIn("Final project review: status=resolved, needs_rework=false", state)
        self.assertIn("final pass", state)
        self.assertNotIn("Current implementation request", state)

    def test_model_request_retrier_reports_exhaustion(self) -> None:
        output = io.StringIO()
        retrier = ModelRequestRetrier(attempts=2, sleep_seconds=0, sleep=lambda _seconds: None, stream=output)

        with self.assertRaisesRegex(RuntimeError, "failed after 2 attempts"):
            retrier.run(lambda: (_ for _ in ()).throw(TimeoutError("slow model server")))

        self.assertIn("attempt 1/2", output.getvalue())

    def test_reasoning_content_is_preserved_before_final_content(self) -> None:
        text = format_assistant_message(
            {
                "reasoning_content": "Check requirements, then produce JSON.",
                "content": "{\"status\":\"resolved\"}",
            },
            preserve_reasoning=True,
        )

        self.assertIn("<think>", text)
        self.assertIn("Check requirements", text)
        self.assertTrue(text.rstrip().endswith("{\"status\":\"resolved\"}"))
        self.assertEqual(extract_json_object(text)["status"], "resolved")

    def test_reasoning_content_can_be_suppressed(self) -> None:
        text = format_assistant_message(
            {
                "reasoning_content": "private reasoning",
                "content": "{\"status\":\"resolved\"}",
            },
            preserve_reasoning=False,
        )

        self.assertEqual(text, "{\"status\":\"resolved\"}")

    def test_reasoning_content_is_not_duplicated_for_legacy_content(self) -> None:
        text = format_assistant_message(
            {
                "reasoning_content": "already present",
                "content": "<think>\nalready present\n</think>\n{\"status\":\"resolved\"}",
            },
            preserve_reasoning=True,
        )

        self.assertEqual(text.count("<think>"), 1)
        self.assertEqual(extract_json_object(text)["status"], "resolved")

    def test_json_extractor_recovers_first_balanced_object_from_noisy_output(self) -> None:
        payload = extract_json_object(
            "<think>not json { nope }</think>\n"
            "{\"status\":\"resolved\",\"needs_rework\":false}\n"
            "trailing duplicate-ish text {broken"
        )

        self.assertEqual(payload["status"], "resolved")
        self.assertFalse(payload["needs_rework"])

    def test_json_extractor_ignores_incomplete_json_inside_think_block(self) -> None:
        payload = extract_json_object(
            "<think>\n"
            "I should return {\"status\":\"resolved\", \"needs\n"
            "</think>\n"
            "{\"status\":\"resolved\",\"needs_rework\":false}"
        )

        self.assertEqual(payload["status"], "resolved")
        self.assertFalse(payload["needs_rework"])

    def test_json_extractor_uses_last_corrected_object(self) -> None:
        payload = extract_json_object(
            "```json\n{\"status\":\"needs_rework\",\"summary\":\"first\"}\n```\n"
            "Wait, corrected object:\n"
            "```json\n{\"status\":\"resolved\",\"summary\":\"second\"}\n```"
        )

        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(payload["summary"], "second")

    def test_json_extractor_repairs_invalid_html_escape(self) -> None:
        payload = extract_json_object(
            "{\"status\":\"needs_rework\",\"summary\":\"Fix </div\\> in HTML.\"}"
        )

        self.assertEqual(payload["summary"], "Fix </div> in HTML.")

    def test_json_extractor_ignores_channel_wrapper(self) -> None:
        payload = extract_json_object(
            "<|channel>thought<channel|>{\"status\":\"resolved\",\"summary\":\"ok\"}"
        )

        self.assertEqual(payload["status"], "resolved")

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

    def test_command_object_accepts_plain_string_cmd_without_shell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = run_commands(
                root,
                [{"cmd": "python -c \"print('string command accepted')\""}],
                timeout_seconds=30,
                max_timeout_seconds=300,
            )

            self.assertEqual(results[0]["command"][:2], ["python", "-c"])
            self.assertEqual(results[0]["returncode"], 0)
            self.assertIn("string command accepted", results[0]["stdout"])

    def test_missing_command_becomes_failed_evidence_instead_of_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = run_commands(
                root,
                [["definitely-not-installed-local-tool-xyz", "--version"]],
                timeout_seconds=30,
                max_timeout_seconds=300,
            )

            self.assertEqual(results[0]["returncode"], 127)
            self.assertIn("command not found", results[0]["stderr"])

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

    def test_compaction_memory_strips_think_blocks(self) -> None:
        cleaned = _clean_compaction_memory("<think>private reasoning</think>\nKeep this decision.")

        self.assertEqual(cleaned, "Keep this decision.")

    def test_compaction_memory_strips_unclosed_think_blocks(self) -> None:
        cleaned = _clean_compaction_memory("<think>private reasoning without close")

        self.assertIn("Compaction produced no usable memory", cleaned)

    def test_compaction_memory_strips_channel_wrappers(self) -> None:
        cleaned = _clean_compaction_memory("<|channel>thought<channel|>Keep this decision.")

        self.assertEqual(cleaned, "Keep this decision.")

    def test_compaction_memory_strips_multiline_channel_wrappers(self) -> None:
        cleaned = _clean_compaction_memory("<|channel>thought\n<channel|>Keep this decision.")

        self.assertEqual(cleaned, "Keep this decision.")

    def test_compaction_rejects_useless_tiny_memory(self) -> None:
        self.assertTrue(_compaction_memory_is_too_weak("fallible_thought"))
        self.assertTrue(_compaction_memory_is_too_weak("ok"))
        self.assertFalse(_compaction_memory_is_too_weak("Requirement: keep the tested validation script and the accepted browser evidence."))

    def test_compaction_rejects_raw_implementation_payload_memory(self) -> None:
        raw_payload = json.dumps({
            "plan_note": "S1 complete",
            "files": [{"path": "README.md", "content": "x" * 2000}],
            "commands": [["test", "-f", "README.md"]],
            "test_evidence": ["README exists"],
            "resolution_request": "none",
        })

        self.assertTrue(_compaction_memory_is_too_weak(raw_payload))

    def test_deterministic_compaction_clips_long_lines(self) -> None:
        compacted = deterministic_compact("short\n" + ("x" * 5000) + "\nend")

        self.assertIn("truncated long compaction line", compacted)
        self.assertLess(len(compacted), 3000)

    def test_compaction_preserves_authoritative_rework_directive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "work"
            workspace.mkdir()
            cfg_path = write_config(root, workspace, "compact directive", "Build anything.")
            config = load_config(cfg_path)
            conversation = Conversation(root / "conversation.jsonl")
            conversation.append("system", "durable system prompt")
            conversation.append("user", "PROJECT DESIGN: build a platformer")
            conversation.append(
                "user",
                'NEXT_IMPLEMENTATION_DIRECTIVE:\n{"status":"needs_rework","needs_rework":true,'
                '"summary":"S1 still references missing files; fix those references."}',
            )
            conversation.append(
                "user",
                "IMPLEMENTATION_AGENT_REQUEST:\nIMPLEMENT_PLAN_STEP_PHASE step_id=S1 attempt=3\nRetry the step.",
            )
            bad_compactor = ScriptedClient(["S1 is complete. All validation passed."])

            maybe_compact(
                conversation,
                config,
                bad_compactor,
                context_window=100,
                incoming_tokens=1000,
                pinned_context="Pinned plan: S1 still pending; requirements and research remain authoritative.",
                force=True,
            )

            active_text = (root / "conversation.jsonl").read_text(encoding="utf-8")
            self.assertIn("AUTHORITATIVE_RECENT_CONTROL_STATE", active_text)
            self.assertIn("status=needs_rework", active_text)
            self.assertIn("needs_rework=true", active_text)
            self.assertIn("step_id=S1 attempt=3", active_text)
            self.assertIn("overrides any older compacted prose", active_text)
            self.assertIn("PINNED_WORKFLOW_STATE", active_text)
            self.assertIn("S1 still pending", active_text)

    def test_compaction_preserves_initial_request_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "work"
            cfg_path = write_config(root, workspace, "initial task", "Build a very specific artifact named alpha.")
            config = load_config(cfg_path)
            conversation = Conversation(root / "conversation.jsonl")
            conversation.append("system", "durable system prompt")
            conversation.append("user", "PROJECT DESIGN: initial task\n\nBuild a very specific artifact named alpha.")
            for index in range(12):
                conversation.append("assistant", f"older implementation chatter {index}")
            bad_compactor = ScriptedClient(["ok"])

            self.assertIn("PROJECT DESIGN", initial_request_context(conversation.turns))
            maybe_compact(
                conversation,
                config,
                bad_compactor,
                context_window=100,
                incoming_tokens=1000,
                pinned_context="Pinned state",
                force=True,
            )

            active_text = (root / "conversation.jsonl").read_text(encoding="utf-8")
            self.assertIn("INITIAL_REQUEST_CONTEXT", active_text)
            self.assertIn("Build a very specific artifact named alpha", active_text)

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

    def test_load_config_accepts_minimal_project_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = root / "minimal.json"
            cfg_path.write_text(json.dumps({
                "runtime": {"workspace": "out/minimal"},
                "project_design": {
                    "title": "Minimal",
                    "prompt": "Build a tiny checked project.",
                },
            }), encoding="utf-8")

            cfg = load_config(cfg_path, repo_root=root)

            self.assertEqual(cfg.runtime.workspace, (root / "out/minimal").resolve())
            self.assertEqual(cfg.project_design.title, "Minimal")
            self.assertEqual(cfg.implementation_model.max_tokens, 32768)
            self.assertTrue(cfg.runtime.docker_isolation)
            self.assertEqual(cfg.loop.max_approach_reattempts, 5)
            self.assertEqual(cfg.phases.analysis.max_iterations, 2)
            self.assertEqual(cfg.implementation_model.reasoning_budget_tokens, 4096)

    def test_model_profile_aliases_resolve_mtp_models(self) -> None:
        fast = resolve_profile("fast")
        qwen_alias = resolve_profile("qwen-26b-qat-mtp")

        self.assertEqual(fast.name, "gemma4-26b-a4b-qat-mtp")
        self.assertIn("MTP", fast.draft_path)
        self.assertEqual(qwen_alias.name, "qwen3.6-27b-mtp")
        self.assertEqual(qwen_alias.spec_type, "draft-mtp")

    def test_load_config_can_override_model_base_urls_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            config_path = write_config(root, workspace, "docker network", "Build anything.")
            data = json.loads(config_path.read_text(encoding="utf-8"))
            data["feedback_model"] = dict(data["implementation_model"])
            data["feedback_model"]["base_url"] = "http://127.0.0.1:2/v1"
            config_path.write_text(json.dumps(data), encoding="utf-8")

            previous_impl = os.environ.get("AGENT_IMPLEMENTATION_BASE_URL")
            previous_feedback = os.environ.get("AGENT_FEEDBACK_BASE_URL")
            try:
                os.environ["AGENT_IMPLEMENTATION_BASE_URL"] = "http://agentic-qwen36-server:8161/v1"
                os.environ["AGENT_FEEDBACK_BASE_URL"] = "http://agentic-reviewer:9000/v1/"

                cfg = load_config(config_path, repo_root=root)
            finally:
                if previous_impl is None:
                    os.environ.pop("AGENT_IMPLEMENTATION_BASE_URL", None)
                else:
                    os.environ["AGENT_IMPLEMENTATION_BASE_URL"] = previous_impl
                if previous_feedback is None:
                    os.environ.pop("AGENT_FEEDBACK_BASE_URL", None)
                else:
                    os.environ["AGENT_FEEDBACK_BASE_URL"] = previous_feedback

            self.assertEqual(cfg.implementation_model.base_url, "http://agentic-qwen36-server:8161/v1")
            self.assertEqual(cfg.feedback_model.base_url, "http://agentic-reviewer:9000/v1")

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

    def test_tool_call_verifier_blocks_destructive_command_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()

            results = agent._run_verified_commands(
                [["dd", "if=/dev/zero", "of=/dev/sda", "bs=1M", "count=1"]],
                source="unit_test",
                context={"purpose": "prove destructive commands are blocked"},
            )

            self.assertEqual(results[0]["returncode"], 126)
            self.assertTrue(results[0]["blocked_by_tool_verifier"])
            self.assertIn("dd", results[0]["stderr"])
            transcript = (workspace / ".agent_state" / "conversation.jsonl").read_text(encoding="utf-8")
            self.assertIn("TOOL_CALL_VERIFICATION_RESULT", transcript)

    def test_tool_call_verifier_blocks_malformed_curl_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()

            results = agent._run_verified_commands(
                [["curl", "-s", "-X", "POST", "-d", "{\"message\": \"unterminated}", "http://127.0.0.1:9"]],
                source="unit_test",
                context={"purpose": "prove malformed JSON payloads are blocked"},
            )

            self.assertEqual(results[0]["returncode"], 126)
            self.assertTrue(results[0]["blocked_by_tool_verifier"])
            self.assertIn("JSON payload is malformed", results[0]["stderr"])

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

    def test_output_only_prompt_overrides_extra_quality_deliverables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Count strings over {A,B,C,D} with exactly two vowels "
                    "if A is the only vowel. Put only the integer answer in ANSWER.txt."
                ),
            )

            self.assertFalse(agent._default_quality_policy_applies())
            self.assertTrue(agent._explicit_artifact_only_constraint())
            self.assertIn("output-only", agent._default_quality_instruction())

    def test_filename_dot_output_only_prompt_overrides_extra_quality_deliverables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Consider integers n from 1 to 120 and return "
                    "the sum of values that pass the filters."
                ),
            )

            self.assertFalse(agent._default_quality_policy_applies())
            self.assertTrue(agent._explicit_artifact_only_constraint())

    def test_computed_answer_plan_rejects_shape_only_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Consider integers n from 1 to 120. "
                    "Keep n if n is divisible by exactly one of 3, 5, 7, and return the sum."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Computed answer")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Create answer file",
                    "description": "Compute the requested value and write ANSWER.txt.",
                    "depends_on": [],
                    "acceptance_criteria": ["ANSWER.txt contains the single integer output."],
                    "validation_commands": [[
                        "python",
                        "-c",
                        "from pathlib import Path; s=Path('ANSWER.txt').read_text().strip(); assert s.isdigit()",
                    ]],
                    "status": "pending",
                }
            ]

            findings = agent._plan_structural_findings()

            self.assertIn("shape-only", "\n".join(findings))
            self.assertIn("semantic validation", "\n".join(findings))

    def test_computed_answer_plan_allows_semantic_validator_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Count the 4-character strings over alphabet {A,B,C,D} "
                    "that satisfy the listed constraints."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Computed answer")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Create answer and semantic validator",
                    "description": "Compute the requested value and validate it by independent enumeration.",
                    "depends_on": [],
                    "acceptance_criteria": ["ANSWER.txt matches the independently recomputed count."],
                    "validation_commands": [["python", "validate_answer.py"]],
                    "status": "pending",
                }
            ]

            findings = agent._plan_structural_findings()

            self.assertNotIn("shape-only", "\n".join(findings))

    def test_computed_answer_plan_allows_format_precursor_with_later_semantic_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Consider integers n from 1 to 120. "
                    "Keep n if n is divisible by exactly one of 3, 5, 7, and return the sum."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Computed answer")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Create candidate answer",
                    "description": "Compute the requested value and write ANSWER.txt.",
                    "depends_on": [],
                    "acceptance_criteria": ["ANSWER.txt exists and contains a single integer."],
                    "validation_commands": [[
                        "python",
                        "-c",
                        "from pathlib import Path; s=Path('ANSWER.txt').read_text().strip(); assert s.isdigit()",
                    ]],
                    "status": "pending",
                },
                {
                    "id": "S2",
                    "title": "Semantically verify answer",
                    "description": "Run a validator that independently recomputes the requested sum.",
                    "depends_on": ["S1"],
                    "acceptance_criteria": ["ANSWER.txt matches the independently recomputed sum."],
                    "validation_commands": [["python", "verify.py"]],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            self.assertNotIn("shape-only", "\n".join(findings))

    def test_computed_answer_plan_rejects_hardcoded_expected_answer_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Consider integers n from 1 to 120. "
                    "Keep n if n is divisible by exactly one of 3, 5, 7, and return the sum."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Computed answer")
            agent.requirements["planning_confirmation"] = {
                "is_feasible": True,
                "is_clear": True,
                "is_verifiable": True,
                "verification_strategy": "Run validate.py and assert ANSWER.txt matches expected value 1751.",
            }
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Create answer and validator",
                    "description": "Create validate.py to compare ANSWER.txt against the expected value 1751.",
                    "depends_on": [],
                    "acceptance_criteria": ["ANSWER.txt contains the correct integer 1751."],
                    "validation_commands": [["python", "validate.py"]],
                    "status": "pending",
                }
            ]

            findings = agent._plan_structural_findings()

            self.assertIn("hard-code", "\n".join(findings))
            self.assertIn("recomputes", "\n".join(findings))

    def test_computed_answer_plan_rejects_vague_validator_without_semantic_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Consider integers n from 1 to 120. "
                    "Keep n if n is divisible by exactly one of 3, 5, 7, and return the sum."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Computed answer")
            agent.requirements["planning_confirmation"] = {
                "is_feasible": True,
                "is_clear": True,
                "is_verifiable": True,
                "verification_strategy": "Run validate.py to check that ANSWER.txt contains the correct integer sum.",
            }
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Create answer and validator",
                    "description": "Create solution.py and validate.py to verify the result and output format.",
                    "depends_on": [],
                    "acceptance_criteria": ["ANSWER.txt contains only the correct integer sum."],
                    "validation_commands": [["python", "validate.py"]],
                    "status": "pending",
                }
            ]

            findings = agent._plan_structural_findings()

            self.assertIn("semantic validation", "\n".join(findings))
            self.assertIn("recomputes", "\n".join(findings))

    def test_requirements_review_enforces_hardcoded_answer_validation_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Consider integers n from 1 to 120. "
                    "Keep n if n is divisible by exactly one of 3, 5, 7, and return the sum."
                ),
                feedback_responses=[
                    json.dumps({
                        "status": "resolved",
                        "needs_rework": False,
                        "summary": "Looks complete.",
                        "required_changes": [],
                    })
                ],
            )
            agent.initialize()
            requirements = base_requirements("Computed answer")
            requirements["planning_confirmation"] = {
                "is_feasible": True,
                "is_clear": True,
                "is_verifiable": True,
                "verification_strategy": "Run validate.py and assert ANSWER.txt matches expected value 1751.",
            }
            requirements["plan"] = [
                {
                    "id": "S1",
                    "title": "Create answer and validator",
                    "description": "Create validate.py to compare ANSWER.txt against the expected value 1751.",
                    "depends_on": [],
                    "acceptance_criteria": ["ANSWER.txt contains the correct integer 1751."],
                    "validation_commands": [["python", "validate.py"]],
                }
            ]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "needs_requirements_change")
            self.assertIn("hard-code", "\n".join(review["required_changes"]))

    def test_only_as_domain_fact_does_not_override_default_quality_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt="Build a counting helper where A is the only vowel in the alphabet.",
            )

            self.assertTrue(agent._default_quality_policy_applies())
            self.assertFalse(agent._explicit_artifact_only_constraint())

    def test_plan_refinement_preserves_active_step_validation_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            active_step = {
                "id": "S1",
                "title": "Fix validation command",
                "description": "Original step with stale validation.",
                "depends_on": [],
                "acceptance_criteria": ["validation command is current"],
                "validation_commands": [["python", "-c", "raise SystemExit(1)"]],
                "status": "pending",
            }
            agent.plan_steps = [
                active_step,
                {
                    "id": "S2",
                    "title": "Future step",
                    "description": "Keep future work ordered.",
                    "depends_on": ["S1"],
                    "acceptance_criteria": ["future validation"],
                    "validation_commands": [["python", "-c", "print('future')"]],
                    "status": "pending",
                },
            ]

            refined = [
                {
                    **active_step,
                    "validation_commands": [["python", "-c", "print('fresh reviewer evidence')"]],
                },
                agent.plan_steps[1],
            ]

            agent.plan_steps = agent._merge_refined_plan_steps(agent.plan_steps, refined)

            self.assertIs(agent.plan_steps[0], active_step)
            self.assertIs(agent._next_pending_step(), active_step)
            self.assertEqual(
                active_step["validation_commands"],
                [["python", "-c", "print('fresh reviewer evidence')"]],
            )

    def test_problem_analysis_phase_runs_before_planning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            analysis_payload = {
                "problem_restatement": "Build a checked artifact after analysis.",
                "domain_and_constraints": ["No external source is required."],
                "initial_source_check": {
                    "sources_checked": ["configured prompt"],
                    "source_gaps": [],
                    "freshness_risks": [],
                },
                "possible_solution_paths": [
                    {
                        "id": "A",
                        "description": "Small direct implementation.",
                        "advantages": ["fast"],
                        "risks": ["limited scope"],
                        "verification_strategy": "unit test",
                    },
                    {
                        "id": "B",
                        "description": "Package structure first.",
                        "advantages": ["scales"],
                        "risks": ["more files"],
                        "verification_strategy": "unit and docs checks",
                    },
                ],
                "recommended_path": {
                    "path_id": "A",
                    "rationale": "The task is small.",
                    "fallback_trigger": "Use B if scope grows.",
                },
                "analysis_quality": {
                    "is_comprehensive": True,
                    "is_domain_aware": True,
                    "is_actionable_for_planning": True,
                    "remaining_unknowns": [],
                },
            }
            implementation = ScriptedClient([json.dumps(analysis_payload)])
            agent = FeedbackLoopAgent(
                load_config(write_config(root, workspace, "analysis", "Build a checked artifact."), repo_root=root),
                implementation_client=implementation,
                feedback_client=ScriptedClient(),
            )
            agent.initialize()

            result = agent._analysis_phase()

            self.assertEqual(result["status"], "resolved")
            self.assertEqual(agent.problem_analysis["recommended_path"]["path_id"], "A")
            prompt = implementation.calls[0]["messages"][-1]["content"]
            self.assertIn("PROBLEM_ANALYSIS_PHASE", prompt)
            self.assertIn("multiple solution paths", prompt)

    def test_analysis_contract_demonstrates_two_solution_paths(self) -> None:
        self.assertIn('"id": "A"', ANALYSIS_CONTRACT)
        self.assertIn('"id": "B"', ANALYSIS_CONTRACT)
        self.assertIn("at least two", ANALYSIS_CONTRACT)

    def test_approach_review_can_request_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            retry_review = {
                "status": "try_another_approach",
                "needs_rework": True,
                "summary": "Periodic check requires another pass.",
                "decision": "retry_with_new_approach",
                "recommended_next_approach": "Watch the log again from the last checkpoint.",
                "evidence_reviewed": ["final review"],
                "runbook_updates": ["last checked line 10"],
            }
            agent = load_test_agent(root, workspace, feedback_responses=[json.dumps(retry_review)])
            agent.initialize()

            review = agent._approach_review_phase(1, [], {"status": "resolved", "iterations": []})

            self.assertTrue(agent._approach_review_requests_retry(review))
            self.assertEqual(review["recommended_next_approach"], "Watch the log again from the last checkpoint.")

    def test_research_structure_skeleton_step_satisfies_quality_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, title="platformer", prompt="Build a browser platformer.")
            agent.initialize()
            agent.requirements = base_requirements("Platformer")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Research, project structure, and skeleton creation",
                    "description": (
                        "Research HTML5 Canvas game patterns from available knowledge, document notes, "
                        "and create the initial project skeleton."
                    ),
                    "depends_on": [],
                    "acceptance_criteria": [
                        "README.md exists with project description, structure overview, and research notes.",
                        "The project structure is documented and follows a clean separation of concerns.",
                    ],
                    "validation_commands": [["python", "validate_s1.py"]],
                    "status": "pending",
                }
            ]

            findings = agent._plan_structural_findings()

            self.assertNotIn("First plan step", "\n".join(findings))

    def test_plan_validation_rejects_harness_state_file_as_project_deliverable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, title="platformer", prompt="Build a browser platformer.")
            agent.initialize()
            agent.requirements = base_requirements("Platformer")
            agent.plan_steps = [
                {
                    "id": "S1a",
                    "title": "Research and architecture notes",
                    "description": "Create RESEARCH.md with platformer architecture notes and level schema.",
                    "depends_on": [],
                    "acceptance_criteria": [
                        "RESEARCH.md exists with Canvas, physics, and Playwright notes.",
                        "RESEARCH.md defines the level data schema.",
                    ],
                    "validation_commands": [["python", "tests/validate_research_md.py"]],
                    "status": "pending",
                }
            ]

            findings = agent._plan_structural_findings()

            self.assertIn("harness-owned state file RESEARCH.md", "\n".join(findings))
            self.assertIn("PROJECT_RESEARCH.md", "\n".join(findings))

    def test_implementation_payload_cannot_overwrite_harness_state_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            payload = {
                "plan_note": "created project README without touching harness state",
                "files": [
                    {"path": "PLAN.md", "content": "model-owned plan should not replace harness state"},
                    {"path": "REQUIREMENTS.md", "content": "model-owned requirements should not replace harness state"},
                    {"path": "README.md", "content": "# Project README\n"},
                ],
                "commands": [["test", "-f", "README.md"]],
                "test_evidence": ["README exists"],
                "resolution_request": "none",
            }
            impl = ScriptedClient([json.dumps(payload)])
            agent = FeedbackLoopAgent(
                load_config(write_config(root, workspace, "state guard", "Build a tiny project."), repo_root=root),
                implementation_client=impl,
                feedback_client=ScriptedClient(),
            )
            agent.initialize()
            agent._requirements_path().write_text("# Original requirements\n", encoding="utf-8")
            original_plan = agent._plan_path().read_text(encoding="utf-8")
            original_requirements = agent._requirements_path().read_text(encoding="utf-8")

            result = agent._implementation_pass(
                {
                    "id": "S1",
                    "title": "Create README",
                    "description": "Create project README.",
                    "acceptance_criteria": ["README exists"],
                    "validation_commands": [["test", "-f", "README.md"]],
                },
                1,
            )

            self.assertEqual((workspace / "README.md").read_text(encoding="utf-8"), "# Project README\n")
            self.assertEqual(agent._plan_path().read_text(encoding="utf-8"), original_plan + "\n- [S1 attempt 1] created project README without touching harness state\n")
            self.assertEqual(agent._requirements_path().read_text(encoding="utf-8"), original_requirements)
            self.assertEqual(result["skipped_harness_files"], ["PLAN.md", "REQUIREMENTS.md"])
            findings = agent._evidence_findings(
                {"id": "S1", "validation_commands": [["test", "-f", "README.md"]]},
                result,
                {"validation_results": result["commands"], "git": {"meaningful_changed_paths": ["README.md"]}},
            )
            self.assertIn("harness-owned state files", "\n".join(findings))

    def test_implementation_prompt_tells_model_how_to_create_empty_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            implementation = ScriptedClient([
                json.dumps({
                    "plan_note": "No-op prompt inspection.",
                    "files": [],
                    "commands": [],
                    "test_evidence": [],
                    "resolution_request": "none",
                })
            ])
            agent = FeedbackLoopAgent(
                load_config(write_config(root, workspace, "directory scaffold", "Create empty directories."), repo_root=root),
                implementation_client=implementation,
                feedback_client=ScriptedClient(),
            )
            agent.initialize()
            agent.requirements = base_requirements("Directory scaffold")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Create empty directories",
                    "description": "Create game/js and tests directories.",
                    "depends_on": [],
                    "acceptance_criteria": ["game/js exists", "tests exists"],
                    "validation_commands": [["test", "-d", "game/js"], ["test", "-d", "tests"]],
                    "status": "pending",
                }
            ]

            agent._implementation_pass(agent.plan_steps[0], 1)

            prompt = "\n".join(message["content"] for message in implementation.calls[0]["messages"])
            self.assertIn("files` payload creates files, not empty directories", prompt)
            self.assertIn(".gitkeep", prompt)
            self.assertIn("game/js/.gitkeep", prompt)

    def test_step_reference_check_ignores_harness_docs_and_planned_future_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, title="platformer", prompt="Build a browser platformer.")
            agent.initialize()
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Structure",
                    "description": "Create README.md and directories.",
                    "acceptance_criteria": ["README exists"],
                    "validation_commands": [["test", "-f", "README.md"]],
                },
                {
                    "id": "S2",
                    "title": "Validation suite",
                    "description": "Create tests/run_validation.py and game/index.html.",
                    "acceptance_criteria": ["tests/run_validation.py exists"],
                    "validation_commands": [["python", "tests/run_validation.py"]],
                },
            ]
            evidence = {
                "workspace_files": [
                    {"path": "PLAN.md", "content": "Future step will create game/index.html and tests/run_validation.py."},
                    {
                        "path": "README.md",
                        "content": (
                            "Run later with http://localhost:8080/game/index.html and "
                            "python tests/run_validation.py once S2 is implemented. "
                            "Installation notes may mention https://dot.net/v1/dotnet-install.sh "
                            "and $HOME/.dotnet without creating local artifact references."
                        ),
                    },
                ],
            }

            step_findings = agent._workspace_reference_findings(evidence, allow_planned_future_refs=True)
            final_findings = agent._workspace_reference_findings(evidence, allow_planned_future_refs=False)

            self.assertEqual(step_findings, [])
            self.assertIn("tests/run_validation.py", "\n".join(final_findings))
            self.assertNotIn("PLAN.md references", "\n".join(final_findings))
            self.assertNotIn("8080/game/index.html", "\n".join(final_findings))
            self.assertNotIn("dot.net/v1", "\n".join(final_findings))
            self.assertNotIn("HOME/.dotnet", "\n".join(final_findings))

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

    def test_plan_validation_rejects_plain_command_for_intentional_residual_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S2",
                "title": "Fix syntax errors",
                "description": "Fix import blockers while leaving the known logic failure for the next step.",
                "depends_on": [],
                "acceptance_criteria": [
                    "The test suite runs without syntax or import errors.",
                    "The test suite still fails due to the logic error.",
                ],
                "validation_commands": [["python", "-m", "unittest", "discover", "-v"]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertIn("expected failure path", "\n".join(findings))

    def test_plan_validation_rejects_plain_command_for_logic_failure_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S2",
                "title": "Investigation and initial error resolution",
                "description": "Fix syntax blockers, then run tests to expose remaining calculator failures.",
                "depends_on": [],
                "acceptance_criteria": [
                    "The test suite runs without syntax or import errors.",
                    "The failure logs clearly indicate logic errors in the calculator.",
                ],
                "validation_commands": [["python", "-m", "unittest", "discover", "-v"]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertIn("expected failure path", "\n".join(findings))

    def test_plan_validation_accepts_wrapper_for_partial_failure_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S2",
                "title": "Fix syntax and import errors",
                "description": "The suite may still fail on logic tests, but it must run without SyntaxError.",
                "depends_on": [],
                "acceptance_criteria": [
                    "The test suite runs without syntax or import errors.",
                    "Logic tests may still fail, but the suite completes its run.",
                ],
                "validation_commands": [{
                    "cmd": [
                        "python",
                        "-c",
                        "import subprocess; r = subprocess.run(['python', '-m', 'unittest'], capture_output=True, text=True); exit(0 if 'SyntaxError' not in r.stderr else 1)",
                    ]
                }],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertNotIn("expected failure path", "\n".join(findings))

    def test_final_review_skips_accepted_transient_expected_failure_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            step = {
                "id": "S2",
                "title": "Fix syntax and import errors",
                "description": "Tests can run but may still fail on logic assertions until a later step.",
                "depends_on": [],
                "acceptance_criteria": [
                    "Syntax errors are fixed.",
                    "The suite may still fail due to the known logic error.",
                ],
                "validation_commands": [
                    ["python", "-m", "compileall", "."],
                    {"cmd": ["python", "-m", "unittest", "discover", "-v"], "expected_returncode": 1},
                ],
                "status": "resolved",
            }
            agent.plan_steps = [step]
            step_results = [{
                "step_id": "S2",
                "status": "resolved",
                "attempts": [{
                    "review": {"status": "resolved"},
                    "implementation": {
                        "commands": [
                            {
                                "command": ["python", "-m", "unittest", "discover", "-v"],
                                "returncode": 1,
                                "expected_returncode": 1,
                                "returncode_matches_expected": True,
                                "timed_out": False,
                            }
                        ],
                    },
                }],
            }]

            evidence = agent._final_feedback_tool_evidence(step_results)
            validation = evidence["step_validations"][0]

            self.assertEqual(validation["accepted_validation_commands_run"], [])
            skipped = json.dumps(validation["accepted_validation_commands_skipped"])
            self.assertIn("intermediate expected failure", skipped)
            self.assertIn("unittest", json.dumps(validation["final_validation_commands_skipped"]))

    def test_error_identification_step_allows_observed_failing_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S1",
                "title": "Research, planning, and error identification",
                "description": "Run the current broken test suite and document failures before fixing code.",
                "depends_on": [],
                "acceptance_criteria": [
                    "Test suite execution results are documented.",
                    "Syntax/import error is identified.",
                ],
                "validation_commands": [["python", "-m", "unittest", "discover", "-v"]],
                "status": "pending",
            }
            implementation = {
                "written": ["AGENT_RESEARCH.md"],
                "commands": [{
                    "command": ["python", "-m", "unittest", "discover", "-v"],
                    "returncode": 1,
                    "expected_returncode": 0,
                    "timed_out": False,
                }],
                "raw": {"test_evidence": ["observed failing tests"]},
            }
            feedback_evidence = {
                "validation_results": [{
                    "command": ["python", "-m", "unittest", "discover", "-v"],
                    "returncode": 1,
                    "expected_returncode": 0,
                    "timed_out": False,
                }],
                "workspace_files": [{"path": "AGENT_RESEARCH.md", "content": "failures", "size": 8}],
                "git": {"meaningful_changed_paths": ["AGENT_RESEARCH.md"]},
            }

            findings = agent._evidence_findings(step, implementation, feedback_evidence)

            self.assertNotIn("returned 1 but expected 0", "\n".join(findings))

    def test_plan_validation_rejects_python_double_m_typo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S2",
                "title": "Fix syntax errors",
                "description": "Fix import blockers while leaving the known logic failure for the next step.",
                "depends_on": [],
                "acceptance_criteria": ["The test suite still fails due to the logic error."],
                "validation_commands": [{
                    "cmd": ["python", "-mm", "unittest", "discover", "-v"],
                    "expected_returncode": 1,
                }],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertIn("malformed Python flag", "\n".join(findings))

    def test_plan_validation_rejects_py_compile_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S2",
                "title": "Validate syntax",
                "description": "Check package syntax before logic fixes.",
                "depends_on": [],
                "acceptance_criteria": ["All package files compile."],
                "validation_commands": [["python", "-m", "py_compile", "."]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertIn("py_compile", "\n".join(findings))
            self.assertIn("directory", "\n".join(findings))

    def test_plan_validation_rejects_one_line_try_except_python_c(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S5",
                "title": "Validate error handling",
                "description": "Check that mean raises ValueError for empty input.",
                "depends_on": [],
                "acceptance_criteria": ["Error handling is validated."],
                "validation_commands": [[
                    "python",
                    "-c",
                    "raised = False; try: mean([]); except ValueError: raised = True; assert raised",
                ]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertIn("try/except", "\n".join(findings))
            self.assertIn("validation script", "\n".join(findings))

    def test_partial_failure_step_allows_compileall_success_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S2",
                "title": "Fix syntax and import errors",
                "description": "Fix syntax while remaining failures are logic-related.",
                "depends_on": [],
                "acceptance_criteria": [
                    "All syntax and import errors are resolved.",
                    "Remaining failures are logic-related, not syntax/import-related.",
                ],
                "validation_commands": [
                    {"cmd": ["python", "-m", "compileall", "."], "expected_returncode": 0},
                    {"cmd": ["python", "-m", "unittest", "discover", "-v"], "expected_returncode": 1},
                ],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertEqual(findings, [])

    def test_plan_validation_rejects_node_playwright_assumption_in_default_container(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.requirements = base_requirements("Browser validation")
            agent.requirements["assumptions"].append("Playwright will be installed via npm and used with its Node.js API.")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Browser validation",
                "description": "Create a Playwright validation script.",
                "depends_on": [],
                "acceptance_criteria": [
                    "No external dependencies are required beyond Node.js and Playwright.",
                    "The browser validation produces a screenshot and JSON report.",
                ],
                "validation_commands": [["python", "validation/validate.py"]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            self.assertTrue(any("Python Playwright" in item and "Node/npm" in item for item in findings))

    def test_environment_guard_allows_explicit_no_node_npm_npx_wording(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.requirements = base_requirements("Browser validation")
            agent.requirements["assumptions"].append(
                "All validation uses Python Playwright sync API with no Node.js/npm/npx tooling."
            )
            agent.plan_steps = [{
                "id": "S1",
                "title": "Browser validation",
                "description": "Create a Python Playwright validation script without Node/npm/npx.",
                "depends_on": [],
                "acceptance_criteria": [
                    "No Node.js, npm, npx, or @playwright/test dependencies are used.",
                    "The browser validation produces a screenshot and JSON report.",
                ],
                "validation_commands": [["python", "validation/validate.py"]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            self.assertFalse(any("Node/npm" in item for item in findings))

    def test_environment_guard_allows_explicit_dependency_setup_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                title="Node browser validation",
                prompt="Build a Node.js browser test project and install required npm dependencies inside Docker.",
            )
            agent.requirements = base_requirements("Node browser validation")
            agent.requirements["assumptions"].append(
                "The plan includes a bounded dependency setup step because Node/npm are not assumed to exist."
            )
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Dependency setup",
                    "description": "Install Node.js/npm dependencies inside the Docker agent container.",
                    "depends_on": [],
                    "acceptance_criteria": ["npm install completes and writes dependency evidence."],
                    "validation_commands": [["test", "-f", "package.json"]],
                    "status": "pending",
                },
                {
                    "id": "S2",
                    "title": "Node validation",
                    "description": "Run the requested Node.js validation stack.",
                    "depends_on": ["S1"],
                    "acceptance_criteria": ["npx playwright test writes a report."],
                    "validation_commands": [["npm", "test"]],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            self.assertFalse(any("default agent Docker image provides" in item for item in findings))

    def test_feedback_review_rejects_expected_failure_from_malformed_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Malformed expected failure")
            step = {
                "id": "S2",
                "title": "Fix syntax errors",
                "description": "Fix import blockers while leaving the known logic failure for the next step.",
                "depends_on": [],
                "acceptance_criteria": ["The test suite still fails due to the logic error."],
                "validation_commands": [{
                    "cmd": ["python", "-mm", "unittest", "discover", "-v"],
                    "expected_returncode": 1,
                }],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])
            (workspace / "placeholder.py").write_text("print('changed')\n", encoding="utf-8")

            review = agent._step_review_pass(
                step,
                1,
                {"written": ["placeholder.py"], "commands": [], "raw": {"test_evidence": ["expected failure checked"]}},
                "hard_pushback",
            )

            self.assertEqual(review["status"], "needs_plan_change")
            required = "\n".join(review["required_changes"])
            self.assertIn("python -mm", required)
            self.assertIn("No module named m", required)

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

    def test_correction_payload_omits_large_review_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            review = {
                "status": "needs_rework",
                "needs_rework": True,
                "summary": "Final review found two concrete issues.",
                "required_changes": ["Fix A", "Fix B"],
                "deterministic_evidence_findings": ["Guardrail C"],
                "feedback_tool_evidence": {
                    "kind": "final_feedback_tools",
                    "workspace_files": [{"path": "huge.txt", "content": "x" * 100000, "size": 100000}],
                    "step_validations": [{
                        "step_id": "S1",
                        "validation_results": [{
                            "command": ["python", "check.py"],
                            "stdout": "y" * 100000,
                            "stderr": "z" * 100000,
                        }],
                    }],
                    "git": {"diff": "d" * 100000},
                },
            }

            compact = agent._compact_review_for_correction(review)
            encoded = json.dumps(compact)

            self.assertEqual(compact["required_changes"], ["Fix A", "Fix B"])
            self.assertEqual(compact["deterministic_evidence_findings"], ["Guardrail C"])
            self.assertNotIn("feedback_tool_evidence", compact)
            self.assertNotIn("feedback_tool_evidence_summary", compact)
            self.assertNotIn("review_truncation_note", compact)
            self.assertNotIn("x" * 5000, encoded)
            self.assertLess(len(encoded), 3000)

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

    def test_feedback_review_routes_py_compile_directory_to_plan_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Syntax validation")
            step = {
                "id": "T1",
                "title": "Validate syntax",
                "description": "Validate package syntax.",
                "depends_on": [],
                "acceptance_criteria": ["all files compile"],
                "validation_commands": [["python", "-m", "py_compile", "."]],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])
            (workspace / "module.py").write_text("VALUE = 1\n", encoding="utf-8")

            review = agent._step_review_pass(
                step,
                1,
                {"written": ["module.py"], "commands": [], "raw": {"test_evidence": []}},
                "hard_pushback",
            )

            self.assertEqual(review["status"], "needs_plan_change")
            self.assertIn("py_compile", "\n".join(review["required_changes"]))

    def test_final_review_rejects_broken_local_doc_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Documentation consistency")
            step = {
                "id": "T1",
                "title": "Write notes",
                "description": "Document the fix.",
                "depends_on": [],
                "acceptance_criteria": ["BUGFIX_NOTES.md references real project files"],
                "validation_commands": [["test", "-f", "BUGFIX_NOTES.md"]],
                "status": "resolved",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])
            (workspace / "invoice_calc").mkdir()
            (workspace / "invoice_calc" / "discounts.py").write_text("def ok(): pass\n", encoding="utf-8")
            (workspace / "BUGFIX_NOTES.md").write_text(
                "Fixed missing colon in `invoice_calc/disintcounts.py`.\n",
                encoding="utf-8",
            )

            review = agent._final_project_review(
                1,
                [{"step_id": "T1", "status": "resolved", "attempts": [{"implementation": {"commands": []}}]}],
            )

            self.assertEqual(review["status"], "needs_rework")
            self.assertIn("invoice_calc/disintcounts.py", "\n".join(review["required_changes"]))

    def test_final_review_prompt_prefers_evidence_over_manual_derivation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Final review evidence discipline")
            step = {
                "id": "T1",
                "title": "Write checked answer",
                "description": "Write answer.txt and validate it.",
                "depends_on": [],
                "acceptance_criteria": ["answer.txt exists"],
                "validation_commands": [["test", "-f", "answer.txt"]],
                "status": "resolved",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])
            (workspace / "answer.txt").write_text("42\n", encoding="utf-8")

            agent._final_project_review(
                1,
                [{"step_id": "T1", "status": "resolved", "attempts": [{"implementation": {"commands": []}}]}],
            )

            prompt = agent.feedback_client.calls[-1]["messages"][-1]["content"]
            self.assertIn("reviewer-owned validation results", prompt)
            self.assertIn("do not spend the final review re-solving algorithmic tasks", prompt)
            self.assertIn("requesting stronger validation evidence", prompt)

    def test_step_review_prompt_prefers_validation_over_manual_derivation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Step review evidence discipline")
            step = {
                "id": "T1",
                "title": "Write checked answer",
                "description": "Write answer.txt and validate it.",
                "depends_on": [],
                "acceptance_criteria": ["answer.txt exists"],
                "validation_commands": [["test", "-f", "answer.txt"]],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])
            (workspace / "answer.txt").write_text("42\n", encoding="utf-8")

            agent._step_review_pass(
                step,
                1,
                {
                    "written": ["answer.txt"],
                    "commands": [],
                    "raw": {"test_evidence": ["answer.txt validation requested"]},
                },
                "hard_pushback",
            )

            prompt = agent.feedback_client.calls[-1]["messages"][-1]["content"]
            self.assertIn("reviewer-owned validation results as primary evidence", prompt)
            self.assertIn("do not spend the review re-solving exact-answer tasks", prompt)
            self.assertIn("request stronger validation evidence", prompt)

    def test_browser_guidance_prefers_python_playwright_without_node_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            guidance = agent._browser_validation_guidance()

            self.assertIn("sync_playwright", guidance)
            self.assertIn("assume there is no", guidance)
            self.assertIn("npx", guidance)

    def test_browser_step_detection_uses_whole_word_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            docs_step = {
                "title": "Create README documentation",
                "description": "Write usage guide and installation guide.",
                "acceptance_criteria": ["README has usage examples."],
            }
            browser_step = {
                "title": "Validate browser UI",
                "description": "Open HTML in a browser and click a button.",
                "acceptance_criteria": ["Screenshot evidence exists."],
            }

            self.assertFalse(agent._looks_like_browser_step(docs_step))
            self.assertTrue(agent._looks_like_browser_step(browser_step))

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
            self.assertIn("focused directly verifiable change", "\n".join(review["required_changes"]))
            self.assertIn("parse_error", review)

    def test_reasoning_only_feedback_acceptance_does_not_force_json_repair_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, feedback_responses=[])
            agent.initialize()

            review = agent._extract_json_or_retry(
                "<think>The plan is feasible, clear, and verifiable. I'll accept.</think>",
                phase="PLAN_VALIDATION_PHASE",
                contract='{"status":"resolved|needs_plan_change","required_changes":["specific change"]}',
                feedback=True,
            )

            self.assertEqual(review["status"], "resolved")
            self.assertFalse(review["needs_rework"])
            self.assertTrue(review["inferred_from_malformed_response"])
            self.assertEqual(agent.feedback_client.calls, [])

    def test_reasoning_only_final_review_completion_does_not_force_json_repair_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, feedback_responses=[])
            agent.initialize()

            review = agent._extract_json_or_retry(
                "<think>The project is complete and meets all requirements. The implementation is solid.</think>",
                phase="FINAL_PROJECT_REVIEW_PHASE",
                contract='{"status":"resolved|needs_rework","required_changes":["specific change"]}',
                feedback=True,
            )

            self.assertEqual(review["status"], "resolved")
            self.assertFalse(review["needs_rework"])
            self.assertTrue(review["inferred_from_malformed_response"])
            self.assertEqual(agent.feedback_client.calls, [])

    def test_long_reasoning_final_review_completion_does_not_force_json_repair_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, feedback_responses=[])
            agent.initialize()

            raw = (
                "<think>Everything looks correct. The implementation is complete and verified. "
                "All steps resolved.</think>\n"
                + "Candidate cross-check note. " * 400
            )
            review = agent._extract_json_or_retry(
                raw,
                phase="FINAL_PROJECT_REVIEW_PHASE",
                contract='{"status":"resolved|needs_rework","required_changes":["specific change"]}',
                feedback=True,
            )

            self.assertEqual(review["status"], "resolved")
            self.assertFalse(review["needs_rework"])
            self.assertTrue(review["inferred_from_malformed_response"])
            self.assertEqual(agent.feedback_client.calls, [])

    def test_reasoning_only_tool_call_approval_does_not_force_json_repair_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, feedback_responses=[])
            agent.initialize()

            review = agent._extract_json_or_retry(
                "<think>The command is correctly targeted and bounded. I'll approve it.</think>",
                phase="TOOL_CALL_VERIFICATION_PHASE",
                contract='{"status":"approved|blocked","commands":[{"index":0,"decision":"approved|blocked"}]}',
                feedback=True,
            )

            self.assertEqual(review["status"], "approved")
            self.assertEqual(review["commands"], [])
            self.assertTrue(review["inferred_from_malformed_response"])
            self.assertEqual(agent.feedback_client.calls, [])

    def test_reasoning_only_tool_call_block_does_not_force_json_repair_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    "<think>`python logic.py` will not create ANSWER.txt, so I will block this command.</think>"
                ],
            )
            agent.initialize()

            review = agent._tool_call_verification_phase(
                [["python", "logic.py"], ["python", "verify_answer.py"]],
                source="implementation",
                context={"purpose": "test"},
            )

            self.assertEqual(review["status"], "blocked")
            self.assertEqual([item["decision"] for item in review["commands"]], ["blocked", "blocked"])
            self.assertTrue(review["inferred_from_malformed_response"])
            self.assertEqual(len(agent.feedback_client.calls), 1)

    def test_reasoning_only_feedback_rejection_uses_json_repair_not_generic_rework(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            repair = {
                "status": "needs_rework",
                "needs_rework": True,
                "summary": "Validation command references a verifier script that the plan never creates.",
                "required_changes": ["Add a plan step that creates verify_solution.py before running it."],
            }
            agent = load_test_agent(root, workspace, feedback_responses=[json.dumps(repair)])
            agent.initialize()

            review = agent._extract_json_or_retry(
                (
                    "<think>The validation command is flawed: it runs verify_solution.py, "
                    "but the plan never creates that script. I will request rework.</think>"
                ),
                phase="REQUIREMENTS_REVIEW_PHASE",
                contract='{"status":"resolved|needs_rework","required_changes":["specific change"]}',
                feedback=True,
            )

            self.assertEqual(review["status"], "needs_rework")
            self.assertIn("verifier script", review["summary"])
            self.assertEqual(review["required_changes"], repair["required_changes"])
            self.assertEqual(len(agent.feedback_client.calls), 1)

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

    def test_malformed_implementation_repair_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                implementation_responses=[
                    json.dumps({"files": [], "commands": [], "resolution_request": "none"})
                ],
            )
            agent.initialize()

            raw = "not json " + ("as_noted_in_the_enoughs_" * 2000)
            payload = agent._extract_json_or_retry(
                raw,
                phase="IMPLEMENTATION_PHASE",
                contract='{"files":[],"commands":[]}',
            )

            self.assertEqual(payload["files"], [])
            self.assertLessEqual(agent.impl_client.calls[-1]["max_tokens"], 6144)
            repair_prompt = agent.impl_client.calls[-1]["messages"][-1]["content"]
            self.assertLess(len(repair_prompt), 5000)
            self.assertIn("[long-token-truncated]", repair_prompt)

    def test_plan_review_rejects_malformed_shell_test_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Create file",
                    "description": "Create output.txt.",
                    "acceptance_criteria": ["output.txt exists"],
                    "validation_commands": [["test", "-F", "output.txt"]],
                }
            ]

            findings = agent._plan_structural_findings()

            self.assertIn("malformed shell test flag '-F'", "\n".join(findings))

    def test_plan_review_rejects_malformed_grep_max_count_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Check heading",
                    "description": "Check a markdown heading exists.",
                    "acceptance_criteria": ["README.md has the heading"],
                    "validation_commands": [["grep", "-md", "# Heading", "README.md"]],
                }
            ]

            findings = agent._plan_structural_findings()

            self.assertIn("malformed grep max-count flag", "\n".join(findings))

    def test_feedback_review_routes_malformed_grep_validation_to_plan_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Malformed grep validation")
            step = {
                "id": "S1",
                "title": "Write analysis",
                "description": "Write a markdown analysis file.",
                "depends_on": [],
                "acceptance_criteria": ["ANALYSIS.md contains a Long-term section"],
                "validation_commands": [["grep", "-md", "# Long-term", "ANALYSIS.md"]],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])
            (workspace / "ANALYSIS.md").write_text("# Long-term\n", encoding="utf-8")

            review = agent._step_review_pass(
                step,
                1,
                {"written": ["ANALYSIS.md"], "commands": [], "raw": {"test_evidence": ["analysis exists"]}},
                "hard_pushback",
            )

            self.assertEqual(review["status"], "needs_plan_change")
            required = "\n".join(review["required_changes"])
            self.assertIn("grep", required)
            self.assertIn("max-count", required)

    def test_write_files_serializes_structured_json_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            written = write_files(
                workspace,
                [
                    {
                        "path": "SOURCES.json",
                        "content": [{"url": "https://example.com", "title": "Example"}],
                    }
                ],
            )

            self.assertEqual(written, ["SOURCES.json"])
            payload = json.loads((workspace / "SOURCES.json").read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["url"], "https://example.com")

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

    def test_final_review_accepts_rerun_step_validation_when_plan_path_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Final review stale plan validation")
            tests_dir = workspace / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_core_logic.py").write_text(
                "import unittest\n\n"
                "class CoreLogicTests(unittest.TestCase):\n"
                "    def test_passes(self):\n"
                "        self.assertEqual(2 + 2, 4)\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n",
                encoding="utf-8",
            )
            step = {
                "id": "S2",
                "title": "Core logic implementation",
                "description": "Implement core logic and validate it with unit tests.",
                "depends_on": [],
                "acceptance_criteria": ["Core logic tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_core_logic.py"]],
                "status": "resolved",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])

            review = agent._final_project_review(
                1,
                [{
                    "step_id": "S2",
                    "status": "resolved",
                    "attempts": [{
                        "review": {"status": "resolved"},
                        "implementation": {
                            "commands": [{
                                "command": ["python", "-m", "unittest", "discover", "tests"],
                                "returncode": 0,
                                "expected_returncode": 0,
                                "timed_out": False,
                                "stdout": "",
                                "stderr": "OK",
                            }],
                        },
                    }],
                }],
            )

            self.assertEqual(review["status"], "resolved")
            self.assertEqual(review["deterministic_evidence_findings"], [])
            validations = review["feedback_tool_evidence"]["step_validations"]
            self.assertEqual(validations[0]["validation_results"][0]["returncode"], 1)
            self.assertEqual(validations[0]["accepted_validation_results"][0]["returncode"], 0)

    def test_final_review_can_rescue_skipped_step_with_fresh_final_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Final correction rescued skipped step")
            (workspace / "index.html").write_text("<!doctype html><title>ok</title>\n", encoding="utf-8")
            step = {
                "id": "S2",
                "title": "Static website content",
                "description": "Create final HTML content.",
                "depends_on": [],
                "acceptance_criteria": ["index.html exists"],
                "validation_commands": [["test", "-f", "index.html"]],
                "status": "skipped_with_note",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])

            review = agent._final_project_review(
                1,
                [{"step_id": "S2", "status": "skipped_with_note", "attempts": [{"implementation": {"commands": []}}]}],
            )

            self.assertEqual(review["status"], "resolved")
            self.assertEqual(review["deterministic_evidence_findings"], [])
            final_status = agent._final_status(
                [{"step_id": "S2", "status": "skipped_with_note", "attempts": []}],
                {"status": "resolved", "iterations": [{"review": review}]},
            )
            self.assertEqual(final_status, "resolved")

    def test_final_review_still_reports_skipped_step_without_final_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Skipped step remains unproven")
            step = {
                "id": "S2",
                "title": "Static website content",
                "description": "Create final HTML content.",
                "depends_on": [],
                "acceptance_criteria": ["index.html exists"],
                "validation_commands": [["test", "-f", "index.html"]],
                "status": "skipped_with_note",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])

            review = agent._final_project_review(
                1,
                [{"step_id": "S2", "status": "skipped_with_note", "attempts": [{"implementation": {"commands": []}}]}],
            )

            self.assertEqual(review["status"], "needs_rework")
            self.assertTrue(any("ended with status skipped_with_note" in item for item in review["deterministic_evidence_findings"]))

    def test_final_review_still_rejects_stale_plan_when_accepted_validation_now_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Final review failed replacement validation")
            tests_dir = workspace / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_core_logic.py").write_text(
                "import unittest\n\n"
                "class CoreLogicTests(unittest.TestCase):\n"
                "    def test_fails(self):\n"
                "        self.assertEqual(2 + 2, 5)\n",
                encoding="utf-8",
            )
            step = {
                "id": "S2",
                "title": "Core logic implementation",
                "description": "Implement core logic and validate it with unit tests.",
                "depends_on": [],
                "acceptance_criteria": ["Core logic tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_core_logic.py"]],
                "status": "resolved",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])

            review = agent._final_project_review(
                1,
                [{
                    "step_id": "S2",
                    "status": "resolved",
                    "attempts": [{
                        "review": {"status": "resolved"},
                        "implementation": {
                            "commands": [{
                                "command": ["python", "-m", "unittest", "discover", "tests"],
                                "returncode": 0,
                                "expected_returncode": 0,
                                "timed_out": False,
                                "stdout": "",
                                "stderr": "OK",
                            }],
                        },
                    }],
                }],
            )

            self.assertEqual(review["status"], "needs_rework")
            self.assertTrue(any("accepted validation returned" in item for item in review["deterministic_evidence_findings"]))

    def test_final_review_skips_transient_expected_failure_validations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Existing-project repair")
            step = {
                "id": "S2",
                "title": "Fix syntax and import errors",
                "description": "The suite should run without syntax/import errors while remaining logic tests still fail.",
                "depends_on": [],
                "acceptance_criteria": [
                    "The test runner successfully discovers and attempts to run tests.",
                    "The suite still fails due to the remaining logic error.",
                ],
                "validation_commands": [{
                    "cmd": ["python", "-c", "print('final state is now healthy')"],
                    "expected_returncode": 1,
                }],
                "status": "resolved",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])

            review = agent._final_project_review(
                1,
                [{"step_id": "S2", "status": "resolved", "attempts": [{"implementation": {"commands": []}}]}],
            )

            self.assertEqual(review["status"], "resolved")
            self.assertEqual(review["deterministic_evidence_findings"], [])
            validations = review["feedback_tool_evidence"]["step_validations"]
            self.assertEqual(validations[0]["validation_results"], [])
            self.assertEqual(len(validations[0]["final_validation_commands_skipped"]), 1)

    def test_final_review_still_runs_final_negative_path_validations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("CLI negative-path behavior")
            step = {
                "id": "T1",
                "title": "Implement invalid input error behavior",
                "description": "Invalid CLI input must exit with code 2.",
                "depends_on": [],
                "acceptance_criteria": ["Invalid input exits 2 with a clear error message."],
                "validation_commands": [{
                    "cmd": ["python", "-c", "import sys; print('invalid input', file=sys.stderr); sys.exit(2)"],
                    "expected_returncode": 2,
                }],
                "status": "resolved",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])

            review = agent._final_project_review(
                1,
                [{"step_id": "T1", "status": "resolved", "attempts": [{"implementation": {"commands": []}}]}],
            )

            self.assertEqual(review["status"], "resolved")
            validations = review["feedback_tool_evidence"]["step_validations"]
            self.assertEqual(validations[0]["final_validation_commands_skipped"], [])
            self.assertEqual(validations[0]["validation_results"][0]["returncode"], 2)

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

    def test_web_research_builds_focused_queries_from_long_prompt(self) -> None:
        prompt = (
            "Use web research to review current economic and central-bank news as of 2026-05-02, "
            "then write a concise but deep analysis of how the news may affect interest rates. "
            "The output project must include ANALYSIS.md, SOURCES.json, a validation script, "
            "README instructions, and strict source citation checks."
        )

        queries = search_queries_for_prompt(prompt)

        self.assertGreaterEqual(len(queries), 2)
        self.assertLessEqual(max(len(query) for query in queries), 240)
        combined = " ".join(queries).lower()
        self.assertIn("central-bank", combined)
        self.assertIn("interest", combined)
        self.assertNotIn("validation script", queries[0].lower())
        self.assertNotIn("federal reserve interest rates may 2026 news inflation jobs", combined)

    def test_web_research_does_not_put_pdf_binary_in_prompt_excerpt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = root / "site"
            site.mkdir()
            (site / "report.pdf").write_bytes(b"%PDF-1.7\n" + bytes(range(32)) * 200)
            cfg = load_config(write_config(root, root / "workspace", "pdf research", "Research a PDF."), repo_root=root).web_research

            with local_http_server(site) as base_url:
                result = fetch_page(f"{base_url}/report.pdf", cfg)

            self.assertEqual(result["status"], "error")
            self.assertIn("Unsupported non-text content type", result["error"])
            self.assertEqual(result["excerpt"], "")
            compact = compact_research_for_prompt({"status": "failed", "requested": True, "targets": [result]})
            self.assertNotIn("%PDF", compact)
            self.assertNotIn("\\ufffd", compact)

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
