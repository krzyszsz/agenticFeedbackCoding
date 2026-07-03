from __future__ import annotations

import contextlib
from dataclasses import replace
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
import types
import unittest
import urllib.error
from typing import Any

import scripts.run_benchmarks as run_benchmarks
from feedback_agent.agent import (
    ANALYSIS_CONTRACT,
    ANALYSIS_REVIEW_CONTRACT,
    APPROACH_REVIEW_CONTRACT,
    EXECUTABLE_DELIVERABLE_GUIDANCE,
    FEEDBACK_SYSTEM_PROMPT,
    IMPLEMENTATION_CONTRACT,
    JSON_OUTPUT_RULES,
    PLAN_REFINEMENT_CONTRACT,
    REQUIREMENTS_CONTRACT,
    REVIEW_DECISION_OUTPUT_GUIDANCE,
    REVIEW_CHALLENGE_GUIDANCE,
    TOOL_CALL_VERIFICATION_CONTRACT,
    TOOL_PROGRESS_REVIEW_CONTRACT,
    VALIDATION_COMMAND_RULES,
    FeedbackLoopAgent,
    _review_prompt_guidance,
)
from feedback_agent.compaction import (
    _bounded_recent_turn_count,
    _clean_compaction_memory,
    _compaction_memory_conflicts_with_control_state,
    _compaction_memory_is_too_weak,
    _source_for_compaction,
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
        phase = self._phase_from_messages(messages)
        if self.responses:
            if (
                phase == "TOOL_CALL_VERIFICATION_PHASE"
                and not self._is_json_repair(messages)
                and not self._looks_like_tool_verification_response(self.responses[0])
            ):
                return self._default_tool_verification_response(messages)
            return self.responses.pop(0)
        if phase == "TOOL_CALL_VERIFICATION_PHASE" and not self._is_json_repair(messages):
            return self._default_tool_verification_response(messages)
        if phase == "TOOL_PROGRESS_REVIEW_PHASE":
            return json.dumps({
                "status": "continue",
                "decision": "continue",
                "summary": "Scripted progress review allowed the command to continue.",
                "evidence": ["No scripted stop condition was supplied."],
                "risks": [],
                "next_check_seconds": 30,
            })
        if phase == "APPROACH_REVIEW_PHASE":
            return json.dumps({
                "status": "resolved",
                "needs_rework": False,
                "summary": "Scripted approach review kept the result.",
                "decision": "keep_result",
                "evidence_reviewed": ["final_review:summary"],
                "runbook_updates": [],
            })
        return json.dumps({
            "status": "resolved",
            "needs_rework": False,
            "summary": "Scripted review accepted the evidence.",
            "required_changes": [],
            "verification_evidence": ["reviewer-owned validation evidence inspected"],
        })

    @staticmethod
    def _phase_from_messages(messages: list[dict[str, str]]) -> str:
        content = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                break
        for phase in (
            "TOOL_CALL_VERIFICATION_PHASE",
            "TOOL_PROGRESS_REVIEW_PHASE",
            "APPROACH_REVIEW_PHASE",
        ):
            lines = content.splitlines()
            if any(line.startswith(phase) or line.startswith(f"{phase}_JSON_REPAIR") for line in lines[:3]):
                return phase
        return ""

    @staticmethod
    def _is_json_repair(messages: list[dict[str, str]]) -> bool:
        for message in reversed(messages):
            if message.get("role") == "user":
                return any("_JSON_REPAIR" in line for line in message.get("content", "").splitlines()[:3])
        return False

    @staticmethod
    def _looks_like_tool_verification_response(raw: str) -> bool:
        try:
            payload = extract_json_object(raw)
        except Exception:
            return True
        if str(payload.get("status") or "") in {"approved", "blocked", "needs_revision"}:
            return True
        if "commands" not in payload:
            return False
        commands = payload.get("commands")
        if not isinstance(commands, list):
            return True
        return isinstance(commands, list) and any(
            isinstance(item, dict)
            for item in commands
        )

    @staticmethod
    def _default_tool_verification_response(messages: list[dict[str, str]]) -> str:
        content = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                break
        try:
            prompt = extract_json_object(content)
        except Exception:
            prompt = {}
        commands = prompt.get("commands") if isinstance(prompt, dict) else []
        decisions = []
        for index, item in enumerate(commands if isinstance(commands, list) else []):
            if isinstance(item, dict) and "index" in item:
                command_index = item.get("index")
            else:
                command_index = index
            decisions.append({
                "index": command_index,
                "decision": "approved",
                "risk_level": "low",
                "reason": "Scripted test default approval for a bounded tool call.",
            })
        return json.dumps({
            "status": "approved",
            "summary": "Scripted test default approved tool calls.",
            "commands": decisions,
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


def write_config(
    root: Path,
    workspace: Path,
    title: str,
    prompt: str,
    *,
    deterministic_semantic_scope_checks: bool = True,
) -> Path:
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
            "deterministic_semantic_scope_checks": deterministic_semantic_scope_checks,
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
    deterministic_semantic_scope_checks: bool = True,
) -> FeedbackLoopAgent:
    cfg = load_config(
        write_config(
            root,
            workspace,
            title,
            prompt,
            deterministic_semantic_scope_checks=deterministic_semantic_scope_checks,
        ),
        repo_root=root,
    )
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
    def test_review_normalization_canonicalizes_list_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            agent = load_test_agent(root, workspace)

            review = agent._normalize_review({
                "status": "resolved",
                "summary": "accepted",
                "required_changes": "none",
                "verification_evidence": "unit tests passed",
                "evidence_reviewed": "final review evidence",
                "runbook_updates": None,
            })

            self.assertEqual(review["required_changes"], ["none"])
            self.assertEqual(review["verification_evidence"], ["unit tests passed"])
            self.assertEqual(review["evidence_reviewed"], ["final review evidence"])
            self.assertEqual(review["runbook_updates"], [])

    def test_implementation_payload_normalization_canonicalizes_test_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            agent = load_test_agent(root, workspace)

            payload = agent._normalize_implementation_payload({
                "plan_note": "done",
                "files": {"path": "x.txt", "content": "x"},
                "commands": ["test", "-f", "x.txt"],
                "test_evidence": "checked x.txt",
                "resolution_request": "none",
            })

            self.assertEqual(payload["test_evidence"], ["checked x.txt"])
            self.assertEqual(payload["files"], [{"path": "x.txt", "content": "x"}])
            self.assertEqual(payload["commands"], [["test", "-f", "x.txt"]])

    def test_benchmark_suite_selection_preserves_declared_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            suite_path = Path(tmp) / "suites.json"
            suite_path.write_text(json.dumps({
                "suites": {
                    "ordered": {
                        "task_ids": ["second", "first", "second"],
                    }
                }
            }), encoding="utf-8")
            tasks = [
                {"id": "first", "title": "First"},
                {"id": "second", "title": "Second"},
            ]

            suite_ids = run_benchmarks.load_suite_ids(suite_path, "ordered")
            selected = run_benchmarks.select_tasks(tasks, suite_ids, None)

            self.assertEqual([task["id"] for task in selected], ["second", "first"])

    def test_benchmark_explicit_task_ids_override_suite_selection(self) -> None:
        suite_ids = ["first", "second"]
        explicit_ids = ["third"]

        selected_ids = run_benchmarks.resolve_selection_ids(suite_ids, explicit_ids)

        self.assertEqual(selected_ids, ["third"])

    def test_benchmark_uses_suite_selection_when_no_explicit_task_ids(self) -> None:
        suite_ids = ["first", "second"]

        selected_ids = run_benchmarks.resolve_selection_ids(suite_ids, [])

        self.assertEqual(selected_ids, ["first", "second"])

    def test_benchmark_config_applies_web_research_and_task_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            task = {
                "id": "research-demo",
                "title": "Research demo",
                "category": "research",
                "prompt": "Use fetched source evidence.",
                "web_research": True,
                "config_overrides": {
                    "runtime": {
                        "docker_user": "root",
                        "command_timeout_seconds": 180,
                    },
                    "phases": {
                        "implementation": {"max_iterations": 9},
                    },
                },
            }

            cfg = run_benchmarks.benchmark_config(
                task,
                repo_root=root,
                workspace=workspace,
                implementation_profile="gemma4-26b-a4b-qat-mtp",
                feedback_profile=None,
                docker_isolation=True,
                reasoning_budget_tokens=None,
                max_tokens=4096,
                feedback_response_max_tokens=2048,
                print_transcript=False,
                live_turn_max_chars=0,
            )

            self.assertTrue(cfg["web_research"]["enabled"])
            self.assertTrue(cfg["mcp_tools"]["web_scraping"])
            self.assertEqual(cfg["implementation_model"]["max_tokens"], 4096)
            self.assertTrue(cfg["implementation_model"]["send_reasoning_budget"])
            self.assertEqual(cfg["runtime"]["feedback_response_max_tokens"], 2048)
            self.assertEqual(cfg["runtime"]["docker_user"], "root")
            self.assertEqual(cfg["runtime"]["command_timeout_seconds"], 180)
            self.assertEqual(cfg["runtime"]["command_progress_review_interval_seconds"], 300)
            self.assertEqual(cfg["runtime"]["command_progress_review_min_interval_seconds"], 30)
            self.assertEqual(cfg["phases"]["requirements_refinement"]["max_iterations"], 4)
            self.assertEqual(cfg["phases"]["plan_validation"]["max_iterations"], 4)
            self.assertEqual(cfg["phases"]["implementation"]["max_iterations"], 9)

    def test_benchmark_result_matching_separates_direct_and_harness_modes(self) -> None:
        result = {
            "run_mode": "single-shot",
            "task_id": "demo",
            "implementation_profile": "gemma4-26b-a4b-qat-mtp",
            "feedback_profile": None,
            "reasoning_budget_tokens": 2048,
            "max_tokens": 4096,
            "feedback_response_max_tokens": 2048,
        }

        self.assertTrue(run_benchmarks.result_matches_run(
            result,
            run_mode="single-shot",
            task_id="demo",
            implementation_profile="gemma4-26b-a4b-qat-mtp",
            feedback_profile=None,
            reasoning_budget_tokens=2048,
            max_tokens=4096,
            feedback_response_max_tokens=2048,
        ))
        self.assertFalse(run_benchmarks.result_matches_run(
            result,
            run_mode="harness",
            task_id="demo",
            implementation_profile="gemma4-26b-a4b-qat-mtp",
            feedback_profile=None,
            reasoning_budget_tokens=2048,
            max_tokens=4096,
            feedback_response_max_tokens=2048,
        ))

    def test_benchmark_resume_keeps_results_from_other_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "results.json").write_text(json.dumps({
                "results": [
                    {
                        "run_mode": "single-shot",
                        "task_id": "demo",
                        "implementation_profile": "gemma4-26b-a4b-qat-mtp",
                        "feedback_profile": None,
                        "reasoning_budget_tokens": 2048,
                        "max_tokens": 8192,
                        "feedback_response_max_tokens": 4096,
                    },
                    {
                        "run_mode": "harness",
                        "task_id": "demo",
                        "implementation_profile": "gemma4-26b-a4b-qat-mtp",
                        "feedback_profile": None,
                        "reasoning_budget_tokens": 2048,
                        "max_tokens": 8192,
                        "feedback_response_max_tokens": 4096,
                    },
                ]
            }), encoding="utf-8")

            results = run_benchmarks.load_resume_results(
                output_dir,
                run_mode="harness",
                selected_tasks=[{"id": "demo"}],
                implementation_profile="gemma4-26b-a4b-qat-mtp",
                feedback_profile=None,
                reasoning_budget_tokens=2048,
                max_tokens=8192,
                feedback_response_max_tokens=4096,
            )

            self.assertEqual({result["run_mode"] for result in results}, {"single-shot", "harness"})

    def test_benchmark_resume_reruns_non_passing_results(self) -> None:
        self.assertTrue(run_benchmarks.should_skip_existing_result({"grade": "pass"}))
        self.assertFalse(run_benchmarks.should_skip_existing_result({"grade": "fail"}))
        self.assertFalse(run_benchmarks.should_skip_existing_result({"grade": "timeout"}))
        self.assertFalse(run_benchmarks.should_skip_existing_result(None))

    def test_single_shot_benchmark_writes_model_files_without_harness(self) -> None:
        captured: dict[str, Any] = {}

        class FakeClient:
            def __init__(self, cfg: Any) -> None:
                captured["cfg"] = cfg

            def chat(
                self,
                messages: list[dict[str, str]],
                *,
                max_tokens: int | None = None,
                temperature: float | None = None,
            ) -> str:
                captured["messages"] = messages
                captured["max_tokens"] = max_tokens
                captured["temperature"] = temperature
                return json.dumps({
                    "files": [{"path": "ANSWER.txt", "content": "ok"}],
                    "notes": "created answer",
                    "self_check": ["answer file present"],
                })

        original_client = run_benchmarks.OpenAICompatClient
        try:
            run_benchmarks.OpenAICompatClient = FakeClient  # type: ignore[assignment]
            with tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                task = {
                    "id": "demo",
                    "title": "Demo",
                    "category": "algorithmic_exact",
                    "prompt": "Create ANSWER.txt only.",
                }
                returncode, _elapsed, output, metadata = run_benchmarks.run_single_shot(
                    workspace,
                    task,
                    implementation_profile="gemma4-26b-a4b-qat-mtp",
                    reasoning_budget_tokens=2048,
                    max_tokens=4096,
                )
                self.assertEqual(returncode, 0)
                self.assertEqual((workspace / "ANSWER.txt").read_text(encoding="utf-8"), "ok")
                self.assertEqual(metadata["files_written"], ["ANSWER.txt"])
                self.assertIn("SINGLE SHOT RESPONSE", output)
                self.assertIn("Create ANSWER.txt only.", captured["messages"][1]["content"])
                self.assertEqual(captured["max_tokens"], 4096)
        finally:
            run_benchmarks.OpenAICompatClient = original_client  # type: ignore[assignment]

    def test_benchmark_harness_sets_feedback_docker_url_for_pair_runs(self) -> None:
        captured: dict[str, Any] = {}

        class FakePopen:
            returncode = 0

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                captured["args"] = args
                captured["kwargs"] = kwargs
                self.pid = 12345

            def communicate(self, timeout: int | None = None) -> tuple[str, None]:
                captured["timeout"] = timeout
                return "ok", None

        def fake_run(*args: Any, **kwargs: Any) -> types.SimpleNamespace:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return types.SimpleNamespace(returncode=0, stdout="ok")

        original_popen = run_benchmarks.subprocess.Popen
        original_run = run_benchmarks.subprocess.run
        try:
            run_benchmarks.subprocess.Popen = FakePopen  # type: ignore[assignment]
            run_benchmarks.subprocess.run = fake_run  # type: ignore[assignment]
            returncode, _elapsed, output = run_benchmarks.run_harness(
                Path.cwd(),
                Path("config.json"),
                implementation_profile="gemma4-26b-a4b-qat-mtp",
                feedback_profile="gemma4-31b-qat-mtp",
                timeout_seconds=123,
                stream_output=False,
            )
        finally:
            run_benchmarks.subprocess.Popen = original_popen  # type: ignore[assignment]
            run_benchmarks.subprocess.run = original_run  # type: ignore[assignment]

        self.assertEqual(returncode, 0)
        self.assertEqual(output, "ok")
        self.assertEqual(captured["timeout"], 123)
        self.assertEqual(
            captured["kwargs"]["env"]["AGENT_FEEDBACK_BASE_URL"],
            "http://agentic-gemma4-31b-mtp-server:8162/v1",
        )
        self.assertIn("AGENT_CONTAINER_NAME", captured["kwargs"]["env"])

    def test_benchmark_timeout_removes_named_agent_container(self) -> None:
        captured: dict[str, Any] = {"run_calls": []}

        class FakePopen:
            returncode = None

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                captured["popen_kwargs"] = kwargs
                self.pid = 12345
                self.calls = 0

            def communicate(self, timeout: int | None = None) -> tuple[str, None]:
                self.calls += 1
                if self.calls == 1:
                    raise subprocess.TimeoutExpired(cmd="scripts/run_agent.sh", timeout=timeout or 0, output="partial")
                return " after-cleanup", None

        def fake_run(*args: Any, **kwargs: Any) -> types.SimpleNamespace:
            captured["run_calls"].append((args, kwargs))
            return types.SimpleNamespace(returncode=0, stdout="removed")

        original_popen = run_benchmarks.subprocess.Popen
        original_run = run_benchmarks.subprocess.run
        try:
            run_benchmarks.subprocess.Popen = FakePopen  # type: ignore[assignment]
            run_benchmarks.subprocess.run = fake_run  # type: ignore[assignment]
            returncode, _elapsed, output = run_benchmarks.run_harness(
                Path.cwd(),
                Path("config.json"),
                implementation_profile="gemma4-26b-a4b-qat-mtp",
                feedback_profile=None,
                timeout_seconds=1,
                stream_output=False,
            )
        finally:
            run_benchmarks.subprocess.Popen = original_popen  # type: ignore[assignment]
            run_benchmarks.subprocess.run = original_run  # type: ignore[assignment]

        self.assertEqual(returncode, 124)
        self.assertIn("partial after-cleanup", output)
        container_name = captured["popen_kwargs"]["env"]["AGENT_CONTAINER_NAME"]
        docker_call = captured["run_calls"][0][0][0]
        self.assertEqual(docker_call, ["docker", "rm", "-f", container_name])

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

    def test_initial_workspace_context_exposes_sources_and_skips_harness_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "median.py").write_text("def median(values):\n    return values[0]\n", encoding="utf-8")
            (workspace / "test_median.py").write_text("import unittest\n", encoding="utf-8")
            (workspace / "PLAN.md").write_text("harness control file\n", encoding="utf-8")
            agent = load_test_agent(root, workspace)

            context = agent._initial_workspace_context_for_prompt()
            files = {item["path"]: item for item in context["files"]}

            self.assertEqual(context["status"], "available")
            self.assertIn("median.py", files)
            self.assertIn("test_median.py", files)
            self.assertNotIn("PLAN.md", files)
            self.assertIn("return values[0]", files["median.py"]["content"])

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

    def test_latest_control_state_does_not_accept_status_only_final_review(self) -> None:
        turns = [
            Turn("user", "IMPLEMENTATION_AGENT_REQUEST:\nIMPLEMENT_PLAN_STEP_PHASE step_id=S6 attempt=2"),
            Turn("user", "FEEDBACK_AGENT_REQUEST:\nFINAL_PROJECT_REVIEW_PHASE"),
            Turn("assistant", "FEEDBACK_AGENT_RESPONSE:\n{\"status\":\"resolved\"}"),
        ]

        state = latest_control_state(turns)

        self.assertIn("Final project review response is off-contract", state)
        self.assertNotIn("Final project review: status=resolved", state)

    def test_latest_control_state_does_not_accept_placeholder_review_summary(self) -> None:
        turns = [
            Turn("user", "IMPLEMENTATION_AGENT_REQUEST:\nIMPLEMENT_PLAN_STEP_PHASE step_id=S6 attempt=2"),
            Turn("user", "FEEDBACK_AGENT_REQUEST:\nFINAL_PROJECT_REVIEW_PHASE"),
            Turn(
                "assistant",
                "FEEDBACK_AGENT_RESPONSE:\n"
                "{\"status\":\"resolved\",\"needs_rework\":false,\"summary\":\"whole project review\"}",
            ),
        ]

        state = latest_control_state(turns)

        self.assertIn("Final project review response is off-contract", state)
        self.assertNotIn("Final project review: status=resolved", state)

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

    def test_json_extractor_recovers_after_unclosed_think_with_stray_brace(self) -> None:
        payload = extract_json_object(
            "<think>\n"
            "The set is \\{3, 5, 7\\} and this thought tag is never closed.\n"
            "{\"status\":\"resolved\",\"summary\":\"Digit sum uses $\\\\text{sum\\_digits}$\"}"
        )

        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(payload["summary"], "Digit sum uses $\\text{sum_digits}$")

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

    def test_json_extractor_rejects_nested_object_inside_malformed_container(self) -> None:
        with self.assertRaisesRegex(ValueError, "Only nested JSON objects"):
            extract_json_object(
                '{"planning_confirmation":{"is_feasible":true,'
                '"is_clear":true,"is_verifiable":true}, "broken": bad}'
            )

    def test_json_extractor_ignores_channel_wrapper(self) -> None:
        payload = extract_json_object(
            "<|channel>thought<channel|>{\"status\":\"resolved\",\"summary\":\"ok\"}"
        )

        self.assertEqual(payload["status"], "resolved")

    def test_requirements_json_shape_mismatch_uses_repair_instead_of_fragment(self) -> None:
        valid_requirements = {
            "project_summary": "Create a checked artifact.",
            "refined_requirements": ["Create ANSWER.txt containing ok."],
            "assumptions": [],
            "open_questions": [],
            "planning_confirmation": {
                "is_feasible": True,
                "is_clear": True,
                "is_verifiable": True,
                "verification_strategy": "Check the file content with a bounded shell assertion.",
                "remaining_risks": [],
            },
            "plan": [
                {
                    "id": "S1",
                    "title": "Create file",
                    "description": "Write ANSWER.txt.",
                    "depends_on": [],
                    "acceptance_criteria": ["ANSWER.txt contains ok."],
                    "validation_commands": [["bash", "-lc", "test \"$(cat ANSWER.txt)\" = ok"]],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, implementation_responses=[json.dumps(valid_requirements)])
            agent.initialize()

            payload = agent._extract_json_or_retry(
                json.dumps(valid_requirements["planning_confirmation"]),
                phase="REQUIREMENTS_REFINEMENT_PHASE",
                contract=REQUIREMENTS_CONTRACT,
            )

            self.assertEqual(payload["project_summary"], "Create a checked artifact.")
            self.assertEqual(payload["plan"][0]["id"], "S1")
            self.assertEqual(len(agent.impl_client.calls), 1)

    def test_artifact_only_requirements_repair_prefers_json_safe_expression_validator(self) -> None:
        valid_requirements = {
            "project_summary": "Create a checked answer artifact.",
            "refined_requirements": ["Create ANSWER.txt containing the computed answer."],
            "assumptions": [],
            "open_questions": [],
            "planning_confirmation": {
                "is_feasible": True,
                "is_clear": True,
                "is_verifiable": True,
                "verification_strategy": "Compare ANSWER.txt to an independently recomputed value.",
                "remaining_risks": [],
            },
            "plan": [{
                "id": "S1",
                "title": "Create answer",
                "description": "Write ANSWER.txt and validate it.",
                "depends_on": [],
                "acceptance_criteria": ["ANSWER.txt contains the expected integer."],
                "validation_commands": [[
                    "python",
                    "-c",
                    "from pathlib import Path; actual=Path('ANSWER.txt').read_text().strip(); assert actual.isdigit(); print(actual)",
                ]],
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt="Create ANSWER.txt only. Return the integer answer.",
                implementation_responses=[json.dumps(valid_requirements)],
            )
            agent.initialize()

            payload = agent._extract_json_or_retry(
                "not json",
                phase="REQUIREMENTS_REFINEMENT_PHASE",
                contract=REQUIREMENTS_CONTRACT,
            )
            repair_prompt = agent.impl_client.calls[-1]["messages"][-1]["content"]

            self.assertEqual(payload["plan"][0]["id"], "S1")
            self.assertIn("Command protocol:", repair_prompt)
            self.assertIn("Commands are data, not prose", repair_prompt)
            self.assertIn("Validation must observe the requested behavior", repair_prompt)
            self.assertNotIn("Do not use def/class/for/while/try/with", repair_prompt)
            self.assertNotIn("Use multiline shell/heredoc commands", repair_prompt)
            self.assertNotIn("prefer simple argv checks, correctly wrapped multiline shell commands", repair_prompt)

    def test_validation_rules_render_literal_newline_escape(self) -> None:
        self.assertIn("Commands are data, not prose", VALIDATION_COMMAND_RULES)
        self.assertIn("one `bash -lc` script string", VALIDATION_COMMAND_RULES)
        self.assertIn("count, absence, uniqueness, ordering, idempotence", VALIDATION_COMMAND_RULES)
        self.assertNotIn("as `\n`", VALIDATION_COMMAND_RULES)
        self.assertNotIn("Comprehensions and generator expressions", VALIDATION_COMMAND_RULES)
        self.assertNotIn("Artifact-only prompts have a stricter command shape", VALIDATION_COMMAND_RULES)

    def test_command_contract_prefers_plain_argv_for_default_success(self) -> None:
        self.assertIn("Use a plain argv list for ordinary commands", VALIDATION_COMMAND_RULES)
        self.assertIn("The `cmd` value is always an argv", VALIDATION_COMMAND_RULES)
        self.assertIn("not pre-expanded by the", VALIDATION_COMMAND_RULES)
        self.assertIn("Long-running or open-ended validation", VALIDATION_COMMAND_RULES)
        self.assertNotIn("Do not loop directly on `proc.stdout.readline()`", VALIDATION_COMMAND_RULES)
        self.assertNotIn("one separate command", VALIDATION_COMMAND_RULES)
        self.assertNotIn("copied temporary workspace", VALIDATION_COMMAND_RULES)
        self.assertIn('not {"cmd": "bash -lc ..."}', IMPLEMENTATION_CONTRACT)
        self.assertIn('"expected_returncode": 2', IMPLEMENTATION_CONTRACT)
        self.assertIn("The `files[].content` values are JSON strings", IMPLEMENTATION_CONTRACT)
        self.assertIn("Raw content like", IMPLEMENTATION_CONTRACT)
        self.assertNotIn('"expected_returncode": 0, "timeout_seconds": 120', IMPLEMENTATION_CONTRACT)

    def test_validation_rules_do_not_require_error_text_with_expected_returncode(self) -> None:
        self.assertIn("Expected failure checks", VALIDATION_COMMAND_RULES)
        self.assertIn("`expected_returncode`", VALIDATION_COMMAND_RULES)
        self.assertIn("intended non-zero outcome", VALIDATION_COMMAND_RULES)
        self.assertNotIn("Require error-text assertions only when", VALIDATION_COMMAND_RULES)

    def test_json_repair_prompt_simplifies_default_success_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            valid_payload = {
                "plan_note": "Recovered with a plain command.",
                "files": [],
                "commands": [["bash", "-lc", "true"]],
                "test_evidence": [],
                "resolution_request": "none",
            }
            agent = load_test_agent(
                root,
                workspace,
                implementation_responses=[json.dumps(valid_payload)],
            )
            agent.initialize()

            payload = agent._extract_json_or_retry(
                '{"commands":[{"cmd":["bash","-lc","true","expected_returncode":0}]}',
                phase="IMPLEMENT_PLAN_STEP_PHASE",
                contract=IMPLEMENTATION_CONTRACT,
            )
            repair_prompt = agent.impl_client.calls[-1]["messages"][-1]["content"]

            self.assertEqual(payload["commands"], [["bash", "-lc", "true"]])
            self.assertIn("Previous response tail for recovery:", repair_prompt)
            self.assertIn('"expected_returncode":0', repair_prompt)
            self.assertIn("Commands are data, not prose", repair_prompt)
            self.assertIn("Use a plain argv list for ordinary commands", repair_prompt)
            self.assertIn("The `cmd` value is always an argv", repair_prompt)
            self.assertNotIn("return it as a plain argv array, not a command object", repair_prompt)
            self.assertNotIn("one command object per distinct failing invocation", repair_prompt)
            self.assertIn("files[].content` values are JSON strings", repair_prompt)

    def test_json_repair_omits_repetitive_implementation_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            valid_payload = {
                "plan_note": "Recovered with a minimal payload.",
                "files": [],
                "commands": [],
                "test_evidence": [],
                "resolution_request": "none",
            }
            agent = load_test_agent(
                root,
                workspace,
                implementation_responses=[json.dumps(valid_payload)],
            )
            agent.initialize()
            repeated_line = "generated repeated source fragment that should not be copied\n"
            raw = "not json\n" + (repeated_line * 80)
            agent.conversation.append("assistant", "IMPLEMENTATION_AGENT_RESPONSE:\n" + raw)

            payload = agent._extract_json_or_retry(
                raw,
                phase="IMPLEMENT_PLAN_STEP_PHASE",
                contract=IMPLEMENTATION_CONTRACT,
            )
            repair_prompt = agent.impl_client.calls[-1]["messages"][-1]["content"]
            full_model_context = "\n".join(message["content"] for message in agent.impl_client.calls[-1]["messages"])

            self.assertEqual(payload["plan_note"], "Recovered with a minimal payload.")
            self.assertIn("Previous response recovery note:", repair_prompt)
            self.assertIn("Discard that text instead of continuing it", repair_prompt)
            self.assertNotIn("Previous response tail for recovery:", repair_prompt)
            self.assertNotIn(repeated_line.strip(), repair_prompt)
            self.assertNotIn(repeated_line.strip(), full_model_context)
            active_transcript = "\n".join(turn.content for turn in agent.conversation.turns)
            self.assertIn("malformed implementation response omitted", active_transcript)
            self.assertIn("Original response length:", active_transcript)

    def test_json_repair_omits_reasoning_and_fake_tool_call_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            valid_payload = {
                "plan_note": "Recovered after protocol repair.",
                "files": [],
                "commands": [["python", "-c", "print('ok')"]],
                "test_evidence": [],
                "resolution_request": "none",
            }
            agent = load_test_agent(
                root,
                workspace,
                implementation_responses=[json.dumps(valid_payload)],
            )
            agent.initialize()
            raw = (
                "<think>private scratch text guessed the answer 42</think>\n"
                "<|tool_call>call:mcp_tools.shell_executor.execute_command("
                "{\"cmd\": \"python -c 'print(42)'\"})<tool_call|>"
            )
            agent.conversation.append("assistant", "IMPLEMENTATION_AGENT_RESPONSE:\n" + raw)

            payload = agent._extract_json_or_retry(
                raw,
                phase="IMPLEMENT_PLAN_STEP_PHASE",
                contract=IMPLEMENTATION_CONTRACT,
            )
            repair_prompt = agent.impl_client.calls[-1]["messages"][-1]["content"]
            full_model_context = "\n".join(message["content"] for message in agent.impl_client.calls[-1]["messages"])
            active_transcript = "\n".join(turn.content for turn in agent.conversation.turns)

            self.assertEqual(payload["plan_note"], "Recovered after protocol repair.")
            self.assertIn("Previous response recovery note:", repair_prompt)
            self.assertNotIn("Previous response tail for recovery:", repair_prompt)
            for forbidden in ("private scratch text", "guessed the answer 42", "<|tool_call>", "mcp_tools"):
                self.assertNotIn(forbidden, repair_prompt)
                self.assertNotIn(forbidden, full_model_context)
            self.assertIn("visible reasoning, chat-template markers, or fake tool-call syntax", active_transcript)

    def test_json_repair_omits_repetitive_requirements_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            valid_payload = {
                "project_summary": "Recovered requirements.",
                "refined_requirements": ["Create the requested script."],
                "assumptions": [],
                "open_questions": [],
                "planning_confirmation": {
                    "is_feasible": True,
                    "is_clear": True,
                    "is_verifiable": True,
                    "verification_strategy": "Run the requested script.",
                    "remaining_risks": [],
                },
                "plan": [
                    {
                        "id": "S1",
                        "title": "Create script",
                        "description": "Create and validate the requested script.",
                        "depends_on": [],
                        "acceptance_criteria": ["The script runs."],
                        "validation_commands": [["python", "script.py"]],
                    }
                ],
            }
            agent = load_test_agent(
                root,
                workspace,
                implementation_responses=[json.dumps(valid_payload)],
            )
            agent.initialize()
            repeated_line = "* Wait, the no unrequested documentation rule: I am not adding any.\n"
            raw = "<think>\n" + (repeated_line * 80)
            agent.conversation.append("assistant", "IMPLEMENTATION_AGENT_RESPONSE:\n" + raw)

            payload = agent._extract_json_or_retry(
                raw,
                phase="REQUIREMENTS_REFINEMENT_PHASE",
                contract=REQUIREMENTS_CONTRACT,
            )
            repair_prompt = agent.impl_client.calls[-1]["messages"][-1]["content"]
            full_model_context = "\n".join(message["content"] for message in agent.impl_client.calls[-1]["messages"])

            self.assertEqual(payload["project_summary"], "Recovered requirements.")
            self.assertIn("Previous response recovery note:", repair_prompt)
            self.assertNotIn("Previous response tail for recovery:", repair_prompt)
            self.assertNotIn(repeated_line.strip(), repair_prompt)
            self.assertNotIn(repeated_line.strip(), full_model_context)

    def test_workflow_state_marks_rejected_requirements_draft_unaccepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = {
                "project_summary": "Draft that incorrectly adds a public --script switch.",
                "refined_requirements": [
                    "validate_huge_output.py may accept --script for negative validation."
                ],
                "assumptions": [],
                "planning_confirmation": {
                    "is_feasible": True,
                    "is_clear": True,
                    "is_verifiable": True,
                    "verification_strategy": "Run with --script.",
                    "remaining_risks": [],
                },
            }
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Rejected draft",
                    "status": "pending",
                    "description": "Uses bad_producer.py and --script.",
                    "depends_on": [],
                    "acceptance_criteria": ["Validation uses --script."],
                    "validation_commands": [["python", "validate_huge_output.py", "--script", "huge_output.py"]],
                }
            ]
            agent._write_requirements_doc()
            agent._write_plan_doc()
            agent.last_requirements_review = {
                "status": "needs_requirements_change",
                "summary": "Public validation-only switch was invented.",
                "required_changes": ["Remove --script and use a temporary fixture without changing the public API."],
            }

            state = agent._workflow_state_for_prompt()

            self.assertIn("Latest requirements review:", state)
            self.assertIn("current PLAN/REQUIREMENTS files are unaccepted draft evidence", state)
            self.assertIn("Remove --script", state)
            self.assertIn("Unaccepted requirements draft summary", state)
            self.assertIn("[unaccepted draft omitted from pinned context", state)
            self.assertNotIn("bad_producer.py", state)

    def test_requirements_review_suppresses_unsupported_syntax_only_objection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            validator = (
                "n_range = range(1, 121); "
                "valid_n = [n for n in n_range if sum(1 for d in [3, 5, 7] if n % d == 0) == 1 "
                "and sum(int(digit) for digit in str(n)) % 2 != 0]; "
                "expected = sum(valid_n); actual = int(open('ANSWER.txt').read().strip()); "
                "assert actual == expected"
            )
            requirements = {
                "project_summary": "Compute an exact answer in ANSWER.txt.",
                "refined_requirements": [
                    "Create ANSWER.txt only.",
                    "Compare the artifact to an independently recomputed value.",
                ],
                "assumptions": [],
                "open_questions": [],
                "planning_confirmation": {
                    "is_feasible": True,
                    "is_clear": True,
                    "is_verifiable": True,
                    "verification_strategy": "Recompute the value and compare it to ANSWER.txt.",
                    "remaining_risks": [],
                },
                "plan": [{
                    "id": "S1",
                    "title": "Write and validate answer",
                    "description": "Write ANSWER.txt and verify it by recomputing the requested calculation.",
                    "depends_on": [],
                    "acceptance_criteria": ["ANSWER.txt contains the correct recomputed integer."],
                    "validation_commands": [["python", "-c", validator]],
                }],
            }
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Consider integers n from 1 to 120. Keep n if n is divisible "
                    "by exactly one of 3, 5, 7, and the digit sum of n is odd."
                ),
                feedback_responses=[
                    json.dumps({
                        "status": "needs_rework",
                        "needs_rework": True,
                        "summary": (
                            "The python -c command is too complex and generator expressions may be parser problems."
                        ),
                        "required_changes": [
                            "Replace it with a simpler single-line expression check."
                        ],
                    })
                ],
            )
            agent.initialize()

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "resolved")
            self.assertFalse(review["needs_rework"])
            self.assertIn("syntax-only reviewer objection was ignored", review["summary"])
            self.assertIn("suppressed_reviewer_findings", review)

    def test_plan_validation_suppresses_unsupported_syntax_only_objection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            validator = (
                "expected = sum(n for n in range(1, 121) if sum(1 for d in [3, 5, 7] if n % d == 0) == 1 "
                "and sum(int(c) for c in str(n)) % 2 == 1); "
                "actual = int(open('ANSWER.txt').read().strip()); "
                "assert actual == expected, f'expected={expected} actual={actual}'"
            )
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Consider integers n from 1 to 120. Keep n if n is divisible "
                    "by exactly one of 3, 5, 7, and the digit sum of n is odd."
                ),
                feedback_responses=[
                    json.dumps({
                        "status": "needs_plan_change",
                        "needs_rework": True,
                        "summary": "The python -c validator is too complex because it uses generator expressions.",
                        "required_changes": ["Simplify the validator syntax."],
                    })
                ],
            )
            agent.initialize()
            agent.requirements = base_requirements("Exact answer")
            agent.requirements["planning_confirmation"]["verification_strategy"] = (
                "Recompute the requested calculation and compare it to ANSWER.txt."
            )
            agent.plan_steps = [{
                "id": "S1",
                "title": "Write and validate answer",
                "description": "Write ANSWER.txt and verify it by recomputing the requested calculation.",
                "depends_on": [],
                "acceptance_criteria": ["ANSWER.txt contains the correct recomputed integer."],
                "validation_commands": [["python", "-c", validator]],
                "status": "pending",
            }]

            review = agent._plan_validation_review(1)

            self.assertEqual(review["status"], "resolved")
            self.assertFalse(review["needs_rework"])
            self.assertIn("syntax-only reviewer objection was ignored", review["summary"])

            agent.conversation.append(
                "user",
                "IMPLEMENTATION_AGENT_REQUEST:\nIMPLEMENT_PLAN_STEP_PHASE step_id=S1 attempt=1",
            )
            state = latest_control_state(agent.conversation.turns)
            self.assertIn("Last reviewer response: status=resolved, needs_rework=false", state)
            self.assertIn("syntax-only reviewer objection was ignored", state)
            self.assertNotIn("status=needs_plan_change", state)

    def test_default_policy_does_not_suppress_reviewer_prose_by_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            validator = (
                "expected = sum(n for n in range(1, 121)); "
                "actual = int(open('ANSWER.txt').read().strip()); "
                "assert actual == expected, f'expected={expected} actual={actual}'"
            )
            agent = load_test_agent(
                root,
                workspace,
                prompt="Create ANSWER.txt with a computed integer.",
                feedback_responses=[
                    json.dumps({
                        "status": "needs_plan_change",
                        "needs_rework": True,
                        "summary": "The python -c validator is too complex because it uses generator expressions.",
                        "required_changes": ["Simplify the validator syntax."],
                    })
                ],
            )
            agent.config = replace(
                agent.config,
                quality_policy=replace(
                    agent.config.quality_policy,
                    deterministic_semantic_scope_checks=False,
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Exact answer")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Write and validate answer",
                "description": "Write ANSWER.txt and verify it.",
                "depends_on": [],
                "acceptance_criteria": ["ANSWER.txt contains the requested integer."],
                "validation_commands": [["python", "-c", validator]],
                "status": "pending",
            }]

            review = agent._plan_validation_review(1)

            self.assertEqual(review["status"], "needs_plan_change")
            self.assertNotIn("suppressed_reviewer_findings", review)

    def test_default_plan_prompt_checks_exclude_legacy_phrase_scope_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.config = replace(
                agent.config,
                quality_policy=replace(
                    agent.config.quality_policy,
                    deterministic_semantic_scope_checks=False,
                ),
            )

            checks = "\n".join(agent._plan_validation_prompt_checks())

            self.assertIn("validation evidence proves the requested behavior", checks)
            self.assertNotIn("computed-answer artifact tasks", checks)
            self.assertNotIn("artifact-only prompts", checks)
            self.assertNotIn("named scripts keep", checks)
            self.assertNotIn("caller-visible representation constraints", checks)

    def test_plan_validation_short_circuits_deterministic_findings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Consider integers n from 1 to 120. "
                    "Return the sum of kept n as a single integer."
                ),
                feedback_responses=[
                    json.dumps({
                        "status": "needs_plan_change",
                        "needs_rework": True,
                        "summary": "Do not let this exact command leak into the directive.",
                        "required_changes": [
                            "Use this full replacement: [\"python\", \"-c\", \"print('recipe')\"]"
                        ],
                    })
                ],
            )
            agent.initialize()
            agent.requirements = base_requirements("Exact answer")
            agent.requirements["planning_confirmation"]["verification_strategy"] = (
                "Recompute the requested calculation and compare it to ANSWER.txt."
            )
            agent.plan_steps = [{
                "id": "S1",
                "title": "Write and validate answer",
                "description": "Write ANSWER.txt and verify it by recomputing the requested calculation.",
                "depends_on": [],
                "acceptance_criteria": ["ANSWER.txt contains the correct recomputed integer."],
                "validation_commands": [[
                    "python",
                    "-c",
                    "expected = 1878; actual = int(open('ANSWER.txt').read().strip()); "
                    "import sys; sys.exit(0 if expected == actual else 1)",
                ]],
                "status": "pending",
            }]

            review = agent._plan_validation_review(1)

            self.assertEqual(review["status"], "needs_plan_change")
            self.assertIn("Deterministic plan checks", review["summary"])
            self.assertIn("diagnostic output", "\n".join(review["required_changes"]))
            self.assertNotIn("full replacement", "\n".join(review["required_changes"]))
            self.assertEqual(agent.feedback_client.calls, [])

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

    def test_command_timeout_can_be_left_to_progress_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            started = time.monotonic()

            results = run_commands(
                root,
                [{"cmd": ["bash", "-c", "echo open-ended; sleep 60"], "timeout_seconds": 0}],
                timeout_seconds=1,
                max_timeout_seconds=1,
                progress_callback=lambda snapshot: {
                    "decision": "terminate",
                    "summary": "The test progress reviewer owns this open-ended command.",
                    "evidence": [
                        f"hard_timeout_disabled={snapshot.get('hard_timeout_disabled')}",
                        snapshot.get("stdout", ""),
                    ],
                    "next_check_seconds": 1,
                },
                progress_interval_seconds=1,
                progress_min_interval_seconds=1,
            )
            elapsed = time.monotonic() - started

            self.assertLess(elapsed, 8)
            self.assertFalse(results[0]["timed_out"])
            self.assertIsNone(results[0]["timeout_seconds"])
            self.assertTrue(results[0]["hard_timeout_disabled"])
            self.assertTrue(results[0]["stopped_by_progress_review"])
            self.assertEqual(results[0]["returncode"], 125)
            self.assertIn("open-ended", results[0]["stdout"])
            self.assertIn("hard_timeout_disabled=True", results[0]["progress_reviews"][0]["evidence"][0])

    def test_positive_command_timeout_is_unclamped_when_max_timeout_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = run_commands(
                root,
                [{"cmd": ["python", "-c", "print('unclamped')"], "timeout_seconds": 999}],
                timeout_seconds=1,
                max_timeout_seconds=0,
            )

            self.assertEqual(results[0]["returncode"], 0)
            self.assertEqual(results[0]["timeout_seconds"], 999)
            self.assertIn("unclamped", results[0]["stdout"])

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

    def test_running_command_can_be_stopped_by_progress_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            started = time.monotonic()

            results = run_commands(
                root,
                [["bash", "-c", "echo waiting; sleep 60"]],
                timeout_seconds=30,
                max_timeout_seconds=30,
                progress_callback=lambda snapshot: {
                    "decision": "terminate",
                    "summary": "Command is waiting in a test-controlled hopeless state.",
                    "evidence": [snapshot["stdout"]],
                    "next_check_seconds": 1,
                },
                progress_interval_seconds=1,
                progress_min_interval_seconds=1,
            )
            elapsed = time.monotonic() - started

            self.assertLess(elapsed, 8)
            self.assertFalse(results[0]["timed_out"])
            self.assertTrue(results[0]["stopped_by_progress_review"])
            self.assertEqual(results[0]["returncode"], 125)
            self.assertIn("waiting", results[0]["stdout"])
            self.assertIn("command stopped by progress review", results[0]["stderr"])
            self.assertEqual(results[0]["progress_reviews"][0]["decision"], "terminate")

    def test_agent_progress_reviewer_uses_workflow_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt="Build a checked utility without hanging.",
                feedback_responses=[
                    json.dumps({
                        "status": "terminate",
                        "decision": "terminate",
                        "summary": "The command is waiting for unavailable input.",
                        "evidence": ["stderr asks for input"],
                        "risks": ["Continuing will not produce useful validation evidence."],
                        "next_check_seconds": 5,
                    })
                ],
            )
            agent.initialize()
            agent.requirements = base_requirements()
            step = {
                "id": "S1",
                "title": "Validate utility",
                "description": "Run a bounded validation command.",
                "acceptance_criteria": ["The command terminates with useful evidence."],
                "validation_commands": [],
                "status": "pending",
            }
            agent.plan_steps = [step]

            callback = agent._running_tool_progress_reviewer(
                source="implementation",
                context={"step": step, "purpose": "unit test progress review"},
            )
            assert callback is not None
            decision = callback({
                "command_index": 0,
                "command": ["python", "waits.py"],
                "cwd": str(workspace),
                "elapsed_seconds": 120,
                "timeout_seconds": 300,
                "review_count": 1,
                "returncode": None,
                "stdout": "",
                "stderr": "waiting for input",
                "stdout_bytes": 0,
                "stderr_bytes": 17,
                "stdout_truncated": False,
                "stderr_truncated": False,
            })

            self.assertEqual(decision["decision"], "terminate")
            self.assertIn("running_command", decision)
            self.assertEqual(decision["running_command"]["elapsed_seconds"], 120)
            self.assertEqual(decision["running_command"]["stderr_excerpt"], "waiting for input")
            call_text = json.dumps(agent.feedback_client.calls[-1]["messages"])
            self.assertIn("TOOL_PROGRESS_REVIEW_PHASE", call_text)
            self.assertIn("Build a checked utility", call_text)
            self.assertIn("waiting for input", call_text)
            transcript = (workspace / ".agent_state" / "conversation.full.jsonl").read_text(encoding="utf-8")
            self.assertIn("TOOL_PROGRESS_REVIEW_RESULT", transcript)
            self.assertIn("waiting for input", transcript)
            self.assertIn("running_command", transcript)

    def test_command_timeout_kills_background_child_after_shell_exits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            started = time.monotonic()
            results = run_commands(
                root,
                [[
                    "bash",
                    "-c",
                    "sleep 60 & echo $! > child.pid; exit 0",
                ]],
                timeout_seconds=1,
                max_timeout_seconds=1,
            )
            elapsed = time.monotonic() - started

            self.assertLess(elapsed, 4)
            self.assertTrue(results[0]["timed_out"])
            self.assertEqual(results[0]["returncode"], 124)
            child_pid = (root / "child.pid").read_text(encoding="utf-8").strip()
            time.sleep(0.2)
            ps = subprocess.run(["ps", "-p", child_pid], capture_output=True, text=True)
            self.assertNotEqual(ps.returncode, 0)

    def test_command_completion_cleans_background_children(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = run_commands(
                root,
                [[
                    "bash",
                    "-c",
                    "sleep 60 >/dev/null 2>&1 & echo $! > child.pid; exit 7",
                ]],
                timeout_seconds=30,
                max_timeout_seconds=30,
            )

            self.assertFalse(results[0]["timed_out"])
            self.assertEqual(results[0]["returncode"], 7)
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

    def test_deterministic_compaction_omits_nested_memory_and_prompt_contracts(self) -> None:
        text = _source_for_compaction([
            Turn(
                "system",
                "Compacted durable memory from earlier turns. Preserve these decisions.\n\n"
                "INITIAL_REQUEST_CONTEXT:\nuser: PROJECT DESIGN: old task\n\n"
                "AUTHORITATIVE_RECENT_CONTROL_STATE:\n- stale control state",
            ),
            Turn("user", "PROJECT DESIGN: current task\n\nCreate ANSWER.txt only."),
            Turn(
                "user",
                "IMPLEMENTATION_AGENT_REQUEST:\nPROBLEM_ANALYSIS_PHASE iteration=1\n"
                "Return strict JSON only:\n{\n"
                '"problem_restatement": "concise restatement of the user request"\n'
                "}\nARTIFACT_ONLY_CONSTRAINT:\nrepeated generated prompt text",
            ),
            Turn("assistant", "IMPLEMENTATION_AGENT_RESPONSE:\n{\"problem_restatement\":\"current task\"}"),
        ])

        compacted = deterministic_compact(text)

        self.assertIn("PROJECT DESIGN: current task", compacted)
        self.assertIn("IMPLEMENTATION_AGENT_REQUEST: PROBLEM_ANALYSIS_PHASE iteration=1", compacted)
        self.assertIn("generated harness prompt omitted", compacted)
        self.assertNotIn("INITIAL_REQUEST_CONTEXT", compacted)
        self.assertNotIn("Return strict JSON only", compacted)
        self.assertNotIn("concise restatement", compacted)
        self.assertNotIn("stale control state", compacted)

    def test_deterministic_compaction_summarizes_rejected_payloads_without_raw_commands(self) -> None:
        text = _source_for_compaction([
            Turn("user", "PROJECT DESIGN: exact artifact\n\nCreate ANSWER.txt only."),
            Turn(
                "assistant",
                "IMPLEMENTATION_AGENT_RESPONSE:\n"
                + json.dumps({
                    "project_summary": "Old rejected requirements.",
                    "plan": [
                        {
                            "id": "S1",
                            "validation_commands": [
                                [
                                    "bash",
                                    "-lc",
                                    "python -c 'open(\"ANSWER.txt\", \"w\").write(\"bad\")'",
                                ]
                            ],
                        }
                    ],
                }),
            ),
            Turn(
                "user",
                "FEEDBACK_AGENT_RESPONSE:\n"
                + json.dumps({
                    "status": "needs_rework",
                    "needs_rework": True,
                    "summary": "Validation wrote the requested artifact; fix the plan.",
                }),
            ),
            Turn(
                "assistant",
                "IMPLEMENTATION_AGENT_RESPONSE:\n"
                + json.dumps({
                    "plan_note": "Corrected artifact-only implementation.",
                    "files": [{"path": "ANSWER.txt", "content": "1878"}],
                    "commands": [["bash", "-lc", "test \"$(cat ANSWER.txt)\" = 1878"]],
                    "resolution_request": "none",
                }),
            ),
        ])

        compacted = deterministic_compact(text)

        self.assertIn("PROJECT DESIGN: exact artifact", compacted)
        self.assertIn("Feedback response: status=needs_rework", compacted)
        self.assertIn("Validation wrote the requested artifact", compacted)
        self.assertIn("Implementation response: present; raw payload omitted", compacted)
        self.assertIn("Implementation response: plan_note=Corrected artifact-only implementation", compacted)
        self.assertIn("files=ANSWER.txt", compacted)
        self.assertNotIn("open(\"ANSWER.txt\", \"w\")", compacted)
        self.assertNotIn("test \"$(cat ANSWER.txt)\"", compacted)
        self.assertNotIn('"commands"', compacted)

    def test_initial_request_context_recovers_project_design_from_compacted_memory(self) -> None:
        turns = [
            Turn(
                "system",
                "Compacted durable memory from earlier turns. Preserve these decisions.\n\n"
                "INITIAL_REQUEST_CONTEXT:\n"
                "system: old system prompt\n\n"
                "user: PROJECT DESIGN: retained task\n\nBuild alpha.\n\n"
                "AUTHORITATIVE_RECENT_CONTROL_STATE:\n- step pending",
            ),
            Turn(
                "user",
                "IMPLEMENTATION_AGENT_REQUEST:\nPROBLEM_ANALYSIS_PHASE iteration=1\nReturn strict JSON only:",
            ),
        ]

        context = initial_request_context(turns)

        self.assertIn("PROJECT DESIGN: retained task", context)
        self.assertIn("Build alpha", context)
        self.assertNotIn("IMPLEMENTATION_AGENT_REQUEST", context)
        self.assertNotIn("AUTHORITATIVE_RECENT_CONTROL_STATE", context)

    def test_initial_request_context_from_nested_memory_stops_before_summaries(self) -> None:
        turns = [
            Turn(
                "system",
                "Compacted durable memory from earlier turns. Preserve these decisions.\n\n"
                "INITIAL_REQUEST_CONTEXT:\n"
                "user: PROJECT DESIGN: Nested parity puzzle\n\n"
                "Create ANSWER.txt only. Return the required integer.\n\n"
                "COMPACTED_WORKFLOW_MEMORY:\n"
                "**Initial User Request**\n"
                "Create `ANSWER.txt` containing the sum.\n\n"
                "**Requirements**\n"
                "- Generated summary that should not become initial context.\n\n"
                "Deterministic fallback compaction was used because model compaction failed.\n"
                "Important early context:\n"
                "user: FEEDBACK_AGENT_REQUEST: PROBLEM_ANALYSIS_REVIEW_PHASE\n"
                "[generated harness prompt omitted from compaction source]\n\n"
                "AUTHORITATIVE_RECENT_CONTROL_STATE:\n"
                "- step pending",
            ),
        ]

        context = initial_request_context(turns)

        self.assertIn("PROJECT DESIGN: Nested parity puzzle", context)
        self.assertIn("Create ANSWER.txt only", context)
        self.assertNotIn("COMPACTED_WORKFLOW_MEMORY", context)
        self.assertNotIn("Initial User Request", context)
        self.assertNotIn("Generated summary", context)
        self.assertNotIn("Deterministic fallback compaction", context)
        self.assertNotIn("FEEDBACK_AGENT_REQUEST", context)
        self.assertNotIn("AUTHORITATIVE_RECENT_CONTROL_STATE", context)

    def test_deterministic_compaction_does_not_duplicate_short_context(self) -> None:
        compacted = deterministic_compact(
            "system: Previous compacted-memory block omitted; fresh context appended separately.\n"
            "user: first useful line\n"
            "assistant: second useful line\n"
            "user: third useful line"
        )

        self.assertEqual(compacted.count("user: first useful line"), 1)
        self.assertEqual(compacted.count("assistant: second useful line"), 1)
        self.assertNotIn("Previous compacted-memory block omitted", compacted)

    def test_compaction_recent_tail_respects_token_budget(self) -> None:
        turns = [
            Turn("user", "older context " * 2000),
            Turn("assistant", "large tool evidence " * 4000),
            Turn("user", "latest request"),
            Turn("assistant", "latest response"),
        ]

        keep = _bounded_recent_turn_count(turns, max_turns=4, max_tokens=20)

        self.assertEqual(keep, 2)

    def test_compaction_uses_hard_token_ceiling_below_context_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "work"
            cfg_path = write_config(root, workspace, "compact ceiling", "Build anything.")
            config = load_config(cfg_path)
            conversation = Conversation(root / "conversation.jsonl")
            conversation.append("system", "durable system prompt")
            conversation.append("user", "PROJECT DESIGN: compact ceiling\n\nBuild anything.")
            conversation.append("assistant", "old verbose evidence " * 9000)
            conversation.append("user", "latest request")
            compactor = ScriptedClient(["Requirement: preserve compacted project context and latest request."])

            compacted = maybe_compact(
                conversation,
                config,
                compactor,
                context_window=131072,
                incoming_tokens=1,
                pinned_context="Pinned state",
            )

            self.assertTrue(compacted)
            self.assertIn("Compacted durable memory", (root / "conversation.jsonl").read_text(encoding="utf-8"))

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
            self.assertNotIn("S1 is complete. All validation passed.", active_text)
            self.assertIn("PINNED_WORKFLOW_STATE", active_text)
            self.assertIn("S1 still pending", active_text)

    def test_compaction_rejects_success_summary_that_conflicts_with_control_state(self) -> None:
        control_state = (
            "AUTHORITATIVE_RECENT_CONTROL_STATE:\n"
            "- Current implementation request: step_id=S1 attempt=2.\n"
            "- Last reviewer response: status=needs_rework, needs_rework=true.\n"
            "- Reviewer summary: Validation failed: expected 1878 but got 2110."
        )

        self.assertTrue(
            _compaction_memory_conflicts_with_control_state(
                "S1 is complete. All validation passed. The answer was confirmed correct.",
                control_state,
            )
        )
        self.assertFalse(
            _compaction_memory_conflicts_with_control_state(
                "S1 implementation claimed success, but reviewer validation failed with a mismatch and needs rework.",
                control_state,
            )
        )

    def test_control_state_prefers_requirements_rework_directive_over_stale_resolved_feedback(self) -> None:
        turns = [
            Turn("user", "PROJECT DESIGN: exact artifact\n\nCreate ANSWER.txt only."),
            Turn(
                "user",
                'FEEDBACK_AGENT_RESPONSE:\n{"status":"resolved","needs_rework":false,'
                '"summary":"Requirements look complete."}',
            ),
            Turn(
                "user",
                "REQUIREMENTS_REWORK_DIRECTIVE:\nRevise requirements using this review:\n"
                '{"status":"needs_requirements_change","needs_rework":true,'
                '"summary":"Deterministic requirements checks found invalid validation commands."}',
            ),
            Turn(
                "user",
                "IMPLEMENTATION_AGENT_REQUEST:\nREQUIREMENTS_REFINEMENT_PHASE iteration=2\nRetry requirements.",
            ),
        ]

        state = latest_control_state(turns)

        self.assertIn("Last requirements rework directive", state)
        self.assertIn("status=needs_requirements_change", state)
        self.assertIn("needs_rework=true", state)
        self.assertIn("invalid validation commands", state)
        self.assertNotIn("Requirements look complete", state)

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
            self.assertIn("COMPACTED_WORKFLOW_MEMORY", active_text)
            self.assertIn("Build a very specific artifact named alpha", active_text)
            self.assertLess(active_text.index("INITIAL_REQUEST_CONTEXT"), active_text.index("COMPACTED_WORKFLOW_MEMORY"))

    def test_compaction_omits_raw_system_prompt_from_initial_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "work"
            cfg_path = write_config(root, workspace, "system prompt omission", "Build alpha.")
            config = load_config(cfg_path)
            conversation = Conversation(root / "conversation.jsonl")
            conversation.append(
                "system",
                "You are a generated harness control prompt. " + ("Do not preserve this. " * 300),
            )
            conversation.append("user", "PROJECT DESIGN: system prompt omission\n\nBuild alpha.")
            conversation.append("assistant", "older evidence")
            bad_compactor = ScriptedClient(["ok"])

            context = initial_request_context(conversation.turns)
            maybe_compact(
                conversation,
                config,
                bad_compactor,
                context_window=100,
                incoming_tokens=1000,
                force=True,
            )

            active_text = (root / "conversation.jsonl").read_text(encoding="utf-8")
            self.assertIn("PROJECT DESIGN: system prompt omission", context)
            self.assertNotIn("generated harness control prompt", context)
            self.assertIn("PROJECT DESIGN: system prompt omission", active_text)
            self.assertNotIn("generated harness control prompt", active_text)

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
            self.assertFalse(cfg.quality_policy.deterministic_semantic_scope_checks)
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
            self.assertEqual(agent.feedback_client.calls, [])

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
            self.assertEqual(agent.feedback_client.calls, [])

    def test_tool_call_verifier_blocks_invalid_inline_python_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()

            results = agent._run_verified_commands(
                [[
                    "python3",
                    "-c",
                    "import itertools; count=sum(1 for item in itertools.product('AB', repeat=2); assert count == 4",
                ]],
                source="unit_test",
                context={"purpose": "prove invalid inline Python is blocked"},
            )

            self.assertEqual(results[0]["returncode"], 126)
            self.assertTrue(results[0]["blocked_by_tool_verifier"])
            self.assertIn("static syntax check", results[0]["stderr"])
            self.assertEqual(agent.feedback_client.calls, [])

    def test_tool_call_verifier_blocks_inline_python_unreachable_after_return(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()

            results = agent._run_verified_commands(
                [[
                    "python3",
                    "-c",
                    "def check(): return True; open('ANSWER.txt','w').write('1'); check()",
                ]],
                source="unit_test",
                context={"purpose": "prove unreachable inline Python proof work is blocked"},
            )

            self.assertEqual(results[0]["returncode"], 126)
            self.assertTrue(results[0]["blocked_by_tool_verifier"])
            self.assertIn("unreachable statements after return", results[0]["stderr"])
            self.assertEqual(agent.feedback_client.calls, [])

    def test_tool_call_verifier_blocks_malformed_quoted_heredoc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()

            results = agent._run_verified_commands(
                [["bash", "-lc", "python3 - <<'PY'\nprint('ok')\n'PY"]],
                source="unit_test",
                context={"purpose": "prove malformed here-docs are blocked"},
            )

            self.assertEqual(results[0]["returncode"], 126)
            self.assertTrue(results[0]["blocked_by_tool_verifier"])
            self.assertIn("here-doc", results[0]["stderr"])
            self.assertIn("quoted delimiter", results[0]["stderr"])
            self.assertEqual(agent.feedback_client.calls, [])

    def test_tool_call_verifier_blocks_artifact_only_heredoc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, prompt="Create ANSWER.txt only. Return the integer answer.")
            agent.initialize()

            results = agent._run_verified_commands(
                [["bash", "-lc", "python3 - <<'PY'\nprint('ok')\nPY"]],
                source="step_feedback_validation",
                context={"purpose": "artifact-only validation should stay one-line or temporary"},
            )

            self.assertEqual(results[0]["returncode"], 126)
            self.assertTrue(results[0]["blocked_by_tool_verifier"])
            self.assertIn("artifact-only", results[0]["stderr"])
            self.assertIn("shell here-doc", results[0]["stderr"])
            self.assertEqual(agent.feedback_client.calls, [])

    def test_tool_call_verifier_blocks_metadata_inside_argv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()

            results = agent._run_verified_commands(
                [["bash", "-lc", "true", "expected_returncode", "0"]],
                source="unit_test",
                context={"purpose": "prove command metadata belongs outside argv"},
            )

            self.assertEqual(results[0]["returncode"], 126)
            self.assertTrue(results[0]["blocked_by_tool_verifier"])
            self.assertIn("inside the argv list", results[0]["stderr"])
            self.assertEqual(agent.feedback_client.calls, [])

    def test_tool_call_verifier_blocks_open_ended_command_without_progress_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.config = replace(
                agent.config,
                runtime=replace(agent.config.runtime, command_progress_review_interval_seconds=0),
            )
            agent.initialize()

            results = agent._run_verified_commands(
                [{"cmd": ["bash", "-lc", "echo waiting; sleep 60"], "timeout_seconds": 0}],
                source="implementation",
                context={"purpose": "open-ended command without progress review should not run"},
            )

            self.assertEqual(results[0]["returncode"], 126)
            self.assertTrue(results[0]["blocked_by_tool_verifier"])
            self.assertIn("progress review is disabled", results[0]["stderr"])
            self.assertEqual(agent.feedback_client.calls, [])

    def test_tool_call_verifier_blocks_artifact_mutating_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, prompt="Create ANSWER.txt only. Return the integer answer.")
            agent.initialize()

            results = agent._run_verified_commands(
                [["python", "-c", "open('ANSWER.txt', 'w').write('42')"]],
                source="implementation",
                context={"purpose": "artifact-only implementation command"},
            )

            self.assertEqual(results[0]["returncode"], 126)
            self.assertTrue(results[0]["blocked_by_tool_verifier"])
            self.assertIn("explicitly requested artifact", results[0]["stderr"])
            self.assertIn("files", results[0]["stderr"])
            self.assertEqual(agent.feedback_client.calls, [])

    def test_tool_call_verifier_blocks_workspace_source_mutating_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()

            results = agent._run_verified_commands(
                [[
                    "bash",
                    "-lc",
                    "cp generator.py generator.py.bak; echo 'print(\"wrong\")' > generator.py; python validate.py; mv generator.py.bak generator.py",
                ]],
                source="implementation",
                context={"purpose": "negative-path validation should not mutate project source"},
            )

            self.assertEqual(results[0]["returncode"], 126)
            self.assertTrue(results[0]["blocked_by_tool_verifier"])
            self.assertIn("workspace source path `generator.py`", results[0]["stderr"])
            self.assertIn("do not infer that direct executability is required", results[0]["stderr"])
            self.assertEqual(agent.feedback_client.calls, [])

    def test_tool_call_verifier_blocks_swallowed_expected_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()

            results = agent._run_verified_commands(
                [["bash", "-lc", "python validate_cli.py --bad-input || exit 0"]],
                source="implementation",
                context={"purpose": "negative-path validation must prove failure"},
            )

            self.assertEqual(results[0]["returncode"], 126)
            self.assertTrue(results[0]["blocked_by_tool_verifier"])
            self.assertIn("mask a validation failure", results[0]["stderr"])
            self.assertEqual(agent.feedback_client.calls, [])

    def test_tool_call_verifier_blocks_off_contract_json_without_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    json.dumps({
                        "plan_note": "wrong schema",
                        "commands": [{"cmd": ["bash", "-lc", "echo ok"]}],
                    })
                ],
            )
            agent.initialize()

            results = agent._run_verified_commands(
                [["python", "-c", "print('should not run')"]],
                source="implementation",
                context={"purpose": "verify malformed verifier responses are conservative"},
            )

            self.assertEqual(results[0]["returncode"], 126)
            self.assertTrue(results[0]["blocked_by_tool_verifier"])
            self.assertIn("malformed after JSON repair", results[0]["stderr"])
            self.assertEqual(len(agent.feedback_client.calls), 3)

    def test_tool_call_verifier_allows_missing_status_with_explicit_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    json.dumps({
                        "summary": "Command is bounded.",
                        "commands": [
                            {
                                "index": 0,
                                "decision": "approved",
                                "risk_level": "low",
                                "reason": "Explicit per-command approval.",
                            }
                        ],
                    }),
                    json.dumps({
                        "status": "approved",
                        "summary": "Protocol repair supplied the missing status.",
                        "commands": [
                            {
                                "index": 0,
                                "decision": "approved",
                                "risk_level": "low",
                                "reason": "Explicit per-command approval after protocol repair.",
                            }
                        ],
                    })
                ],
            )
            agent.initialize()

            results = agent._run_verified_commands(
                [["python", "-c", "print('ok')"]],
                source="implementation",
                context={"purpose": "verify explicit command decision"},
            )

            self.assertEqual(results[0]["returncode"], 0)
            self.assertIn("ok", results[0]["stdout"])
            self.assertFalse(results[0].get("blocked_by_tool_verifier", False))
            self.assertEqual(len(agent.feedback_client.calls), 2)

    def test_tool_call_verifier_blocks_incomplete_approved_decision_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    json.dumps({
                        "status": "approved",
                        "summary": "Both commands look safe.",
                        "commands": [
                            {
                                "index": 0,
                                "decision": "approved",
                                "risk_level": "low",
                                "reason": "Only the first supplied command was reviewed.",
                            }
                        ],
                    })
                ],
            )
            agent.initialize()

            results = agent._run_verified_commands(
                [
                    ["python", "-c", "print('first')"],
                    ["python", "-c", "print('second')"],
                ],
                source="implementation",
                context={"purpose": "verify incomplete approved verifier output is conservative"},
            )

            self.assertEqual([result["returncode"] for result in results], [126, 126])
            self.assertTrue(all(result["blocked_by_tool_verifier"] for result in results))
            self.assertIn("omitted explicit decisions", results[0]["stderr"])
            self.assertIn("omitted explicit decisions", results[1]["stderr"])
            self.assertEqual(len(agent.feedback_client.calls), 1)

    def test_tool_call_verifier_requires_current_decision_for_repeated_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            command = ["python", "-c", "print('ok')"]
            approval = {
                "status": "approved",
                "summary": "The command is safe and bounded.",
                "commands": [
                    {
                        "index": 0,
                        "decision": "approved",
                        "risk_level": "low",
                        "reason": "Runs a bounded Python print command.",
                    }
                ],
            }
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    json.dumps(approval),
                    json.dumps(approval),
                    json.dumps(approval),
                ],
            )
            agent.initialize()

            first = agent._run_verified_commands(
                [command],
                source="implementation",
                context={"purpose": "first validation run"},
            )
            second = agent._run_verified_commands(
                [command],
                source="final_feedback_validation",
                context={"purpose": "repeat validation run"},
            )
            third = agent._run_verified_commands(
                [command],
                source="final_feedback_validation",
                context={"purpose": "repeat validation run again"},
            )

            self.assertEqual(first[0]["returncode"], 0)
            self.assertEqual(second[0]["returncode"], 0)
            self.assertEqual(third[0]["returncode"], 0)
            self.assertIn("ok", first[0]["stdout"])
            self.assertIn("ok", second[0]["stdout"])
            self.assertIn("ok", third[0]["stdout"])
            self.assertFalse(second[0]["tool_verification"].get("reason", "").startswith("Reused prior"))
            self.assertEqual(len(agent.feedback_client.calls), 3)

    def test_tool_call_verifier_allows_weak_documentation_content_command_for_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            step = {
                "id": "S1",
                "title": "Create documentation",
                "description": "Create README.md and design_notes.md.",
                "depends_on": [],
                "acceptance_criteria": [
                    "README.md exists and contains usage instructions.",
                    "design_notes.md exists and explains the implementation approach.",
                    "Files are not empty.",
                ],
                "validation_commands": [],
                "status": "pending",
            }
            agent = load_test_agent(root, workspace)
            agent.initialize()
            (workspace / "README.md").write_text("Usage\n", encoding="utf-8")
            (workspace / "design_notes.md").write_text("placeholder\n", encoding="utf-8")

            results = agent._run_verified_commands(
                [{
                    "cmd": [
                        "bash",
                        "-lc",
                        "test -f README.md && test -f design_notes.md && grep -q 'Usage' README.md",
                    ],
                    "expected_returncode": 0,
                }],
                source="implementation",
                context={"step": step, "purpose": "documentation validation"},
            )

            self.assertEqual(results[0]["returncode"], 0)
            self.assertFalse(results[0].get("blocked_by_tool_verifier", False))
            self.assertEqual(len(agent.feedback_client.calls), 1)

    def test_reviewer_validation_runs_even_when_documentation_content_check_is_shallow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            step = {
                "id": "S1",
                "title": "Fix bug and document it",
                "description": "Fix calc.py and create BUGFIX_NOTES.md explaining the fix.",
                "depends_on": [],
                "acceptance_criteria": [
                    "BUGFIX_NOTES.md documents the identified bug and applied fix.",
                    "The unittest suite passes.",
                ],
                "validation_commands": [],
                "status": "pending",
            }
            agent = load_test_agent(root, workspace)
            agent.initialize()
            (workspace / "BUGFIX_NOTES.md").write_text("Fixed median even-length calculation.\n", encoding="utf-8")

            results = agent._run_verified_commands(
                [["test", "-f", "BUGFIX_NOTES.md"]],
                source="step_feedback_validation",
                context={"step": step, "purpose": "reviewer-owned validation commands for the current plan step"},
            )

            self.assertEqual(results[0]["returncode"], 0)
            self.assertFalse(results[0].get("blocked_by_tool_verifier", False))
            self.assertEqual(len(agent.feedback_client.calls), 1)

    def test_tool_call_verifier_allows_discovery_when_documentation_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            step = {
                "id": "S1",
                "title": "Fix bug and document it",
                "description": "Inspect the project, fix the bug, and create BUGFIX_NOTES.md explaining the fix.",
                "depends_on": [],
                "acceptance_criteria": [
                    "BUGFIX_NOTES.md documents the identified bug and applied fix.",
                    "The unittest suite passes.",
                ],
                "validation_commands": [],
                "status": "pending",
            }
            agent = load_test_agent(root, workspace)
            agent.initialize()

            results = agent._run_verified_commands(
                [["ls", "-R"]],
                source="implementation",
                context={"step": step, "purpose": "discover existing files before editing"},
            )

            self.assertEqual(results[0]["returncode"], 0)
            self.assertFalse(results[0].get("blocked_by_tool_verifier", False))

    def test_tool_call_verifier_blocks_only_unsafe_commands_in_mixed_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()

            results = agent._run_verified_commands(
                [
                    ["python", "-c", "print('ok')"],
                    [
                        "bash",
                        "-lc",
                        "printf 'print(\"wrong\")\\n' > generator.py",
                    ],
                ],
                source="implementation",
                context={"purpose": "safe validation plus invalid negative-path source mutation"},
            )

            self.assertEqual(results[0]["returncode"], 0)
            self.assertIn("ok", results[0]["stdout"])
            self.assertFalse(results[0].get("blocked_by_tool_verifier", False))
            self.assertEqual(results[1]["returncode"], 126)
            self.assertTrue(results[1]["blocked_by_tool_verifier"])
            self.assertIn("workspace source path `generator.py`", results[1]["stderr"])
            self.assertEqual(len(agent.feedback_client.calls), 1)

    def test_tool_call_verifier_allows_tmp_fixture_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()

            results = agent._run_verified_commands(
                [[
                    "bash",
                    "-lc",
                    "printf '%s\n' 'print(123)' > /tmp/agentic_fixture.py && python /tmp/agentic_fixture.py",
                ]],
                source="implementation",
                context={"purpose": "temporary fixture validation outside the workspace"},
            )

            self.assertEqual(results[0]["returncode"], 0)
            self.assertFalse(results[0].get("blocked_by_tool_verifier", False))
            self.assertIn("123", results[0]["stdout"])

    def test_tool_call_verifier_blocks_unwrapped_git_diff_no_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()

            results = agent._run_verified_commands(
                [[
                    "bash",
                    "-lc",
                    "python -m pytest tests/test_slugify.py && git diff --no-index /dev/null tests/test_slugify.py",
                ]],
                source="implementation",
                context={"purpose": "prove unwrapped diff evidence is blocked"},
            )

            self.assertEqual(results[0]["returncode"], 126)
            self.assertTrue(results[0]["blocked_by_tool_verifier"])
            self.assertIn("git diff --no-index", results[0]["stderr"])
            self.assertEqual(agent.feedback_client.calls, [])

    def test_git_diff_no_index_with_expected_returncode_can_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            (workspace / "example.txt").write_text("changed\n", encoding="utf-8")

            results = agent._run_verified_commands(
                [{"cmd": ["git", "diff", "--no-index", "/dev/null", "example.txt"], "expected_returncode": 1}],
                source="implementation",
                context={"purpose": "collect standalone diff evidence"},
            )

            self.assertEqual(results[0]["returncode"], 1)
            self.assertTrue(results[0]["returncode_matches_expected"])
            self.assertFalse(results[0].get("blocked_by_tool_verifier", False))
            self.assertEqual(len(agent.feedback_client.calls), 1)

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

    def test_visible_reasoning_is_not_reused_as_durable_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            impl = ScriptedClient(["<think>secret answer scratchpad 24</think>\n{\"note\":\"implementation turn\"}"])
            reviewer = ScriptedClient()
            cfg = load_config(write_config(root, workspace, "chat continuity", "Build anything."), repo_root=root)
            agent = FeedbackLoopAgent(cfg, implementation_client=impl, feedback_client=reviewer)
            agent.initialize()

            raw = agent._implementation_chat("Create the first file.")
            agent._feedback_chat("Review the first file.")

            feedback_context = "\n".join(message["content"] for message in reviewer.calls[0]["messages"])
            transcript = (workspace / ".agent_state" / "conversation.jsonl").read_text(encoding="utf-8")
            self.assertIn("secret answer scratchpad", raw)
            self.assertIn("visible reasoning omitted", feedback_context)
            self.assertNotIn("secret answer scratchpad", feedback_context)
            self.assertNotIn("secret answer scratchpad", transcript)
            self.assertIn("implementation turn", transcript)

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

    def test_plan_validation_rejects_documentation_content_existence_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, title="docs", prompt="Create README and design notes.")
            agent.initialize()
            agent.requirements = base_requirements("Documentation")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Create documentation",
                    "description": "Create README.md and design_notes.md.",
                    "depends_on": [],
                    "acceptance_criteria": [
                        "README.md exists and contains usage instructions.",
                        "design_notes.md exists and explains the implementation approach.",
                        "Files are not empty.",
                    ],
                    "validation_commands": [[
                        "bash",
                        "-lc",
                        "test -f README.md && test -f design_notes.md && grep -q 'Usage' README.md",
                    ]],
                    "status": "pending",
                }
            ]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("design_notes.md", text)
            self.assertIn("requires content evidence", text)

    def test_plan_validation_rejects_bugfix_notes_existence_only_with_semantic_checks_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Fix the existing Python median implementation. Preserve the test intent, "
                    "add a short BUGFIX_NOTES.md, and verify with unittest."
                ),
                deterministic_semantic_scope_checks=True,
            )
            agent.initialize()
            agent.requirements = base_requirements("Existing bugfix")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Fix median logic and document the fix",
                "description": (
                    "Update calc.py to correctly calculate the median for even-length lists "
                    "and create BUGFIX_NOTES.md to document the change."
                ),
                "depends_on": [],
                "acceptance_criteria": [
                    "calc.py contains the corrected median logic.",
                    "BUGFIX_NOTES.md exists and describes the fix.",
                    "python3 -m unittest test_calc.py exits with status 0.",
                ],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "python3 -m unittest test_calc.py && test -f BUGFIX_NOTES.md",
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("BUGFIX_NOTES.md", text)
            self.assertIn("requires content evidence", text)

    def test_plan_validation_accepts_documentation_content_checks_for_each_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, title="docs", prompt="Create README and design notes.")
            agent.initialize()
            agent.requirements = base_requirements("Documentation")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Create documentation",
                    "description": "Create README.md and design_notes.md.",
                    "depends_on": [],
                    "acceptance_criteria": [
                        "README.md exists and contains usage instructions.",
                        "design_notes.md exists and explains the implementation approach.",
                        "Files are not empty.",
                    ],
                    "validation_commands": [[
                        "bash",
                        "-lc",
                        "grep -q 'Usage' README.md && grep -q 'Implementation' design_notes.md",
                    ]],
                    "status": "pending",
                }
            ]

            findings = agent._plan_structural_findings()

            self.assertNotIn("requires content evidence", "\n".join(findings))

    def test_plan_validation_accepts_grep_regex_alternation_for_documentation_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, title="docs", prompt="Create README.md.")
            agent.initialize()
            agent.requirements = base_requirements("Documentation")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Create README",
                "description": "Create README.md documentation.",
                "depends_on": [],
                "acceptance_criteria": ["README.md contains Usage and Arguments sections."],
                "validation_commands": [["bash", "-lc", "grep -qE 'Usage|Arguments' README.md"]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            self.assertNotIn("requires content evidence", "\n".join(findings))

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

    def test_computed_answer_plan_allows_shape_only_validation_when_semantic_checks_disabled(self) -> None:
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
                deterministic_semantic_scope_checks=False,
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

            text = "\n".join(findings)
            self.assertNotIn("shape-only", text)
            self.assertNotIn("semantic validation", text)

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
                    "title": "Create answer and semantic validation",
                    "description": "Compute the requested value and validate it by independent enumeration.",
                    "depends_on": [],
                    "acceptance_criteria": ["ANSWER.txt matches the independently recomputed count."],
                    "validation_commands": [[
                        "python",
                        "-c",
                        "from itertools import product; from pathlib import Path; count=sum(1 for s in product('ABCD', repeat=4)); assert Path('ANSWER.txt').read_text().strip() == str(count)",
                    ]],
                    "status": "pending",
                }
            ]

            findings = agent._plan_structural_findings()

            self.assertNotIn("shape-only", "\n".join(findings))

    def test_plan_rejects_precedence_prone_arithmetic_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Count the two-digit numbers whose digit product "
                    "is divisible by 6."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Computed answer")
            step = {
                "id": "S1",
                "title": "Create answer and semantic validation",
                "description": "Compute the requested value and validate it by independent enumeration.",
                "depends_on": [],
                "acceptance_criteria": ["ANSWER.txt matches the independently recomputed count."],
                "validation_commands": [[
                    "python",
                    "-c",
                    (
                        "expected=sum(1 for n in range(10, 100) if ((n // 10 * n % 10) % 6 == 0)); "
                        "actual=open('ANSWER.txt').read().strip(); "
                        "assert actual == str(expected), f'expected={expected} actual={actual}'"
                    ),
                ]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(
                step,
                computed_answer_semantic_validation_present=True,
            )

            text = "\n".join(findings)
            self.assertIn("operator precedence", text)
            self.assertIn("named intermediate variables", text)

            step["validation_commands"] = [[
                "python",
                "-c",
                (
                    "expected=sum(1 for n in range(10, 100) if (((n // 10) * (n % 10)) % 6 == 0)); "
                    "actual=open('ANSWER.txt').read().strip(); "
                    "assert actual == str(expected), f'expected={expected} actual={actual}'"
                ),
            ]]

            findings = agent._validation_command_findings(
                step,
                computed_answer_semantic_validation_present=True,
            )

            self.assertNotIn("operator precedence", "\n".join(findings))

    def test_plan_rejects_raw_text_numeric_comparison_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Consider integers n from 1 to 120 and return "
                    "the requested sum as a single integer."
                ),
            )
            step = {
                "id": "S1",
                "title": "Create answer and semantic validation",
                "description": "Compute the requested value and validate it by independent enumeration.",
                "depends_on": [],
                "acceptance_criteria": ["ANSWER.txt matches the independently recomputed sum."],
                "validation_commands": [[
                    "python",
                    "-c",
                    (
                        "import sys; s=sum(n for n in range(1,121)); "
                        "actual=open('ANSWER.txt').read().strip(); "
                        "sys.exit(0 if s == actual else (print(f'Mismatch: expected {s}, got {actual}') or 1))"
                    ),
                ]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(
                step,
                computed_answer_semantic_validation_present=True,
            )

            text = "\n".join(findings)
            self.assertIn("raw file text", text)
            self.assertIn("numeric expression", text)

    def test_plan_allows_explicit_numeric_or_string_conversion_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Consider integers n from 1 to 120 and return "
                    "the requested sum as a single integer."
                ),
            )
            step = {
                "id": "S1",
                "title": "Create answer and semantic validation",
                "description": "Compute the requested value and validate it by independent enumeration.",
                "depends_on": [],
                "acceptance_criteria": ["ANSWER.txt matches the independently recomputed sum."],
                "validation_commands": [[
                    "python",
                    "-c",
                    (
                        "s=sum(n for n in range(1,121)); actual=open('ANSWER.txt').read().strip(); "
                        "assert str(s) == actual, f'expected={s} actual={actual}'"
                    ),
                ]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(
                step,
                computed_answer_semantic_validation_present=True,
            )

            self.assertNotIn("raw file text", "\n".join(findings))

            step["validation_commands"] = [[
                "python",
                "-c",
                (
                    "s=sum(n for n in range(1,121)); actual=int(open('ANSWER.txt').read().strip()); "
                    "assert s == actual, f'expected={s} actual={actual}'"
                ),
            ]]

            findings = agent._validation_command_findings(
                step,
                computed_answer_semantic_validation_present=True,
            )

            self.assertNotIn("raw file text", "\n".join(findings))

    def test_computed_answer_plan_rejects_silent_semantic_mismatch_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Consider integers n from 1 to 120. "
                    "Keep n if n is divisible by exactly one of 3, 5, 7 and return the sum."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Computed answer")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Create answer and semantic validation",
                    "description": "Compute the requested value and validate it by independent enumeration.",
                    "depends_on": [],
                    "acceptance_criteria": ["ANSWER.txt matches the independently recomputed count."],
                    "validation_commands": [[
                        "bash",
                        "-lc",
                        "python3 -c 'total=sum(n for n in range(1, 121)); exit(0 if str(total)==open(\"ANSWER.txt\").read().strip() else 1)'",
                    ]],
                    "status": "pending",
                }
            ]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("semantic validation exits non-zero on mismatch without diagnostic output", text)
            self.assertIn("expected/actual", text)

            agent.plan_steps[0]["validation_commands"] = [[
                "bash",
                "-lc",
                (
                    "python3 -c 'total=sum(n for n in range(1, 121)); "
                    "actual=open(\"ANSWER.txt\").read().strip(); "
                    "print(f\"expected={total} actual={actual}\") if str(total)!=actual else None; "
                    "exit(0 if str(total)==actual else 1)'"
                ),
            ]]

            findings = agent._plan_structural_findings()

            self.assertNotIn("semantic validation exits non-zero", "\n".join(findings))

    def test_computed_answer_plan_allows_silent_semantic_mismatch_validator_when_semantic_checks_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Consider integers n from 1 to 120. "
                    "Keep n if n is divisible by exactly one of 3, 5, 7 and the digit sum is odd. "
                    "Return the sum as a single integer."
                ),
            )
            agent.config = replace(
                agent.config,
                quality_policy=replace(
                    agent.config.quality_policy,
                    deterministic_semantic_scope_checks=False,
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Computed answer")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Create answer and semantic validation",
                "description": "Compute the requested value and validate it by independent enumeration.",
                "depends_on": [],
                "acceptance_criteria": ["ANSWER.txt matches the independently recomputed sum."],
                "validation_commands": [[
                    "python",
                    "-c",
                    (
                        "import sys; expected = sum(n for n in range(1, 121) "
                        "if sum(1 for d in [3, 5, 7] if n % d == 0) == 1 "
                        "and sum(int(d) for d in str(n)) % 2 != 0); "
                        "f=open('ANSWER.txt', 'r'); actual = int(f.read().strip()); "
                        "sys.exit(0 if expected == actual else 1)"
                    ),
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertNotIn("semantic validation exits non-zero on mismatch without diagnostic output", text)
            self.assertNotIn("expected/actual", text)

    def test_computed_answer_plan_rejects_bare_assert_mismatch_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Consider integers n from 1 to 120. "
                    "Keep n if n is divisible by exactly one of 3, 5, 7 and return the sum."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Computed answer")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Create answer and semantic validation",
                "description": "Compute the requested value and validate it by independent enumeration.",
                "depends_on": [],
                "acceptance_criteria": ["ANSWER.txt matches the independently recomputed count."],
                "validation_commands": [[
                    "python",
                    "-c",
                    (
                        "s=sum(n for n in range(1, 121) if sum(n%x==0 for x in [3,5,7])==1); "
                        "target=int(open('ANSWER.txt').read().strip()); assert s == target"
                    ),
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            self.assertIn(
                "semantic validation exits non-zero on mismatch without diagnostic output",
                "\n".join(findings),
            )

    def test_artifact_only_validation_allows_tmp_validator_that_reads_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Consider integers n from 1 to 120. "
                    "Return the computed sum as a single integer."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Computed answer")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Create answer and semantic validation",
                    "description": "Compute the requested value and validate it using a temporary validator.",
                    "depends_on": [],
                    "acceptance_criteria": ["ANSWER.txt matches an independently recomputed sum."],
                    "validation_commands": [[
                        "sh",
                        "-c",
                        (
                            "printf '%s\n' \"from pathlib import Path\" "
                            "\"expected = sum(n for n in range(1, 121))\" "
                            "\"actual = Path('ANSWER.txt').read_text().strip()\" "
                            "\"assert actual == str(expected)\" > /tmp/validate_answer.py "
                            "&& python /tmp/validate_answer.py"
                        ),
                    ]],
                    "status": "pending",
                }
            ]

            findings = agent._plan_structural_findings()

            self.assertNotIn("appears to write or mutate", "\n".join(findings))

    def test_artifact_only_plan_rejects_shell_heredoc_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, prompt="Create ANSWER.txt only. Return the integer answer.")
            agent.initialize()
            step = {
                "id": "S1",
                "title": "Create answer",
                "description": "Create and validate ANSWER.txt.",
                "depends_on": [],
                "acceptance_criteria": ["ANSWER.txt contains the correct answer."],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "python3 - <<'PY'\nfrom pathlib import Path\nassert Path('ANSWER.txt').read_text().strip()\nPY",
                ]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertIn("artifact-only", "\n".join(findings))
            self.assertIn("shell here-doc", "\n".join(findings))

    def test_artifact_only_plan_rejects_non_deliverable_probe_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt="Create ANSWER.txt only. Compute a sum and return it as a single integer.",
            )
            agent.initialize()
            agent.requirements = base_requirements("Computed answer")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Calculate and verify the sum",
                    "description": "Print the computed sum to stdout.",
                    "depends_on": [],
                    "acceptance_criteria": ["The sum is printed to stdout."],
                    "validation_commands": [["python", "-c", "print(42)"]],
                    "status": "pending",
                },
                {
                    "id": "S2",
                    "title": "Deliver and validate ANSWER.txt",
                    "description": "Write ANSWER.txt and validate its content.",
                    "depends_on": ["S1"],
                    "acceptance_criteria": ["ANSWER.txt contains the computed integer."],
                    "validation_commands": [["python", "-c", "assert open('ANSWER.txt').read().strip()"]],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("Artifact-only plan splits non-deliverable work", text)
            self.assertIn("S1", text)

    def test_artifact_only_validation_rejects_shell_redirect_to_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, prompt="Create ANSWER.txt only. Return the integer answer.")
            agent.initialize()

            mutated = agent._validation_command_appears_to_mutate_artifact([
                "sh",
                "-c",
                "printf 42 > ANSWER.txt",
            ])

            self.assertTrue(mutated)

    def test_artifact_only_validation_rejects_python_write_to_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, prompt="Create ANSWER.txt only. Return the integer answer.")
            agent.initialize()

            mutated = agent._validation_command_appears_to_mutate_artifact([
                "python",
                "-c",
                "from pathlib import Path; Path('ANSWER.txt').write_text('42\\n')",
            ])

            self.assertTrue(mutated)

    def test_artifact_only_guidance_is_not_duplicated_in_phase_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            requirements_payload = base_requirements("Create ANSWER.txt only.")
            requirements_payload["refined_requirements"] = ["Create ANSWER.txt only."]
            requirements_payload["plan"] = [{
                "id": "S1",
                "title": "Create answer",
                "description": "Create ANSWER.txt.",
                "depends_on": [],
                "acceptance_criteria": ["ANSWER.txt contains the answer."],
                "validation_commands": [["python", "-c", "from pathlib import Path; assert Path('ANSWER.txt').read_text().strip()"]],
            }]
            implementation_response = {
                "plan_note": "Created ANSWER.txt.",
                "files": [{"path": "ANSWER.txt", "content": "42"}],
                "commands": [],
                "test_evidence": ["No command evidence requested in this prompt-capture test."],
                "resolution_request": "none",
            }
            agent = load_test_agent(
                root,
                workspace,
                prompt="Create ANSWER.txt only. Return the integer answer.",
                implementation_responses=[json.dumps(requirements_payload), json.dumps(implementation_response)],
            )
            agent.initialize()

            agent._requirements_refinement_phase()
            requirements_prompt = agent.impl_client.calls[0]["messages"][-1]["content"]
            self.assertEqual(requirements_prompt.count("ARTIFACT_ONLY_CONSTRAINT:"), 1)
            self.assertIn("must read or compare the actual requested artifact", requirements_prompt)
            self.assertIn("must not be used as a substitute artifact", requirements_prompt)

            step = {
                "id": "S1",
                "title": "Create answer",
                "description": "Create ANSWER.txt.",
                "depends_on": [],
                "acceptance_criteria": ["ANSWER.txt contains the answer."],
                "validation_commands": [],
                "status": "pending",
            }
            agent.plan_steps = [step]
            agent.requirements = requirements_payload
            agent._implementation_pass(step, attempt=1)
            implementation_prompt = agent.impl_client.calls[1]["messages"][-1]["content"]
            self.assertEqual(implementation_prompt.count("ARTIFACT_ONLY_CONSTRAINT:"), 1)
            self.assertIn("must read or compare the actual requested artifact", implementation_prompt)
            self.assertIn("must not be used as a substitute artifact", implementation_prompt)

    def test_computed_answer_plan_rejects_print_only_semantic_command(self) -> None:
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
                    "title": "Create answer and semantic validation",
                    "description": "Compute the requested value and validate it by independent enumeration against ANSWER.txt.",
                    "depends_on": [],
                    "acceptance_criteria": ["ANSWER.txt matches the independently recomputed count."],
                    "validation_commands": [[
                        "python",
                        "-c",
                        "from itertools import product; print(sum(1 for s in product('ABCD', repeat=4)))",
                    ]],
                    "status": "pending",
                }
            ]

            findings = agent._plan_structural_findings()

            self.assertIn("compares the artifact", "\n".join(findings))

    def test_artifact_only_plan_rejects_printed_value_without_requested_file(self) -> None:
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
                    "title": "Compute and verify the sum",
                    "description": "Calculate the sum for the range 1-120 and print it.",
                    "depends_on": [],
                    "acceptance_criteria": ["The sum for 1-120 is printed."],
                    "validation_commands": [[
                        "python",
                        "-c",
                        "valid=lambda n: sum(1 for x in [3,5,7] if n%x==0)==1; print(sum(n for n in range(1,121) if valid(n)))",
                    ]],
                    "status": "pending",
                }
            ]

            findings = agent._plan_structural_findings()
            text = "\n".join(findings)

            self.assertIn("does not preserve the explicitly requested artifact", text)
            self.assertIn("validation does not inspect the explicitly requested artifact", text)

    def test_computed_answer_plan_allows_loop_based_semantic_validator(self) -> None:
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
                    "title": "Create answer",
                    "description": "Compute the requested value and write ANSWER.txt.",
                    "depends_on": [],
                    "acceptance_criteria": ["ANSWER.txt exists and contains an integer."],
                    "validation_commands": [["test", "-f", "ANSWER.txt"]],
                    "status": "pending",
                },
                {
                    "id": "S2",
                    "title": "Semantic validation of ANSWER.txt",
                    "description": "Re-calculate the count using loops and compare it to ANSWER.txt.",
                    "depends_on": ["S1"],
                    "acceptance_criteria": ["The value in ANSWER.txt matches the re-calculated count."],
                    "validation_commands": [[
                        "python",
                        "-c",
                        "alphabet=['A','B','C','D']; count=0; [count:=count+1 for s in [a+b+c+d for a in alphabet for b in alphabet for c in alphabet for d in alphabet] if s.count('A')==2 and all(s[i]!=s[i+1] for i in range(3))]; actual=open('ANSWER.txt').read().strip(); exit(0 if actual == str(count) else 1)",
                    ]],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            self.assertNotIn("does not explicitly require semantic validation", "\n".join(findings))

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
                    "validation_commands": [[
                        "python",
                        "-c",
                        "from pathlib import Path; total=sum(n for n in range(1, 121)); assert Path('ANSWER.txt').read_text().strip() == str(total)",
                    ]],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            self.assertNotIn("shape-only", "\n".join(findings))

    def test_computed_answer_plan_rejects_split_compute_without_artifact_comparison(self) -> None:
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
                    "title": "Brute-force verification and calculation",
                    "description": "Use itertools.product to enumerate valid strings and print the count.",
                    "depends_on": [],
                    "acceptance_criteria": ["The script outputs the correct integer count."],
                    "validation_commands": [[
                        "python",
                        "-c",
                        "from itertools import product; print(sum(1 for _ in product('ABCD', repeat=4)))",
                    ]],
                    "status": "pending",
                },
                {
                    "id": "S2",
                    "title": "Create ANSWER.txt",
                    "description": "Write the verified integer result to ANSWER.txt.",
                    "depends_on": ["S1"],
                    "acceptance_criteria": ["ANSWER.txt exists and contains only the integer result."],
                    "validation_commands": [["cat", "ANSWER.txt"]],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            self.assertIn("compares the artifact", "\n".join(findings))

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

    def test_computed_answer_requirements_allow_hardcoded_answer_when_semantic_checks_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Consider integers n from 1 to 120. "
                    "Keep n if n is divisible by exactly one of 3, 5, 7, and the digit sum of n is odd. "
                    "Return the sum of all kept n as a single integer."
                ),
                deterministic_semantic_scope_checks=False,
            )
            agent.initialize()
            requirements = base_requirements("Computed answer")
            requirements["refined_requirements"] = [
                "Calculate the requested sum.",
                "Store the resulting sum as a single integer in ANSWER.txt.",
            ]
            requirements["assumptions"] = [
                "The expected mathematical result for the sum is 1778.",
            ]
            requirements["planning_confirmation"] = {
                "is_feasible": True,
                "is_clear": True,
                "is_verifiable": True,
                "verification_strategy": (
                    "Check that ANSWER.txt contains exactly one integer and that the integer "
                    "matches the expected mathematical result (1778)."
                ),
                "remaining_risks": [],
            }
            requirements["plan"] = [{
                "id": "S1",
                "title": "Create answer and validator",
                "description": "Create ANSWER.txt and compare it against the expected value.",
                "depends_on": [],
                "acceptance_criteria": ["ANSWER.txt contains the correct integer 1778."],
                "validation_commands": [[
                    "python",
                    "-c",
                    "actual=open('ANSWER.txt').read().strip(); assert actual == '1778'",
                ]],
                "status": "pending",
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "resolved")
            self.assertNotIn("hard-code", "\n".join(review.get("required_changes", [])))
            self.assertNotIn("recomputes", "\n".join(review.get("required_changes", [])))

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

    def test_artifact_only_plan_rejects_helper_workspace_files(self) -> None:
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
            agent.requirements = base_requirements("Artifact-only answer")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Create solver and answer",
                    "description": "Create solver.py to calculate the answer and write ANSWER.txt.",
                    "depends_on": [],
                    "acceptance_criteria": ["solver.py exists", "ANSWER.txt exists"],
                    "validation_commands": [["python", "solver.py"]],
                    "status": "pending",
                },
                {
                    "id": "S2",
                    "title": "Create verifier",
                    "description": "Create validate.py for independent enumeration.",
                    "depends_on": ["S1"],
                    "acceptance_criteria": ["validate.py confirms the answer."],
                    "validation_commands": [["python", "validate.py"]],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("extra workspace artifact", text)
            self.assertIn("solver.py", text)
            self.assertIn("validate.py", text)

    def test_bounded_named_artifact_plan_rejects_unrequested_validation_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create CURL_NOTES.md explaining a safe pattern for sending complex JSON "
                    "with curl by writing the JSON to a file."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Curl notes")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Create documentation and validation script",
                    "description": "Create CURL_NOTES.md and validate_curl_pattern.py.",
                    "depends_on": [],
                    "acceptance_criteria": [
                        "CURL_NOTES.md exists.",
                        "validate_curl_pattern.py proves the curl pattern with a local server.",
                    ],
                    "validation_commands": [["bash", "-lc", "python3 -m pip install requests && python3 validate_curl_pattern.py"]],
                    "status": "pending",
                }
            ]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("unrequested workspace helper artifact", text)
            self.assertIn("validate_curl_pattern.py", text)
            self.assertIn("hides package installation", text)

    def test_bounded_named_artifact_plan_allows_requested_validator_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create huge_output.py that can print lines and validate_huge_output.py "
                    "that runs it with a bounded line count."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Huge output")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Create scripts",
                    "description": "Create huge_output.py and validate_huge_output.py.",
                    "depends_on": [],
                    "acceptance_criteria": [
                        "huge_output.py prints the requested number of lines.",
                        "validate_huge_output.py checks a bounded line count.",
                    ],
                    "validation_commands": [["python", "validate_huge_output.py"]],
                    "status": "pending",
                }
            ]

            findings = agent._plan_structural_findings()

            self.assertNotIn("unrequested workspace helper artifact", "\n".join(findings))

    def test_artifact_only_plan_ignores_dotted_python_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, prompt="Create ANSWER.txt only. Return the integer answer.")

            refs = agent._file_references_in_text(
                "python -c \"import os, itertools; os.path.exists('ANSWER.txt'); "
                "itertools.product('ABCD'); s.count('A')\""
            )

            self.assertEqual(refs, {"ANSWER.txt"})

    def test_artifact_only_write_guard_blocks_helper_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, prompt="Create ANSWER.txt only. Return the integer answer.")

            allowed, skipped = agent._split_model_writable_files([
                {"path": "ANSWER.txt", "content": "24\n"},
                {"path": "solver.py", "content": "print(24)\n"},
            ])

            self.assertEqual([item["path"] for item in allowed], ["ANSWER.txt"])
            self.assertEqual(skipped, ["solver.py"])

    def test_artifact_only_final_review_rejects_extra_workspace_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, prompt="Create ANSWER.txt only. Return the integer answer.")

            findings = agent._artifact_only_workspace_findings({
                "workspace_files": [
                    {"path": "ANSWER.txt", "content": "24\n"},
                    {"path": ".gitignore", "content": ".agent_state/\n"},
                    {"path": "solver.py", "content": "print(24)\n"},
                    {"path": "PLAN.md", "content": "# harness\n"},
                ]
            })

            self.assertIn("extra project artifact", "\n".join(findings))
            self.assertIn("solver.py", "\n".join(findings))
            self.assertNotIn(".gitignore", "\n".join(findings))
            self.assertNotIn("gitignore", "\n".join(findings))

    def test_artifact_only_final_review_ignores_harness_gitignore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, prompt="Create ANSWER.txt only. Return the integer answer.")

            findings = agent._artifact_only_workspace_findings({
                "workspace_files": [
                    {"path": "./ANSWER.txt", "content": "24\n"},
                    {"path": ".gitignore", "content": ".agent_state/\n"},
                ]
            })

            self.assertEqual(findings, [])

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

    def test_requirements_review_rejects_hardcoded_answer_with_semantic_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Count the 4-character strings over alphabet {A,B,C,D} "
                    "that contain exactly two vowels if A is the only vowel and have no adjacent equal characters."
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
                "verification_strategy": "Use brute force enumeration, then assert ANSWER.txt matches expected value 24.",
            }
            requirements["plan"] = [
                {
                    "id": "S1",
                    "title": "Create answer",
                    "description": "Use itertools.product to enumerate valid strings and write ANSWER.txt.",
                    "depends_on": [],
                    "acceptance_criteria": ["ANSWER.txt contains exactly 24."],
                    "validation_commands": [["python3", "-c", "import itertools; print('semantic enumeration')"]],
                }
            ]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "needs_requirements_change")
            self.assertIn("without embedding the final numeric answer", "\n".join(review["required_changes"]))

    def test_requirements_review_applies_plan_structural_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Count the 4-character strings over alphabet {A,B,C,D} "
                    "that contain exactly two vowels if A is the only vowel and have no adjacent equal characters."
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
                "verification_strategy": "Generate ANSWER.txt, then recompute and compare the artifact.",
            }
            requirements["plan"] = [
                {
                    "id": "S1",
                    "title": "Generate answer",
                    "description": "Generate ANSWER.txt.",
                    "depends_on": [],
                    "acceptance_criteria": ["ANSWER.txt exists."],
                    "validation_commands": [
                        {
                            "cmd": "python -c \"open('ANSWER.txt', 'w').write('24')\"",
                            "timeout_seconds": 10,
                        }
                    ],
                },
                {
                    "id": "S2",
                    "title": "Semantic validation",
                    "description": "Recompute and compare.",
                    "depends_on": ["S1"],
                    "acceptance_criteria": ["The recomputed value matches ANSWER.txt."],
                    "validation_commands": [[
                        "python",
                        "-c",
                        "count=24; with open('ANSWER.txt') as f: actual=int(f.read()); exit(0 if actual == count else 1)",
                    ]],
                },
            ]

            review = agent._requirements_review(1, requirements)

            text = "\n".join(review["required_changes"])
            self.assertEqual(review["status"], "needs_requirements_change")
            self.assertIn("string-valued cmd", text)
            self.assertIn("appears to write or mutate", text)
            self.assertIn("invalid one-line `python -c` compound statement", text)

    def test_simple_helper_prompt_uses_proportional_quality_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt="Build a counting helper where A is the only vowel in the alphabet.",
            )

            self.assertFalse(agent._default_quality_policy_applies())
            self.assertFalse(agent._explicit_artifact_only_constraint())

    def test_negative_readme_mention_does_not_trigger_quality_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create huge_output.py that can print a configurable number of lines and "
                    "validate_huge_output.py that runs it with a bounded line count and asserts "
                    "the output format. Do not rely on dumping huge output into README."
                ),
            )

            self.assertFalse(agent._default_quality_policy_applies())
            self.assertEqual(
                agent._default_quality_policy_reason(),
                "bounded utility/script prompt without requested extra quality deliverables",
            )

    def test_bounded_plan_rejects_redundant_final_verification_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create huge_output.py that can print a configurable number of lines and "
                    "validate_huge_output.py that runs it with a bounded line count and asserts "
                    "the output format. Do not rely on dumping huge output into README."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Bounded utility")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Implement scripts",
                    "description": "Create huge_output.py and validate_huge_output.py.",
                    "depends_on": [],
                    "acceptance_criteria": ["Both scripts work."],
                    "validation_commands": [["python", "validate_huge_output.py", "--count", "1000"]],
                    "status": "pending",
                },
                {
                    "id": "S2",
                    "title": "Final Verification",
                    "description": "Run a comprehensive check of error cases and success cases.",
                    "depends_on": ["S1"],
                    "acceptance_criteria": ["Success and error cases pass."],
                    "validation_commands": [["python", "validate_huge_output.py", "--count", "1000"]],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            self.assertIn("standalone final verification/QA step", "\n".join(findings))

    def test_artifact_only_plan_rejects_separate_verification_step_for_same_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Consider integers n from 1 to 120. "
                    "Keep n if n is divisible by exactly one of 3, 5, 7, and the digit sum of n is odd. "
                    "Return the sum of all kept n as a single integer."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Artifact-only answer")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Compute and write the sum to ANSWER.txt",
                    "description": "Write ANSWER.txt with the requested single integer.",
                    "depends_on": [],
                    "acceptance_criteria": ["ANSWER.txt exists.", "ANSWER.txt contains a single integer."],
                    "validation_commands": [["python", "-c", "val=open('ANSWER.txt').read().strip(); assert val.isdigit(), val"]],
                    "status": "pending",
                },
                {
                    "id": "S2",
                    "title": "Verify the correctness of ANSWER.txt",
                    "description": "Recalculate the sum and compare it to the value in ANSWER.txt.",
                    "depends_on": ["S1"],
                    "acceptance_criteria": ["The calculated sum matches ANSWER.txt."],
                    "validation_commands": [[
                        "python",
                        "-c",
                        "total=sum(n for n in range(1,121) if sum(n%d==0 for d in [3,5,7])==1 and sum(int(d) for d in str(n))%2); actual=int(open('ANSWER.txt').read()); assert total==actual, (total, actual)",
                    ]],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            self.assertIn("standalone final verification/QA step", "\n".join(findings))

    def test_requested_verifier_file_step_is_not_treated_as_redundant_qa(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create huge_output.py that can print a configurable number of lines and "
                    "validate_huge_output.py that runs it with a bounded line count and asserts "
                    "the output format."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Bounded utility")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Implement huge_output.py",
                    "description": "Create the output generator.",
                    "depends_on": [],
                    "acceptance_criteria": ["Generator prints requested lines."],
                    "validation_commands": [["python", "huge_output.py", "--lines", "3"]],
                    "status": "pending",
                },
                {
                    "id": "S2",
                    "title": "Implement validate_huge_output.py",
                    "description": "Create validate_huge_output.py to validate streamed output.",
                    "depends_on": ["S1"],
                    "acceptance_criteria": ["validate_huge_output.py validates output."],
                    "validation_commands": [["python", "validate_huge_output.py", "--count", "1000"]],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            self.assertNotIn("standalone final verification/QA step", "\n".join(findings))

    def test_bounded_plan_rejects_unrequested_test_suite_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create huge_output.py that can print a configurable number of lines and "
                    "validate_huge_output.py that runs it with a bounded line count and asserts "
                    "the output format."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Bounded utility")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Implement generator and validator",
                    "description": "Create huge_output.py, validate_huge_output.py, and test_suite.py.",
                    "depends_on": [],
                    "acceptance_criteria": ["test_suite.py verifies success and failure modes."],
                    "validation_commands": [["python", "test_suite.py"]],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("unrequested test deliverable", text)
            self.assertIn("test_suite.py", text)

    def test_explicit_test_request_allows_test_suite_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create huge_output.py and validate_huge_output.py. Include tests in test_suite.py."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Bounded utility")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Implement generator, validator, and tests",
                    "description": "Create huge_output.py, validate_huge_output.py, and test_suite.py.",
                    "depends_on": [],
                    "acceptance_criteria": ["test_suite.py verifies success and failure modes."],
                    "validation_commands": [["python", "test_suite.py"]],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            self.assertNotIn("unrequested test deliverable", "\n".join(findings))

    def test_prompt_implied_direct_script_invocation_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create huge_output.py that can print a configurable number of lines and "
                    "validate_huge_output.py that runs it with a bounded line count and asserts "
                    "the output format."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Bounded utility")
            agent.requirements["refined_requirements"] = [
                "validate_huge_output.py must accept a command and count argument.",
            ]
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Implement generator and validator",
                    "description": "Create the scripts.",
                    "depends_on": [],
                    "acceptance_criteria": ["Validator works."],
                    "validation_commands": [["python", "validate_huge_output.py", "5000"]],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("validate_huge_output.py", text)
            self.assertIn("direct invocation", text)

    def test_prompt_implied_direct_script_invocation_allows_optional_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create huge_output.py that can print a configurable number of lines and "
                    "validate_huge_output.py that runs it with a bounded line count and asserts "
                    "the output format."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Bounded utility")
            agent.requirements["refined_requirements"] = [
                "validate_huge_output.py must support direct invocation without any positional arguments.",
                "validate_huge_output.py must accept an optional --count flag to override the default bounded count.",
            ]
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Implement generator and validator",
                    "description": "Create the scripts.",
                    "depends_on": [],
                    "acceptance_criteria": ["Validator works with defaults and optional count."],
                    "validation_commands": [
                        ["python", "validate_huge_output.py"],
                        ["python", "validate_huge_output.py", "--count", "1000"],
                    ],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            self.assertNotIn("direct invocation", "\n".join(findings))

    def test_prompt_implied_direct_script_invocation_allows_shell_chain_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create huge_output.py that can print a configurable number of lines and "
                    "validate_huge_output.py that runs it with a bounded line count and asserts "
                    "the output format."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Bounded utility")
            agent.requirements["refined_requirements"] = [
                "validate_huge_output.py must support direct invocation without any positional arguments.",
                "validate_huge_output.py must accept an optional count argument to override the default bounded count.",
            ]
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Implement generator and validator",
                    "description": "Create the scripts.",
                    "depends_on": [],
                    "acceptance_criteria": ["Validator works with defaults and optional count."],
                    "validation_commands": [
                        [
                            "bash",
                            "-lc",
                            "python huge_output.py 5 | head -n 5 && "
                            "python validate_huge_output.py 100 && "
                            "python validate_huge_output.py && "
                            "python validate_huge_output.py 10000",
                        ],
                    ],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            self.assertNotIn("direct invocation", "\n".join(findings))

    def test_prompt_implied_direct_script_invocation_rejects_shell_chain_without_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create huge_output.py that can print a configurable number of lines and "
                    "validate_huge_output.py that runs it with a bounded line count and asserts "
                    "the output format."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Bounded utility")
            agent.requirements["refined_requirements"] = [
                "validate_huge_output.py must support direct invocation without any positional arguments.",
                "validate_huge_output.py must accept an optional count argument to override the default bounded count.",
            ]
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Implement generator and validator",
                    "description": "Create the scripts.",
                    "depends_on": [],
                    "acceptance_criteria": ["Validator works with optional count."],
                    "validation_commands": [
                        ["bash", "-lc", "python validate_huge_output.py 100 && python validate_huge_output.py 10000"],
                    ],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            self.assertIn("direct invocation", "\n".join(findings))

    def test_configurable_script_argument_does_not_force_direct_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt="Create huge_output.py that can print a configurable number of lines.",
            )
            agent.initialize()
            agent.requirements = base_requirements("Bounded utility")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Implement generator",
                    "description": "Create huge_output.py.",
                    "depends_on": [],
                    "acceptance_criteria": ["Generator accepts a count."],
                    "validation_commands": [["python", "huge_output.py", "5"]],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            self.assertNotIn("direct invocation", "\n".join(findings))

    def test_provided_input_cli_does_not_force_direct_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python CLI tool named palindrome.py that checks whether one provided "
                    "string is a palindrome."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Input-taking CLI")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Implement CLI",
                    "description": "Create palindrome.py.",
                    "depends_on": [],
                    "acceptance_criteria": ["CLI accepts one string argument."],
                    "validation_commands": [["python", "palindrome.py", "race car"]],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            self.assertNotIn("direct invocation", "\n".join(findings))

    def test_prompt_implied_file_path_rejects_invented_required_file_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create wait_for_file.py. It should poll for a file path until "
                    "--timeout-seconds expires, sleep --interval-seconds between checks, "
                    "and exit 0 when the file appears or 2 on timeout."
                ),
                deterministic_semantic_scope_checks=True,
            )
            agent.initialize()
            agent.requirements = base_requirements("Timeout-friendly command")
            agent.requirements["refined_requirements"] = [
                "wait_for_file.py accepts --file, --timeout-seconds, and --interval-seconds.",
            ]
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Implement wait_for_file.py",
                    "description": "Create the polling CLI.",
                    "depends_on": [],
                    "acceptance_criteria": ["CLI detects an appearing file."],
                    "validation_commands": [[
                        "bash",
                        "-lc",
                        "touch target.txt && python wait_for_file.py --file target.txt --timeout-seconds 1 --interval-seconds 0.1",
                    ]],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("primary file path input", text)
            self.assertIn("`--file`", text)

    def test_requirements_review_allows_invented_primary_file_flag_with_semantic_checks_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create wait_for_file.py. It should poll for a file path until "
                    "--timeout-seconds expires, sleep --interval-seconds between checks, "
                    "print periodic status, and exit 0 when the file appears or 2 on timeout."
                ),
                deterministic_semantic_scope_checks=False,
            )
            agent.initialize()
            requirements = base_requirements("Timeout-friendly command")
            requirements["refined_requirements"] = [
                "Command-line arguments: --file, --timeout-seconds, and --interval-seconds.",
                "Tests use unittest and complete quickly.",
            ]
            requirements["plan"] = [
                {
                    "id": "S1",
                    "title": "Implement wait_for_file.py",
                    "description": "Create the polling CLI and tests.",
                    "depends_on": [],
                    "acceptance_criteria": ["Tests pass."],
                    "validation_commands": [["python3", "-m", "unittest", "test_wait_for_file.py"]],
                }
            ]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "resolved")
            text = "\n".join(review.get("required_changes", []))
            self.assertNotIn("primary file path input", text)
            self.assertNotIn("`--file`", text)

    def test_prompt_implied_file_path_accepts_positional_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create wait_for_file.py. It should poll for a file path until "
                    "--timeout-seconds expires, sleep --interval-seconds between checks."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Timeout-friendly command")
            agent.requirements["refined_requirements"] = [
                "wait_for_file.py accepts a file path plus optional --timeout-seconds and --interval-seconds.",
            ]
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Implement wait_for_file.py",
                    "description": "Create the polling CLI.",
                    "depends_on": [],
                    "acceptance_criteria": ["CLI detects an appearing file."],
                    "validation_commands": [[
                        "bash",
                        "-lc",
                        "touch target.txt && python wait_for_file.py target.txt --timeout-seconds 1 --interval-seconds 0.1",
                    ]],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            self.assertNotIn("primary file path input", "\n".join(findings))

    def test_prompt_named_file_flag_is_not_treated_as_invented(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create wait_for_file.py. It should accept --file for a file path and "
                    "--timeout-seconds for the timeout."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Timeout-friendly command")
            agent.requirements["refined_requirements"] = [
                "wait_for_file.py accepts --file and --timeout-seconds.",
            ]
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Implement wait_for_file.py",
                    "description": "Create the polling CLI.",
                    "depends_on": [],
                    "acceptance_criteria": ["CLI detects an appearing file."],
                    "validation_commands": [[
                        "bash",
                        "-lc",
                        "touch target.txt && python wait_for_file.py --file target.txt --timeout-seconds 1",
                    ]],
                    "status": "pending",
                },
            ]

            findings = agent._plan_structural_findings()

            self.assertNotIn("primary file path input", "\n".join(findings))

    def test_validation_command_rejects_unsupported_stdout_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S1",
                "title": "CLI checks",
                "description": "Check exact CLI output.",
                "validation_commands": [
                    {
                        "cmd": ["python", "palindrome.py", "race car"],
                        "expected_output": "true",
                    }
                ],
            }

            findings = agent._validation_command_findings(step)

            text = "\n".join(findings)
            self.assertIn("unsupported assertion metadata", text)
            self.assertIn("expected_output", text)

    def test_explicit_tests_and_readme_trigger_quality_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a tiny Python CLI slugify.py. It should print the slug. "
                    "Include tests and README."
                ),
            )

            self.assertTrue(agent._default_quality_policy_applies())
            self.assertEqual(
                agent._default_quality_policy_reason(),
                "prompt requests quality deliverables or project-level scope",
            )
            self.assertFalse(agent._default_quality_policy_requires_research_structure_step())

    def test_default_run_does_not_keyword_classify_quality_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a browser app with tests, README, architecture notes, and a clean project structure."
                ),
                deterministic_semantic_scope_checks=False,
            )

            self.assertFalse(agent._default_quality_policy_applies())
            self.assertFalse(agent._default_quality_policy_requires_research_structure_step())
            self.assertIn("keyword-based quality-scope classification disabled", agent._default_quality_policy_reason())

    def test_append_in_script_prompt_does_not_trigger_project_quality_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create watch_and_react.sh. It should take the watched file path as its first "
                    "argument, read PATTERN from the environment, check the file every "
                    "WATCH_INTERVAL_SECONDS seconds, append a line containing the matched pattern "
                    "to actions.log once the pattern appears, and stop after MAX_POLLS when that "
                    "environment variable is set."
                ),
            )

            self.assertFalse(agent._default_quality_policy_applies())
            self.assertEqual(
                agent._default_quality_policy_reason(),
                "bounded utility/script prompt without requested extra quality deliverables",
            )

    def test_short_design_notes_prompt_does_not_force_research_structure_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python CLI tool named palindrome.py. Include reusable core function, "
                    "unittest coverage, README documentation, and a short design notes file. "
                    "Keep it complete, well structured, and easy to validate."
                ),
            )

            self.assertTrue(agent._default_quality_policy_applies())
            self.assertFalse(agent._default_quality_policy_requires_research_structure_step())

    def test_explicit_architecture_prompt_requires_research_structure_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a browser platformer and document the project structure, architecture, "
                    "and separation of concerns before implementation."
                ),
            )

            self.assertTrue(agent._default_quality_policy_applies())
            self.assertTrue(agent._default_quality_policy_requires_research_structure_step())

    def test_prompt_contracts_warn_against_unrequested_api_overconstraint(self) -> None:
        contract_text = "\n".join([
            REQUIREMENTS_CONTRACT,
            PLAN_REFINEMENT_CONTRACT,
            IMPLEMENTATION_CONTRACT,
        ])

        self.assertIn("Scope boundary", contract_text)
        self.assertIn("Do not turn an unspecified detail", contract_text)
        self.assertIn("validation-only interface", contract_text)
        self.assertIn("Do not invent public API details", contract_text)
        self.assertIn("File existence alone is enough only when existence is the", contract_text)
        self.assertNotIn("return container/record type", contract_text)
        self.assertNotIn("caller-visible\ncontainer choice is not specified", contract_text)
        self.assertNotIn("compound statements", contract_text)
        self.assertIn("never place metadata keys inside an argv list", contract_text)
        self.assertIn("machine-readable `commands` field", contract_text)
        self.assertIn("concise diagnostic evidence", contract_text)

    def test_tool_verifier_normalization_ignores_unsupplied_command_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            review = agent._normalize_tool_verification(
                {
                    "status": "approved",
                    "summary": "Commands 2 and 3 prove the failure path.",
                    "commands": [
                        {"index": 0, "decision": "approved", "risk_level": "low", "reason": "safe"},
                        {"index": 1, "decision": "approved", "risk_level": "low", "reason": "safe"},
                    ],
                },
                [["python", "ok.py"], ["python", "validate.py"]],
                [],
            )

            self.assertEqual(review["status"], "approved")
            self.assertIn("supplied command indexes only", review["summary"])
            self.assertEqual([item["index"] for item in review["commands"]], [0, 1])

    def test_evidence_findings_reject_missing_negative_path_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Negative path evidence")
            step = {
                "id": "S1",
                "title": "Implement validator",
                "description": "Validator exits with non-zero code when output is incorrect.",
                "depends_on": [],
                "acceptance_criteria": [
                    "validate_output.py exits 0 when output is correct.",
                    "validate_output.py exits with non-zero code when output is incorrect.",
                ],
                "validation_commands": [["python", "validate_output.py"]],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])

            findings = agent._evidence_findings(
                step,
                {
                    "written": ["validate_output.py"],
                    "commands": [],
                    "raw": {"test_evidence": ["success path checked"]},
                },
                {
                    "validation_results": [
                        {
                            "command": ["python", "validate_output.py"],
                            "returncode": 0,
                            "expected_returncode": 0,
                            "returncode_matches_expected": True,
                            "timed_out": False,
                            "stdout": "ok\n",
                            "stderr": "",
                        }
                    ],
                    "workspace_files": [],
                    "git": {"enabled": False, "meaningful_changed_paths": ["validate_output.py"]},
                },
            )

            self.assertTrue(any("only proves the success path" in item for item in findings))

    def test_evidence_findings_accept_negative_path_expected_returncode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Negative path evidence")
            step = {
                "id": "S1",
                "title": "Implement CLI",
                "description": "CLI exits with non-zero code for invalid input.",
                "depends_on": [],
                "acceptance_criteria": ["Invalid input exits with code 2."],
                "validation_commands": [{
                    "cmd": ["python", "cli.py", "--invalid"],
                    "expected_returncode": 2,
                }],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])

            findings = agent._evidence_findings(
                step,
                {
                    "written": ["cli.py"],
                    "commands": [],
                    "raw": {"test_evidence": ["negative path checked"]},
                },
                {
                    "validation_results": [
                        {
                            "command": ["python", "cli.py", "--invalid"],
                            "returncode": 2,
                            "expected_returncode": 2,
                            "returncode_matches_expected": True,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "invalid input\n",
                        }
                    ],
                    "workspace_files": [],
                    "git": {"enabled": False, "meaningful_changed_paths": ["cli.py"]},
                },
            )

            self.assertFalse(any("only proves the success path" in item for item in findings))

    def test_evidence_findings_reject_weak_documentation_content_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Documentation evidence")
            step = {
                "id": "S1",
                "title": "Create documentation",
                "description": "Create README.md and design_notes.md.",
                "depends_on": [],
                "acceptance_criteria": [
                    "README.md exists and contains usage instructions.",
                    "design_notes.md exists and explains the implementation approach.",
                    "Files are not empty.",
                ],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "test -f README.md && test -f design_notes.md && grep -q 'Usage' README.md",
                ]],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])

            findings = agent._evidence_findings(
                step,
                {
                    "written": ["README.md", "design_notes.md"],
                    "commands": [],
                    "raw": {"test_evidence": ["documentation validation requested"]},
                },
                {
                    "validation_results": [
                        {
                            "command": [
                                "bash",
                                "-lc",
                                "test -f README.md && test -f design_notes.md && grep -q 'Usage' README.md",
                            ],
                            "returncode": 0,
                            "expected_returncode": 0,
                            "returncode_matches_expected": True,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "",
                        }
                    ],
                    "workspace_files": [],
                    "git": {"enabled": False, "meaningful_changed_paths": ["README.md", "design_notes.md"]},
                },
            )

            text = "\n".join(findings)
            self.assertIn("design_notes.md", text)
            self.assertIn("requires content evidence", text)

    def test_evidence_findings_reject_weak_bugfix_notes_with_semantic_checks_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, deterministic_semantic_scope_checks=False)
            agent.initialize()
            agent.requirements = base_requirements("Existing bugfix")
            step = {
                "id": "S1",
                "title": "Fix median logic and document the fix",
                "description": (
                    "Update calc.py to correctly calculate the median for even-length lists "
                    "and create BUGFIX_NOTES.md to document the change."
                ),
                "depends_on": [],
                "acceptance_criteria": [
                    "calc.py contains the corrected median logic.",
                    "BUGFIX_NOTES.md exists and describes the fix.",
                    "python3 -m unittest test_calc.py exits with status 0.",
                ],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "python3 -m unittest test_calc.py && test -f BUGFIX_NOTES.md",
                ]],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])

            findings = agent._evidence_findings(
                step,
                {
                    "written": ["calc.py", "BUGFIX_NOTES.md"],
                    "commands": [{
                        "command": ["bash", "-lc", "python3 -m unittest test_calc.py && test -f BUGFIX_NOTES.md"],
                        "returncode": 0,
                        "expected_returncode": 0,
                        "returncode_matches_expected": True,
                        "timed_out": False,
                        "stdout": "",
                        "stderr": "",
                    }],
                    "raw": {"test_evidence": ["unittest passed and BUGFIX_NOTES.md exists"]},
                },
                {
                    "validation_results": [{
                        "command": ["bash", "-lc", "python3 -m unittest test_calc.py && test -f BUGFIX_NOTES.md"],
                        "returncode": 0,
                        "expected_returncode": 0,
                        "returncode_matches_expected": True,
                        "timed_out": False,
                        "stdout": "",
                        "stderr": "",
                    }],
                    "workspace_files": [
                        {"path": "BUGFIX_NOTES.md", "content": "Fixed median even-length calculation.\n"},
                    ],
                    "git": {"enabled": False, "meaningful_changed_paths": ["calc.py", "BUGFIX_NOTES.md"]},
                },
            )

            text = "\n".join(findings)
            self.assertIn("BUGFIX_NOTES.md", text)
            self.assertIn("requires content evidence", text)

    def test_failed_implementation_self_check_is_not_hard_gate_when_reviewer_validation_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            step = {
                "id": "S1",
                "title": "Implement CLI",
                "description": "Create a CLI and tests.",
                "depends_on": [],
                "acceptance_criteria": [
                    "CLI prints transformed output.",
                    "Unit tests pass.",
                ],
                "validation_commands": [["python", "-m", "unittest", "test_cli.py"]],
                "status": "pending",
            }

            findings = agent._evidence_findings(
                step,
                {
                    "written": ["cli.py", "test_cli.py"],
                    "commands": [
                        {
                            "command": ["bash", "-lc", "python cli.py --help | grep -q 'exact lower-case phrase'"],
                            "returncode": 1,
                            "expected_returncode": 0,
                            "returncode_matches_expected": False,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "",
                        }
                    ],
                    "raw": {"test_evidence": ["implementation-side help grep failed"]},
                },
                {
                    "validation_results": [
                        {
                            "command": ["python", "-m", "unittest", "test_cli.py"],
                            "returncode": 0,
                            "expected_returncode": 0,
                            "returncode_matches_expected": True,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "Ran 4 tests in 0.1s\n\nOK\n",
                        }
                    ],
                    "workspace_files": [],
                    "git": {"enabled": False, "meaningful_changed_paths": ["cli.py", "test_cli.py"]},
                },
            )

            self.assertNotIn("Implementation command returned", "\n".join(findings))

    def test_requirements_review_rejects_unvalidated_public_output_representation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module pairs.py with normalize_pairs(pairs). "
                    "It must normalize each numeric pair by sorting the two values "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Pair normalization")
            requirements["refined_requirements"] = [
                "Module `pairs.py` must contain `normalize_pairs(pairs)`.",
                "Output format: a list of lists.",
            ]
            requirements["assumptions"] = ["The output format will be a list of lists for consistency."]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Research and Architecture",
                "description": "Research the transformation pattern and write DESIGN.md.",
                "depends_on": [],
                "acceptance_criteria": ["DESIGN.md exists"],
                "validation_commands": [["test", "-f", "DESIGN.md"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "needs_requirements_change")
            text = "\n".join(review["required_changes"])
            self.assertIn("`normalize_pairs`", text)
            self.assertIn("canonical caller-visible output representation", text)

    def test_requirements_review_rejects_unresolved_alternative_assumption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt="Create monitor_disk.sh that checks disk space for a chosen filesystem.",
            )
            agent.initialize()
            requirements = base_requirements("Disk monitor")
            requirements["refined_requirements"] = [
                "`monitor_disk.sh` checks disk space using df.",
                "The script prints status output.",
            ]
            requirements["assumptions"] = [
                "The script will check the root filesystem or the current directory filesystem; "
                "let's assume root for the primary check, or better, current directory for generality."
            ]
            requirements["open_questions"] = []
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement disk monitor",
                "description": "Create the shell script.",
                "depends_on": [],
                "acceptance_criteria": ["Script runs."],
                "validation_commands": [["bash", "-lc", "test -x ./monitor_disk.sh && ./monitor_disk.sh | head -n 1"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "needs_requirements_change")
            text = "\n".join(review["required_changes"])
            self.assertIn("unresolved alternatives", text)
            self.assertIn("Choose one explicit assumption", text)

    def test_requirements_review_rejects_unrequested_pretty_json_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create normalize_config.py. It reads a JSON object file path, recursively sorts object keys, "
                    "removes keys with null values, keeps list order, and writes normalized JSON to stdout. Include tests."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Config normalizer")
            requirements["refined_requirements"] = [
                "Read a JSON file path argument.",
                "Write normalized JSON to stdout.",
                "Output JSON should be pretty-printed with 4-space indentation for readability.",
            ]
            requirements["assumptions"] = [
                "The output JSON should be pretty-printed with 4-space indentation.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement normalizer",
                "description": "Create normalize_config.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_normalize_config.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "needs_requirements_change")
            text = "\n".join(review["required_changes"])
            self.assertIn("machine-readable JSON stdout", text)
            self.assertIn("compact deterministic JSON", text)

    def test_requirements_pretty_json_check_allows_compact_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create normalize_config.py. It reads a JSON object file path, recursively sorts object keys, "
                    "removes keys with null values, keeps list order, and writes normalized JSON to stdout. Include tests."
                ),
            )
            requirements = base_requirements("Config normalizer")
            requirements["project_summary"] = (
                "Normalize JSON configuration files and output compact deterministic JSON to stdout."
            )
            requirements["refined_requirements"] = [
                "Write compact deterministic JSON to stdout.",
                "Do not pretty-print stdout or add indentation.",
            ]
            requirements["open_questions"] = [{
                "question": "Should the output be pretty-printed?",
                "resolution_strategy": "assume",
                "decision": "No, output will be compact JSON to preserve the machine-readable stdout contract.",
            }]
            requirements["assumptions"] = [
                "The output is compact JSON with no indentation.",
            ]

            findings = agent._stdout_json_format_requirements_findings(requirements)

            self.assertEqual([], findings)

    def test_requirements_review_does_not_phrase_match_unrequested_list_null_removal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create normalize_config.py. It reads a JSON object file path, recursively sorts object keys, "
                    "removes keys with null values, keeps list order, and writes normalized JSON to stdout. Include tests."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Config normalizer")
            requirements["refined_requirements"] = [
                "Read a JSON file path argument.",
                "Remove keys with null values from objects.",
                "Filter out None values from lists before writing output.",
                "Write normalized JSON to stdout.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement normalizer",
                "description": "Create normalize_config.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_normalize_config.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "resolved")
            text = "\n".join(review["required_changes"])
            self.assertNotIn("list null-element removal", text)
            self.assertNotIn("original user request did not name", text)

    def test_evidence_findings_flag_generated_test_list_null_scope_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create normalize_config.py. It reads a JSON object file path, recursively sorts object keys, "
                    "removes keys with null values, keeps list order, and writes normalized JSON to stdout. Include tests."
                ),
            )
            agent.initialize()
            step = {
                "id": "S1",
                "title": "Implement normalizer",
                "description": "Create normalize_config.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_normalize_config.py"]],
            }
            result = {
                "command": ["python", "-m", "unittest", "test_normalize_config.py"],
                "returncode": 1,
                "expected_returncode": 0,
                "timed_out": False,
                "stdout": "",
                "stderr": (
                    "FAIL: test_list_order_and_nesting\n"
                    "AssertionError: {'a': [3, 1, 2], 'b': {'x': [2, None, 1], 'y': 1}} "
                    "!= {'a': [3, 1, 2], 'b': {'x': [2, 1], 'y': 1}}\n"
                ),
            }

            findings = agent._evidence_findings(
                step,
                {"written": ["normalize_config.py", "test_normalize_config.py"], "commands": []},
                {"validation_results": [result], "workspace_files": []},
            )

            text = "\n".join(findings)
            self.assertIn("generated validation expects", text)
            self.assertIn("validator repair", text)
            self.assertIn("instead of changing implementation behavior", text)

    def test_evidence_findings_allow_explicit_list_null_removal_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create normalize_config.py. It reads a JSON object file path, recursively sorts object keys, "
                    "removes keys with null values, removes null values from lists, keeps list order, and writes "
                    "normalized JSON to stdout. Include tests."
                ),
            )
            agent.initialize()
            step = {
                "id": "S1",
                "title": "Implement normalizer",
                "description": "Create normalize_config.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_normalize_config.py"]],
            }
            result = {
                "command": ["python", "-m", "unittest", "test_normalize_config.py"],
                "returncode": 1,
                "expected_returncode": 0,
                "timed_out": False,
                "stdout": "",
                "stderr": "AssertionError: {'x': [2, None, 1]} != {'x': [2, 1]}\n",
            }

            findings = agent._evidence_findings(
                step,
                {"written": ["normalize_config.py", "test_normalize_config.py"], "commands": []},
                {"validation_results": [result], "workspace_files": []},
            )

            self.assertNotIn("generated validation expects", "\n".join(findings))

    def test_requirements_review_rejects_public_pair_api_shape_without_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module pairs.py with normalize_pairs(pairs). "
                    "It must normalize each numeric pair by sorting the two values, "
                    "preserve each input pair's list or tuple type in the returned pairs, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Pair normalization")
            requirements["refined_requirements"] = [
                "Implement `normalize_pairs(pairs)` in `pairs.py`.",
                "Input `pairs` is a list of iterables, e.g., `[[1, 3], [2, 6]]`.",
                "Output is a list of normalized pairs, e.g., `[[1, 3], [2, 6]]`.",
            ]
            requirements["assumptions"] = ["Input format can be lists or tuples."]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement pair normalization",
                "description": "Create pairs.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_pairs.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "needs_requirements_change")
            text = "\n".join(review["required_changes"])
            self.assertIn("Do not invent a concrete return container", text)
            self.assertIn("semantic values across representative input shapes", text)
            self.assertIn("without requiring a specific pair container shape", text)

    def test_requirements_review_short_circuits_deterministic_findings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
                feedback_responses=[
                    json.dumps({
                        "status": "resolved",
                        "needs_rework": False,
                        "summary": "This should not be used when deterministic findings exist.",
                    })
                ],
            )
            agent.initialize()
            requirements = base_requirements("Interval merge")
            requirements["refined_requirements"] = [
                "Implement `merge_intervals(pairs)` in `intervals.py`.",
                "Output format: a list of lists.",
            ]
            requirements["assumptions"] = ["The output format will be a list of lists for consistency."]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "needs_requirements_change")
            self.assertIn("Deterministic requirements checks", review["summary"])
            self.assertEqual(agent.feedback_client.calls, [])

    def test_requirements_review_rejects_invented_canonical_shape_for_flexible_pair_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Interval merge")
            requirements["refined_requirements"] = [
                "Implement `merge_intervals(pairs)` in `intervals.py`.",
                "Input `pairs` is a list of lists or list of tuples of two integers.",
                "Output must be a list of lists of integers, regardless of whether input contains tuples or lists.",
                "Unit tests must explicitly verify that the output is a list of list.",
            ]
            requirements["assumptions"] = [
                "The canonical output shape and type is a list of lists of integers.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py and tests.",
                "depends_on": [],
                "acceptance_criteria": [
                    "Unit tests cover lists and tuples.",
                    "Unit tests check both value and representation.",
                ],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "needs_requirements_change")
            text = "\n".join(review["required_changes"])
            self.assertIn("`merge_intervals`", text)
            self.assertIn("canonical caller-visible output representation", text)
            self.assertIn("user did not request that representation", text)

    def test_requirements_review_rejects_list_tuple_canonical_shape_even_with_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Interval merge")
            requirements["refined_requirements"] = [
                "Implement `merge_intervals(pairs: list[list[int] | tuple[int, int]]) -> list[tuple[int, int]]`.",
                "Output must be a `list` of `tuples`, even if the input contains lists.",
                "Intervals are closed and touching intervals merge.",
            ]
            requirements["assumptions"] = [
                "Input format: a list of iterables (lists or tuples) representing intervals.",
                "Output format: a list of tuples representing the merged intervals.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py and tests.",
                "depends_on": [],
                "acceptance_criteria": [
                    "`merge_intervals([[1, 2], [2, 3]])` returns `[(1, 3)]`.",
                ],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "needs_requirements_change")
            text = "\n".join(review["required_changes"])
            self.assertIn("canonical caller-visible output representation", text)
            self.assertIn("list-of-tuples, list-of-lists", text)

    def test_requirements_review_rejects_output_format_assumption_without_open_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Interval merge")
            requirements["refined_requirements"] = [
                "Module `intervals.py` must contain a function `merge_intervals(pairs)`.",
                "`merge_intervals` must accept a list of integer pairs.",
                "`merge_intervals` must return a list of merged integer pairs.",
                "The function must raise a `ValueError` if any input interval has `start > end`.",
            ]
            requirements["assumptions"] = [
                "Input format is a list of iterables, where each iterable contains exactly two integers.",
                "Output format is a list of tuples.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "needs_requirements_change")
            self.assertIn(
                "canonical caller-visible output representation",
                "\n".join(review["required_changes"]),
            )

    def test_requirements_review_rejects_slash_form_input_with_output_list_example(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Interval merge")
            requirements["refined_requirements"] = [
                "Implement `merge_intervals(pairs)` in `intervals.py`.",
                "Input: A list of lists/tuples representing closed integer intervals.",
                "Output: A list of merged intervals, e.g., `[[1, 4], [5, 6]]`.",
            ]
            requirements["assumptions"] = [
                "Input is a list of iterables (lists or tuples) of integers.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "needs_requirements_change")
            self.assertIn(
                "canonical caller-visible output representation",
                "\n".join(review["required_changes"]),
            )

    def test_requirements_review_rejects_numbered_element_list_container_for_pair_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Interval merge")
            requirements["refined_requirements"] = [
                "Implement `merge_intervals(pairs)` in `intervals.py`.",
                "Input `pairs` is an iterable of 2-element iterables.",
                "Output is a list of 2-element lists representing the merged intervals.",
                "The function must raise `ValueError` if any interval has `start > end`.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py and tests.",
                "depends_on": [],
                "acceptance_criteria": [
                    "`merge_intervals([[1, 2], [2, 3]])` returns `[[1, 3]]`.",
                    "`merge_intervals([(1, 2), (2, 3)])` returns `[[1, 3]]`.",
                ],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "needs_requirements_change")
            text = "\n".join(review["required_changes"])
            self.assertIn("canonical caller-visible output representation", text)
            self.assertIn("user did not request that representation", text)

    def test_requirements_review_accepts_generic_sequence_pair_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Interval merge")
            requirements["refined_requirements"] = [
                "Function `merge_intervals(pairs)` must accept a sequence of pairs.",
                "Function must return a sequence of merged pairs.",
                "Function must merge touching closed intervals.",
                "Function must raise `ValueError` if any pair has `start > end`.",
            ]
            requirements["assumptions"] = [
                "The implementation will handle input/output as sequences, e.g., lists or tuples, and maintain semantic interval values.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py, tests, and README.",
                "depends_on": [],
                "acceptance_criteria": [
                    "`merge_intervals` returns expected merged interval values.",
                    "`test_intervals.py` covers boundary cases.",
                ],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "resolved")
            self.assertNotIn("caller-visible output representation", "\n".join(review["required_changes"]))

    def test_requirements_review_accepts_representation_neutral_open_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Interval merge")
            requirements["refined_requirements"] = [
                "Module `intervals.py` must contain a function `merge_intervals(pairs)`.",
                "`merge_intervals` accepts an iterable of pairs, where each pair is an iterable of two integers.",
                "`merge_intervals` returns an iterable of merged intervals, where each interval is an iterable of two integers.",
                "Input validation: Raise `ValueError` if any interval's start is greater than its end.",
                "Merging logic: Intervals are closed; overlapping or touching intervals must be merged.",
                "Unit tests must use the standard library `unittest` framework.",
                "A `README.md` must be provided with usage instructions and examples.",
            ]
            requirements["open_questions"] = [{
                "question": (
                    "Should the output container type or the input/output pair type be a fixed part "
                    "of the API?"
                ),
                "resolution_strategy": "skip",
                "decision": (
                    "The requirements specify an iterable interface; the specific container type is "
                    "not a requirement and will be left to the implementation's discretion."
                ),
            }]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement interval merge and tests",
                "description": "Create intervals.py and test_intervals.py.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "resolved")
            self.assertNotIn("caller-visible output representation", "\n".join(review["required_changes"]))

    def test_requirements_review_rejects_default_pytest_without_runner_convention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create wait_for_file.py. It should poll for a file path until "
                    "--timeout-seconds expires and include fast tests."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Timeout-friendly command")
            requirements["refined_requirements"] = [
                "Implement `wait_for_file.py` using Python's standard library.",
                "Provide fast tests for success and timeout behavior.",
            ]
            requirements["assumptions"] = [
                "The testing framework will be `pytest` because it is common for Python projects.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement wait_for_file.py and tests",
                "description": "Create the script and pytest tests.",
                "depends_on": [],
                "acceptance_criteria": ["The pytest suite passes."],
                "validation_commands": [["pytest", "test_wait_for_file.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "needs_requirements_change")
            self.assertIn("pytest", "\n".join(review["required_changes"]))

    def test_requirements_review_allows_negative_pytest_mention_with_unittest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create wait_for_file.py. It should poll for a file path until "
                    "--timeout-seconds expires and include fast tests."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Timeout-friendly command")
            requirements["refined_requirements"] = [
                "Implement `wait_for_file.py` using Python's standard library.",
                "Provide tests using `unittest`, not pytest.",
            ]
            requirements["assumptions"] = [
                "Use the standard-library unittest runner to avoid adding pytest as a dependency.",
            ]
            requirements["open_questions"] = [{
                "question": "Choice of test runner",
                "resolution_strategy": "assume",
                "decision": "Use unittest instead of pytest for compatibility with a blank workspace.",
            }]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement wait_for_file.py and tests",
                "description": "Create the script and unittest tests.",
                "depends_on": [],
                "acceptance_criteria": ["The unittest suite passes."],
                "validation_commands": [["python", "-m", "unittest", "discover", "-s", "tests"]],
            }]

            findings = agent._requirements_test_runner_consistency_findings(requirements)

            self.assertEqual(findings, [])

    def test_test_runner_consistency_allows_prompt_requested_pytest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt="Create a Python CLI and test it with pytest.",
            )
            agent.initialize()
            requirements = base_requirements("Pytest requested")
            requirements["assumptions"] = ["Use pytest because the user requested it."]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement CLI and pytest tests",
                "description": "Create pytest tests for the CLI.",
                "depends_on": [],
                "acceptance_criteria": ["pytest passes."],
                "validation_commands": [["pytest", "test_cli.py"]],
            }]

            findings = agent._requirements_test_runner_consistency_findings(requirements)

            self.assertEqual(findings, [])

    def test_requirements_review_accepts_neutral_container_implementation_detail_assumption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Interval merge")
            requirements["refined_requirements"] = [
                "Implement `merge_intervals(pairs)` in `intervals.py`.",
                "Input `pairs` is an iterable of pairs of integers.",
                "Output is an iterable of pairs of integers representing the merged intervals.",
                "Raise `ValueError` if any interval has `start > end`.",
                "Merge intervals that overlap or touch.",
                "Provide unit tests in `test_intervals.py` using the `unittest` framework.",
                "Provide a `README.md` with a description and usage example.",
            ]
            requirements["assumptions"] = [
                "The specific container types (e.g., list, tuple) used for the input and output are "
                "implementation details and not part of the functional requirements.",
                "Empty input returns an empty iterable.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement interval merge and unit tests",
                "description": "Create intervals.py, test_intervals.py, and README.md.",
                "depends_on": [],
                "acceptance_criteria": [
                    "`merge_intervals` returns expected merged interval values.",
                    "`merge_intervals` raises `ValueError` for invalid intervals.",
                    "`README.md` documents usage.",
                ],
                "validation_commands": [
                    ["python", "-m", "unittest", "test_intervals.py"],
                    ["grep", "-q", "Usage", "README.md"],
                ],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "resolved")
            self.assertNotIn("caller-visible output representation", "\n".join(review["required_changes"]))

    def test_requirements_review_accepts_generic_collection_of_interval_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Interval merge")
            requirements["refined_requirements"] = [
                "The module `intervals.py` must provide a function `merge_intervals(pairs)`.",
                "The function must merge intervals that overlap or share a boundary.",
                "The function must raise a `ValueError` if any interval's start is greater than its end.",
                "The function must return a collection of merged intervals.",
                "The project must include unit tests in `test_intervals.py`.",
                "The project must include a `README.md` file.",
            ]
            requirements["assumptions"] = [
                "The input `pairs` is an iterable of two-element iterables of integers.",
                "The output is a collection of two-element iterables of integers.",
            ]
            requirements["plan"] = [
                {
                    "id": "S1",
                    "title": "Implement core logic and unit tests",
                    "description": "Create intervals.py with the merging logic and test_intervals.py.",
                    "depends_on": [],
                    "acceptance_criteria": [
                        "`intervals.py` exists and contains `merge_intervals`.",
                        "`test_intervals.py` exists and passes all tests.",
                    ],
                    "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
                },
                {
                    "id": "S2",
                    "title": "Create documentation",
                    "description": "Create README.md with usage instructions.",
                    "depends_on": ["S1"],
                    "acceptance_criteria": ["`README.md` contains a Usage section."],
                    "validation_commands": [["grep", "-q", "Usage", "README.md"]],
                },
            ]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "resolved")
            self.assertNotIn("caller-visible output representation", "\n".join(review["required_changes"]))

    def test_requirements_review_does_not_phrase_match_unrequested_adjacency_scope_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Interval merge")
            requirements["refined_requirements"] = [
                "Module `intervals.py` must contain `merge_intervals(pairs)`.",
                "`merge_intervals` must merge overlapping and adjacent closed integer intervals.",
                "`merge_intervals` must raise `ValueError` if any interval has `start > end`.",
                "Include unit tests and README.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py, tests, and README.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "resolved")
            self.assertNotIn("adjacency/contiguity behavior", "\n".join(review["required_changes"]))

    def test_requirements_review_does_not_phrase_match_unrequested_log_rotation_scope_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create watch_log.sh. It should tail a log file path, poll every configurable interval, "
                    "remember the last checked line in .watch_state, and print TRIGGERED when a pattern appears."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Log watcher")
            requirements["refined_requirements"] = [
                "The script must remember the last checked line in `.watch_state`.",
                "The script must detect and handle log rotation and truncation.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement log watcher",
                "description": "Create watch_log.sh.",
                "depends_on": [],
                "acceptance_criteria": ["watch_log.sh exists and is executable."],
                "validation_commands": [["test", "-x", "watch_log.sh"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "resolved")
            self.assertNotIn("log rotation/truncation handling", "\n".join(review["required_changes"]))

    def test_requirements_review_allows_log_rotation_as_negative_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create watch_log.sh. It should tail a log file path, poll every configurable interval, "
                    "remember the last checked line in .watch_state, and print TRIGGERED when a pattern appears."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Log watcher")
            requirements["refined_requirements"] = [
                "The script must remember the last checked line in `.watch_state`.",
                "The script does not implement logic for log rotation or truncation.",
            ]
            requirements["assumptions"] = [
                "The log file is assumed append-only; rotation/truncation handling is out of scope.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement log watcher",
                "description": "Create watch_log.sh.",
                "depends_on": [],
                "acceptance_criteria": ["watch_log.sh exists and is executable."],
                "validation_commands": [["test", "-x", "watch_log.sh"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertNotIn("log rotation/truncation handling", "\n".join(review["required_changes"]))

    def test_requirements_review_does_not_phrase_match_positive_log_truncation_behavior_after_negative_clause(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create watch_log.sh. It should tail a log file path, poll every configurable interval, "
                    "remember the last checked line in .watch_state, and print TRIGGERED when a pattern appears."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Log watcher")
            requirements["refined_requirements"] = [
                "The script must remember the last checked line in `.watch_state`.",
                (
                    "Log rotation/truncation behavior is not required and will not be implemented; "
                    "the script will start from line 1 if the file is shorter than the state."
                ),
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement log watcher",
                "description": "Create watch_log.sh.",
                "depends_on": [],
                "acceptance_criteria": ["watch_log.sh exists and is executable."],
                "validation_commands": [["test", "-x", "watch_log.sh"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "resolved")
            self.assertNotIn("log rotation/truncation handling", "\n".join(review["required_changes"]))

    def test_requirements_review_does_not_phrase_match_log_truncation_behavior_in_open_question_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create watch_log.sh. It should tail a log file path, poll every configurable interval, "
                    "remember the last checked line in .watch_state, and print TRIGGERED when a pattern appears."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Log watcher")
            requirements["refined_requirements"] = [
                "The script must remember the last checked line in `.watch_state`.",
                "The script must print TRIGGERED when the pattern appears.",
            ]
            requirements["open_questions"] = [{
                "question": "How should the script handle a log file that becomes shorter than the state?",
                "resolution_strategy": "assume",
                "decision": (
                    "Behavior is not specified; implementation will treat it as an edge case where "
                    "the script starts from line 1 if the file is shorter than the state."
                ),
            }]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement log watcher",
                "description": "Create watch_log.sh.",
                "depends_on": [],
                "acceptance_criteria": ["watch_log.sh exists and is executable."],
                "validation_commands": [["test", "-x", "watch_log.sh"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "resolved")
            self.assertNotIn("log rotation/truncation handling", "\n".join(review["required_changes"]))

    def test_requirements_review_allows_log_truncation_undefined_open_question_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create watch_log.sh. It should tail a log file path, poll every configurable interval, "
                    "remember the last checked line in .watch_state, and print TRIGGERED when a pattern appears."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Log watcher")
            requirements["refined_requirements"] = [
                "The script must be named `watch_log.sh` and be executable.",
                "The script must remember the last checked line in `.watch_state`.",
                "The script must print TRIGGERED when the pattern appears.",
            ]
            requirements["assumptions"] = [
                "The script does not handle log rotation or file truncation.",
            ]
            requirements["open_questions"] = [{
                "question": "How should the script behave if the log file is truncated?",
                "resolution_strategy": "skip",
                "decision": (
                    "Behavior on truncation is not defined and remains outside the current requirement."
                ),
            }]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement log watcher",
                "description": "Create watch_log.sh, README.md, and validate.py.",
                "depends_on": [],
                "acceptance_criteria": [
                    "watch_log.sh exists and is executable.",
                    "validate.py proves pattern detection and state persistence.",
                ],
                "validation_commands": [["bash", "-lc", "test -x ./watch_log.sh && python3 validate.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertNotIn("log rotation/truncation handling", "\n".join(review["required_changes"]))

    def test_step_evidence_does_not_phrase_match_unrequested_log_truncation_implementation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create watch_log.sh. It should tail a log file path, poll every configurable interval, "
                    "remember the last checked line in .watch_state, and print TRIGGERED when a pattern appears."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Log watcher")
            agent.requirements["refined_requirements"] = [
                "The script must remember the last checked line in `.watch_state`.",
                "The script must print TRIGGERED when the pattern appears.",
            ]
            step = {
                "id": "S1",
                "title": "Implement log watcher",
                "description": "Create watch_log.sh, README.md, and validate.sh.",
                "depends_on": [],
                "acceptance_criteria": ["watch_log.sh detects new pattern matches and resumes from .watch_state."],
                "validation_commands": [["bash", "-lc", "test -x ./watch_log.sh && ./validate.sh"]],
            }

            findings = agent._evidence_findings(
                step,
                {
                    "written": ["watch_log.sh", "README.md", "validate.sh"],
                    "commands": [],
                    "raw": {
                        "plan_note": "Fixed truncation handling and added validation for it.",
                        "files": [
                            {
                                "path": "watch_log.sh",
                                "content": "#!/bin/bash\n# Handle log rotation/truncation by resetting state.\n",
                            }
                        ],
                    },
                },
                {
                    "validation_results": [{
                        "command": ["bash", "-lc", "test -x ./watch_log.sh && ./validate.sh"],
                        "returncode": 0,
                        "expected_returncode": 0,
                        "returncode_matches_expected": True,
                        "timed_out": False,
                        "stdout": "ALL TESTS PASSED\n",
                        "stderr": "",
                    }],
                    "workspace_files": [
                        {
                            "path": "README.md",
                            "content": "If the log file is truncated, the script resets its state.",
                        },
                        {
                            "path": "watch_log.sh",
                            "content": "#!/bin/bash\n# Handle log rotation/truncation by resetting state.\n",
                        },
                    ],
                    "git": {"enabled": True, "meaningful_changed_paths": ["watch_log.sh", "README.md", "validate.sh"]},
                },
            )

            text = "\n".join(findings)
            self.assertNotIn("Step evidence adds log rotation/truncation handling", text)

    def test_step_evidence_allows_removed_log_truncation_repair_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create watch_log.sh. It should tail a log file path, poll every configurable interval, "
                    "remember the last checked line in .watch_state, and print TRIGGERED when a pattern appears."
                ),
            )
            agent.initialize()

            findings = agent._unrequested_scope_expansion_findings(
                {
                    "workspace_files": [{
                        "path": "README.md",
                        "content": "The script assumes append-only logs; rotation/truncation handling is out of scope.",
                    }],
                    "plan_note": "Removed log truncation/rotation handling from watch_log.sh.",
                },
                source_label="Step evidence",
            )

            self.assertEqual(findings, [])

    def test_final_evidence_does_not_phrase_match_unrequested_log_truncation_implementation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create watch_log.sh. It should tail a log file path, poll every configurable interval, "
                    "remember the last checked line in .watch_state, and print TRIGGERED when a pattern appears."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Log watcher")
            agent.requirements["refined_requirements"] = [
                "The script must remember the last checked line in `.watch_state`.",
                "The script must print TRIGGERED when the pattern appears.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement log watcher",
                "description": "Create watch_log.sh, README.md, and validate.sh.",
                "depends_on": [],
                "acceptance_criteria": ["watch_log.sh detects new pattern matches and resumes from .watch_state."],
                "validation_commands": [["bash", "-lc", "test -x ./watch_log.sh && ./validate.sh"]],
            }]
            evidence = {
                "workspace_files": [{
                    "path": "README.md",
                    "content": "If the log file is truncated, the script resets its state.",
                }],
                "step_validations": [{
                    "step_id": "S1",
                    "validation_results": [{
                        "command": ["bash", "-lc", "test -x ./watch_log.sh && ./validate.sh"],
                        "returncode": 0,
                        "expected_returncode": 0,
                        "returncode_matches_expected": True,
                        "timed_out": False,
                        "stdout": "ALL TESTS PASSED\n",
                        "stderr": "",
                    }],
                }],
            }

            findings = agent._project_evidence_findings(
                [{"step_id": "S1", "status": "resolved", "attempts": [{"implementation": {"commands": []}}]}],
                evidence,
            )

            self.assertNotIn("Final project evidence adds log rotation/truncation handling", "\n".join(findings))

    def test_final_evidence_allows_log_truncation_as_negative_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create watch_log.sh. It should tail a log file path, poll every configurable interval, "
                    "remember the last checked line in .watch_state, and print TRIGGERED when a pattern appears."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Log watcher")
            agent.requirements["refined_requirements"] = [
                "The script must remember the last checked line in `.watch_state`.",
                "The script does not handle log rotation or truncation.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement log watcher",
                "description": "Create watch_log.sh, README.md, and validate.sh.",
                "depends_on": [],
                "acceptance_criteria": ["watch_log.sh detects new pattern matches and resumes from .watch_state."],
                "validation_commands": [["bash", "-lc", "test -x ./watch_log.sh && ./validate.sh"]],
            }]
            evidence = {
                "workspace_files": [{
                    "path": "README.md",
                    "content": "The script does not handle log rotation or truncation. Append-only logs are assumed.",
                }],
                "step_validations": [{
                    "step_id": "S1",
                    "validation_results": [{
                        "command": ["bash", "-lc", "test -x ./watch_log.sh && ./validate.sh"],
                        "returncode": 0,
                        "expected_returncode": 0,
                        "returncode_matches_expected": True,
                        "timed_out": False,
                        "stdout": "ALL TESTS PASSED\n",
                        "stderr": "",
                    }],
                }],
            }

            findings = agent._project_evidence_findings(
                [{"step_id": "S1", "status": "resolved", "attempts": [{"implementation": {"commands": []}}]}],
                evidence,
            )

            self.assertNotIn("log rotation/truncation handling", "\n".join(findings))

    def test_requirements_review_does_not_phrase_match_unrequested_public_state_file_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create watch_log.sh. It should tail a log file path, poll every configurable interval, "
                    "remember the last checked line in .watch_state, and print TRIGGERED when a pattern appears."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Log watcher")
            requirements["refined_requirements"] = [
                "The script must remember the last checked line in `.watch_state`.",
                "The script must support an optional `--state-file` argument for validation isolation.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement log watcher",
                "description": "Create watch_log.sh with `--state-file` support.",
                "depends_on": [],
                "acceptance_criteria": ["watch_log.sh exists and is executable."],
                "validation_commands": [["test", "-x", "watch_log.sh"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "resolved")
            self.assertNotIn("public state-file option", "\n".join(review["required_changes"]))

    def test_requirements_review_does_not_phrase_match_unrequested_public_state_short_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create watch_log.sh. It should tail a log file path, poll every configurable interval, "
                    "remember the last checked line in .watch_state, and print TRIGGERED when a pattern appears."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Log watcher")
            requirements["refined_requirements"] = [
                "The script must remember the last checked line in `.watch_state`.",
                "The script must support `--state` to choose a different state file path.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement log watcher",
                "description": "Create watch_log.sh with `--state` support.",
                "depends_on": [],
                "acceptance_criteria": ["watch_log.sh supports `--state`."],
                "validation_commands": [["test", "-x", "watch_log.sh"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "resolved")
            self.assertNotIn("public state-file option", "\n".join(review["required_changes"]))

    def test_requirements_review_accepts_unrequested_adjacency_as_open_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Interval merge")
            requirements["refined_requirements"] = [
                "Module `intervals.py` must contain `merge_intervals(pairs)`.",
                "`merge_intervals` must merge overlapping closed integer intervals.",
                "`merge_intervals` must raise `ValueError` if any interval has `start > end`.",
                "Include unit tests and README.",
            ]
            requirements["open_questions"] = [{
                "question": "Whether adjacent intervals should merge is not specified.",
                "resolution_strategy": "skip",
                "decision": "Do not add adjacency behavior unless the user requests it.",
            }]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py, tests, and README.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertNotIn("adjacency/contiguity behavior", "\n".join(review["required_changes"]))

    def test_requirements_review_accepts_user_requested_adjacency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge overlapping and adjacent closed integer intervals, "
                    "validate start <= end, and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Interval merge")
            requirements["refined_requirements"] = [
                "Module `intervals.py` must contain `merge_intervals(pairs)`.",
                "`merge_intervals` must merge overlapping and adjacent closed integer intervals.",
                "`merge_intervals` must raise `ValueError` if any interval has `start > end`.",
                "Include unit tests and README.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py, tests, and README.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "resolved")

    def test_requirements_review_does_not_treat_consecutive_characters_as_interval_adjacency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a tiny Python CLI slugify.py. It should read one command-line argument, "
                    "lowercase it, replace runs of non-alphanumeric characters with one hyphen, "
                    "trim hyphens, and print the slug. Include tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Slug CLI")
            requirements["refined_requirements"] = [
                "`slugify.py` must accept exactly one command-line argument.",
                "It must replace one or more consecutive non-alphanumeric characters with one hyphen.",
                "It must lowercase and trim leading and trailing hyphens.",
                "Include tests and README.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement slugify CLI, tests, and README",
                "description": "Create slugify.py, test_slugify.py, and README.md.",
                "depends_on": [],
                "acceptance_criteria": [
                    "`python slugify.py 'Hello, World!'` outputs `hello-world`.",
                    "Unit tests cover missing and extra argument error paths.",
                    "README.md contains a Usage section.",
                ],
                "validation_commands": [
                    ["python", "-m", "unittest", "test_slugify.py"],
                    ["bash", "-lc", "python slugify.py 'Hello, World!' | grep -q '^hello-world$'"],
                    ["bash", "-lc", "err=$(python slugify.py 2>&1); status=$?; test $status -ne 0 && printf '%s' \"$err\" | grep -qi usage"],
                    {"cmd": ["python", "slugify.py", "one", "two"], "expected_returncode": 2},
                    ["bash", "-lc", "grep -q 'Usage' README.md"],
                ],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertNotIn("adjacency/contiguity behavior", "\n".join(review["required_changes"]))

    def test_requirements_review_accepts_neutral_open_question_with_standard_collection_wording(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Interval merge")
            requirements["refined_requirements"] = [
                "The function `merge_intervals(pairs)` must merge overlapping or touching closed integer intervals.",
                "The function must raise `ValueError` if any interval has `start > end`.",
                "The function must correctly handle empty input, single intervals, nested intervals, and overlapping intervals.",
                "Include unit tests and a `README.md`.",
            ]
            requirements["assumptions"] = [
                "Input `pairs` is an iterable of two-element iterables of integers.",
                "Intervals are closed.",
            ]
            requirements["open_questions"] = [{
                "question": "What is the exact container type for input and output?",
                "resolution_strategy": "dilute",
                "decision": (
                    "The implementation will use a standard Python collection to represent the results; "
                    "the requirement is semantic and does not mandate a specific container type or input-type preservation."
                ),
            }]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement core logic, tests, and documentation",
                "description": "Create intervals.py, test_intervals.py, and README.md.",
                "depends_on": [],
                "acceptance_criteria": [
                    "`test_intervals.py` passes all tests.",
                    "`README.md` contains usage examples and complexity information.",
                ],
                "validation_commands": [
                    ["python", "-m", "unittest", "test_intervals.py"],
                    ["bash", "-lc", "grep -q 'Usage' README.md && grep -q 'Complexity' README.md"],
                ],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "resolved")
            self.assertNotIn("caller-visible output representation", "\n".join(review["required_changes"]))

    def test_requirements_review_rejects_open_question_decision_that_assumes_output_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Interval merge")
            requirements["refined_requirements"] = [
                "Module `intervals.py` must contain a function `merge_intervals(pairs)`.",
                "`merge_intervals` accepts an iterable of pairs, where each pair is an iterable of two integers.",
                "`merge_intervals` returns an iterable of merged intervals, where each interval is an iterable of two integers.",
                "Input validation: raise `ValueError` if any interval's start is greater than its end.",
                "Merging logic: intervals are closed; overlapping or touching intervals must be merged.",
            ]
            requirements["open_questions"] = [{
                "question": "What is the exact container type for the output intervals?",
                "resolution_strategy": "assume",
                "decision": "Assume list of lists for implementation to allow progress.",
            }]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement interval merge and tests",
                "description": "Create intervals.py and test_intervals.py.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "needs_requirements_change")
            text = "\n".join(review["required_changes"])
            self.assertIn("canonical caller-visible output representation", text)
            self.assertIn("user did not request that representation", text)

    def test_requirements_review_accepts_generic_list_of_merged_pairs_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Interval merge")
            requirements["refined_requirements"] = [
                "Implement `merge_intervals(pairs)` in `intervals.py`.",
                "The function must return a list of merged, non-overlapping, sorted pairs.",
                "The function must merge touching closed intervals.",
                "The function must raise `ValueError` if any pair has `start > end`.",
                "Unit tests must cover success cases and the ValueError path.",
            ]
            requirements["assumptions"] = [
                "Pairs are pair-like interval values; no concrete tuple/list output container is required.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement interval merge and unit tests",
                "description": "Create intervals.py and unit tests covering invalid intervals.",
                "depends_on": [],
                "acceptance_criteria": [
                    "`merge_intervals` returns expected merged interval values.",
                    "`merge_intervals` raises `ValueError` for invalid intervals.",
                ],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "resolved")
            self.assertNotIn("caller-visible output representation", "\n".join(review["required_changes"]))

    def test_requirements_review_accepts_explicit_shape_preservation_for_pair_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module pairs.py with normalize_pairs(pairs). "
                    "It must normalize each numeric pair by sorting the two values, "
                    "preserve each input pair's list or tuple type in the returned pairs, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Pair normalization")
            requirements["refined_requirements"] = [
                "Implement `normalize_pairs(pairs)` in `pairs.py`.",
                "Accept a list of iterables, e.g., `[[1, 3], [2, 6]]` or `[(1, 3), (2, 6)]`.",
                "Return normalized pairs using the same iterable type used by the input pairs.",
                "Include unit tests for tuple preservation and list preservation.",
            ]
            requirements["assumptions"] = [
                "The output format (inner iterable type) will match the input format.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement pair normalization",
                "description": "Create pairs.py and tests.",
                "depends_on": [],
                "acceptance_criteria": [
                    "`normalize_pairs([(3, 1), (6, 2)])` returns `[(1, 3), (2, 6)]`.",
                    "`normalize_pairs([[3, 1], [6, 2]])` returns `[[1, 3], [2, 6]]`.",
                ],
                "validation_commands": [["python", "-m", "unittest", "test_pairs.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertNotIn("caller-visible output representation", "\n".join(review["required_changes"]))

    def test_requirements_review_rejects_unrequested_shape_preservation_assumption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Interval merge")
            requirements["refined_requirements"] = [
                "Implement `merge_intervals(pairs)` function.",
                "Merge overlapping closed integer intervals.",
                "Validate that for every interval in `pairs`, the first element is less than or equal to the second.",
                "Include unit tests using `unittest`.",
            ]
            requirements["assumptions"] = [
                "The input and output should preserve the natural shape of the input, "
                "for example list-pair input returns list-pair output and tuple-pair input returns tuple-pair output.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement interval merge and tests",
                "description": "Create intervals.py and test_intervals.py.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "needs_requirements_change")
            text = "\n".join(review["required_changes"])
            self.assertIn("same-input-type preservation policy", text)
            self.assertNotIn("natural shape", text.lower())

    def test_requirements_review_rejects_type_signature_even_with_shape_preservation_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Interval merge")
            requirements["refined_requirements"] = [
                "Implement `merge_intervals(pairs: list[list[int]]) -> list[list[int]]` in `intervals.py`.",
                "Merge overlapping closed integer intervals.",
                "Include unit tests using `unittest`.",
            ]
            requirements["assumptions"] = [
                "The output should maintain the same structure as the input list of lists/tuples.",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement interval merge and tests",
                "description": "Create intervals.py and test_intervals.py.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "needs_requirements_change")
            self.assertIn("canonical caller-visible output representation", "\n".join(review["required_changes"]))

    def test_step_evidence_rejects_implementation_shape_conversion_for_flexible_pair_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Interval merge")
            agent.requirements["refined_requirements"] = [
                "Function `merge_intervals(pairs)` must accept pair-like intervals.",
                "Function must return merged interval values.",
            ]
            step = {
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
                "status": "pending",
            }

            findings = agent._evidence_findings(
                step,
                {"written": ["intervals.py", "test_intervals.py"], "commands": [], "raw": {}},
                {
                    "validation_results": [
                        {
                            "command": ["python", "-m", "unittest", "test_intervals.py"],
                            "returncode": 0,
                            "expected_returncode": 0,
                            "returncode_matches_expected": True,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "OK\n",
                        }
                    ],
                    "workspace_files": [
                        {
                            "path": "intervals.py",
                            "content": (
                                "def merge_intervals(pairs):\n"
                                "    \"\"\"Returns merged intervals as lists.\"\"\"\n"
                                "    return [list(pair) for pair in pairs]\n"
                            ),
                        },
                        {
                            "path": "test_intervals.py",
                            "content": (
                                "from intervals import merge_intervals\n"
                                "def test_input_with_tuples():\n"
                                "    assert merge_intervals([(1, 2), (2, 3)]) == [[1, 3]]\n"
                            ),
                        },
                    ],
                    "git": {"enabled": True, "meaningful_changed_paths": ["intervals.py", "test_intervals.py"]},
                },
            )

            text = "\n".join(findings)
            self.assertIn("canonical output representation", text)
            self.assertIn("`merge_intervals`", text)

    def test_step_evidence_accepts_documented_canonical_shape_with_representative_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Interval merge")
            agent.requirements["refined_requirements"] = [
                "Function `merge_intervals(pairs)` must accept pair-like intervals.",
                "Function must return merged interval values.",
            ]
            step = {
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
                "status": "pending",
            }

            findings = agent._evidence_findings(
                step,
                {"written": ["intervals.py", "test_intervals.py", "README.md"], "commands": [], "raw": {}},
                {
                    "validation_results": [
                        {
                            "command": ["python", "-m", "unittest", "test_intervals.py"],
                            "returncode": 0,
                            "expected_returncode": 0,
                            "returncode_matches_expected": True,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "OK\n",
                        },
                        {
                            "command": ["grep", "-q", "list of lists", "README.md"],
                            "returncode": 0,
                            "expected_returncode": 0,
                            "returncode_matches_expected": True,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "",
                        },
                    ],
                    "workspace_files": [
                        {
                            "path": "intervals.py",
                            "content": (
                                "def merge_intervals(pairs):\n"
                                "    merged = []\n"
                                "    for interval in pairs:\n"
                                "        merged.append(list(interval))\n"
                                "    return merged\n"
                            ),
                        },
                        {
                            "path": "test_intervals.py",
                            "content": (
                                "from intervals import merge_intervals\n"
                                "def test_list_input_returns_lists():\n"
                                "    assert merge_intervals([[1, 2], [2, 3]]) == [[1, 2], [2, 3]]\n"
                                "def test_tuple_input_returns_lists():\n"
                                "    result = merge_intervals([(1, 2), (2, 3)])\n"
                                "    assert isinstance(result[0], list)\n"
                            ),
                        },
                        {
                            "path": "README.md",
                            "content": (
                                "## Returns\n\n"
                                "A list of lists, where each inner list is a merged pair.\n"
                            ),
                        },
                    ],
                    "git": {"enabled": True, "meaningful_changed_paths": ["intervals.py", "test_intervals.py", "README.md"]},
                },
            )

            self.assertNotIn("canonical output representation", "\n".join(findings))

    def test_step_evidence_accepts_source_documented_canonical_shape_before_readme_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Interval merge")
            agent.requirements["refined_requirements"] = [
                "Function `merge_intervals(pairs)` must accept pair-like intervals.",
                "Function must return merged interval values.",
                "README.md will be created in a later documentation step.",
            ]
            step = {
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
                "status": "pending",
            }

            findings = agent._evidence_findings(
                step,
                {"written": ["intervals.py", "test_intervals.py"], "commands": [], "raw": {}},
                {
                    "validation_results": [
                        {
                            "command": ["python", "-m", "unittest", "test_intervals.py"],
                            "returncode": 0,
                            "expected_returncode": 0,
                            "returncode_matches_expected": True,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "OK\n",
                        }
                    ],
                    "workspace_files": [
                        {
                            "path": "intervals.py",
                            "content": (
                                "def merge_intervals(pairs):\n"
                                "    \"\"\"Return merged intervals as a list of lists.\"\"\"\n"
                                "    return [list(pair) for pair in pairs]\n"
                            ),
                        },
                        {
                            "path": "test_intervals.py",
                            "content": (
                                "from intervals import merge_intervals\n"
                                "def test_list_input_returns_lists():\n"
                                "    assert merge_intervals([[1, 2], [2, 3]]) == [[1, 2], [2, 3]]\n"
                                "def test_tuple_input_returns_lists():\n"
                                "    assert merge_intervals([(1, 2), (2, 3)]) == [[1, 2], [2, 3]]\n"
                            ),
                        },
                    ],
                    "git": {"enabled": True, "meaningful_changed_paths": ["intervals.py", "test_intervals.py"]},
                },
            )

            self.assertNotIn("canonical output representation", "\n".join(findings))

    def test_step_evidence_accepts_returns_list_as_tuples_docstring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Interval merge")
            agent.requirements["refined_requirements"] = [
                "Function `merge_intervals(pairs)` must accept pair-like intervals.",
                "Function must return merged interval values.",
            ]
            step = {
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
                "status": "pending",
            }

            findings = agent._evidence_findings(
                step,
                {"written": ["intervals.py", "test_intervals.py"], "commands": [], "raw": {}},
                {
                    "validation_results": [
                        {
                            "command": ["python", "-m", "unittest", "test_intervals.py"],
                            "returncode": 0,
                            "expected_returncode": 0,
                            "returncode_matches_expected": True,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "OK\n",
                        }
                    ],
                    "workspace_files": [
                        {
                            "path": "intervals.py",
                            "content": (
                                "def merge_intervals(pairs):\n"
                                "    \"\"\"\n"
                                "    Returns:\n"
                                "        A list of merged intervals as tuples (start, end).\n"
                                "    \"\"\"\n"
                                "    return [tuple(pair) for pair in pairs]\n"
                            ),
                        },
                        {
                            "path": "test_intervals.py",
                            "content": (
                                "import unittest\n"
                                "from intervals import merge_intervals\n"
                                "class TestMergeIntervals(unittest.TestCase):\n"
                                "    def test_mixed_input_types(self):\n"
                                "        self.assertEqual(merge_intervals([[1, 3], [2, 4]]), [(1, 4)])\n"
                                "        self.assertEqual(merge_intervals([(1, 3), (2, 4)]), [(1, 4)])\n"
                            ),
                        },
                    ],
                    "git": {"enabled": True, "meaningful_changed_paths": ["intervals.py", "test_intervals.py"]},
                },
            )

            self.assertNotIn("canonical output representation", "\n".join(findings))

    def test_step_evidence_accepts_variable_based_canonical_shape_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Interval merge")
            agent.requirements["refined_requirements"] = [
                "Function `merge_intervals(pairs)` must accept pair-like intervals.",
                "Function must return merged interval values.",
            ]
            step = {
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
                "status": "pending",
            }

            findings = agent._evidence_findings(
                step,
                {"written": ["intervals.py", "test_intervals.py", "README.md"], "commands": [], "raw": {}},
                {
                    "validation_results": [
                        {
                            "command": ["python", "-m", "unittest", "test_intervals.py"],
                            "returncode": 0,
                            "expected_returncode": 0,
                            "returncode_matches_expected": True,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "OK\n",
                        }
                    ],
                    "workspace_files": [
                        {
                            "path": "intervals.py",
                            "content": (
                                "def merge_intervals(pairs):\n"
                                "    \"\"\"\n"
                                "    Merge closed integer intervals.\n"
                                "\n"
                                "    Returns:\n"
                                "        A list of tuples, where each tuple is a merged interval.\n"
                                "    \"\"\"\n"
                                "    return [tuple(pair) for pair in pairs]\n"
                            ),
                        },
                        {
                            "path": "test_intervals.py",
                            "content": (
                                "import unittest\n"
                                "from intervals import merge_intervals\n"
                                "class TestMergeIntervals(unittest.TestCase):\n"
                                "    def test_input_types_output_type(self):\n"
                                "        input_lists = [[1, 3], [2, 4]]\n"
                                "        input_tuples = [(1, 3), (2, 4)]\n"
                                "        expected = [(1, 4)]\n"
                                "        res_lists = merge_intervals(input_lists)\n"
                                "        self.assertEqual(res_lists, expected)\n"
                                "        self.assertTrue(all(isinstance(item, tuple) for item in res_lists))\n"
                                "        res_tuples = merge_intervals(input_tuples)\n"
                                "        self.assertEqual(res_tuples, expected)\n"
                                "        self.assertTrue(all(isinstance(item, tuple) for item in res_tuples))\n"
                            ),
                        },
                        {
                            "path": "README.md",
                            "content": "## Return Type\n\nReturns a list of tuples for merged intervals.\n",
                        },
                    ],
                    "git": {"enabled": True, "meaningful_changed_paths": ["intervals.py", "test_intervals.py", "README.md"]},
                },
            )

            self.assertNotIn("canonical output representation", "\n".join(findings))

    def test_step_evidence_accepts_expected_variable_shape_for_list_and_tuple_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Interval merge")
            agent.requirements["refined_requirements"] = [
                "Function `merge_intervals(pairs)` must accept pair-like intervals.",
                "Function must return merged interval values.",
            ]
            step = {
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
                "status": "pending",
            }

            findings = agent._evidence_findings(
                step,
                {"written": ["intervals.py", "test_intervals.py"], "commands": [], "raw": {}},
                {
                    "validation_results": [
                        {
                            "command": ["python", "-m", "unittest", "test_intervals.py"],
                            "returncode": 0,
                            "expected_returncode": 0,
                            "returncode_matches_expected": True,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "OK\n",
                        }
                    ],
                    "workspace_files": [
                        {
                            "path": "intervals.py",
                            "content": (
                                "def merge_intervals(pairs):\n"
                                "    \"\"\"Return merged intervals as a list of lists.\"\"\"\n"
                                "    return [list(pair) for pair in pairs]\n"
                            ),
                        },
                        {
                            "path": "test_intervals.py",
                            "content": (
                                "import unittest\n"
                                "from intervals import merge_intervals\n"
                                "class TestMergeIntervals(unittest.TestCase):\n"
                                "    def test_input_types_canonical_output(self):\n"
                                "        expected = [[1, 4]]\n"
                                "        self.assertEqual(merge_intervals([(1, 3), (2, 4)]), expected)\n"
                                "        self.assertEqual(merge_intervals([[1, 3], [2, 4]]), expected)\n"
                            ),
                        },
                    ],
                    "git": {"enabled": True, "meaningful_changed_paths": ["intervals.py", "test_intervals.py"]},
                },
            )

            self.assertNotIn("canonical output representation", "\n".join(findings))

    def test_step_evidence_rejects_mixed_input_as_replacement_for_list_pair_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Interval merge")
            agent.requirements["refined_requirements"] = [
                "Function `merge_intervals(pairs)` must accept pair-like intervals.",
                "Function must return merged interval values.",
            ]
            step = {
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
                "status": "pending",
            }

            findings = agent._evidence_findings(
                step,
                {"written": ["intervals.py", "test_intervals.py"], "commands": [], "raw": {}},
                {
                    "validation_results": [
                        {
                            "command": ["python", "-m", "unittest", "test_intervals.py"],
                            "returncode": 0,
                            "expected_returncode": 0,
                            "returncode_matches_expected": True,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "OK\n",
                        }
                    ],
                    "workspace_files": [
                        {
                            "path": "intervals.py",
                            "content": (
                                "def merge_intervals(pairs):\n"
                                "    \"\"\"Return merged intervals as a list of lists.\"\"\"\n"
                                "    return [list(pair) for pair in pairs]\n"
                            ),
                        },
                        {
                            "path": "test_intervals.py",
                            "content": (
                                "from intervals import merge_intervals\n"
                                "def test_input_types():\n"
                                "    expected = [[1, 4]]\n"
                                "    assert merge_intervals([(1, 3), (2, 4)]) == expected\n"
                                "    assert merge_intervals([(1, 3), [2, 4]]) == expected\n"
                            ),
                        },
                    ],
                    "git": {"enabled": True, "meaningful_changed_paths": ["intervals.py", "test_intervals.py"]},
                },
            )

            text = "\n".join(findings)
            self.assertIn("canonical output representation", text)
            self.assertIn("list of list pairs", text)

    def test_step_evidence_accepts_list_of_intervals_as_lists_documentation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Interval merge")
            agent.requirements["refined_requirements"] = [
                "Function `merge_intervals(pairs)` must accept pair-like intervals.",
                "Function must return merged interval values.",
            ]
            step = {
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
                "status": "pending",
            }

            findings = agent._evidence_findings(
                step,
                {"written": ["intervals.py", "test_intervals.py", "README.md"], "commands": [], "raw": {}},
                {
                    "validation_results": [
                        {
                            "command": ["python", "-m", "unittest", "test_intervals.py"],
                            "returncode": 0,
                            "expected_returncode": 0,
                            "returncode_matches_expected": True,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "OK\n",
                        }
                    ],
                    "workspace_files": [
                        {
                            "path": "intervals.py",
                            "content": (
                                "def merge_intervals(pairs):\n"
                                "    \"\"\"\n"
                                "    Merge closed integer intervals.\n"
                                "\n"
                                "    Returns:\n"
                                "        A list of merged intervals, where each interval is a list of two integers.\n"
                                "    \"\"\"\n"
                                "    merged = []\n"
                                "    for interval in pairs:\n"
                                "        merged.append(list(interval))\n"
                                "    return merged\n"
                            ),
                        },
                        {
                            "path": "test_intervals.py",
                            "content": (
                                "from intervals import merge_intervals\n"
                                "def test_input_types():\n"
                                "    assert merge_intervals([(1, 2), (2, 3)]) == [[1, 3]]\n"
                                "    assert merge_intervals([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]\n"
                            ),
                        },
                        {
                            "path": "README.md",
                            "content": (
                                "# Interval Merger\n\n"
                                "The function returns a list of merged intervals as lists of two integers.\n"
                            ),
                        },
                    ],
                    "git": {"enabled": True, "meaningful_changed_paths": ["intervals.py", "test_intervals.py", "README.md"]},
                },
            )

            self.assertNotIn("canonical output representation", "\n".join(findings))

    def test_step_evidence_rejects_documented_canonical_shape_without_representative_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Interval merge")
            agent.requirements["refined_requirements"] = [
                "Function `merge_intervals(pairs)` must accept pair-like intervals.",
                "Function must return merged interval values.",
            ]
            step = {
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
                "status": "pending",
            }

            findings = agent._evidence_findings(
                step,
                {"written": ["intervals.py", "test_intervals.py", "README.md"], "commands": [], "raw": {}},
                {
                    "validation_results": [
                        {
                            "command": ["python", "-m", "unittest", "test_intervals.py"],
                            "returncode": 0,
                            "expected_returncode": 0,
                            "returncode_matches_expected": True,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "OK\n",
                        }
                    ],
                    "workspace_files": [
                        {
                            "path": "intervals.py",
                            "content": (
                                "def merge_intervals(pairs):\n"
                                "    return [list(pair) for pair in pairs]\n"
                            ),
                        },
                        {
                            "path": "test_intervals.py",
                            "content": (
                                "from intervals import merge_intervals\n"
                                "def test_list_input_returns_lists():\n"
                                "    assert merge_intervals([[1, 2], [2, 3]]) == [[1, 2], [2, 3]]\n"
                            ),
                        },
                        {
                            "path": "README.md",
                            "content": "## Returns\n\nA list of lists.\n",
                        },
                    ],
                    "git": {"enabled": True, "meaningful_changed_paths": ["intervals.py", "test_intervals.py", "README.md"]},
                },
            )

            self.assertIn("canonical output representation", "\n".join(findings))

    def test_step_evidence_rejects_shape_conversion_despite_prose_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Interval merge")
            agent.requirements["refined_requirements"] = [
                "Function `merge_intervals(pairs)` must accept pair-like intervals.",
                "Function must return merged interval values.",
            ]
            step = {
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
                "status": "pending",
            }

            findings = agent._evidence_findings(
                step,
                {"written": ["intervals.py", "test_intervals.py", "README.md"], "commands": [], "raw": {}},
                {
                    "validation_results": [
                        {
                            "command": ["python", "-m", "unittest", "test_intervals.py"],
                            "returncode": 0,
                            "expected_returncode": 0,
                            "returncode_matches_expected": True,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "OK\n",
                        }
                    ],
                    "workspace_files": [
                        {
                            "path": "intervals.py",
                            "content": (
                                "def merge_intervals(pairs):\n"
                                "    merged = []\n"
                                "    for interval in pairs:\n"
                                "        merged.append(list(interval))\n"
                                "    return merged\n"
                            ),
                        },
                        {
                            "path": "test_intervals.py",
                            "content": (
                                "from intervals import merge_intervals\n"
                                "def test_list_input():\n"
                                "    assert merge_intervals([[1, 2], [2, 3]]) == [[1, 2], [2, 3]]\n"
                            ),
                        },
                        {
                            "path": "README.md",
                            "content": "The function preserves the natural shape of the input pairs.\n",
                        },
                    ],
                    "git": {"enabled": True, "meaningful_changed_paths": ["intervals.py", "test_intervals.py", "README.md"]},
                },
            )

            self.assertIn("canonical output representation", "\n".join(findings))

    def test_step_evidence_rejects_quiet_source_level_pair_shape_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Interval merge")
            agent.requirements["refined_requirements"] = [
                "Function `merge_intervals(pairs)` must accept pair-like intervals.",
                "Function must return merged interval values.",
            ]
            step = {
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
                "status": "pending",
            }

            findings = agent._evidence_findings(
                step,
                {"written": ["intervals.py", "test_intervals.py"], "commands": [], "raw": {}},
                {
                    "validation_results": [
                        {
                            "command": ["python", "-m", "unittest", "test_intervals.py"],
                            "returncode": 0,
                            "expected_returncode": 0,
                            "returncode_matches_expected": True,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "OK\n",
                        }
                    ],
                    "workspace_files": [
                        {
                            "path": "intervals.py",
                            "content": (
                                "def merge_intervals(pairs):\n"
                                "    merged = []\n"
                                "    for interval in pairs:\n"
                                "        if not merged or interval[0] > merged[-1][1]:\n"
                                "            merged.append(list(interval))\n"
                                "        else:\n"
                                "            merged[-1][1] = max(merged[-1][1], interval[1])\n"
                                "    return merged\n"
                            ),
                        },
                        {
                            "path": "test_intervals.py",
                            "content": (
                                "from intervals import merge_intervals\n"
                                "def test_list_input():\n"
                                "    assert merge_intervals([[1, 2], [2, 3]]) == [[1, 3]]\n"
                            ),
                        },
                    ],
                    "git": {"enabled": True, "meaningful_changed_paths": ["intervals.py", "test_intervals.py"]},
                },
            )

            self.assertIn("canonical output representation", "\n".join(findings))

    def test_step_evidence_accepts_conversion_when_requirements_name_return_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Interval merge")
            agent.requirements["refined_requirements"] = [
                "Implement `merge_intervals(pairs: list[list[int]]) -> list[list[int]]` in `intervals.py`.",
                "Function must merge touching closed intervals.",
                "Function must raise `ValueError` if any pair has `start > end`.",
            ]
            step = {
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
                "status": "pending",
            }

            findings = agent._evidence_findings(
                step,
                {"written": ["intervals.py"], "commands": [], "raw": {}},
                {
                    "validation_results": [
                        {
                            "command": ["python", "-m", "unittest", "test_intervals.py"],
                            "returncode": 0,
                            "expected_returncode": 0,
                            "returncode_matches_expected": True,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "OK\n",
                        }
                    ],
                    "workspace_files": [
                        {
                            "path": "intervals.py",
                            "content": (
                                "def merge_intervals(pairs: list[list[int]]) -> list[list[int]]:\n"
                                "    merged = []\n"
                                "    for interval in pairs:\n"
                                "        merged.append(list(interval))\n"
                                "    return merged\n"
                            ),
                        },
                    ],
                    "git": {"enabled": True, "meaningful_changed_paths": ["intervals.py"]},
                },
            )

            self.assertNotIn("canonical output representation", "\n".join(findings))

    def test_step_evidence_accepts_shape_preservation_for_flexible_pair_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Interval merge")
            step = {
                "id": "S1",
                "title": "Implement interval merge",
                "description": "Create intervals.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
                "status": "pending",
            }

            findings = agent._evidence_findings(
                step,
                {"written": ["intervals.py", "test_intervals.py"], "commands": [], "raw": {}},
                {
                    "validation_results": [
                        {
                            "command": ["python", "-m", "unittest", "test_intervals.py"],
                            "returncode": 0,
                            "expected_returncode": 0,
                            "returncode_matches_expected": True,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "OK\n",
                        }
                    ],
                    "workspace_files": [
                        {
                            "path": "intervals.py",
                            "content": (
                                "def merge_intervals(pairs):\n"
                                "    \"\"\"Preserve the input pair container type in merged output.\"\"\"\n"
                                "    return pairs\n"
                            ),
                        },
                        {
                            "path": "test_intervals.py",
                            "content": (
                                "from intervals import merge_intervals\n"
                                "def test_tuple_input_preserves_tuples():\n"
                                "    assert merge_intervals([(1, 2), (2, 3)]) == [(1, 3)]\n"
                                "def test_list_input_preserves_lists():\n"
                                "    assert merge_intervals([[1, 2], [2, 3]]) == [[1, 3]]\n"
                            ),
                        },
                    ],
                    "git": {"enabled": True, "meaningful_changed_paths": ["intervals.py", "test_intervals.py"]},
                },
            )

            self.assertNotIn("canonical output representation", "\n".join(findings))

    def test_step_evidence_rejects_unrequested_pretty_json_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create normalize_config.py. It reads a JSON object file path, recursively sorts object keys, "
                    "removes keys with null values, keeps list order, and writes normalized JSON to stdout. Include tests."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Config normalizer")
            step = {
                "id": "S1",
                "title": "Implement normalizer",
                "description": "Create normalize_config.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass.", "CLI writes normalized JSON to stdout."],
                "validation_commands": [["python", "-m", "unittest", "test_normalize_config.py"]],
                "status": "pending",
            }

            findings = agent._evidence_findings(
                step,
                {"written": ["normalize_config.py", "test_normalize_config.py"], "commands": [], "raw": {}},
                {
                    "validation_results": [
                        {
                            "command": ["python", "-m", "unittest", "test_normalize_config.py"],
                            "returncode": 0,
                            "expected_returncode": 0,
                            "returncode_matches_expected": True,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "OK\n",
                        }
                    ],
                    "workspace_files": [
                        {
                            "path": "normalize_config.py",
                            "content": (
                                "import json\n"
                                "def main():\n"
                                "    data = {'a': 1}\n"
                                "    print(json.dumps(data, indent=4))\n"
                            ),
                        },
                        {
                            "path": "test_normalize_config.py",
                            "content": "def test_placeholder():\n    assert True\n",
                        },
                    ],
                    "git": {"enabled": True, "meaningful_changed_paths": ["normalize_config.py", "test_normalize_config.py"]},
                },
            )

            text = "\n".join(findings)
            self.assertIn("pretty-print or indent JSON stdout", text)
            self.assertIn("compact deterministic JSON", text)

    def test_requirements_review_doc_summary_omits_stale_shape_wording(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Interval merge")
            agent.requirements["refined_requirements"] = [
                "Function `merge_intervals(pairs)` must return merged interval values.",
            ]
            review = {
                "status": "resolved",
                "summary": (
                    "The agent now preserves the natural shape of the input by assuming the return type "
                    "will match the input's container type, e.g. list of lists."
                ),
            }

            sanitized = agent._requirements_review_for_doc(review)

            self.assertNotIn("match the input", sanitized["summary"])
            self.assertIn("refined requirements above are authoritative", sanitized["summary"])

    def test_requirements_review_rejects_flexible_but_unverified_public_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module pairs.py with normalize_pairs(pairs). "
                    "It must normalize each numeric pair by sorting the two values "
                    "and include unit tests and README."
                ),
            )
            agent.initialize()
            requirements = base_requirements("Pair normalization")
            requirements["refined_requirements"] = [
                "Implement `normalize_pairs(pairs)` in `pairs.py`.",
                "Input `pairs` is an iterable of pair-like iterables.",
                "The output format is a collection of normalized records, e.g. a list of lists or list of tuples.",
            ]
            requirements["assumptions"] = [
                "The output format is a collection of normalized records (e.g., a list of lists or list of tuples).",
            ]
            requirements["plan"] = [{
                "id": "S1",
                "title": "Implement pair normalization",
                "description": "Create pairs.py and tests.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_pairs.py"]],
            }]

            review = agent._requirements_review(1, requirements)

            self.assertEqual(review["status"], "needs_requirements_change")
            self.assertIn("canonical caller-visible output representation", "\n".join(review["required_changes"]))

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

    def test_stale_reviewer_validation_with_passing_corrected_command_requests_plan_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, feedback_responses=[
                json.dumps({
                    "status": "resolved",
                    "needs_rework": False,
                    "summary": "Reviewer would accept without deterministic stale-plan evidence.",
                    "required_changes": [],
                    "verification_evidence": ["implementation command evidence inspected"],
                })
            ])
            agent.initialize()
            agent.requirements = base_requirements("CLI validation")
            step = {
                "id": "S3",
                "title": "Testing and CLI Verification",
                "description": "Run CLI checks.",
                "depends_on": [],
                "acceptance_criteria": ["CLI output matches expected slugs."],
                "validation_commands": [{
                    "cmd": [
                        "python",
                        "-c",
                        "import subprocess; subprocess.run(['python', 'slugify.py', '---Test---'], check=True)",
                    ]
                }],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])
            (workspace / "slugify.py").write_text(
                "import argparse\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('text')\n"
                "args = parser.parse_args()\n"
                "print(args.text.strip('-').lower())\n",
                encoding="utf-8",
            )
            implementation = {
                "written": [],
                "commands": [{
                    "command": ["python", "slugify.py", "--", "---Test---"],
                    "returncode": 0,
                    "expected_returncode": 0,
                    "returncode_matches_expected": True,
                    "timed_out": False,
                    "stdout": "test\n",
                    "stderr": "",
                }],
                "raw": {},
            }

            review = agent._step_review_pass(step, 2, implementation, "hard_pushback")

            self.assertEqual(review["status"], "needs_plan_change")
            self.assertIn("stale or misaligned", "\n".join(review["required_changes"]))

    def test_blocked_logically_flawed_reviewer_validation_is_plan_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            result = {
                "command": ["python", "-c", "print('stale validation')"],
                "returncode": 126,
                "expected_returncode": 0,
                "timed_out": False,
                "stdout": "",
                "stderr": (
                    "Tool call blocked before execution by verification step: "
                    "The reviewer-owned validation command is logically flawed and does not actually verify the failure mode."
                ),
            }
            implementation_commands = [{
                "command": ["python", "validate_all.py"],
                "returncode": 0,
                "expected_returncode": 0,
                "timed_out": False,
            }]

            self.assertTrue(agent._looks_like_stale_or_misaligned_plan_validation_result(
                result,
                implementation_commands,
            ))

    def test_blocked_arithmetic_precedence_reviewer_validation_is_plan_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            result = {
                "command": ["python", "-c", "print('stale validation')"],
                "returncode": 126,
                "expected_returncode": 0,
                "timed_out": False,
                "stdout": "",
                "stderr": (
                    "Tool call blocked before execution by verification step: "
                    "The expression `(n // 10 * n % 10) % 6 == 0` is mathematically incorrect "
                    "due to operator precedence and does not calculate the product of digits."
                ),
            }
            implementation_commands = [{
                "command": ["python", "-c", "assert open('ANSWER.txt').read().strip() == '11'"],
                "returncode": 0,
                "expected_returncode": 0,
                "timed_out": False,
            }]

            self.assertTrue(agent._looks_like_stale_or_misaligned_plan_validation_result(
                result,
                implementation_commands,
            ))

    def test_step_review_suppresses_false_negative_path_shell_objection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, feedback_responses=[
                json.dumps({
                    "status": "approved",
                    "summary": "The command safely verifies the expected failure path.",
                    "commands": [
                        {
                            "index": 0,
                            "decision": "approved",
                            "risk_level": "low",
                            "reason": "The command is non-destructive and bounded.",
                        }
                    ],
                }),
                json.dumps({
                    "status": "needs_rework",
                    "needs_rework": True,
                    "summary": (
                        "The validation command used to verify the non-zero exit status "
                        "requirement is logically flawed and the test results are contradictory."
                    ),
                    "required_changes": [
                        "Fix the validation command to correctly verify that slugify.py exits with a non-zero status.",
                    ],
                    "cross_check_questions": [
                        "Does `(python3 slugify.py > /dev/null 2>&1 && exit 1 || exit 0)` actually return 0?",
                    ],
                    "verification_evidence": [
                        "The subshell logic returns 1 if the command under test returns 1.",
                    ],
                })
            ])
            agent.initialize()
            agent.requirements = base_requirements("CLI validation")
            step = {
                "id": "S1",
                "title": "CLI validation",
                "description": "Validate expected failure behavior.",
                "depends_on": [],
                "acceptance_criteria": ["CLI exits non-zero when no argument is provided."],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "python3 -m unittest test_slugify.py && (python3 slugify.py > /dev/null 2>&1 && exit 1 || exit 0)",
                ]],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])
            (workspace / "slugify.py").write_text(
                "import sys\n"
                "if len(sys.argv) != 2:\n"
                "    sys.exit(1)\n"
                "print(sys.argv[1].lower())\n",
                encoding="utf-8",
            )
            (workspace / "test_slugify.py").write_text(
                "import unittest\n"
                "\n"
                "class TestSlugify(unittest.TestCase):\n"
                "    def test_placeholder(self):\n"
                "        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            implementation = {
                "written": ["slugify.py", "test_slugify.py"],
                "commands": [{
                    "command": [
                        "bash",
                        "-lc",
                        "python3 -m unittest test_slugify.py && (python3 slugify.py > /dev/null 2>&1 && exit 1 || exit 0)",
                    ],
                    "returncode": 0,
                    "expected_returncode": 0,
                    "returncode_matches_expected": True,
                    "timed_out": False,
                    "stdout": "",
                    "stderr": ".\n----------------------------------------------------------------------\nRan 1 test in 0.000s\n\nOK\n",
                }],
                "raw": {},
            }

            review = agent._step_review_pass(step, 1, implementation, "hard_pushback")

            self.assertEqual(review["status"], "resolved")
            self.assertIn("expected-failure shell wrapper", review["summary"])
            self.assertEqual(review["required_changes"], [])
            self.assertEqual(
                review["suppressed_reviewer_findings"][0]["reason"],
                "unsupported_negative_path_shell_objection",
            )

    def test_tool_call_verification_blocks_negative_path_pipeline_without_status_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            findings = agent._deterministic_tool_call_findings([
                ["bash", "-lc", "python slugify.py 2>&1 | grep -q 'required'"],
            ])

            self.assertEqual(findings[0]["index"], 0)
            self.assertIn("without checking the command-under-test exit status", findings[0]["reason"])

    def test_tool_call_verification_blocks_grep_v_absence_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, prompt="Build monitor_disk.sh and README.md.")
            step = {
                "id": "S1",
                "title": "Implement disk monitor",
                "description": "Create monitor_disk.sh and README.md.",
                "acceptance_criteria": [
                    "Script avoids ACTION_REQUIRED above threshold.",
                    "Script reaches Check 2/2 with MAX_CHECKS=2.",
                ],
            }

            findings = agent._deterministic_tool_call_findings(
                [[
                    "bash",
                    "-lc",
                    "CHECK_INTERVAL_SECONDS=1 MAX_CHECKS=2 MIN_FREE_PERCENT=0 ./monitor_disk.sh | grep -v 'ACTION_REQUIRED' | grep -q 'Check 2/2'",
                ]],
                context={"step": step},
            )

            self.assertEqual(findings[0]["index"], 0)
            self.assertIn("filtering it out with `grep -v`", findings[0]["reason"])

    def test_tool_call_verification_blocks_grep_option_like_pattern_without_separator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            findings = agent._deterministic_tool_call_findings([
                ["bash", "-lc", "grep -q '--data-binary @' CURL_NOTES.md"],
            ])

            self.assertEqual(findings[0]["index"], 0)
            self.assertIn("starts with `--`", findings[0]["reason"])
            self.assertIn("grep -q -- PATTERN FILE", findings[0]["reason"])

    def test_tool_call_verification_allows_success_pipeline_with_failure_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            findings = agent._deterministic_tool_call_findings([
                [
                    "bash",
                    "-lc",
                    "python slugify.py 'Hello World!' | grep -q '^hello-world$' && echo 'CLI OK' || (echo 'CLI FAIL' && exit 1)",
                ],
            ])

            self.assertFalse(findings)

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
        self.assertIn("standard-library test runner", ANALYSIS_CONTRACT)
        self.assertIn("do not speculate", ANALYSIS_CONTRACT)
        self.assertIn("Scope boundary", ANALYSIS_CONTRACT)
        self.assertIn("Do not turn an unspecified detail", ANALYSIS_CONTRACT)

    def test_analysis_structural_findings_reject_unrequested_public_io_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )

            findings = agent._analysis_structural_findings({
                "problem_restatement": "Build a module with merge_intervals.",
                "domain_and_constraints": [
                    "Input: A list of integer pairs.",
                    "Output: A list of merged integer pairs.",
                ],
                "possible_solution_paths": [
                    {"id": "A", "description": "Sort interval values and merge."},
                    {"id": "B", "description": "Use a sweep-line approach."},
                ],
                "recommended_path": {"path_id": "A"},
                "analysis_quality": {
                    "is_comprehensive": True,
                    "is_domain_aware": True,
                    "is_actionable_for_planning": True,
                },
            })

            self.assertIn(
                "Analysis invents a concrete caller-visible input/output representation",
                "\n".join(findings),
            )

    def test_analysis_structural_findings_do_not_phrase_match_unrequested_adjacency_scope_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )

            findings = agent._analysis_structural_findings({
                "problem_restatement": "Build a module with merge_intervals.",
                "domain_and_constraints": [
                    "Input values are closed integer intervals.",
                    "Output values are merged closed integer intervals.",
                ],
                "possible_solution_paths": [
                    {"id": "A", "description": "Sort interval values and merge overlapping or adjacent intervals."},
                    {"id": "B", "description": "Use a sweep-line approach over interval boundaries."},
                ],
                "recommended_path": {"path_id": "A"},
                "analysis_quality": {
                    "is_comprehensive": True,
                    "is_domain_aware": True,
                    "is_actionable_for_planning": True,
                },
            })

            self.assertNotIn("adjacency/contiguity behavior", "\n".join(findings))

    def test_analysis_structural_findings_accept_unrequested_adjacency_as_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )

            findings = agent._analysis_structural_findings({
                "problem_restatement": "Build a module with merge_intervals.",
                "domain_and_constraints": [
                    "Input values are closed integer intervals.",
                    "Output values are merged closed integer intervals.",
                ],
                "initial_source_check": {
                    "sources_checked": ["none"],
                    "source_gaps": [],
                    "freshness_risks": [],
                },
                "possible_solution_paths": [
                    {"id": "A", "description": "Sort interval values and merge overlapping intervals."},
                    {"id": "B", "description": "Use a sweep-line approach over interval boundaries."},
                ],
                "recommended_path": {"path_id": "A"},
                "analysis_quality": {
                    "is_comprehensive": True,
                    "is_domain_aware": True,
                    "is_actionable_for_planning": True,
                    "remaining_unknowns": [
                        "Whether adjacent intervals should be merged is not specified by the request."
                    ],
                },
            })

            self.assertNotIn("adjacency/contiguity behavior", "\n".join(findings))

    def test_analysis_structural_findings_accept_unrequested_adjacency_as_fallback_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )

            findings = agent._analysis_structural_findings({
                "problem_restatement": "Build a module with merge_intervals.",
                "domain_and_constraints": [
                    "Input values are closed integer intervals.",
                    "Output values are merged closed integer intervals.",
                ],
                "initial_source_check": {
                    "sources_checked": ["none"],
                    "source_gaps": [],
                    "freshness_risks": [],
                },
                "possible_solution_paths": [
                    {"id": "A", "description": "Sort interval values and merge overlapping intervals."},
                    {"id": "B", "description": "Use a sweep-line approach over interval boundaries."},
                ],
                "recommended_path": {
                    "path_id": "A",
                    "fallback_trigger": "Use another approach if requirements later include adjacent intervals.",
                },
                "analysis_quality": {
                    "is_comprehensive": True,
                    "is_domain_aware": True,
                    "is_actionable_for_planning": True,
                    "remaining_unknowns": [],
                },
            })

            self.assertNotIn("adjacency/contiguity behavior", "\n".join(findings))

    def test_analysis_structural_findings_reject_hidden_representation_assumption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )

            findings = agent._analysis_structural_findings({
                "problem_restatement": "Build a module with merge_intervals.",
                "domain_and_constraints": [
                    "Input values are closed integer intervals with unspecified representation.",
                    "Output values are merged closed integer intervals.",
                ],
                "initial_source_check": {
                    "sources_checked": ["none"],
                    "source_gaps": [
                        "Specific data structure for pairs, e.g., list of tuples or list of lists, is not defined."
                    ],
                    "freshness_risks": [],
                },
                "possible_solution_paths": [
                    {"id": "A", "description": "Sort interval values and merge."},
                    {"id": "B", "description": "Use a sweep-line approach."},
                ],
                "recommended_path": {"path_id": "A"},
                "analysis_quality": {
                    "is_comprehensive": True,
                    "is_domain_aware": True,
                    "is_actionable_for_planning": True,
                    "remaining_unknowns": [
                        "Exact representation of pairs is unknown; I will assume a list of tuples/lists."
                    ],
                },
            })

            self.assertIn(
                "Analysis invents a concrete caller-visible input/output representation",
                "\n".join(findings),
            )

    def test_analysis_structural_findings_accept_unresolved_representation_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build a Python module intervals.py with merge_intervals(pairs). "
                    "It must merge closed integer intervals, validate start <= end, "
                    "and include unit tests and README."
                ),
            )

            findings = agent._analysis_structural_findings({
                "problem_restatement": "Build a module with merge_intervals.",
                "domain_and_constraints": [
                    "Input values are closed integer intervals with unspecified representation.",
                    "Output values are merged closed integer intervals.",
                ],
                "initial_source_check": {
                    "sources_checked": ["none"],
                    "source_gaps": [
                        "Specific data structure for pairs, e.g., list of tuples or list of lists, is not explicitly defined."
                    ],
                    "freshness_risks": [],
                },
                "possible_solution_paths": [
                    {"id": "A", "description": "Sort interval values and merge."},
                    {"id": "B", "description": "Use a sweep-line approach."},
                ],
                "recommended_path": {"path_id": "A"},
                "analysis_quality": {
                    "is_comprehensive": True,
                    "is_domain_aware": True,
                    "is_actionable_for_planning": True,
                    "remaining_unknowns": [
                        "The exact data structure for pairs, e.g., list of tuples vs list of lists."
                    ],
                },
            })

            self.assertNotIn(
                "Analysis invents a concrete caller-visible input/output representation",
                "\n".join(findings),
            )

    def test_analysis_structural_findings_reject_dependency_free_path_with_pytest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            findings = agent._analysis_structural_findings({
                "problem_restatement": "Build a small Python CLI with tests.",
                "possible_solution_paths": [
                    {
                        "id": "A",
                        "description": "Use the Python standard library for a tiny dependency-free CLI and unittest or pytest for testing.",
                    },
                    {
                        "id": "B",
                        "description": "Use a third-party package if advanced behavior is required.",
                    },
                ],
                "recommended_path": {"path_id": "A"},
                "analysis_quality": {
                    "is_comprehensive": True,
                    "is_domain_aware": True,
                    "is_actionable_for_planning": True,
                },
            })

            self.assertIn(
                "Analysis path A mixes a dependency-free/standard-library approach with an external test runner.",
                findings,
            )

    def test_analysis_structural_findings_accept_third_party_path_compared_to_stdlib(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            findings = agent._analysis_structural_findings({
                "problem_restatement": "Create a JSON normalizer.",
                "possible_solution_paths": [
                    {
                        "id": "A",
                        "description": "Use the Python standard library and unittest.",
                        "advantages": ["No external dependencies."],
                        "verification_strategy": "Run unittest checks.",
                    },
                    {
                        "id": "B",
                        "description": "Use a third-party helper library and pytest.",
                        "advantages": ["May help with unusual inputs."],
                        "risks": ["Adds dependency overhead."],
                        "verification_strategy": "Compare against a standard library implementation.",
                    },
                ],
                "recommended_path": {"path_id": "A"},
                "analysis_quality": {
                    "is_comprehensive": True,
                    "is_domain_aware": True,
                    "is_actionable_for_planning": True,
                },
            })

            self.assertNotIn(
                "Analysis path B mixes a dependency-free/standard-library approach with an external test runner.",
                findings,
            )

    def test_analysis_structural_findings_accept_common_camel_case_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            findings = agent._analysis_structural_findings({
                "problem_restatement": "Do the requested work.",
                "possible_solution_paths": [
                    {"id": "A", "description": "Direct path"},
                    {"id": "B", "description": "Alternative path"},
                ],
                "recommended_path": {"path_id": "A"},
                "analysis_quality": {
                    "is_comprehensive": True,
                    "is_domainAware": True,
                    "is_actionableForPlanning": True,
                },
            })

            self.assertEqual(findings, [])

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
                "evidence_reviewed": ["final_review:summary"],
                "runbook_updates": ["last checked line 10"],
            }
            agent = load_test_agent(root, workspace, feedback_responses=[json.dumps(retry_review)])
            agent.initialize()

            review = agent._approach_review_phase(1, [], {"status": "resolved", "iterations": []})

            self.assertTrue(agent._approach_review_requests_retry(review))
            self.assertEqual(review["recommended_next_approach"], "Watch the log again from the last checkpoint.")

    def test_approach_review_requires_evidence_ids_not_prose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            prose_review = {
                "status": "resolved",
                "needs_rework": False,
                "summary": "Keep the current approach.",
                "decision": "keep_result",
                "evidence_reviewed": ["The validation command proved the answer manually."],
                "runbook_updates": [],
            }
            repaired_review = {
                "status": "resolved",
                "needs_rework": False,
                "summary": "Keep the current approach based on final review evidence.",
                "decision": "keep_result",
                "evidence_reviewed": ["final_review:summary", "final_review:verification_evidence:0"],
                "runbook_updates": [],
            }
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[json.dumps(prose_review), json.dumps(repaired_review)],
            )
            agent.initialize()

            review = agent._approach_review_phase(
                1,
                [{"step_id": "S1", "status": "resolved", "attempts": []}],
                {
                    "status": "resolved",
                    "summary": "Final review accepted command evidence.",
                    "verification_evidence": ["Command returned exit code 0."],
                    "iterations": [{"review": {"status": "resolved", "deterministic_evidence_findings": []}}],
                },
            )

            self.assertIn("final_review:summary", review["evidence_reviewed"])
            self.assertIn("final_review:verification_evidence:0", review["evidence_reviewed"])
            self.assertTrue(all(item.startswith("final_review:") for item in review["evidence_reviewed"]))
            self.assertIn("cited evidence IDs", review["summary"])
            self.assertNotIn("manually", review["summary"])
            self.assertIn("Keep the current approach", review["reviewer_rationale"])
            transcript = (workspace / ".agent_state" / "conversation.jsonl").read_text(encoding="utf-8")
            self.assertIn("APPROACH_REVIEW_RESULT", transcript)
            self.assertIn("Approach review kept the result based on cited evidence IDs", transcript)
            self.assertEqual(len(agent.feedback_client.calls), 2)

    def test_approach_review_cannot_keep_failed_workflow_as_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            keep_failed_review = {
                "status": "resolved",
                "needs_rework": False,
                "summary": "Keep the current approach.",
                "decision": "keep_result",
                "evidence_reviewed": ["final_review:status"],
            }
            agent = load_test_agent(root, workspace, feedback_responses=[json.dumps(keep_failed_review)])
            agent.initialize()
            phase_result = {
                "status": "skipped_with_note",
                "iterations": [{
                    "review": {
                        "status": "needs_rework",
                        "summary": "Validation fixture was created but not wired into the validator.",
                        "required_changes": ["Pass the fixture path through a real option or run a copied workspace."],
                    }
                }],
                "resolution": {"status": "skipped_with_note", "note": "Bounded retries exhausted for requirements."},
            }

            review = agent._approach_review_phase(
                1,
                [{"step_id": "requirements_phase", "status": "cannot_resolve", "phase_result": phase_result}],
                {"status": "cannot_resolve", "iterations": []},
            )

            self.assertEqual(review["status"], "try_another_approach")
            self.assertEqual(review["decision"], "retry_with_new_approach")
            self.assertTrue(agent._approach_review_requests_retry(review))
            self.assertIn("requested another approach", review["summary"])
            self.assertIn("cannot_resolve", review["reviewer_rationale"])
            self.assertIn("Pass the fixture path", json.dumps(review["required_changes"]))
            self.assertIn("Pass the fixture path", review["recommended_next_approach"])

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

    def test_research_design_notes_step_satisfies_quality_gate_for_shell_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, title="disk monitor", prompt="Build a Bash disk monitor.")
            agent.initialize()
            agent.requirements = base_requirements("Disk monitor")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Research and Design",
                    "description": (
                        "Research df -P parsing patterns and write DESIGN_NOTES.md to document the "
                        "implementation strategy and environment variable handling."
                    ),
                    "depends_on": [],
                    "acceptance_criteria": [
                        "DESIGN_NOTES.md exists and contains df -P parsing notes.",
                    ],
                    "validation_commands": [["test", "-f", "DESIGN_NOTES.md"]],
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
            self.assertIn("harness-owned state", "\n".join(findings))

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
            self.assertIn("Structural repair rule", prompt)
            self.assertIn("treat the whole affected file as suspect", prompt)
            self.assertIn("Do not claim a structural repair based only on", prompt)

    def test_next_implementation_directive_includes_structural_repair_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            review = {
                "status": "needs_rework",
                "needs_rework": True,
                "summary": "test_intervals.py still has a SyntaxError.",
                "required_changes": [
                    "Fix line 52 and ensure there are no other delimiter mismatches.",
                ],
                "deterministic_evidence_findings": [
                    "SyntaxError: closing parenthesis ']' does not match opening parenthesis '('",
                ],
            }

            directive = agent._next_implementation_directive(review)

            self.assertIn("NEXT_IMPLEMENTATION_DIRECTIVE", directive)
            self.assertIn("Structural repair rule", directive)
            self.assertIn("treat the whole affected file as suspect", directive)
            self.assertIn("Do not claim a structural repair based only on", directive)
            self.assertIn("Review to apply", directive)
            self.assertIn("SyntaxError", directive)
            self.assertIn("Deterministic evidence findings are authoritative repair blockers", directive)
            self.assertIn("Do not claim a deterministic finding is fixed", directive)

    def test_workflow_state_includes_active_repair_findings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Active repair finding")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Repair script",
                "description": "Fix current implementation.",
                "depends_on": [],
                "acceptance_criteria": ["validation passes"],
                "validation_commands": [["python", "validate.py"]],
                "status": "pending",
            }]
            agent.active_repair_findings = [
                "S1 attempt 2 deterministic: remove unrequested scope expansion before retrying.",
                "S1 attempt 2 reviewer: validation command returned 1.",
            ]

            workflow_state = agent._workflow_state_for_prompt(agent.plan_steps[0])

            self.assertIn("Active repair findings:", workflow_state)
            self.assertIn("remove unrequested scope expansion", workflow_state)
            self.assertIn("validation command returned 1", workflow_state)

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

    def test_step_reference_check_normalizes_dot_slash_artifact_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, title="disk monitor", prompt="Build a disk monitor script.")
            agent.initialize()
            evidence = {
                "workspace_files": [
                    {
                        "path": "README.md",
                        "content": "Run `./monitor_disk.sh --check .` every 30 seconds.",
                    },
                    {
                        "path": "monitor_disk.sh",
                        "content": "#!/usr/bin/env bash\n",
                    },
                ],
            }

            self.assertEqual(agent._workspace_reference_findings(evidence, allow_planned_future_refs=False), [])

            missing_evidence = {
                "workspace_files": [
                    {
                        "path": "README.md",
                        "content": "Run `./missing.sh` after setup.",
                    },
                ],
            }
            findings = agent._workspace_reference_findings(missing_evidence, allow_planned_future_refs=False)

            self.assertIn("`missing.sh`", "\n".join(findings))
            self.assertNotIn("`/missing.sh`", "\n".join(findings))

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

    def test_compact_step_results_labels_claims_when_command_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            step_results = [{
                "step_id": "S1",
                "status": "resolved",
                "attempts": [{
                    "implementation": {
                        "written": ["ANSWER.txt"],
                        "commands": [{
                            "command": ["python", "-c", "open('ANSWER.txt', 'w').write('42')"],
                            "returncode": 126,
                            "expected_returncode": 0,
                            "returncode_matches_expected": False,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "Tool call blocked before execution by verification step: mutates artifact",
                            "blocked_by_tool_verifier": True,
                        }],
                        "raw": {
                            "test_evidence": [
                                "The command calculated the answer and wrote ANSWER.txt."
                            ],
                        },
                    },
                    "review": {
                        "status": "resolved",
                        "summary": "Reviewer-owned validation passed.",
                        "feedback_tool_evidence": {
                            "validation_results": [{
                                "command": ["python", "-c", "assert open('ANSWER.txt').read() == '42'"],
                                "returncode": 0,
                                "expected_returncode": 0,
                                "returncode_matches_expected": True,
                                "timed_out": False,
                                "stdout": "",
                                "stderr": "",
                            }],
                        },
                    },
                }],
            }]

            compact = agent._compact_step_results_for_prompt(step_results)

            self.assertNotIn("test_evidence", compact[0])
            self.assertEqual(compact[0]["implementation_command_summary"]["blocked"], 1)
            self.assertEqual(compact[0]["reviewer_validation_summary"]["passed"], 1)
            self.assertIn("implementation_test_evidence_claims", compact[0])
            self.assertIn("model-provided prose", compact[0]["evidence_note"])

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

    def test_evidence_findings_accept_any_nonzero_when_scope_says_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = {
                "project_summary": "CLI handles invalid input.",
                "refined_requirements": ["The CLI exits with a non-zero code for invalid input."],
                "assumptions": [],
            }
            step = {
                "id": "S1",
                "title": "Implement CLI",
                "description": "Create the CLI and validate bad input.",
                "depends_on": [],
                "acceptance_criteria": ["`cli.py --count abc` exits with a non-zero code."],
                "validation_commands": [{"cmd": ["python", "cli.py", "--count", "abc"], "expected_returncode": 1}],
                "status": "pending",
            }
            result = {
                "command": ["python", "cli.py", "--count", "abc"],
                "returncode": 2,
                "expected_returncode": 1,
                "returncode_matches_expected": False,
                "timed_out": False,
                "stdout": "",
                "stderr": "usage: cli.py --count COUNT\ncli.py: error: invalid int value: 'abc'\n",
            }

            findings = agent._evidence_findings(
                step,
                {"written": ["cli.py"], "commands": [result], "raw": {"test_evidence": ["bad input checked"]}},
                {
                    "validation_results": [result],
                    "workspace_files": [{"path": "cli.py", "content": "print('placeholder')\n"}],
                    "git": {"enabled": False, "meaningful_changed_paths": ["cli.py"]},
                },
            )

            text = "\n".join(findings)
            self.assertNotIn("returned 2 but expected 1", text)
            self.assertNotIn("only proves the success path", text)

    def test_evidence_findings_keep_exact_nonzero_exit_code_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = {
                "project_summary": "CLI handles invalid input.",
                "refined_requirements": ["The CLI exits with code 1 for invalid input."],
                "assumptions": [],
            }
            step = {
                "id": "S1",
                "title": "Implement CLI",
                "description": "Create the CLI and validate bad input.",
                "depends_on": [],
                "acceptance_criteria": ["`cli.py --count abc` exits with code 1."],
                "validation_commands": [{"cmd": ["python", "cli.py", "--count", "abc"], "expected_returncode": 1}],
                "status": "pending",
            }
            result = {
                "command": ["python", "cli.py", "--count", "abc"],
                "returncode": 2,
                "expected_returncode": 1,
                "returncode_matches_expected": False,
                "timed_out": False,
                "stdout": "",
                "stderr": "usage: cli.py --count COUNT\ncli.py: error: invalid int value: 'abc'\n",
            }

            findings = agent._evidence_findings(
                step,
                {"written": ["cli.py"], "commands": [result], "raw": {"test_evidence": ["bad input checked"]}},
                {
                    "validation_results": [result],
                    "workspace_files": [{"path": "cli.py", "content": "print('placeholder')\n"}],
                    "git": {"enabled": False, "meaningful_changed_paths": ["cli.py"]},
                },
            )

            self.assertIn("returned 2 but expected 1", "\n".join(findings))

    def test_computed_answer_silent_validation_failure_requests_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create ANSWER.txt only. Consider integers n from 1 to 120. "
                    "Keep n if n is divisible by exactly one of 3, 5, 7 and return the sum."
                ),
            )
            agent.initialize()
            step = {
                "id": "S1",
                "title": "Create computed answer",
                "description": "Create ANSWER.txt and validate the computed sum.",
                "depends_on": [],
                "acceptance_criteria": ["ANSWER.txt matches the recomputed sum."],
                "validation_commands": [["python", "-c", "raise SystemExit(1)"]],
                "status": "pending",
            }

            findings = agent._evidence_findings(
                step,
                {
                    "written": ["ANSWER.txt"],
                    "commands": [],
                    "raw": {"test_evidence": ["silent failed validation"]},
                },
                {
                    "validation_results": [
                        {
                            "command": ["python", "-c", "raise SystemExit(1)"],
                            "returncode": 1,
                            "expected_returncode": 0,
                            "returncode_matches_expected": False,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "",
                        }
                    ],
                    "workspace_files": [{"path": "ANSWER.txt", "content": "2036"}],
                    "git": {"enabled": False, "meaningful_changed_paths": ["ANSWER.txt"]},
                },
            )

            text = "\n".join(findings)
            self.assertIn("failed without stdout or stderr", text)
            self.assertIn("expected value", text)
            self.assertIn("actual artifact value", text)

    def test_runtime_state_artifact_requests_isolated_validation_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create watch_log.sh. It should remember the last checked "
                    "line in .watch_state while watching a log."
                ),
            )
            agent.initialize()
            step = {
                "id": "S1",
                "title": "Create log watcher",
                "description": "Implement a stateful log watcher.",
                "depends_on": [],
                "acceptance_criteria": ["The watcher remembers the last checked line in .watch_state."],
                "validation_commands": [["bash", "-lc", "./validate.sh"]],
                "status": "pending",
            }

            findings = agent._evidence_findings(
                step,
                {
                    "written": ["watch_log.sh", "validate.sh"],
                    "commands": [],
                    "raw": {"test_evidence": ["validation failed"]},
                },
                {
                    "validation_results": [
                        {
                            "command": ["bash", "-lc", "./validate.sh"],
                            "returncode": 1,
                            "expected_returncode": 0,
                            "returncode_matches_expected": False,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "",
                        }
                    ],
                    "workspace_files": [{"path": "watch_log.sh", "content": "#!/bin/sh\n"}],
                    "git": {"enabled": True, "meaningful_changed_paths": ["watch_log.sh", ".watch_state"]},
                },
            )

            text = "\n".join(findings)
            self.assertIn("runtime state artifact", text)
            self.assertIn(".watch_state", text)
            self.assertIn("temporary working directory", text)
            self.assertIn("public state-file", text)

            self.assertEqual(
                agent._unrequested_runtime_state_artifacts(
                    {"written": [".watch_state"]},
                    {"git": {"meaningful_changed_paths": [".watch_state"]}},
                ),
                [],
            )

    def test_script_validation_silent_failure_requests_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, prompt="Build monitor_disk.sh and README.md.")
            agent.initialize()
            step = {
                "id": "S1",
                "title": "Implement script",
                "description": "Create a script and validate behavior.",
                "depends_on": [],
                "acceptance_criteria": ["The script emits the expected marker."],
                "validation_commands": [["bash", "-lc", "bash monitor_disk.sh | grep -q ACTION_REQUIRED"]],
                "status": "pending",
            }

            findings = agent._evidence_findings(
                step,
                {
                    "written": ["monitor_disk.sh"],
                    "commands": [
                        {
                            "command": ["bash", "-lc", "bash monitor_disk.sh | grep -q ACTION_REQUIRED"],
                            "returncode": 1,
                            "expected_returncode": 0,
                            "returncode_matches_expected": False,
                            "timed_out": False,
                            "stdout": "",
                            "stderr": "",
                        }
                    ],
                    "raw": {"test_evidence": ["silent script validation"]},
                },
                {
                    "validation_results": [],
                    "workspace_files": [{"path": "monitor_disk.sh", "content": "#!/bin/bash\ntrue\n"}],
                    "git": {"enabled": False, "meaningful_changed_paths": ["monitor_disk.sh"]},
                },
            )

            text = "\n".join(findings)
            self.assertIn("failed without stdout or stderr", text)
            self.assertIn("failing sub-check", text)

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

    def test_plan_validation_rejects_swallowed_expected_failure_shell_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Negative path plan")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Validate CLI failure behavior",
                "description": "The validator exits non-zero when count or format is incorrect.",
                "depends_on": [],
                "acceptance_criteria": [
                    "Valid input succeeds.",
                    "Invalid format exits non-zero.",
                ],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "python validate_cli.py && (python validate_cli.py --bad-input || exit 0)",
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("mask an expected failure", text)
            self.assertIn("expected_returncode", text)

    def test_plan_validation_rejects_negative_path_pipeline_without_status_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("CLI negative path plan")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement CLI",
                "description": "The CLI exits with an error when no argument is provided.",
                "depends_on": [],
                "acceptance_criteria": [
                    "Valid input exits 0.",
                    "Missing input exits non-zero with a required-argument message.",
                ],
                "validation_commands": [
                    ["bash", "-lc", "python slugify.py 2>&1 | grep -q 'required'"],
                ],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("without checking the command-under-test exit status", text)
            self.assertIn("expected_returncode", text)

    def test_plan_validation_rejects_grep_v_absence_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, prompt="Build monitor_disk.sh and README.md.")
            agent.initialize()
            agent.requirements = base_requirements("Disk monitor")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement disk monitor",
                "description": "Create monitor_disk.sh and README.md.",
                "depends_on": [],
                "acceptance_criteria": [
                    "Script outputs ACTION_REQUIRED below threshold.",
                    "Script avoids ACTION_REQUIRED above threshold.",
                    "Script reaches Check 2/2 with MAX_CHECKS=2.",
                ],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "CHECK_INTERVAL_SECONDS=1 MAX_CHECKS=2 MIN_FREE_PERCENT=0 ./monitor_disk.sh | grep -v 'ACTION_REQUIRED' | grep -q 'Check 2/2'",
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("filtering it out with `grep -v`", text)
            self.assertIn("Capture the full output", text)

    def test_plan_validation_allows_captured_absence_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, prompt="Build monitor_disk.sh and README.md.")
            agent.initialize()
            agent.requirements = base_requirements("Disk monitor")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement disk monitor",
                "description": "Create monitor_disk.sh and README.md.",
                "depends_on": [],
                "acceptance_criteria": [
                    "Script outputs ACTION_REQUIRED below threshold.",
                    "Script avoids ACTION_REQUIRED above threshold.",
                    "Script reaches Check 2/2 with MAX_CHECKS=2.",
                ],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "out=$(CHECK_INTERVAL_SECONDS=1 MAX_CHECKS=2 MIN_FREE_PERCENT=0 ./monitor_disk.sh); "
                    "printf '%s\\n' \"$out\"; ! printf '%s\\n' \"$out\" | grep -q 'ACTION_REQUIRED'; "
                    "printf '%s\\n' \"$out\" | grep -q 'Check 2/2'",
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            self.assertNotIn("filtering it out with `grep -v`", "\n".join(findings))

    def test_plan_validation_accepts_expected_failure_wrapper_with_final_exit_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Negative path plan")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Validate CLI failure behavior",
                "description": "The validator exits non-zero when count is incorrect.",
                "depends_on": [],
                "acceptance_criteria": [
                    "Incorrect count exits non-zero.",
                ],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "python validate_cli.py --bad-input || exit 0; exit 1",
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            self.assertNotIn("mask an expected failure", "\n".join(findings))

    def test_plan_validation_rejects_missing_negative_path_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("CLI negative path plan")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement CLI validator",
                "description": "The CLI exits non-zero for invalid input.",
                "depends_on": [],
                "acceptance_criteria": [
                    "Valid input exits 0.",
                    "Invalid input exits non-zero with an error message.",
                ],
                "validation_commands": [["bash", "-lc", "python cli.py ok && python cli.py --help"]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("validation_commands only show success-path evidence", text)

    def test_plan_validation_default_policy_does_not_infer_negative_path_from_phrases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.config = replace(
                agent.config,
                quality_policy=replace(
                    agent.config.quality_policy,
                    deterministic_semantic_scope_checks=False,
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("CLI negative path plan")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement CLI validator",
                "description": "The CLI exits non-zero for invalid input.",
                "depends_on": [],
                "acceptance_criteria": [
                    "Valid input exits 0.",
                    "Invalid input exits non-zero with an error message.",
                ],
                "validation_commands": [["bash", "-lc", "python cli.py ok && python cli.py --help"]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertNotIn("validation_commands only show success-path evidence", text)
            self.assertNotIn("negative-path behavior", text)

    def test_plan_validation_does_not_treat_exit_code_zero_as_negative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Fast validator")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement validator",
                "description": "Create validate.py and run it.",
                "depends_on": [],
                "acceptance_criteria": [
                    "validate.py runs to completion with exit code 0.",
                    "All success-path integration tests pass.",
                ],
                "validation_commands": [["python", "validate.py"]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            self.assertNotIn("negative-path behavior", "\n".join(findings))

    def test_plan_validation_rejects_requirements_negative_path_without_plan_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Slug CLI")
            agent.requirements["assumptions"].append(
                "The tool exits non-zero if no argument is provided."
            )
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement slugify.py and tests",
                "description": "Create slugify.py and test_slugify.py.",
                "depends_on": [],
                "acceptance_criteria": [
                    "slugify.py prints transformed output.",
                    "Unit tests pass.",
                ],
                "validation_commands": [
                    ["python", "-m", "unittest", "test_slugify.py"],
                    ["bash", "-lc", "python slugify.py 'Hello World' | grep -q '^hello-world$'"],
                ],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("negative-path behavior", text)
            self.assertIn("missing argument", text)

    def test_plan_validation_rejects_exactly_one_cli_arg_without_no_arg_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Slug CLI")
            agent.requirements["refined_requirements"] = [
                "CLI: `slugify.py <string>` takes exactly one argument.",
                "README.md includes a Usage section.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement slugify.py and docs",
                "description": "Create slugify.py, tests, and README usage text.",
                "depends_on": [],
                "acceptance_criteria": [
                    "slugify.py prints transformed output.",
                    "README.md contains a Usage section.",
                ],
                "validation_commands": [
                    ["python", "-m", "unittest", "discover", "tests"],
                    [
                        "bash",
                        "-lc",
                        "python slugify.py 'Hello World!' | grep -q '^hello-world$' && grep -q 'Usage:' README.md",
                    ],
                ],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("negative-path behavior", text)
            self.assertIn("missing argument", text)

    def test_plan_validation_does_not_turn_single_argument_shape_into_error_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Configurable output")
            agent.requirements["refined_requirements"] = [
                "`huge_output.py` accepts a single command-line argument specifying the number of lines to print.",
                "`huge_output.py` prints the requested number of lines.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement huge_output.py",
                "description": "Create a configurable output script.",
                "depends_on": [],
                "acceptance_criteria": [
                    "`python huge_output.py 5` prints five lines.",
                ],
                "validation_commands": [["bash", "-lc", "python huge_output.py 5 | wc -l | grep -q '^5$'"]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertNotIn("missing argument", text)
            self.assertNotIn("too many arguments", text)
            self.assertNotIn("negative-path behavior", text)

    def test_plan_validation_rejects_exactly_one_positional_arg_without_no_arg_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Slug CLI")
            agent.requirements["refined_requirements"] = [
                "CLI tool `slugify.py` accepts exactly one positional argument.",
                "README.md includes a Usage section.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement slugify.py and docs",
                "description": "Create slugify.py, tests, and README usage text.",
                "depends_on": [],
                "acceptance_criteria": [
                    "slugify.py prints transformed output.",
                    "README.md contains a Usage section.",
                ],
                "validation_commands": [
                    ["python", "-m", "unittest", "discover", "tests"],
                    [
                        "bash",
                        "-lc",
                        "python slugify.py 'Hello World!' | grep -q '^hello-world$' && grep -q 'Usage:' README.md",
                    ],
                ],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("negative-path behavior", text)
            self.assertIn("missing argument", text)
            self.assertIn("too many arguments", text)

    def test_plan_validation_rejects_exactly_one_cli_arg_with_only_no_arg_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Slug CLI")
            agent.requirements["refined_requirements"] = [
                "CLI: `slugify.py <string>` takes exactly one argument.",
                "README.md includes a Usage section.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement slugify.py and docs",
                "description": "Create slugify.py, tests, and README usage text.",
                "depends_on": [],
                "acceptance_criteria": [
                    "slugify.py prints transformed output.",
                    "Tests cover the missing argument error path.",
                    "README.md contains a Usage section.",
                ],
                "validation_commands": [
                    ["python", "-m", "unittest", "discover", "tests"],
                    ["bash", "-lc", "python slugify.py 'Hello World!' | grep -q '^hello-world$'"],
                    [
                        "bash",
                        "-lc",
                        "err=$(python slugify.py 2>&1); status=$?; test $status -ne 0 && printf '%s' \"$err\" | grep -qi usage",
                    ],
                ],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("negative-path behavior", text)
            self.assertNotIn("missing argument", text)
            self.assertIn("too many arguments", text)

    def test_plan_validation_accepts_explicit_test_suite_failure_mode_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Slug CLI")
            agent.requirements["refined_requirements"] = [
                "CLI: `slugify.py <string>` takes exactly one argument.",
                "The unittest suite must cover 0 arguments and >1 arguments.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement slugify.py and tests",
                "description": "Create slugify.py and a unittest suite.",
                "depends_on": [],
                "acceptance_criteria": [
                    "slugify.py prints transformed output.",
                    "The unittest suite passes and covers missing and >1 argument error paths.",
                ],
                "validation_commands": [
                    ["python", "-m", "unittest", "discover", "tests"],
                    ["bash", "-lc", "python slugify.py 'Hello World!' | grep -q '^hello-world$'"],
                ],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertNotIn("negative-path behavior", text)
            self.assertNotIn("missing argument", text)
            self.assertNotIn("too many arguments", text)

    def test_plan_validation_accepts_exactly_one_cli_arg_with_argument_count_wrappers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Slug CLI")
            agent.requirements["refined_requirements"] = [
                "CLI: `slugify.py <string>` takes exactly one argument.",
                "README.md includes a Usage section.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement slugify.py and docs",
                "description": "Create slugify.py, tests, and README usage text.",
                "depends_on": [],
                "acceptance_criteria": [
                    "slugify.py prints transformed output.",
                    "Tests cover the missing argument error path.",
                    "README.md contains a Usage section.",
                ],
                "validation_commands": [
                    ["python", "-m", "unittest", "discover", "tests"],
                    ["bash", "-lc", "python slugify.py 'Hello World!' | grep -q '^hello-world$'"],
                    [
                        "bash",
                        "-lc",
                        "err=$(python slugify.py 2>&1); status=$?; test $status -ne 0 && printf '%s' \"$err\" | grep -qi usage",
                    ],
                    {"cmd": ["python", "slugify.py", "one", "two"], "expected_returncode": 2},
                ],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertNotIn("negative-path behavior", text)
            self.assertNotIn("missing argument", text)
            self.assertNotIn("too many arguments", text)

    def test_plan_validation_accepts_too_many_argument_shell_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Slug CLI")
            agent.requirements["refined_requirements"] = [
                "CLI: `slugify.py <string>` takes exactly one argument.",
                "README.md includes a Usage section.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement slugify.py and docs",
                "description": "Create slugify.py, tests, and README usage text.",
                "depends_on": [],
                "acceptance_criteria": [
                    "slugify.py prints transformed output.",
                    "The CLI exits non-zero for 0 arguments and >1 arguments.",
                    "README.md contains a Usage section.",
                ],
                "validation_commands": [
                    ["python", "-m", "unittest", "test_slugify.py"],
                    ["bash", "-lc", "python slugify.py 'Hello World!' | grep -q '^hello-world$'"],
                    [
                        "bash",
                        "-lc",
                        "err=$(python slugify.py 2>&1); status=$?; "
                        "test $status -ne 0 && printf '%s' \"$err\" | grep -qi usage",
                    ],
                    [
                        "bash",
                        "-lc",
                        "err=$(python slugify.py 'one' 'two' 2>&1); status=$?; "
                        "test $status -ne 0 && printf '%s' \"$err\" | grep -qi usage",
                    ],
                ],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertNotIn("negative-path behavior", text)
            self.assertNotIn("missing argument", text)
            self.assertNotIn("too many arguments", text)

    def test_plan_validation_accepts_no_argument_shell_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Slug CLI")
            agent.requirements["assumptions"].append(
                "The tool exits non-zero if no argument is provided."
            )
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement slugify.py and tests",
                "description": "Create slugify.py and test_slugify.py.",
                "depends_on": [],
                "acceptance_criteria": [
                    "slugify.py prints transformed output.",
                    "Unit tests pass.",
                    "Exits non-zero and prints usage when no command-line argument is provided.",
                ],
                "validation_commands": [
                    ["python", "-m", "unittest", "test_slugify.py"],
                    ["bash", "-lc", "result=$(python3 slugify.py 'Hello, World!'); test \"$result\" = hello-world"],
                    [
                        "bash",
                        "-lc",
                        "ERR=$(python3 slugify.py 2>&1); if [ $? -ne 0 ] && echo \"$ERR\" | grep -qi 'usage'; then exit 0; else exit 1; fi",
                    ],
                ],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertNotIn("negative-path behavior", text)
            self.assertNotIn("missing argument", text)

    def test_plan_validation_accepts_unittest_covering_declared_value_error_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Interval merge plan")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement intervals.py and unit tests",
                "description": "Create merge_intervals and unit tests covering invalid intervals.",
                "depends_on": [],
                "acceptance_criteria": [
                    "merge_intervals returns expected merged interval values.",
                    "merge_intervals raises ValueError for invalid intervals.",
                ],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            self.assertNotIn("validation_commands only show success-path evidence", "\n".join(findings))

    def test_plan_validation_accepts_unittest_for_timeout_exit_code_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Timeout CLI plan")
            agent.requirements["refined_requirements"] = [
                "Tests must verify immediate success, delayed success, timeout behavior, and exit code 2 on timeout.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement timeout CLI and tests",
                "description": "Create wait_for_file.py and unittest coverage for success and timeout paths.",
                "depends_on": [],
                "acceptance_criteria": [
                    "The script exits with code 0 when a file is created mid-wait.",
                    "The script exits with code 2 on timeout.",
                    "`python -m unittest test_wait_for_file.py` passes.",
                ],
                "validation_commands": [["python", "-m", "unittest", "test_wait_for_file.py"]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            self.assertNotIn("validation_commands only show success-path evidence", "\n".join(findings))

    def test_plan_validation_does_not_treat_empty_success_case_as_negative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Interval merge plan")
            agent.requirements["refined_requirements"] = [
                "Unit tests must cover: empty input, single interval, no overlaps, and invalid intervals.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement intervals.py and unit tests",
                "description": "Create merge_intervals and unit tests covering valid edge cases and invalid intervals.",
                "depends_on": [],
                "acceptance_criteria": [
                    "merge_intervals returns [] for empty input.",
                    "merge_intervals raises ValueError for invalid intervals.",
                ],
                "validation_commands": [["python", "-m", "unittest", "test_intervals.py"]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertNotIn("empty input", text)
            self.assertNotIn("negative-path behavior", text)

    def test_plan_validation_rejects_missing_named_failure_mode_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Named failure modes")
            agent.requirements["refined_requirements"] = [
                "The validator exits non-zero if the count is incorrect, the format is incorrect, or invalid arguments are provided."
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement validator",
                "description": "Validate success and one negative path.",
                "depends_on": [],
                "acceptance_criteria": [
                    "Invalid arguments exit non-zero with an error message.",
                ],
                "validation_commands": [
                    ["python", "validate_huge_output.py"],
                    ["bash", "-lc", "python validate_huge_output.py --count abc 2>&1 | grep -i error"],
                ],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("multiple distinct failure modes", text)
            self.assertIn("incorrect count", text)
            self.assertIn("incorrect format", text)
            self.assertIn("different valid argument value", text)

    def test_plan_validation_accepts_covered_named_failure_mode_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Named failure modes")
            agent.requirements["refined_requirements"] = [
                "The validator exits non-zero if the count is incorrect, the format is incorrect, or invalid arguments are provided."
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement validator",
                "description": "Validate success and all named negative paths.",
                "depends_on": [],
                "acceptance_criteria": [
                    "Invalid arguments, incorrect count, and incorrect format exit non-zero.",
                ],
                "validation_commands": [
                    ["python", "validate_huge_output.py"],
                    ["bash", "-lc", "out=$(python validate_huge_output.py --count abc 2>&1); status=$?; test $status -ne 0 && printf '%s' \"$out\" | grep -i error"],
                    ["bash", "-lc", "out=$(python validate_huge_output.py --wrong-count 2>&1); status=$?; test $status -ne 0 && printf '%s' \"$out\" | grep -i 'wrong count'"],
                    ["bash", "-lc", "out=$(python validate_huge_output.py --wrong-format 2>&1); status=$?; test $status -ne 0 && printf '%s' \"$out\" | grep -i 'wrong format'"],
                ],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            self.assertNotIn("multiple distinct failure modes", "\n".join(findings))

    def test_plan_validation_accepts_copied_workspace_bad_format_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Validator format failure")
            agent.requirements["refined_requirements"] = [
                "`validate_huge_output.py` exits non-zero when the line count is incorrect.",
                "`validate_huge_output.py` exits non-zero when the line format is incorrect.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement validator",
                "description": "Create huge_output.py and validate_huge_output.py.",
                "depends_on": [],
                "acceptance_criteria": [
                    "validate_huge_output.py rejects wrong counts and malformed output format.",
                ],
                "validation_commands": [
                    ["python", "validate_huge_output.py"],
                    {
                        "cmd": [
                            "bash",
                            "-lc",
                            "tmp=$(mktemp -d); cp huge_output.py validate_huge_output.py \"$tmp\"; "
                            "echo 'print(\"Line 1\")' > \"$tmp/huge_output.py\"; "
                            "(cd \"$tmp\" && python validate_huge_output.py)",
                        ],
                        "expected_returncode": 1,
                    },
                    {
                        "cmd": [
                            "bash",
                            "-lc",
                            "tmp=$(mktemp -d); cp huge_output.py validate_huge_output.py \"$tmp\"; "
                            "echo 'print(\"Bad Format\")' > \"$tmp/huge_output.py\"; "
                            "(cd \"$tmp\" && python validate_huge_output.py)",
                        ],
                        "expected_returncode": 1,
                    },
                ],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertNotIn("incorrect count", text)
            self.assertNotIn("incorrect format", text)
            self.assertNotIn("do not clearly prove", text)

    def test_plan_validation_rejects_bad_format_without_failure_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Validator format failure")
            agent.requirements["refined_requirements"] = [
                "`validate_huge_output.py` exits non-zero when the line format is incorrect.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement validator",
                "description": "Create huge_output.py and validate_huge_output.py.",
                "depends_on": [],
                "acceptance_criteria": [
                    "validate_huge_output.py rejects malformed output format.",
                ],
                "validation_commands": [
                    ["bash", "-lc", "printf 'Bad Format\\n' > sample.txt && python validate_huge_output.py"],
                ],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("incorrect format", text)
            self.assertIn("do not clearly prove", text)

    def test_plan_validation_accepts_validator_owned_test_plan_for_failure_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Validator test plan")
            agent.requirements["refined_requirements"] = [
                "`validate_huge_output.py` exits non-zero when the line count is incorrect.",
                "`validate_huge_output.py` exits non-zero when the line format is incorrect.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement validator and test suite",
                "description": "Create the generator, validator, and unit tests for validator failure modes.",
                "depends_on": [],
                "acceptance_criteria": [
                    "test_validate_huge_output.py covers incorrect count and incorrect format using temporary generator fixtures.",
                ],
                "validation_commands": [["python", "-m", "unittest", "test_validate_huge_output.py"]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertNotIn("multiple distinct failure modes", text)
            self.assertNotIn("do not clearly prove", text)

    def test_plan_validation_rejects_public_failure_injection_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt="Create huge_output.py and validate_huge_output.py for bounded output validation.",
            )
            agent.initialize()
            agent.requirements = base_requirements("Failure flags")
            agent.requirements["refined_requirements"] = [
                "`huge_output.py` must support `--fail-count` and `--fail-format` for validation.",
                "`validate_huge_output.py` must pass those failure flags through to the generator.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement failure flags",
                "description": "Add public testing flags for failure modes.",
                "depends_on": [],
                "acceptance_criteria": ["`validate_huge_output.py --fail-count` exits non-zero."],
                "validation_commands": [["python", "validate_huge_output.py"]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("public failure-injection/test switches", text)
            self.assertIn("--fail-count", text)

    def test_plan_validation_allows_invalid_option_only_in_validation_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Negative path command")
            agent.requirements["refined_requirements"] = ["The CLI rejects invalid input."]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement CLI",
                "description": "Validate normal and invalid input.",
                "depends_on": [],
                "acceptance_criteria": ["Invalid input exits non-zero."],
                "validation_commands": [
                    ["python", "cli.py", "ok"],
                    {"cmd": ["python", "cli.py", "--bad-input"], "expected_returncode": 2},
                ],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            self.assertNotIn("public failure-injection/test switches", "\n".join(findings))

    def test_plan_validation_rejects_validation_only_public_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt="Create huge_output.py and validate_huge_output.py for bounded output validation.",
            )
            agent.initialize()
            agent.requirements = base_requirements("Validation-only option")
            agent.requirements["refined_requirements"] = [
                "`validate_huge_output.py` must accept optional `--prefix <P>` to facilitate testing of failure modes.",
                "`validate_huge_output.py` exits non-zero when the line format is incorrect.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement validator",
                "description": "Add prefix option for validation.",
                "depends_on": [],
                "acceptance_criteria": ["Incorrect format exits non-zero."],
                "validation_commands": [["python", "validate_huge_output.py"]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("public failure-injection/test switches", text)
            self.assertIn("--prefix", text)

    def test_plan_validation_does_not_blame_requested_flag_near_validation_only_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt="Create huge_output.py that can print a configurable number of lines.",
            )
            agent.initialize()
            agent.requirements = base_requirements("Huge output")
            agent.requirements["refined_requirements"] = [
                "`huge_output.py` must accept `--lines` for the number of lines to print.",
                "`validate_huge_output.py` must validate the generated output.",
            ]
            agent.requirements["assumptions"] = [
                "To allow testing of failure modes, `validate_huge_output.py` will accept `--count` and `--script`.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement validator",
                "description": "Implement bounded validation.",
                "depends_on": [],
                "acceptance_criteria": ["`huge_output.py --lines 5` prints five lines."],
                "validation_commands": [["python", "validate_huge_output.py"]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("public failure-injection/test switches", text)
            self.assertIn("--count", text)
            self.assertIn("--script", text)
            self.assertNotIn("--lines", text)

    def test_plan_validation_rejects_near_duplicate_named_script_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt="Create huge_output.py and validate_huge_output.py.",
            )
            agent.initialize()
            agent.requirements = base_requirements("Script typo")
            agent.requirements["refined_requirements"] = [
                "`huge_huge_output.py` must print lines in the requested format.",
                "`validate_huge_output.py` runs `huge_output.py`.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement scripts",
                "description": "Create huge_huge_output.py and validate_huge_output.py.",
                "depends_on": [],
                "acceptance_criteria": ["Both scripts run."],
                "validation_commands": [["python", "validate_huge_output.py"]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("huge_huge_output.py", text)
            self.assertIn("huge_output.py", text)

    def test_plan_validation_allows_conventional_test_module_for_named_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt="Build a tiny Python CLI slugify.py. Include tests and README.",
            )
            agent.initialize()
            agent.requirements = base_requirements("Slug CLI")
            agent.requirements["refined_requirements"] = [
                "`slugify.py` implements the CLI.",
                "`test_slugify.py` contains unit tests for the CLI.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement slug CLI",
                "description": "Create slugify.py and test_slugify.py.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_slugify.py"]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            self.assertNotIn("test_slugify.py", "\n".join(findings))

    def test_plan_validation_allows_unittest_module_for_named_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt="Build a Python CLI tool named palindrome.py. Include unittest coverage.",
            )
            agent.initialize()
            agent.requirements = base_requirements("Palindrome CLI")
            agent.requirements["refined_requirements"] = [
                "`palindrome.py` implements the CLI.",
                "`test_palindrome.py` contains unittest coverage.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement palindrome CLI",
                "description": "Create palindrome.py and test_palindrome.py.",
                "depends_on": [],
                "acceptance_criteria": ["Unit tests pass."],
                "validation_commands": [["python", "-m", "unittest", "test_palindrome.py"]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            self.assertNotIn("test_palindrome.py", "\n".join(findings))

    def test_plan_validation_accepts_expected_returncode_negative_path_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("CLI negative path plan")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement CLI validator",
                "description": "The CLI exits non-zero for invalid input.",
                "depends_on": [],
                "acceptance_criteria": [
                    "Valid input exits 0.",
                    "Invalid input exits non-zero with an error message.",
                ],
                "validation_commands": [
                    ["python", "cli.py", "ok"],
                    {"cmd": ["python", "cli.py", "--bad-input"], "expected_returncode": 2},
                ],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            self.assertNotIn("validation_commands only show success-path evidence", "\n".join(findings))

    def test_plan_validation_allows_positive_empty_case_with_separate_expected_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Interval merge plan")
            agent.plan_steps = [{
                "id": "S2",
                "title": "Implement core logic",
                "description": "Create merge_intervals with invalid interval validation.",
                "depends_on": [],
                "acceptance_criteria": [
                    "merge_intervals returns [] for empty input.",
                    "merge_intervals raises ValueError for invalid intervals.",
                ],
                "validation_commands": [
                    [
                        "python",
                        "-c",
                        "from intervals import merge_intervals; assert merge_intervals([]) == []; print('SUCCESS')",
                    ],
                    {
                        "cmd": [
                            "python",
                            "-c",
                            "from intervals import merge_intervals; merge_intervals([(3, 1)])",
                        ],
                        "expected_returncode": 1,
                    },
                ],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            self.assertNotIn("expected failure path", "\n".join(findings))

    def test_plan_validation_rejects_silent_captured_subprocess_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S3",
                "title": "Validate generated CLI",
                "description": "Run a generated validator through subprocess.",
                "depends_on": [],
                "acceptance_criteria": ["Validator exits successfully."],
                "validation_commands": [[
                    "python",
                    "-c",
                    "import subprocess; r = subprocess.run(['python', 'validate.py'], capture_output=True, text=True); exit(0 if r.returncode == 0 else 1)",
                ]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertIn("captures subprocess output but discards it", "\n".join(findings))

            step["validation_commands"] = [[
                "python",
                "-c",
                "import subprocess; r = subprocess.run(['python', 'validate.py'], capture_output=True, text=True); assert r.returncode == 0, r.stderr; exit(0)",
            ]]

            findings = agent._validation_command_findings(step)

            self.assertNotIn("captures subprocess output but discards it", "\n".join(findings))

    def test_requirements_review_can_ignore_validator_diagnostic_advisory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("CLI utility")
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Validate generated CLI",
                    "description": "Run a generated validator through subprocess.",
                    "depends_on": [],
                    "acceptance_criteria": ["Validator exits successfully."],
                    "validation_commands": [[
                        "python",
                        "-c",
                        "import subprocess; r = subprocess.run(['python', 'validate.py'], capture_output=True, text=True); exit(0 if r.returncode == 0 else 1)",
                    ]],
                    "status": "pending",
                }
            ]

            strict_findings = agent._plan_structural_findings()
            requirements_findings = agent._plan_structural_findings(include_diagnostic_quality=False)

            self.assertIn("captures subprocess output but discards it", "\n".join(strict_findings))
            self.assertNotIn("captures subprocess output but discards it", "\n".join(requirements_findings))

    def test_missing_args_test_suite_wording_counts_as_negative_path_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = {
                "refined_requirements": [
                    "Exit with non-zero status and a descriptive error message for missing args."
                ],
                "assumptions": [],
                "planning_confirmation": {},
            }
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Implement CLI and tests",
                    "description": "Use unittest to cover success behavior and missing args.",
                    "depends_on": [],
                    "acceptance_criteria": [
                        "Tests cover CLI error handling for missing args."
                    ],
                    "validation_commands": [["python", "-m", "unittest", "test_cli.py"]],
                    "status": "pending",
                }
            ]

            findings = agent._requirements_negative_path_validation_findings()

            self.assertNotIn("negative-path behavior", "\n".join(findings))

    def test_missing_argument_subprocess_argv_list_counts_as_negative_path_coverage(self) -> None:
        command_text = json.dumps([[
            "python",
            "-c",
            (
                "import subprocess; "
                "p = subprocess.run(['python', 'normalize_config.py'], capture_output=True, text=True); "
                "exit(0 if p.returncode != 0 and 'error' in p.stderr else 1)"
            ),
        ]]).lower()

        self.assertTrue(
            FeedbackLoopAgent._command_text_proves_missing_argument(command_text)
        )

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

    def test_plan_validation_rejects_fails_as_expected_command_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                deterministic_semantic_scope_checks=False,
            )
            step = {
                "id": "S1",
                "title": "Discovery and diagnosis",
                "description": "Locate existing files and run the current test suite.",
                "depends_on": [],
                "acceptance_criteria": [
                    "Files containing the implementation and tests are identified.",
                    "The existing test suite is executed and fails as expected.",
                ],
                "validation_commands": [["bash", "-lc", "ls -R && python3 -m unittest discover"]],
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

            self.assertIn("compound statement", "\n".join(findings))
            self.assertIn("validation script", "\n".join(findings))

    def test_plan_validation_rejects_one_line_nested_for_if_python_c(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S2",
                "title": "Semantic verification",
                "description": "Independently recompute an answer.",
                "depends_on": [],
                "acceptance_criteria": ["Answer is semantically validated."],
                "validation_commands": [[
                    "python",
                    "-c",
                    "import itertools; count=0; for p in itertools.product('ABCD', repeat=4): if p.count('A') == 2: count += 1",
                ]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertIn("compound statement", "\n".join(findings))
            self.assertIn("JSON-safe single-line expression validator", "\n".join(findings))

    def test_plan_validation_rejects_shell_wrapped_inline_python_syntax_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S3",
                "title": "Semantic verification",
                "description": "Independently recompute an answer.",
                "depends_on": [],
                "acceptance_criteria": ["Answer is semantically validated."],
                "validation_commands": [[
                    "bash",
                    "-c",
                    "python3 -c \"import itertools; count=sum(1 for item in itertools.product('AB', repeat=2); assert count == 4\"",
                ]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertIn("static syntax check", "\n".join(findings))
            self.assertIn("shell-wrapped python -c", "\n".join(findings))

    def test_plan_validation_rejects_malformed_quoted_heredoc_closer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S3",
                "title": "Semantic verification",
                "description": "Run a multiline Python validator.",
                "depends_on": [],
                "acceptance_criteria": ["Answer is semantically validated."],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "python3 - <<'PY'\nprint('ok')\n'PY",
                ]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertIn("here-doc", "\n".join(findings))
            self.assertIn("closing delimiter line must be exactly `PY`", "\n".join(findings))

    def test_plan_validation_allows_valid_quoted_heredoc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S3",
                "title": "Semantic verification",
                "description": "Run a multiline Python validator.",
                "depends_on": [],
                "acceptance_criteria": ["Answer is semantically validated."],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "python3 - <<'PY'\nprint('ok')\nPY",
                ]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertNotIn("here-doc", "\n".join(findings))

    def test_plan_validation_rejects_inline_python_unreachable_after_return(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S4",
                "title": "Semantic verification",
                "description": "Independently recompute and write the requested answer.",
                "depends_on": [],
                "acceptance_criteria": ["ANSWER.txt contains the recomputed result."],
                "validation_commands": [[
                    "python",
                    "-c",
                    "def compute(): return 42; open('ANSWER.txt','w').write(str(compute()))",
                ]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertIn("unreachable statements after return", "\n".join(findings))
            self.assertIn("function `compute`", "\n".join(findings))

    def test_plan_validation_rejects_metadata_inside_argv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S3",
                "title": "Validate command format",
                "description": "Run a command with metadata.",
                "depends_on": [],
                "acceptance_criteria": ["The command metadata is represented correctly."],
                "validation_commands": [["bash", "-lc", "true", "expected_returncode", "0"]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertIn("inside the argv list", "\n".join(findings))
            self.assertIn("command object", "\n".join(findings))

    def test_plan_validation_rejects_artifact_mutating_validation_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, prompt="Create ANSWER.txt only. Return the integer answer.")
            step = {
                "id": "S1",
                "title": "Calculate answer",
                "description": "Calculate the answer and write ANSWER.txt.",
                "depends_on": [],
                "acceptance_criteria": ["ANSWER.txt exists."],
                "validation_commands": [[
                    "python",
                    "-c",
                    "open('ANSWER.txt', 'w').write('24')",
                ]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertIn("mutate the explicitly requested artifact", "\n".join(findings))

    def test_plan_validation_rejects_artifact_mutating_validation_command_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, prompt="Create ANSWER.txt only. Return the integer answer.")
            agent.config = replace(
                agent.config,
                quality_policy=replace(
                    agent.config.quality_policy,
                    deterministic_semantic_scope_checks=False,
                ),
            )
            step = {
                "id": "S1",
                "title": "Calculate answer",
                "description": "Calculate the answer and write ANSWER.txt.",
                "depends_on": [],
                "acceptance_criteria": ["ANSWER.txt exists."],
                "validation_commands": [[
                    "python",
                    "-c",
                    "open('ANSWER.txt', 'w').write('24')",
                ]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertIn("mutate the explicitly requested artifact", "\n".join(findings))

    def test_plan_validation_rejects_source_mutating_validation_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S2",
                "title": "Validate negative path",
                "description": "Prove the validator rejects a bad generator.",
                "depends_on": [],
                "acceptance_criteria": ["The validator rejects bad generated output."],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "cp huge_output.py huge_output.py.bak; echo 'print(\"Wrong\")' > huge_output.py; python validate_huge_output.py; mv huge_output.py.bak huge_output.py",
                ]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            text = "\n".join(findings)
            self.assertIn("workspace source path `huge_output.py`", text)
            self.assertIn("/tmp fixtures", text)

    def test_plan_validation_rejects_chmod_source_validation_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S1",
                "title": "Validate executable script",
                "description": "Run the generated script.",
                "depends_on": [],
                "acceptance_criteria": ["monitor_disk.sh is executable and runs."],
                "validation_commands": [[
                    "bash",
                    "-c",
                    "chmod +x monitor_disk.sh && ./monitor_disk.sh | head -n 1 | grep -q '.'",
                ]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            text = "\n".join(findings)
            self.assertIn("workspace source path `monitor_disk.sh`", text)
            self.assertIn("shebang", text)
            self.assertIn("test -x", text)

    def test_plan_validation_requires_executable_deliverable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, prompt="Create slugify.py as a directly executable Python CLI.")
            agent.initialize()
            agent.requirements = base_requirements("Slug CLI")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement slugify.py and tests",
                "description": "Create the Python CLI and test suite.",
                "depends_on": [],
                "acceptance_criteria": [
                    "`slugify.py` exists and is executable.",
                    "`python -m unittest discover` passes.",
                ],
                "validation_commands": [["python", "-m", "unittest", "discover"]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("requires `slugify.py` to be executable", text)
            self.assertIn("test -x ./slugify.py", text)
            self.assertIn("actual validation_commands entry", text)
            self.assertIn("validator script will check executability is not command evidence", text)

            agent.plan_steps[0]["validation_commands"] = [
                ["python", "-m", "unittest", "discover"],
                ["test", "-x", "slugify.py"],
            ]

            findings = agent._plan_structural_findings()

            self.assertNotIn("requires `slugify.py` to be executable", "\n".join(findings))

    def test_plan_validation_flags_unrequested_python_executable_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, prompt="Build a Python CLI slugify.py. Run it with python slugify.py.")
            agent.initialize()
            agent.requirements = base_requirements("Slug CLI")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement slugify.py and tests",
                "description": "Create the Python CLI and test suite.",
                "depends_on": [],
                "acceptance_criteria": [
                    "`slugify.py` exists and is executable.",
                    "`python -m unittest discover` passes.",
                ],
                "validation_commands": [["python", "-m", "unittest", "discover"]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("does not clearly require `./` invocation or a shebang", text)
            self.assertIn("revise the acceptance criteria", text)

    def test_plan_validation_allows_relative_source_write_inside_tmp_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S2",
                "title": "Validate negative path with fixture",
                "description": "Prove the validator rejects a bad generator using a temporary copy.",
                "depends_on": [],
                "acceptance_criteria": ["The validator rejects bad generated output."],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "mkdir -p /tmp/fail_count && cp huge_output.py validate_huge_output.py /tmp/fail_count/ && "
                    "(cd /tmp/fail_count && echo 'print(\"Line 1\")' > huge_output.py && "
                    "python validate_huge_output.py 10 && exit 1 || exit 0)",
                ]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertNotIn("workspace source path `huge_output.py`", "\n".join(findings))

    def test_plan_validation_allows_shell_temp_variable_source_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S2",
                "title": "Validate negative path with temp variable",
                "description": "Prove the validator rejects a bad generator using a temporary copy.",
                "depends_on": [],
                "acceptance_criteria": ["The validator rejects bad generated output."],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "TMP=$(mktemp -d); cp validate_huge_output.py $TMP/; "
                    "echo 'print(\"Line 1\")' > $TMP/huge_output.py; "
                    "(cd $TMP && python validate_huge_output.py) > /dev/null 2>&1; "
                    "status=$?; rm -rf $TMP; exit $status",
                ]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertNotIn("workspace source path `$TMP/huge_output.py`", "\n".join(findings))

    def test_plan_validation_rejects_shell_workspace_variable_source_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S2",
                "title": "Validate negative path with workspace variable",
                "description": "This mutates a workspace file through a variable.",
                "depends_on": [],
                "acceptance_criteria": ["The validator rejects bad generated output."],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "WORKSPACE=$(pwd); echo 'print(\"Wrong\")' > $WORKSPACE/huge_output.py; "
                    "python validate_huge_output.py",
                ]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertIn("workspace source path `$WORKSPACE/huge_output.py`", "\n".join(findings))

    def test_plan_validation_accepts_named_failure_tmp_fixture_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Temporary fixture failure modes")
            agent.requirements["refined_requirements"] = [
                "`validate_huge_output.py` exits non-zero when the line count is incorrect.",
                "`validate_huge_output.py` exits non-zero when the line format is incorrect.",
                "`validate_huge_output.py` handles direct invocation without arguments.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement huge output validator",
                "description": "Create generator and streaming validator.",
                "depends_on": [],
                "acceptance_criteria": [
                    "Direct invocation succeeds.",
                    "Incorrect count exits non-zero.",
                    "Incorrect format exits non-zero.",
                ],
                "validation_commands": [
                    ["bash", "-lc", "python validate_huge_output.py 100"],
                    ["bash", "-lc", "python validate_huge_output.py"],
                    [
                        "bash",
                        "-lc",
                        "mkdir -p /tmp/fail_count && cp huge_output.py validate_huge_output.py /tmp/fail_count/ && "
                        "(cd /tmp/fail_count && echo 'import sys; print(\"Line 1\")' > huge_output.py && "
                        "python validate_huge_output.py 10 && exit 1 || exit 0)",
                    ],
                    [
                        "bash",
                        "-lc",
                        "mkdir -p /tmp/fail_fmt && cp huge_output.py validate_huge_output.py /tmp/fail_fmt/ && "
                        "(cd /tmp/fail_fmt && echo 'import sys; print(\"Wrong\")' > huge_output.py && "
                        "python validate_huge_output.py 10 && exit 1 || exit 0)",
                    ],
                ],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertNotIn("workspace source path `huge_output.py`", text)
            self.assertNotIn("multiple distinct failure modes", text)

    def test_plan_validation_accepts_bad_count_fixture_with_preserved_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Temporary fixture failure modes")
            agent.requirements["refined_requirements"] = [
                "`validate_huge_output.py` exits non-zero when the line count is incorrect.",
                "`validate_huge_output.py` exits non-zero when the line format is incorrect.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement huge output validator",
                "description": "Create generator and validator.",
                "depends_on": [],
                "acceptance_criteria": [
                    "Incorrect count exits non-zero.",
                    "Incorrect format exits non-zero.",
                ],
                "validation_commands": [
                    {"cmd": ["python3", "validate_huge_output.py"], "expected_returncode": 0},
                    {
                        "cmd": [
                            "bash",
                            "-lc",
                            "echo 'print(\"Wrong Format\")' > /tmp/bad_fmt.py; "
                            "python3 validate_huge_output.py --tool /tmp/bad_fmt.py; "
                            "status=$?; rm /tmp/bad_fmt.py; exit $status",
                        ],
                        "expected_returncode": 1,
                    },
                    {
                        "cmd": [
                            "bash",
                            "-lc",
                            "echo 'print(\"Line 1: content\")' > /tmp/bad_cnt.py; "
                            "python3 validate_huge_output.py --tool /tmp/bad_cnt.py --count 5; "
                            "status=$?; rm /tmp/bad_cnt.py; exit $status",
                        ],
                        "expected_returncode": 1,
                    },
                ],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertNotIn("multiple distinct failure modes", text)
            self.assertNotIn("incorrect count", text)
            self.assertNotIn("without preserving its status", text)

    def test_plan_validation_accepts_one_line_temp_producer_as_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Temporary count fixture")
            agent.requirements["refined_requirements"] = [
                "`validate_huge_output.py` exits non-zero when the line count is incorrect.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement huge output validator",
                "description": "Create generator and validator.",
                "depends_on": [],
                "acceptance_criteria": [
                    "Incorrect count exits non-zero.",
                ],
                "validation_commands": [
                    ["python3", "validate_huge_output.py"],
                    {
                        "cmd": [
                            "bash",
                            "-lc",
                            "TMP=$(mktemp -d); cp validate_huge_output.py $TMP/; "
                            "echo 'print(\"Line 1\")' > $TMP/huge_output.py; "
                            "(cd $TMP && python validate_huge_output.py) > /dev/null 2>&1; "
                            "status=$?; rm -rf $TMP; exit $status",
                        ],
                        "expected_returncode": 1,
                    },
                ],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertNotIn("incorrect count", text)
            self.assertNotIn("workspace source path `$TMP/huge_output.py`", text)

    def test_plan_validation_rejects_unwired_pythonpath_temp_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Temporary fixture failure modes")
            agent.requirements["refined_requirements"] = [
                "`validate_huge_output.py` exits non-zero when the line count is incorrect.",
                "`validate_huge_output.py` exits non-zero when the line format is incorrect.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement huge output validator",
                "description": "Create generator and validator.",
                "depends_on": [],
                "acceptance_criteria": [
                    "Incorrect count exits non-zero.",
                    "Incorrect format exits non-zero.",
                ],
                "validation_commands": [{
                    "cmd": [
                        "python3",
                        "-c",
                        "import subprocess, os; open('/tmp/mock_huge.py', 'w').write('print(\"Line 1\")'); "
                        "p=subprocess.run(['python3', 'validate_huge_output.py', '--count', '3'], "
                        "env={**os.environ, 'PYTHONPATH': '/tmp'}); exit(0 if p.returncode != 0 else 1)",
                    ],
                    "expected_returncode": 0,
                }],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("temporary fixture or test double", text)
            self.assertIn("PYTHONPATH", text)

    def test_plan_validation_rejects_expected_returncode_shell_cleanup_status_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Temporary fixture failure modes")
            agent.requirements["refined_requirements"] = [
                "`validate_huge_output.py` exits non-zero when the line count is incorrect.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement huge output validator",
                "description": "Create generator and validator.",
                "depends_on": [],
                "acceptance_criteria": ["Incorrect count exits non-zero."],
                "validation_commands": [{
                    "cmd": [
                        "bash",
                        "-lc",
                        "echo 'print(\"Line 1: content\")' > /tmp/bad_cnt.py; "
                        "python3 validate_huge_output.py --tool /tmp/bad_cnt.py --count 5; "
                        "rm /tmp/bad_cnt.py",
                    ],
                    "expected_returncode": 1,
                }],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("without preserving its status", text)

    def test_plan_validation_rejects_shell_cleanup_after_assertion_status_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Temporary fixture validation")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Validate CLI with temporary fixture",
                "description": "Create a temporary input and compare CLI output.",
                "depends_on": [],
                "acceptance_criteria": ["CLI output matches expected normalized JSON."],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "echo '{\"b\": 1}' > /tmp/input.json; "
                    "python3 normalize_config.py /tmp/input.json | grep -q '{\"b\":1}'; "
                    "rm /tmp/input.json",
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("after a validation assertion", text)

            agent.plan_steps[0]["validation_commands"] = [[
                "bash",
                "-lc",
                "tmp=/tmp/input.json; trap 'rm -f $tmp' EXIT; "
                "echo '{\"b\": 1}' > $tmp; "
                "python3 normalize_config.py $tmp | grep -q '{\"b\":1}'",
            ]]

            findings = agent._plan_structural_findings()

            self.assertNotIn("after a validation assertion", "\n".join(findings))

    def test_plan_validation_rejects_success_echo_failure_mask_with_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Timeout-friendly command")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Validate CLI success path",
                "description": "Create a temporary file and run the polling CLI.",
                "depends_on": [],
                "acceptance_criteria": ["The CLI exits 0 when the target file exists."],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "touch test_file.txt && python wait_for_file.py --file test_file.txt "
                    "--timeout-seconds 2 && echo 'Success: Exit 0' || echo 'Fail: Exit 0'; "
                    "rm test_file.txt",
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("mask an assertion failure", text)

    def test_plan_validation_allows_status_preserving_cleanup_after_assertion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Temporary fixture validation")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Validate CLI with temporary fixture",
                "description": "Create a temporary input and compare CLI output.",
                "depends_on": [],
                "acceptance_criteria": ["CLI output matches expected normalized JSON."],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "echo '{\"b\": 1}' > /tmp/input.json; "
                    "python3 normalize_config.py /tmp/input.json | grep -q '{\"b\":1}'; "
                    "status=$?; rm /tmp/input.json; exit $status",
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertNotIn("after a validation assertion", text)
            self.assertNotIn("mask an assertion failure", text)

    def test_plan_validation_rejects_assertion_or_echo_failure_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Output validation")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Validate output text",
                "description": "Run a CLI and check expected output.",
                "depends_on": [],
                "acceptance_criteria": ["CLI output contains READY."],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "python3 app.py | grep -q READY || echo failed",
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            self.assertIn("mask an assertion failure", "\n".join(findings))

    def test_plan_validation_rejects_env_assignment_after_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build monitor_disk.sh. It must support MIN_FREE_PERCENT and "
                    "MAX_CHECKS environment overrides."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Disk monitor")
            agent.requirements["refined_requirements"] = [
                "The script supports MIN_FREE_PERCENT and MAX_CHECKS environment overrides."
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Validate threshold override",
                "description": "Mock df and verify ACTION_REQUIRED.",
                "depends_on": [],
                "acceptance_criteria": ["MIN_FREE_PERCENT controls threshold behavior."],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "PATH=/tmp/mock:$PATH ./monitor_disk.sh MIN_FREE_PERCENT=10 | grep -q ACTION_REQUIRED",
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("MIN_FREE_PERCENT", text)
            self.assertIn("environment override", text)

            agent.plan_steps[0]["validation_commands"] = [[
                "bash",
                "-lc",
                "PATH=/tmp/mock:$PATH MIN_FREE_PERCENT=10 ./monitor_disk.sh | grep -q ACTION_REQUIRED",
            ]]

            findings = agent._plan_structural_findings()

            self.assertNotIn("environment override", "\n".join(findings))

    def test_plan_validation_rejects_workspace_validation_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Disk monitor")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Validate script output",
                "description": "Run script and inspect output.",
                "depends_on": [],
                "acceptance_criteria": ["The validation inspects generated output."],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "./monitor_disk.sh > output.log && grep -q ACTION_REQUIRED output.log",
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("output.log", text)
            self.assertIn("unrequested project artifact", text)

            agent.plan_steps[0]["validation_commands"] = [[
                "bash",
                "-lc",
                "tmp=$(mktemp); trap 'rm -f \"$tmp\"' EXIT; ./monitor_disk.sh > \"$tmp\" && grep -q ACTION_REQUIRED \"$tmp\"",
            ]]

            findings = agent._plan_structural_findings()

            self.assertNotIn("unrequested project artifact", "\n".join(findings))

    def test_plan_validation_allows_quoted_python_greater_than_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Answer validator")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Validate answer",
                "description": "Read ANSWER.txt and assert the value is positive.",
                "depends_on": [],
                "acceptance_criteria": ["ANSWER.txt contains a positive integer."],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "python -c \"val = open('ANSWER.txt').read().strip(); assert val.isdigit() and int(val) > 0\"",
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings(include_diagnostic_quality=False)

            self.assertNotIn("workspace path `0`", "\n".join(findings))
            self.assertNotIn("unrequested project artifact", "\n".join(findings))

    def test_plan_validation_allows_outputs_after_cd_to_mktemp_var(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Log watcher")
            agent.requirements["refined_requirements"] = [
                "README.md contains usage instructions.",
                "watch_log.sh remembers progress in .watch_state.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement watch_log.sh and README",
                "description": "Create the core Bash script with state management and documentation.",
                "depends_on": [],
                "acceptance_criteria": [
                    "`watch_log.sh` exists and is executable.",
                    "`.watch_state` is created/updated.",
                    "`README.md` contains usage instructions and flag descriptions.",
                ],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "PROJ_ROOT=$(pwd); TMP_DIR=$(mktemp -d); "
                    "( trap 'rm -rf \"$TMP_DIR\"' EXIT; cd \"$TMP_DIR\"; "
                    "cp \"$PROJ_ROOT/watch_log.sh\" .; cp \"$PROJ_ROOT/README.md\" .; "
                    "chmod +x watch_log.sh; touch test.log; "
                    "printf MATCH >> test.log; grep -q 'usage' \"$PROJ_ROOT/README.md\"; "
                    "[ -f .watch_state ] || true )",
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertNotIn("workspace path `test.log`", text)
            self.assertNotIn("workspace source path `watch_log.sh`", text)

    def test_plan_validation_accepts_stateful_entrypoint_inside_mktemp_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Log watcher")
            agent.requirements["refined_requirements"] = [
                "watch_log.sh remembers progress in .watch_state.",
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement watch_log.sh",
                "description": "Create the core Bash script with state management.",
                "depends_on": [],
                "acceptance_criteria": [
                    "The script stores and resumes from `.watch_state`.",
                    "The validation runs from isolated temporary state.",
                ],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "PROJ_ROOT=$(pwd); TEMP_DIR=$(mktemp -d); "
                    "trap 'rm -rf \"$TEMP_DIR\"' EXIT; "
                    "cp \"$PROJ_ROOT/watch_log.sh\" \"$TEMP_DIR/\"; "
                    "cd \"$TEMP_DIR\" || exit 1; chmod +x watch_log.sh; "
                    "printf '%s\\n' trigger > test.log; "
                    "timeout 2s ./watch_log.sh test.log trigger 1 | grep -q TRIGGERED",
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            self.assertNotIn("stale workspace state", "\n".join(findings))

    def test_validation_rules_require_stateful_runtime_state_isolation(self) -> None:
        self.assertIn("temporary fixtures/state", VALIDATION_COMMAND_RULES)
        self.assertIn("clean them", VALIDATION_COMMAND_RULES)
        self.assertIn("without hiding failures", VALIDATION_COMMAND_RULES)
        self.assertIn("wire them into the command being tested", VALIDATION_COMMAND_RULES)
        self.assertNotIn(".watch_state", VALIDATION_COMMAND_RULES)

    def test_executable_deliverable_guidance_is_shared_across_prompts(self) -> None:
        self.assertIn("marks shebang files executable", EXECUTABLE_DELIVERABLE_GUIDANCE)
        self.assertIn("test -x ./script", EXECUTABLE_DELIVERABLE_GUIDANCE)
        self.assertIn("Do not use \"executable\" merely to mean", EXECUTABLE_DELIVERABLE_GUIDANCE)
        self.assertIn("python script.py", EXECUTABLE_DELIVERABLE_GUIDANCE)
        for prompt in (REQUIREMENTS_CONTRACT, PLAN_REFINEMENT_CONTRACT, IMPLEMENTATION_CONTRACT):
            self.assertIn("Executable deliverables", prompt)
            self.assertIn("Do not use `chmod`", prompt)

    def test_json_output_rules_discourage_latex_backslash_notation(self) -> None:
        self.assertIn("plain ASCII prose", JSON_OUTPUT_RULES)
        self.assertIn("LaTeX/backslash", JSON_OUTPUT_RULES)
        self.assertIn("\\le", JSON_OUTPUT_RULES)

    def test_json_output_rules_strongly_forbid_markdown_fences(self) -> None:
        self.assertIn("Never wrap the object", JSON_OUTPUT_RULES)
        self.assertIn("```json", JSON_OUTPUT_RULES)
        self.assertIn("first character", JSON_OUTPUT_RULES)

    def test_review_prompts_preserve_review_decision_role(self) -> None:
        compact_guidance = " ".join(REVIEW_DECISION_OUTPUT_GUIDANCE.split())
        self.assertIn("Review decision output", REVIEW_DECISION_OUTPUT_GUIDANCE)
        self.assertIn("do not return a replacement plan", REVIEW_DECISION_OUTPUT_GUIDANCE)
        self.assertIn("do not emit a new implementation payload", compact_guidance)
        for prompt in (
            ANALYSIS_REVIEW_CONTRACT,
            APPROACH_REVIEW_CONTRACT,
            FEEDBACK_SYSTEM_PROMPT,
            TOOL_CALL_VERIFICATION_CONTRACT,
            TOOL_PROGRESS_REVIEW_CONTRACT,
        ):
            self.assertIn("Review decision output", prompt)

    def test_shared_review_prompt_guidance_includes_json_rules_and_evidence_precision(self) -> None:
        guidance = _review_prompt_guidance()

        self.assertIn("Review decision output", guidance)
        self.assertIn("Evidence-bound review check", guidance)
        self.assertIn("Never wrap the object", guidance)
        self.assertIn("first character", guidance)
        self.assertIn("Report evidence", guidance)
        self.assertIn("Generated tests and validators are evidence", guidance)
        self.assertIn("Positive presence evidence is not enough", guidance)
        self.assertIn("one-time side", guidance)

    def test_runtime_feedback_prompts_include_json_output_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    json.dumps({
                        "status": "resolved",
                        "needs_rework": False,
                        "summary": "analysis accepted",
                        "required_changes": [],
                        "quality_questions": [],
                    }),
                    json.dumps({
                        "status": "approved",
                        "summary": "safe validation",
                        "commands": [{
                            "index": 0,
                            "decision": "approved",
                            "risk_level": "low",
                            "reason": "bounded validation",
                        }],
                    }),
                ],
            )

            agent._analysis_review(1, {
                "problem_restatement": "Build a small checked artifact.",
                "initial_source_check": {
                    "sources_checked": ["none"],
                    "source_gaps": [],
                    "freshness_risks": [],
                },
                "possible_solution_paths": [
                    {"id": "A", "description": "direct path", "advantages": [], "risks": [], "verification_strategy": "test"},
                    {"id": "B", "description": "alternate path", "advantages": [], "risks": [], "verification_strategy": "test"},
                ],
                "recommended_path": {
                    "path_id": "A",
                    "rationale": "smallest sufficient path",
                    "fallback_trigger": "new evidence invalidates it",
                },
                "analysis_quality": {
                    "is_comprehensive": True,
                    "is_domain_aware": True,
                    "is_actionable_for_planning": True,
                    "remaining_unknowns": [],
                },
            })
            agent._tool_call_verification_phase(
                [["python3", "-m", "unittest", "test_wait_for_file.py"]],
                source="implementation",
                context={"step": {"id": "S1", "title": "Run tests", "status": "pending"}},
            )

            prompts = [
                "\n".join(message["content"] for message in call["messages"])
                for call in agent.feedback_client.calls
            ]
            self.assertEqual(len(prompts), 2)
            for prompt in prompts:
                self.assertIn("Never wrap the object", prompt)
                self.assertIn("first character", prompt)
                self.assertIn("Review decision output", prompt)

    def test_review_guidance_preserves_model_repair_autonomy(self) -> None:
        self.assertIn("Do not prescribe a complete replacement", REVIEW_CHALLENGE_GUIDANCE)
        self.assertIn("should choose the repair", REVIEW_CHALLENGE_GUIDANCE)

    def test_evidence_review_flags_stale_output_reuse_in_validation_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S1",
                "title": "Implement watcher",
                "acceptance_criteria": ["validator proves behavior after state changes"],
                "validation_commands": [["bash", "validate_watcher.sh"]],
            }
            evidence = {
                "workspace_files": [{
                    "path": "validate_watcher.sh",
                    "content": """#!/usr/bin/env bash
./watcher > "phase_output.txt" &
if grep -q "TRIGGERED" "phase_output.txt" 2>/dev/null; then
  echo first pass
fi
echo "new line" > "$LOG_FILE"
echo "needle" >> "$LOG_FILE"
if grep -q "TRIGGERED" "phase_output.txt" 2>/dev/null; then
  echo second pass
fi
""",
                }],
                "validation_results": [{
                    "command": ["bash", "validate_watcher.sh"],
                    "returncode": 0,
                    "expected_returncode": 0,
                    "timed_out": False,
                }],
                "git": {"meaningful_changed_paths": ["validate_watcher.sh"]},
            }

            findings = agent._evidence_findings(step, {"commands": []}, evidence)

            text = "\n".join(findings)
            self.assertIn("stale evidence", text)
            self.assertIn("phase_output.txt", text)

    def test_evidence_review_allows_cleared_output_before_reassertion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            evidence = {
                "workspace_files": [{
                    "path": "validate_watcher.sh",
                    "content": """#!/usr/bin/env bash
./watcher > "phase_output.txt" &
if grep -q "TRIGGERED" "phase_output.txt" 2>/dev/null; then
  echo first pass
fi
: > "phase_output.txt"
echo "new line" > "$LOG_FILE"
echo "needle" >> "$LOG_FILE"
if grep -q "TRIGGERED" "phase_output.txt" 2>/dev/null; then
  echo second pass
fi
""",
                }],
            }

            findings = agent._stale_validation_evidence_findings(evidence)

            self.assertEqual(findings, [])

    def test_evidence_review_flags_delayed_file_test_that_precreates_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            evidence = {
                "workspace_files": [{
                    "path": "test_wait_for_file.py",
                    "content": """import subprocess, sys, tempfile, threading, time, unittest

class WaitForFileTests(unittest.TestCase):
    def test_file_appears_during_wait(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        def create_file_later():
            time.sleep(1)
            with open(tmp_path, "w", encoding="utf-8") as handle:
                handle.write("ok")

        thread = threading.Thread(target=create_file_later)
        thread.start()
        result = subprocess.run(
            [sys.executable, "wait_for_file.py", tmp_path],
            text=True,
            capture_output=True,
        )
        thread.join()
        self.assertEqual(result.returncode, 0)
""",
                }],
            }

            findings = agent._delayed_resource_validation_findings(evidence)

            text = "\n".join(findings)
            self.assertIn("appears after the watcher starts", text)
            self.assertIn("already", text)

    def test_evidence_review_allows_delayed_file_test_with_absent_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            evidence = {
                "workspace_files": [{
                    "path": "test_wait_for_file.py",
                    "content": """import os, subprocess, sys, tempfile, threading, unittest

class WaitForFileTests(unittest.TestCase):
    def test_file_appears_during_wait(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "later.txt")
            self.assertFalse(os.path.exists(tmp_path))
            timer = threading.Timer(0.05, lambda: open(tmp_path, "w", encoding="utf-8").write("ok"))
            timer.start()
            result = subprocess.run(
                [sys.executable, "wait_for_file.py", tmp_path],
                text=True,
                capture_output=True,
            )
            timer.join()
            self.assertEqual(result.returncode, 0)
""",
                }],
            }

            findings = agent._delayed_resource_validation_findings(evidence)

            self.assertEqual(findings, [])

    def test_timeout_friendly_benchmark_requires_real_delayed_file_detection(self) -> None:
        tasks = run_benchmarks.load_tasks(Path("benchmarks/tasks.json"))
        task = next(item for item in tasks if item["id"] == "tool-004-timeout-friendly")
        post_validation = json.dumps(task["post_validation_commands"])

        self.assertIn("--timeout-seconds", task["prompt"])
        self.assertIn("--interval-seconds", task["prompt"])
        self.assertIn("threading.Timer", post_validation)
        self.assertIn("appears.txt", post_validation)
        self.assertIn("'unrecognized' not in out", post_validation)
        self.assertIn("time.monotonic", post_validation)
        self.assertIn("'waiting' in out", post_validation)
        self.assertNotIn("'timeout' in out", post_validation)

    def test_final_evidence_flags_stale_output_reuse_in_validation_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement watcher",
                "acceptance_criteria": ["validator proves truncation with fresh output"],
            }]
            evidence = {
                "workspace_files": [{
                    "path": "validate_watcher.sh",
                    "content": """#!/usr/bin/env bash
./watcher > "trunc_output.txt" &
if grep -q "TRIGGERED" "trunc_output.txt" 2>/dev/null; then
  echo reached before truncation
fi
echo "new line 1" > "$LOG_FILE"
echo "needle" >> "$LOG_FILE"
if grep -q "TRIGGERED" "trunc_output.txt" 2>/dev/null; then
  echo stale success
fi
""",
                }],
                "step_validations": [{
                    "step_id": "S1",
                    "validation_results": [{
                        "command": ["bash", "validate_watcher.sh"],
                        "returncode": 0,
                        "expected_returncode": 0,
                        "timed_out": False,
                    }],
                }],
            }

            findings = agent._project_evidence_findings(
                [{"step_id": "S1", "status": "resolved", "attempts": [{"implementation": {"commands": []}}]}],
                evidence,
            )

            text = "\n".join(findings)
            self.assertIn("stale evidence", text)
            self.assertIn("trunc_output.txt", text)

    def test_plan_validation_rejects_unisolated_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create watch_log.sh. It should poll a log path, remember the "
                    "last checked line in .watch_state, and print TRIGGERED."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Log watcher")
            agent.requirements["refined_requirements"] = [
                "The script remembers the last checked line in .watch_state."
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement log watcher",
                "description": "Create the script and validate trigger/resume behavior.",
                "depends_on": [],
                "acceptance_criteria": ["watch_log.sh remembers the last checked line in .watch_state."],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "LOG=$(mktemp); trap 'rm -f \"$LOG\"' EXIT; "
                    "printf '%s\\n' needle > \"$LOG\"; "
                    "timeout 2s ./watch_log.sh --file \"$LOG\" --pattern needle | grep -q TRIGGERED",
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("runtime state", text)
            self.assertIn("stale workspace state", text)
            self.assertIn("temporary working directory", text)
            self.assertIn("do not add a public state-file", text)

            agent.plan_steps[0]["validation_commands"] = [[
                "bash",
                "-lc",
                "tmp=$(mktemp -d); trap 'rm -rf \"$tmp\"' EXIT; "
                "LOG=\"$tmp/log\"; printf '%s\\n' needle > \"$LOG\"; "
                "STATE_FILE=\"$tmp/state\" timeout 2s ./watch_log.sh --file \"$LOG\" "
                "--pattern needle | grep -q TRIGGERED",
            ]]

            findings = agent._plan_structural_findings()

            self.assertNotIn("stale workspace state", "\n".join(findings))

    def test_plan_validation_does_not_treat_executable_probe_as_stateful_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create watch_log.sh. It should poll a log path, remember the "
                    "last checked line in .watch_state, and print TRIGGERED."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Log watcher")
            agent.requirements["refined_requirements"] = [
                "The script remembers the last checked line in .watch_state."
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement log watcher",
                "description": "Create the script and a validator that uses a temporary working directory.",
                "depends_on": [],
                "acceptance_criteria": [
                    "watch_log.sh is executable.",
                    "validate.sh verifies state persistence in an isolated temporary directory.",
                ],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "test -x ./watch_log.sh && test -x ./validate.sh && ./validate.sh",
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            self.assertNotIn("stale workspace state", "\n".join(findings))

    def test_plan_validation_rejects_help_only_stateful_validation_precisely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create watch_log.sh. It should remember the last checked "
                    "line in .watch_state and print TRIGGERED."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Log watcher")
            agent.requirements["refined_requirements"] = [
                "The script remembers the last checked line in .watch_state."
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement log watcher",
                "description": "Create the script and README.",
                "depends_on": [],
                "acceptance_criteria": [
                    "watch_log.sh remembers the last checked line in .watch_state.",
                    "README.md contains Usage.",
                ],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "grep -q 'Usage' README.md && ./watch_log.sh --help",
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("only checks help or metadata", text)
            self.assertNotIn("stale workspace state", text)

    def test_plan_validation_stateful_check_is_step_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create watch_log.sh. It should remember the last checked "
                    "line in .watch_state and print TRIGGERED."
                ),
            )
            agent.initialize()
            agent.requirements = base_requirements("Log watcher")
            agent.requirements["refined_requirements"] = [
                "The project has a stateful watcher that remembers .watch_state."
            ]
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement wrapper and README",
                "description": "Create CLI shell and documentation; deeper behavior is validated later.",
                "depends_on": [],
                "acceptance_criteria": [
                    "watch_log.sh exists and is executable.",
                    "README.md contains Usage.",
                    "The script exits non-zero when required arguments are missing.",
                ],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "test -x ./watch_log.sh && grep -q 'Usage' README.md && ./watch_log.sh --help > /dev/null && ! ./watch_log.sh > /dev/null 2>&1",
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertNotIn("stateful workflow", text)
            self.assertNotIn("stale workspace state", text)

    def test_plan_validation_rejects_placeholder_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, prompt="Build a small CLI and validate it.")
            agent.initialize()
            agent.requirements = base_requirements("CLI")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Implement CLI",
                "description": "Create the CLI and validation.",
                "depends_on": [],
                "acceptance_criteria": ["The CLI behavior is validated."],
                "validation_commands": [[
                    "python",
                    "-c",
                    "print('Test logic placeholder passed')",
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("placeholder or stub test logic", text)
            self.assertIn("exercising the requested artifact", text)

    def test_plan_validation_rejects_printf_literal_percent_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Disk monitor")
            agent.plan_steps = [{
                "id": "S1",
                "title": "Validate df parsing",
                "description": "Mock df and verify threshold output.",
                "depends_on": [],
                "acceptance_criteria": ["Mocked df output can include percent capacity values."],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "tmp=$(mktemp -d); "
                    "printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n/dev/mock 1000 950 50 95% /\\n' > \"$tmp/df\"",
                ]],
                "status": "pending",
            }]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("printf", text)
            self.assertIn("%%", text)

            agent.plan_steps[0]["validation_commands"] = [[
                "bash",
                "-lc",
                "tmp=$(mktemp -d); "
                "printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n/dev/mock 1000 950 50 95%% /\\n' > \"$tmp/df\"",
            ]]

            findings = agent._plan_structural_findings()

            self.assertNotIn("unescaped literal `%`", "\n".join(findings))

            agent.plan_steps[0]["validation_commands"] = [[
                "bash",
                "-lc",
                "tmp=$(mktemp -d); "
                "printf '%s\\n' 'Filesystem 1024-blocks Used Available Capacity Mounted on' "
                "'/dev/mock 1000 950 50 95% /' > \"$tmp/df\"",
            ]]

            findings = agent._plan_structural_findings()

            self.assertNotIn("unescaped literal `%`", "\n".join(findings))

    def test_plan_validation_rejects_timeout_wait_builtin_misuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S1",
                "title": "Validate watcher",
                "description": "Validate that a background watcher observes a delayed event.",
                "depends_on": [],
                "acceptance_criteria": ["The watcher reacts after the delayed event is appended."],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "tmp=$(mktemp -d); cd \"$tmp\" || exit 1; "
                    "sleep 0.1 & PID=$!; timeout 5s wait $PID",
                ]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            text = "\n".join(findings)
            self.assertIn("external `timeout`", text)
            self.assertIn("not a standalone program", text)

            step["validation_commands"] = [[
                "timeout",
                "5s",
                "wait",
                "$PID",
            ]]

            findings = agent._validation_command_findings(step)

            self.assertIn("not a standalone program", "\n".join(findings))

    def test_tool_verifier_rejects_shell_cleanup_after_assertion_status_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            findings = agent._deterministic_tool_call_findings([[
                "bash",
                "-lc",
                "echo '{\"b\": 1}' > /tmp/input.json; "
                "python3 normalize_config.py /tmp/input.json | grep -q '{\"b\":1}'; "
                "rm /tmp/input.json",
            ]])

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertIn("after a validation assertion", reasons)

    def test_tool_verifier_rejects_assertion_or_echo_failure_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            findings = agent._deterministic_tool_call_findings([[
                "bash",
                "-lc",
                "python3 app.py | grep -q READY || echo failed",
            ]])

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertIn("mask a validation failure", reasons)

    def test_tool_verifier_rejects_success_echo_failure_mask_with_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            findings = agent._deterministic_tool_call_findings([[
                "bash",
                "-lc",
                "touch test_file.txt && python wait_for_file.py --file test_file.txt "
                "--timeout-seconds 2 && echo 'Success: Exit 0' || echo 'Fail: Exit 0'; "
                "rm test_file.txt",
            ]])

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertIn("mask a validation failure", reasons)

    def test_tool_verifier_rejects_timeout_wait_builtin_misuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            findings = agent._deterministic_tool_call_findings([[
                "bash",
                "-lc",
                "tmp=$(mktemp -d); cd \"$tmp\" || exit 1; sleep 0.1 & PID=$!; timeout 5s wait $PID",
            ]])

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertIn("shell builtin `wait`", reasons)
            self.assertIn("not a standalone program", reasons)

            direct_findings = agent._deterministic_tool_call_findings([[
                "timeout",
                "5s",
                "wait",
                "$PID",
            ]])
            direct_reasons = "\n".join(str(item.get("reason", "")) for item in direct_findings)
            self.assertIn("not a standalone program", direct_reasons)

    def test_tool_verifier_rejects_env_assignment_after_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Build monitor_disk.sh. It must support MIN_FREE_PERCENT and "
                    "MAX_CHECKS environment overrides."
                ),
            )
            agent.requirements = base_requirements("Disk monitor")
            agent.requirements["refined_requirements"] = [
                "The script supports MIN_FREE_PERCENT and MAX_CHECKS environment overrides."
            ]

            findings = agent._deterministic_tool_call_findings([[
                "bash",
                "-lc",
                "PATH=/tmp/mock:$PATH ./monitor_disk.sh MIN_FREE_PERCENT=10 | grep -q ACTION_REQUIRED",
            ]])

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertIn("MIN_FREE_PERCENT", reasons)
            self.assertIn("environment override", reasons)

    def test_tool_verifier_rejects_workspace_validation_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            findings = agent._deterministic_tool_call_findings([[
                "bash",
                "-lc",
                "./monitor_disk.sh > output.log && grep -q ACTION_REQUIRED output.log",
            ]])

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertIn("output.log", reasons)
            self.assertIn("unrequested project artifact", reasons)

    def test_tool_verifier_allows_quoted_python_greater_than_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            findings = agent._deterministic_tool_call_findings([[
                "bash",
                "-lc",
                "python -c \"val = open('ANSWER.txt').read().strip(); assert val.isdigit() and int(val) > 0\"",
            ]])

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertNotIn("workspace path `0`", reasons)
            self.assertNotIn("unrequested project artifact", reasons)

    def test_tool_verifier_rejects_raw_text_numeric_comparison_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            findings = agent._deterministic_tool_call_findings([[
                "python",
                "-c",
                (
                    "s=sum(n for n in range(1,121)); actual=open('ANSWER.txt').read().strip(); "
                    "assert s == actual, f'expected={s} actual={actual}'"
                ),
            ]])

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertIn("raw file text", reasons)
            self.assertIn("numeric expression", reasons)

    def test_tool_verifier_allows_stringified_numeric_comparison_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            findings = agent._deterministic_tool_call_findings([[
                "python",
                "-c",
                (
                    "s=sum(n for n in range(1,121)); actual=open('ANSWER.txt').read().strip(); "
                    "assert str(s) == actual, f'expected={s} actual={actual}'"
                ),
            ]])

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertNotIn("raw file text", reasons)

    def test_tool_verifier_keeps_block_for_wrong_answer_suspicion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            command = [
                "bash",
                "-lc",
                (
                    "python -c 'expected = 1878; actual = int(open(\"ANSWER.txt\").read().strip()); "
                    "assert expected == actual, f\"Expected {expected}, got {actual}\"'"
                ),
            ]
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    json.dumps({
                        "status": "blocked",
                        "summary": "The proposed ANSWER.txt content is incorrect.",
                        "commands": [{
                            "index": 0,
                            "decision": "blocked",
                            "risk_level": "medium",
                            "reason": "The proposed answer content does not match my manually calculated expected sum.",
                        }],
                    })
                ],
            )
            agent.initialize()
            (workspace / "ANSWER.txt").write_text("1034", encoding="utf-8")
            step = {
                "id": "S1",
                "title": "Validate answer",
                "description": "Validate the generated answer.",
                "acceptance_criteria": ["ANSWER.txt contains the correct sum."],
                "validation_commands": [command],
            }

            results = agent._run_verified_commands(
                [command],
                source="implementation",
                context={"step": step, "purpose": "current step validation"},
            )

            self.assertTrue(results[0].get("blocked_by_tool_verifier", False))
            self.assertEqual(results[0]["returncode"], 126)
            self.assertIn("Tool call blocked before execution", results[0]["stderr"])
            self.assertEqual(results[0]["tool_verification"]["decision"], "blocked")

    def test_tool_verifier_keeps_block_for_outer_expansion_misread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            command = [
                "bash",
                "-lc",
                (
                    "tmp=$(mktemp -d); trap 'rm -rf \"$tmp\"' EXIT; "
                    "cd \"$tmp\" || exit 1; printf '%s\\n' ok > result.txt; "
                    "test \"$(pwd)\" = \"$tmp\" && grep -q ok result.txt"
                ),
            ]
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    json.dumps({
                        "status": "blocked",
                        "summary": "The command has a shell expansion issue.",
                        "commands": [{
                            "index": 0,
                            "decision": "blocked",
                            "risk_level": "medium",
                            "reason": (
                                "The command uses double quotes for the bash -lc string, so $(pwd) will be "
                                "expanded by the current shell before execution and may target the wrong path."
                            ),
                        }],
                    })
                ],
            )
            agent.initialize()
            step = {
                "id": "S1",
                "title": "Validate temporary shell fixture",
                "description": "Run a bounded validation script in a temporary directory.",
                "acceptance_criteria": ["The command produces and checks temporary evidence."],
                "validation_commands": [command],
            }

            results = agent._run_verified_commands(
                [command],
                source="implementation",
                context={"step": step, "purpose": "current step validation"},
            )

            self.assertTrue(results[0].get("blocked_by_tool_verifier", False))
            self.assertEqual(results[0]["returncode"], 126)
            self.assertEqual(results[0]["tool_verification"]["decision"], "blocked")

    def test_tool_verifier_keeps_shell_quoting_block_for_validation_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            command = [
                "bash",
                "-lc",
                (
                    "python3 -c 'actual = open(\"ANSWER.txt\").read().strip(); assert actual.isdigit()'"
                ),
            ]
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    json.dumps({
                        "status": "blocked",
                        "summary": "The validation command contains malformed shell quoting.",
                        "commands": [{
                            "index": 0,
                            "decision": "blocked",
                            "risk_level": "high",
                            "reason": (
                                "The command uses nested double quotes that will prematurely terminate the "
                                "shell string and produce a shell syntax error."
                            ),
                        }],
                    })
                ],
            )
            agent.initialize()
            step = {
                "id": "S1",
                "title": "Validate answer",
                "description": "Validate the generated answer.",
                "acceptance_criteria": ["ANSWER.txt contains the correct sum."],
                "validation_commands": [command],
            }

            results = agent._run_verified_commands(
                [command],
                source="implementation",
                context={"step": step, "purpose": "current step validation"},
            )

            self.assertTrue(results[0].get("blocked_by_tool_verifier"))
            self.assertEqual(results[0]["tool_verification"]["decision"], "blocked")
            self.assertIn("syntax error", results[0]["tool_verification"]["reason"])

    def test_tool_verifier_rejects_unisolated_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create watch_log.sh. It should poll a log path, remember the "
                    "last checked line in .watch_state, and print TRIGGERED."
                ),
            )
            agent.requirements = base_requirements("Log watcher")
            agent.requirements["refined_requirements"] = [
                "The script remembers the last checked line in .watch_state."
            ]
            step = {
                "id": "S1",
                "title": "Validate log watcher",
                "description": "Validate trigger and resume behavior.",
                "acceptance_criteria": ["watch_log.sh remembers the last checked line in .watch_state."],
            }

            findings = agent._deterministic_tool_call_findings(
                [[
                    "bash",
                    "-lc",
                    "LOG=$(mktemp); trap 'rm -f \"$LOG\"' EXIT; "
                    "printf '%s\\n' needle > \"$LOG\"; "
                    "timeout 2s ./watch_log.sh --file \"$LOG\" --pattern needle | grep -q TRIGGERED",
                ]],
                context={"step": step},
            )

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertIn("runtime state", reasons)
            self.assertIn("stale workspace state", reasons)

            isolated = agent._deterministic_tool_call_findings(
                [[
                    "bash",
                    "-lc",
                    "tmp=$(mktemp -d); trap 'rm -rf \"$tmp\"' EXIT; "
                    "LOG=\"$tmp/log\"; printf '%s\\n' needle > \"$LOG\"; "
                    "STATE_FILE=\"$tmp/state\" timeout 2s ./watch_log.sh --file \"$LOG\" "
                    "--pattern needle | grep -q TRIGGERED",
                ]],
                context={"step": step},
            )
            isolated_reasons = "\n".join(str(item.get("reason", "")) for item in isolated)
            self.assertNotIn("stale workspace state", isolated_reasons)

    def test_tool_verifier_does_not_treat_diagnostic_echo_as_stateful_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create watch_log.sh. It should poll a log path, remember the "
                    "last checked line in .watch_state, and print TRIGGERED."
                ),
            )
            agent.requirements = base_requirements("Log watcher")
            agent.requirements["refined_requirements"] = [
                "The script remembers the last checked line in .watch_state."
            ]
            step = {
                "id": "S1",
                "title": "Implement log watcher",
                "description": "Create the script and README.",
                "acceptance_criteria": [
                    "watch_log.sh is executable.",
                    "README.md contains Usage.",
                    "watch_log.sh remembers the last checked line in .watch_state.",
                ],
            }

            findings = agent._deterministic_tool_call_findings(
                [{
                    "cmd": [
                        "bash",
                        "-lc",
                        (
                            "test -x ./watch_log.sh || "
                            "(echo 'Error: watch_log.sh is not executable' && exit 1); "
                            "grep -q 'Usage:' README.md || "
                            "(echo 'Error: README.md missing Usage section' && exit 1)"
                        ),
                    ]
                }],
                context={"step": step},
            )

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertNotIn("stale workspace state", reasons)
            self.assertNotIn("runtime state", reasons)

    def test_tool_verifier_rejects_help_only_stateful_validation_precisely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt=(
                    "Create watch_log.sh. It should remember the last checked "
                    "line in .watch_state and print TRIGGERED."
                ),
            )
            agent.requirements = base_requirements("Log watcher")
            step = {
                "id": "S1",
                "title": "Validate log watcher",
                "description": "Validate README and script metadata.",
                "acceptance_criteria": ["watch_log.sh remembers the last checked line in .watch_state."],
            }

            findings = agent._deterministic_tool_call_findings(
                [["bash", "-lc", "grep -q 'Usage' README.md && ./watch_log.sh --help"]],
                context={"step": step},
            )

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertIn("only checks help or metadata", reasons)
            self.assertNotIn("stale workspace state", reasons)

    def test_tool_verifier_rejects_placeholder_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            findings = agent._deterministic_tool_call_findings([[
                "python",
                "-c",
                "print('Test logic placeholder passed')",
            ]])

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertIn("placeholder or stub test logic", reasons)
            self.assertIn("requested artifact", reasons)

    def test_tool_verifier_rejects_stale_output_reuse_in_validation_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "validate_watcher.sh").write_text(
                """#!/usr/bin/env bash
./watcher > "trunc_output.txt" &
if grep -q "TRIGGERED" "trunc_output.txt" 2>/dev/null; then
  echo first pass
fi
echo "new line" > "$LOG_FILE"
echo "needle" >> "$LOG_FILE"
if grep -q "TRIGGERED" "trunc_output.txt" 2>/dev/null; then
  echo stale second pass
fi
""",
                encoding="utf-8",
            )
            agent = load_test_agent(root, workspace)

            findings = agent._deterministic_tool_call_findings([[
                "bash",
                "-lc",
                "grep -q Usage README.md && bash validate_watcher.sh",
            ]])

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertIn("stale evidence", reasons)
            self.assertIn("trunc_output.txt", reasons)

    def test_tool_verifier_rejects_blocking_python_validation_readline_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "validate.py").write_text(
                """import subprocess
import time

proc = subprocess.Popen(
    ["./watch_log.sh", "app.log", "needle", "0.1"],
    stdout=subprocess.PIPE,
    text=True,
)
start = time.time()
while time.time() - start < 2:
    line = proc.stdout.readline()
    if "TRIGGERED" in line:
        break
proc.terminate()
""",
                encoding="utf-8",
            )
            agent = load_test_agent(root, workspace)

            findings = agent._deterministic_tool_call_findings([["python", "validate.py"]])

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertIn("blocking `stdout.readline()`", reasons)
            self.assertIn("long-running or watch-style", reasons)

    def test_tool_verifier_allows_python_validation_communicate_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "validate.py").write_text(
                """import subprocess

proc = subprocess.Popen(
    ["python", "-c", "print('ok')"],
    stdout=subprocess.PIPE,
    text=True,
)
stdout, _ = proc.communicate(timeout=2)
assert "ok" in stdout
""",
                encoding="utf-8",
            )
            agent = load_test_agent(root, workspace)

            findings = agent._deterministic_tool_call_findings([["python", "validate.py"]])

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertNotIn("blocking `stdout.readline()`", reasons)

    def test_tool_verifier_rejects_chmod_source_validation_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            findings = agent._deterministic_tool_call_findings([[
                "bash",
                "-c",
                "chmod +x monitor_disk.sh && ./monitor_disk.sh | head -n 1 | grep -q '.'",
            ]])

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertIn("workspace source path `monitor_disk.sh`", reasons)
            self.assertIn("shebang", reasons)
            self.assertIn("test -x", reasons)

    def test_tool_verifier_rejects_printf_literal_percent_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            findings = agent._deterministic_tool_call_findings([[
                "bash",
                "-lc",
                "tmp=$(mktemp -d); "
                "printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n/dev/mock 1000 950 50 95% /\\n' > \"$tmp/df\"",
            ]])

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertIn("printf", reasons)
            self.assertIn("%%", reasons)

    def test_tool_verifier_rejects_string_valued_command_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            findings = agent._deterministic_tool_call_findings([{
                "cmd": "bash -lc \"echo ok\"",
                "expected_returncode": 0,
            }])

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertIn("string-valued `cmd`", reasons)

    def test_tool_verifier_rejects_expected_returncode_shell_cleanup_status_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            findings = agent._deterministic_tool_call_findings([{
                "cmd": [
                    "bash",
                    "-lc",
                    "echo 'print(\"Line 1: content\")' > /tmp/bad_cnt.py; "
                    "python3 validate_huge_output.py --tool /tmp/bad_cnt.py --count 5; "
                    "rm /tmp/bad_cnt.py",
                ],
                "expected_returncode": 1,
            }])

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertIn("without preserving its status", reasons)

    def test_plan_validation_rejects_relative_source_write_after_workspace_cd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            step = {
                "id": "S2",
                "title": "Validate negative path",
                "description": "Prove the validator rejects a bad generator.",
                "depends_on": [],
                "acceptance_criteria": ["The validator rejects bad generated output."],
                "validation_commands": [[
                    "bash",
                    "-lc",
                    "cd . && echo 'print(\"Wrong\")' > huge_output.py && python validate_huge_output.py",
                ]],
                "status": "pending",
            }

            findings = agent._validation_command_findings(step)

            self.assertIn("workspace source path `huge_output.py`", "\n".join(findings))

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

    def test_malformed_feedback_repair_uses_minimal_second_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    "<|channel>thought\n```json\n{\"status\":\"needs_rework\",\"required_changes\":[",
                    json.dumps({
                        "status": "resolved",
                        "needs_rework": False,
                        "summary": "Minimal repair accepted the reviewer-owned evidence.",
                        "required_changes": [],
                    }),
                ],
            )
            agent.initialize()

            review = agent._extract_json_or_retry(
                "not valid json",
                phase="STEP_REVIEW_PHASE",
                contract='{"status":"resolved|needs_rework","required_changes":["specific change"]}',
                feedback=True,
            )
            repair_prompt = agent.feedback_client.calls[0]["messages"][-1]["content"]
            minimal_prompt = agent.feedback_client.calls[1]["messages"][-1]["content"]

            self.assertEqual(review["status"], "resolved")
            self.assertEqual(review["summary"], "Minimal repair accepted the reviewer-owned evidence.")
            self.assertEqual(len(agent.feedback_client.calls), 2)
            self.assertIn("Previous response recovery note:", repair_prompt)
            self.assertNotIn("Previous response tail for recovery:", repair_prompt)
            self.assertIn("STEP_REVIEW_PHASE_MINIMAL_JSON_REPAIR", minimal_prompt)
            self.assertIn("do not request task changes just because the reviewer response was malformed", minimal_prompt)

    def test_incomplete_final_review_uses_protocol_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    json.dumps({
                        "status": "resolved",
                        "needs_rework": False,
                        "summary": "Final evidence inspected.",
                        "required_changes": [],
                        "verification_evidence": ["reviewer-owned validation passed"],
                    }),
                ],
            )
            agent.initialize()

            review = agent._extract_json_or_retry(
                json.dumps({
                    "status": "resolved",
                    "summary": "Looks good but omitted required final evidence fields.",
                }),
                phase="FINAL_PROJECT_REVIEW_PHASE",
                contract=(
                    '{"status":"resolved|needs_rework|cannot_resolve|needs_requirements_change|'
                    'needs_plan_change|skipped_with_note|resolved_with_compromise",'
                    '"needs_rework":false,"summary":"whole project review",'
                    '"required_changes":["specific final change"],'
                    '"verification_evidence":["evidence reviewed"]}'
                ),
                feedback=True,
                current_question_context="CURRENT FINAL REVIEW PAYLOAD: reviewer-owned validation passed",
            )
            repair_prompt = agent.feedback_client.calls[0]["messages"][-1]["content"]

            self.assertEqual(review["status"], "resolved")
            self.assertEqual(review["verification_evidence"], ["reviewer-owned validation passed"])
            self.assertIn("FINAL_PROJECT_REVIEW_PHASE_JSON_REPAIR", repair_prompt)
            self.assertIn("verification_evidence", repair_prompt)
            self.assertIn("CURRENT FINAL REVIEW PAYLOAD", repair_prompt)

    def test_complete_final_review_can_omit_derived_needs_rework(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, feedback_responses=[])
            agent.initialize()

            review = agent._normalize_review(agent._extract_json_or_retry(
                json.dumps({
                    "status": "resolved",
                    "summary": "Final evidence satisfies the request.",
                    "required_changes": [],
                    "verification_evidence": ["reviewer-owned validation returned exit code 0"],
                }),
                phase="FINAL_PROJECT_REVIEW_PHASE",
                contract=(
                    '{"status":"resolved|needs_rework|cannot_resolve|needs_requirements_change|'
                    'needs_plan_change|skipped_with_note|resolved_with_compromise",'
                    '"needs_rework":false,"summary":"whole project review",'
                    '"required_changes":["specific final change"],'
                    '"verification_evidence":["evidence reviewed"]}'
                ),
                feedback=True,
            ))

            self.assertEqual(review["status"], "resolved")
            self.assertFalse(review["needs_rework"])
            self.assertEqual(len(agent.feedback_client.calls), 0)

    def test_final_review_placeholder_summary_with_evidence_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, feedback_responses=[])
            agent.initialize()

            review = agent._normalize_review(agent._extract_json_or_retry(
                json.dumps({
                    "status": "resolved",
                    "needs_rework": False,
                    "summary": "whole project review",
                    "required_changes": [],
                    "verification_evidence": ["reviewer-owned validation returned exit code 0"],
                }),
                phase="FINAL_PROJECT_REVIEW_PHASE",
                contract=(
                    '{"status":"resolved|needs_rework","needs_rework":false,'
                    '"summary":"concrete final review summary",'
                    '"required_changes":["concrete final change, or empty when resolved"],'
                    '"verification_evidence":["specific command result, file evidence, or reviewer fact"]}'
                ),
                feedback=True,
            ))

            self.assertEqual(review["status"], "resolved")
            self.assertEqual(review["summary"], "Final review resolved based on supplied verification evidence.")
            self.assertEqual(review["verification_evidence"], ["reviewer-owned validation returned exit code 0"])
            self.assertEqual(len(agent.feedback_client.calls), 0)

    def test_malformed_feedback_fallback_records_review_protocol_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            malformed = "<|channel>thought\n```json\n{\"status\":\"needs_rework\",\"required_changes\":["
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[malformed, malformed],
            )
            agent.initialize()

            review = agent._extract_json_or_retry(
                "not valid json",
                phase="STEP_REVIEW_PHASE",
                contract='{"status":"resolved|needs_rework|cannot_resolve","summary":"review summary","required_changes":["specific change"]}',
                feedback=True,
            )

            self.assertEqual(review["status"], "cannot_resolve")
            self.assertTrue(review["review_protocol_error"])
            self.assertIn("Reviewer protocol repair failed", "\n".join(review["required_changes"]))
            self.assertNotIn("focused directly verifiable change", "\n".join(review["required_changes"]))
            self.assertIn("parse_error", review)
            self.assertIn("final_repair_error", review)

    def test_feedback_json_repair_omits_malformed_review_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    json.dumps({
                        "status": "resolved",
                        "needs_rework": False,
                        "summary": "Repair answered from current evidence instead of copied scratch.",
                        "required_changes": [],
                        "verification_evidence": ["reviewer-owned evidence was inspected"],
                    }),
                ],
            )
            agent.initialize()

            review = agent._extract_json_or_retry(
                "<think>Manual scratch says wrong total 1782, but this is not JSON.</think>",
                phase="FINAL_PROJECT_REVIEW_PHASE",
                contract=(
                    '{"status":"resolved|needs_rework","needs_rework":false,'
                    '"summary":"review summary","required_changes":["specific change"],'
                    '"verification_evidence":["evidence reviewed"]}'
                ),
                feedback=True,
            )
            repair_prompt = agent.feedback_client.calls[-1]["messages"][-1]["content"]

            self.assertEqual(review["status"], "resolved")
            self.assertIn("Previous response recovery note:", repair_prompt)
            self.assertNotIn("Previous response tail for recovery:", repair_prompt)
            self.assertNotIn("wrong total 1782", repair_prompt)

    def test_review_schema_placeholders_trigger_json_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    json.dumps({
                        "status": "resolved",
                        "needs_rework": False,
                        "summary": "Final evidence and reviewer-owned validation satisfy the request.",
                        "required_changes": [],
                        "verification_evidence": ["reviewer-owned validation was inspected"],
                    }),
                ],
            )
            agent.initialize()

            review = agent._extract_json_or_retry(
                json.dumps({
                    "status": "resolved",
                    "needs_rework": False,
                    "summary": "whole project review",
                    "required_changes": [],
                }),
                phase="FINAL_PROJECT_REVIEW_PHASE",
                contract=(
                    '{"status":"resolved|needs_rework","needs_rework":false,'
                    '"summary":"whole project review","required_changes":["specific final change"],'
                    '"verification_evidence":["evidence reviewed"]}'
                ),
                feedback=True,
            )
            repair_prompt = agent.feedback_client.calls[-1]["messages"][-1]["content"]

            self.assertEqual(review["summary"], "Final evidence and reviewer-owned validation satisfy the request.")
            self.assertEqual(len(agent.feedback_client.calls), 1)
            self.assertIn("Schema example strings are placeholders", repair_prompt)
            self.assertIn("whole project review", repair_prompt)

    def test_reasoning_only_feedback_acceptance_uses_json_repair_dialogue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    json.dumps({
                        "status": "resolved",
                        "needs_rework": False,
                        "summary": "Protocol repair accepted the plan.",
                        "required_changes": [],
                        "planning_confirmation": {
                            "feasible": True,
                            "clear": True,
                            "verifiable": True,
                            "verification_matrix": [
                                {"step_id": "S1", "how_verified": "Run the listed validation command."}
                            ],
                        },
                    }),
                ],
            )
            agent.initialize()

            review = agent._extract_json_or_retry(
                "<think>The plan is feasible, clear, and verifiable. I'll accept.</think>",
                phase="PLAN_VALIDATION_PHASE",
                contract='{"status":"resolved|needs_plan_change","required_changes":["specific change"]}',
                feedback=True,
            )

            self.assertEqual(review["status"], "resolved")
            self.assertFalse(review["needs_rework"])
            self.assertNotIn("inferred_from_malformed_response", review)
            self.assertEqual(len(agent.feedback_client.calls), 1)

    def test_resolved_plan_review_without_matrix_uses_json_repair_dialogue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            repair = {
                "status": "resolved",
                "needs_rework": False,
                "summary": "Protocol repair supplied concrete plan verification coverage.",
                "required_changes": [],
                "planning_confirmation": {
                    "feasible": True,
                    "clear": True,
                    "verifiable": True,
                    "verification_matrix": [
                        {"step_id": "S1", "how_verified": "Run the step validation command and inspect its result."}
                    ],
                },
            }
            agent = load_test_agent(root, workspace, feedback_responses=[json.dumps(repair)])
            agent.initialize()

            review = agent._extract_json_or_retry(
                '{"status":"resolved","summary":"The plan is acceptable."}',
                phase="PLAN_VALIDATION_PHASE",
                contract=(
                    '{"status":"resolved|needs_plan_change","summary":"review summary",'
                    '"planning_confirmation":{"feasible":true,"clear":true,"verifiable":true,'
                    '"verification_matrix":[{"step_id":"S1","how_verified":"command"}]}}'
                ),
                feedback=True,
            )

            self.assertEqual(review["status"], "resolved")
            self.assertEqual(review["summary"], repair["summary"])
            self.assertEqual(review["planning_confirmation"]["verification_matrix"], repair["planning_confirmation"]["verification_matrix"])
            self.assertEqual(len(agent.feedback_client.calls), 1)

    def test_complete_resolved_plan_review_does_not_need_json_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, feedback_responses=[])
            agent.initialize()

            review = agent._extract_json_or_retry(
                json.dumps({
                    "status": "resolved",
                    "summary": "The plan is feasible, clear, and verifiable.",
                    "planning_confirmation": {
                        "feasible": True,
                        "clear": True,
                        "verifiable": True,
                        "verification_matrix": [
                            {"step_id": "S1", "how_verified": "Run python -m unittest test_app.py."}
                        ],
                    },
                }),
                phase="PLAN_VALIDATION_PHASE",
                contract=(
                    '{"status":"resolved|needs_plan_change","summary":"review summary",'
                    '"planning_confirmation":{"feasible":true,"clear":true,"verifiable":true,'
                    '"verification_matrix":[{"step_id":"S1","how_verified":"command"}]}}'
                ),
                feedback=True,
            )

            self.assertEqual(review["status"], "resolved")
            self.assertEqual(len(agent.feedback_client.calls), 0)

    def test_reasoning_only_final_review_completion_uses_json_repair_dialogue(self) -> None:
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
            self.assertNotIn("inferred_from_malformed_response", review)
            self.assertEqual(len(agent.feedback_client.calls), 1)

    def test_long_reasoning_final_review_completion_uses_json_repair_dialogue(self) -> None:
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
            self.assertNotIn("inferred_from_malformed_response", review)
            self.assertEqual(len(agent.feedback_client.calls), 1)

    def test_malformed_feedback_with_clear_resolved_status_does_not_force_rework(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, feedback_responses=[])
            agent.initialize()
            raw = (
                '```json\n'
                '{\n'
                '  "status": "resolved",\n'
                '  "summary": "The validation command `python -c "print(1)"` returned exit code 0.",\n'
                '  "verification_evidence": ["ANSWER.txt was verified against the expected result."]\n'
                '}\n'
                '```'
            )

            review = agent._extract_json_or_retry(
                raw,
                phase="STEP_REVIEW_PHASE",
                contract='{"status":"resolved|needs_rework","required_changes":["specific change"]}',
                feedback=True,
            )

            self.assertEqual(review["status"], "resolved")
            self.assertFalse(review["needs_rework"])
            self.assertNotIn("inferred_from_malformed_response", review)
            self.assertEqual(len(agent.feedback_client.calls), 1)

    def test_status_only_review_uses_json_repair_dialogue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    json.dumps({
                        "status": "resolved",
                        "needs_rework": False,
                        "summary": "Protocol repair supplied the missing review fields.",
                        "required_changes": [],
                        "verification_evidence": ["reviewer-owned validation was inspected"],
                    }),
                ],
            )
            agent.initialize()

            review = agent._extract_json_or_retry(
                '{"status":"resolved"}',
                phase="FINAL_PROJECT_REVIEW_PHASE",
                contract=(
                    '{"status":"resolved|needs_rework","needs_rework":false,'
                    '"summary":"review summary","required_changes":["specific change"],'
                    '"verification_evidence":["evidence reviewed"]}'
                ),
                feedback=True,
            )

            self.assertEqual(review["status"], "resolved")
            self.assertEqual(review["summary"], "Protocol repair supplied the missing review fields.")
            self.assertEqual(review["required_changes"], [])
            self.assertEqual(len(agent.feedback_client.calls), 1)

    def test_analysis_review_repair_omits_command_and_artifact_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                prompt="Create ANSWER.txt only with the requested result.",
                feedback_responses=[
                    json.dumps({
                        "status": "resolved",
                        "needs_rework": False,
                        "summary": "Analysis compared viable paths and preserved scope.",
                        "required_changes": [],
                        "quality_questions": [],
                    }),
                ],
            )
            agent.initialize()

            review = agent._extract_json_or_retry(
                '{"status":"resolved"}',
                phase="PROBLEM_ANALYSIS_REVIEW_PHASE",
                contract=ANALYSIS_REVIEW_CONTRACT,
                feedback=True,
            )
            repair_prompt = agent.feedback_client.calls[-1]["messages"][-1]["content"]

            self.assertEqual(review["status"], "resolved")
            self.assertIn("Return a review decision object", repair_prompt)
            self.assertNotIn("Command protocol:", repair_prompt)
            self.assertNotIn("Commands are data, not prose", repair_prompt)
            self.assertNotIn("Artifact-only boundary", repair_prompt)
            self.assertNotIn("cannot execute <tool_call>", repair_prompt)
            self.assertNotIn("Per-attempt file limits", repair_prompt)
            self.assertNotIn("files[].content", repair_prompt)

    def test_resolved_review_with_summary_can_omit_required_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace, feedback_responses=[])
            agent.initialize()

            payload = agent._extract_json_or_retry(
                '{"status":"resolved","summary":"Review accepted with enough evidence."}',
                phase="STEP_REVIEW_PHASE",
                contract='{"status":"resolved|needs_rework","summary":"review summary","required_changes":["specific change"]}',
                feedback=True,
            )
            review = agent._normalize_review(payload)

            self.assertEqual(review["status"], "resolved")
            self.assertEqual(review["summary"], "Review accepted with enough evidence.")
            self.assertEqual(review["required_changes"], [])
            self.assertEqual(len(agent.feedback_client.calls), 0)

    def test_invalid_review_status_uses_json_repair_dialogue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    json.dumps({
                        "status": "needs_rework",
                        "needs_rework": True,
                        "summary": "Protocol repair selected an allowed review status.",
                        "required_changes": ["Name the concrete evidence gap."],
                    }),
                ],
            )
            agent.initialize()

            review = agent._extract_json_or_retry(
                json.dumps({
                    "status": "probably_ok",
                    "summary": "Looks fine.",
                    "required_changes": [],
                }),
                phase="STEP_REVIEW_PHASE",
                contract='{"status":"resolved|needs_rework","summary":"review summary","required_changes":["specific change"]}',
                feedback=True,
            )

            self.assertEqual(review["status"], "needs_rework")
            self.assertEqual(review["summary"], "Protocol repair selected an allowed review status.")
            self.assertEqual(len(agent.feedback_client.calls), 1)

    def test_tool_progress_unknown_decision_uses_json_repair_dialogue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    json.dumps({
                        "status": "continue",
                        "decision": "continue",
                        "summary": "Progress is still useful.",
                        "evidence": ["stdout shows new progress"],
                        "risks": [],
                        "next_check_seconds": 30,
                    }),
                ],
            )
            agent.initialize()

            review = agent._extract_json_or_retry(
                json.dumps({
                    "status": "wait",
                    "decision": "wait",
                    "summary": "Let it run.",
                    "evidence": ["some output"],
                    "risks": [],
                    "next_check_seconds": 30,
                }),
                phase="TOOL_PROGRESS_REVIEW_PHASE",
                contract='{"status":"continue|terminate","decision":"continue|terminate","summary":"why","evidence":[],"risks":[]}',
                feedback=True,
            )

            self.assertEqual(review["decision"], "continue")
            self.assertEqual(review["status"], "continue")
            self.assertEqual(len(agent.feedback_client.calls), 1)

    def test_tool_call_unknown_command_decision_uses_json_repair_dialogue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    json.dumps({
                        "status": "approved",
                        "summary": "Protocol repair supplied explicit command decisions.",
                        "commands": [
                            {
                                "index": 0,
                                "decision": "approved",
                                "risk_level": "low",
                                "reason": "bounded read-only validation",
                            }
                        ],
                    }),
                ],
            )
            agent.initialize()

            review = agent._extract_json_or_retry(
                json.dumps({
                    "status": "approved",
                    "summary": "Safe enough.",
                    "commands": [{"index": 0, "decision": "safe", "reason": "read only"}],
                }),
                phase="TOOL_CALL_VERIFICATION_PHASE",
                contract='{"status":"approved|blocked","commands":[{"index":0,"decision":"approved|blocked"}]}',
                feedback=True,
            )

            self.assertEqual(review["commands"][0]["decision"], "approved")
            self.assertEqual(len(agent.feedback_client.calls), 1)

    def test_tool_verification_phase_repairs_unknown_command_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    json.dumps({
                        "status": "approved",
                        "summary": "Uses a non-protocol command decision.",
                        "commands": [{"index": 0, "decision": "safe", "reason": "read only"}],
                    }),
                    json.dumps({
                        "status": "approved",
                        "summary": "Protocol repair supplied an allowed command decision.",
                        "commands": [
                            {
                                "index": 0,
                                "decision": "approved",
                                "risk_level": "low",
                                "reason": "bounded read-only validation",
                            }
                        ],
                    }),
                ],
            )
            agent.initialize()

            review = agent._tool_call_verification_phase(
                [["python", "-c", "print('ok')"]],
                source="implementation",
                context={"purpose": "validate command protocol repair"},
            )

            self.assertEqual(review["status"], "approved")
            self.assertEqual(review["commands"][0]["decision"], "approved")
            self.assertEqual(len(agent.feedback_client.calls), 2)

    def test_tool_verification_minimal_repair_preserves_current_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            current_command = [
                "bash",
                "-lc",
                "export MAX_POLLS=10; ./watch_and_react.sh input.txt",
            ]
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    json.dumps({
                        "status": "needs_revision",
                        "summary": "Off-contract command decision.",
                        "commands": [
                            {
                                "index": 0,
                                "decision": "needs_revision",
                                "risk_level": "medium",
                                "reason": "stale command concern",
                            }
                        ],
                    }),
                    json.dumps({
                        "status": "needs_revision",
                        "summary": "Still off-contract command decision.",
                        "commands": [
                            {
                                "index": 0,
                                "decision": "needs_revision",
                                "risk_level": "medium",
                                "reason": "still stale command concern",
                            }
                        ],
                    }),
                    json.dumps({
                        "status": "approved",
                        "summary": "Minimal repair judged the current supplied command.",
                        "commands": [
                            {
                                "index": 0,
                                "decision": "approved",
                                "risk_level": "low",
                                "reason": "bounded current validation command",
                            }
                        ],
                    }),
                ],
            )
            agent.initialize()
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Implement watcher",
                    "status": "pending",
                    "validation_commands": [
                        [
                            "bash",
                            "-lc",
                            "export MAX_POLLS=3; ./watch_and_react.sh input.txt",
                        ]
                    ],
                }
            ]

            review = agent._tool_call_verification_phase(
                [current_command],
                source="implementation",
                context={"step": agent.plan_steps[0], "purpose": "validate watcher"},
            )

            minimal_prompt = agent.feedback_client.calls[-1]["messages"][-1]["content"]
            self.assertEqual(review["status"], "approved")
            self.assertEqual(review["commands"][0]["command"], current_command)
            self.assertIn("TOOL_CALL_VERIFICATION_PHASE_MINIMAL_JSON_REPAIR", minimal_prompt)
            self.assertIn("authoritative set of proposed commands", minimal_prompt)
            self.assertIn("MAX_POLLS=10", minimal_prompt)
            self.assertIn("MAX_POLLS=3", minimal_prompt)
            current_section = minimal_prompt.split("Current review question and supplied evidence:", 1)[1]
            self.assertIn("MAX_POLLS=10", current_section)

    def test_reasoning_only_tool_call_approval_uses_json_repair_dialogue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    json.dumps({
                        "status": "approved",
                        "summary": "Protocol repair approved the command.",
                        "commands": [],
                    }),
                ],
            )
            agent.initialize()

            review = agent._extract_json_or_retry(
                "<think>The command is correctly targeted and bounded. I'll approve it.</think>",
                phase="TOOL_CALL_VERIFICATION_PHASE",
                contract='{"status":"approved|blocked","commands":[{"index":0,"decision":"approved|blocked"}]}',
                feedback=True,
            )

            self.assertEqual(review["status"], "approved")
            self.assertEqual(review["commands"], [])
            self.assertNotIn("inferred_from_malformed_response", review)
            self.assertEqual(len(agent.feedback_client.calls), 1)

    def test_reasoning_only_tool_verifier_approval_uses_json_repair_dialogue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    "<think>The command is correctly targeted and bounded for this validation. I will approve it.</think>",
                    json.dumps({
                        "status": "approved",
                        "summary": "Protocol repair approved the command.",
                        "commands": [{"index": 0, "decision": "approved", "reason": "bounded validation command"}],
                    }),
                ],
            )
            agent.initialize()

            review = agent._tool_call_verification_phase(
                [["bash", "./watch_log.sh", "--help"]],
                source="implementation",
                context={"purpose": "validate help output"},
            )

            self.assertEqual(review["status"], "approved")
            self.assertEqual(review["commands"][0]["decision"], "approved")
            self.assertNotIn("inferred_from_malformed_response", review)
            self.assertEqual(len(agent.feedback_client.calls), 2)

    def test_tool_verifier_blocks_command_with_protocol_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    json.dumps({
                        "status": "blocked",
                        "summary": "validate.py would hang because it runs an infinite watcher with subprocess.run.",
                        "commands": [
                            {
                                "index": 0,
                                "decision": "blocked",
                                "risk_level": "medium",
                                "reason": "validate.py would hang because it runs an infinite watcher with subprocess.run.",
                            }
                        ],
                    })
                ],
            )
            agent.initialize()

            results = agent._run_verified_commands(
                [["python3", "validate.py"]],
                source="implementation",
                context={"purpose": "run generated validation"},
            )

            self.assertEqual(len(results), 1)
            self.assertTrue(results[0]["blocked_by_tool_verifier"])
            self.assertEqual(results[0]["returncode"], 126)
            self.assertIn("validate.py would hang", results[0]["stderr"])
            self.assertEqual(len(agent.feedback_client.calls), 1)

    def test_reasoning_only_tool_call_block_uses_json_repair_dialogue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    "<think>`python logic.py` will not create ANSWER.txt, so I will block this command.</think>",
                    json.dumps({
                        "status": "blocked",
                        "summary": "Protocol repair blocked the supplied commands.",
                        "commands": [
                            {"index": 0, "decision": "blocked", "risk_level": "medium", "reason": "does not verify the requested evidence"},
                            {"index": 1, "decision": "blocked", "risk_level": "medium", "reason": "depends on missing prior output"},
                        ],
                    }),
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
            self.assertNotIn("inferred_from_malformed_response", review)
            self.assertEqual(len(agent.feedback_client.calls), 2)

    def test_reasoning_only_validation_tool_block_uses_json_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            repaired_review = {
                "status": "approved",
                "summary": "The reviewer-owned validation command is bounded and matches the plan.",
                "commands": [
                    {
                        "index": 0,
                        "decision": "approved",
                        "risk_level": "low",
                        "reason": "It only prints the calculated value for this validation-only step.",
                    }
                ],
            }
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[
                    "<think>The command does not create ANSWER.txt, so I will block this command.</think>",
                    json.dumps(repaired_review),
                ],
            )
            agent.initialize()

            review = agent._tool_call_verification_phase(
                [["python", "-c", "print(24)"]],
                source="step_feedback_validation",
                context={"purpose": "reviewer-owned validation command"},
            )

            self.assertEqual(review["status"], "approved")
            self.assertEqual(review["commands"][0]["decision"], "approved")
            self.assertEqual(len(agent.feedback_client.calls), 2)

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

    def test_feedback_json_repair_prompt_stays_in_reviewer_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            repair = {
                "status": "needs_rework",
                "needs_rework": True,
                "summary": "Requirements review must ask for a corrected validation command.",
                "required_changes": ["Replace malformed validation with a bounded command."],
            }
            agent = load_test_agent(root, workspace, feedback_responses=[json.dumps(repair)])
            agent.initialize()

            review = agent._extract_json_or_retry(
                json.dumps({
                    "project_summary": "Wrong payload shape",
                    "refined_requirements": [],
                    "assumptions": [],
                    "planning_confirmation": {},
                    "plan": [],
                }),
                phase="REQUIREMENTS_REVIEW_PHASE",
                contract='{"status":"resolved|needs_rework","required_changes":["specific change"]}',
                feedback=True,
            )

            prompt = agent.feedback_client.calls[-1]["messages"][-1]["content"]
            self.assertEqual(review["status"], "needs_rework")
            self.assertIn("This is a feedback/review phase", prompt)
            self.assertIn("do not return requirements, plan, files, or implementation payloads", prompt)
            self.assertIn("Do not tell the implementation agent to add review-only fields", prompt)
            self.assertEqual(review["required_changes"], repair["required_changes"])

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

    def test_structured_token_caps_reserve_room_after_reasoning_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            model = replace(
                agent.config.implementation_model,
                max_tokens=8192,
                reasoning_budget_tokens=4096,
            )
            runtime = replace(agent.config.runtime, feedback_response_max_tokens=4096)
            agent.config = replace(agent.config, implementation_model=model, runtime=runtime)

            self.assertEqual(agent._implementation_payload_tokens(), 8192)
            self.assertEqual(agent._structured_control_tokens(), 8192)
            self.assertEqual(agent._feedback_response_tokens(agent.config.implementation_model), 8192)

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

    def test_plan_review_rejects_grep_option_like_pattern_without_separator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Check curl notes",
                    "description": "Check a documentation file mentions a curl flag.",
                    "acceptance_criteria": ["CURL_NOTES.md mentions --data-binary @file."],
                    "validation_commands": [["bash", "-lc", "grep -q '--data-binary @' CURL_NOTES.md"]],
                }
            ]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("starts with `--`", text)
            self.assertIn("grep -q -- PATTERN FILE", text)

    def test_plan_review_rejects_malformed_bash_validation_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Check curl notes",
                    "description": "Check a documentation file mentions a curl flag.",
                    "acceptance_criteria": ["CURL_NOTES.md mentions --data-binary @file."],
                    "validation_commands": [[
                        "bash",
                        "-lc",
                        (
                            "if [ -f CURL_NOTES.md ]; then echo ok; else exit 1; fi && "
                            "grep -q -- '--data-binary @' CURL_NOTES.md; else exit 1; fi"
                        ),
                    ]],
                }
            ]

            findings = agent._plan_structural_findings()

            text = "\n".join(findings)
            self.assertIn("shell syntax", text)
            self.assertIn("static parse check", text)

    def test_plan_review_accepts_parseable_bash_validation_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.plan_steps = [
                {
                    "id": "S1",
                    "title": "Check curl notes",
                    "description": "Check a documentation file mentions a curl flag.",
                    "acceptance_criteria": ["CURL_NOTES.md mentions --data-binary @file."],
                    "validation_commands": [[
                        "bash",
                        "-lc",
                        (
                            "test -f CURL_NOTES.md && "
                            "grep -q -- '--data-binary @' CURL_NOTES.md && "
                            "grep -q -- 'curl' CURL_NOTES.md"
                        ),
                    ]],
                }
            ]

            findings = agent._plan_structural_findings()

            self.assertNotIn("static parse check", "\n".join(findings))

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

    def test_write_files_marks_shebang_files_executable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            written = write_files(
                workspace,
                [
                    {
                        "path": "monitor_disk.sh",
                        "content": "#!/bin/sh\nprintf '%s\\n' ok\n",
                    }
                ],
            )

            self.assertEqual(written, ["monitor_disk.sh"])
            self.assertTrue((workspace / "monitor_disk.sh").stat().st_mode & 0o111)

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

    def test_step_review_uses_accepted_validation_when_plan_command_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Stale step validation")
            (workspace / "answer.txt").write_text("done\n", encoding="utf-8")
            accepted_command = [
                "python",
                "-c",
                "from pathlib import Path; assert Path('answer.txt').read_text().strip() == 'done'; print('accepted ok')",
            ]
            step = {
                "id": "T5",
                "title": "Create answer",
                "description": "Write answer.txt and validate it.",
                "depends_on": [],
                "acceptance_criteria": ["answer.txt contains done."],
                "validation_commands": [["python", "-m", "unittest", "missing_plan_tests.py"]],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])

            review = agent._step_review_pass(
                step,
                2,
                {
                    "written": ["answer.txt"],
                    "commands": [{
                        "command": accepted_command,
                        "returncode": 0,
                        "expected_returncode": 0,
                        "timed_out": False,
                        "stdout": "accepted ok\n",
                        "stderr": "",
                    }],
                    "raw": {"test_evidence": ["accepted validation passed"]},
                },
                "hard_pushback",
            )

            self.assertEqual(review["status"], "resolved")
            self.assertEqual(review["deterministic_evidence_findings"], [])
            evidence = review["feedback_tool_evidence"]
            self.assertNotEqual(evidence["validation_results"][0]["returncode"], 0)
            self.assertEqual(evidence["accepted_validation_results"][0]["returncode"], 0)

    def test_step_review_uses_case_insensitive_accepted_grep_when_plan_grep_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("README validation")
            (workspace / "README.md").write_text(
                "# Example\n\n## Usage\n\nRun it.\n\n## Tests\n\nRun tests.\n",
                encoding="utf-8",
            )
            accepted_command = ["bash", "-lc", "grep -qi 'usage' README.md && grep -qi 'tests' README.md"]
            step = {
                "id": "S2",
                "title": "Create README.md",
                "description": "Create README documentation.",
                "depends_on": [],
                "acceptance_criteria": ["README.md contains usage and tests sections."],
                "validation_commands": [["bash", "-lc", "grep -q 'usage' README.md && grep -q 'tests' README.md"]],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])

            review = agent._step_review_pass(
                step,
                2,
                {
                    "written": ["README.md"],
                    "commands": [{
                        "command": accepted_command,
                        "returncode": 0,
                        "expected_returncode": 0,
                        "returncode_matches_expected": True,
                        "timed_out": False,
                        "stdout": "",
                        "stderr": "",
                    }],
                    "raw": {"test_evidence": ["case-insensitive README validation passed"]},
                },
                "hard_pushback",
            )

            self.assertEqual(review["status"], "resolved")
            self.assertEqual(review["deterministic_evidence_findings"], [])
            evidence = review["feedback_tool_evidence"]
            self.assertEqual(evidence["validation_results"][0]["returncode"], 1)
            self.assertEqual(evidence["accepted_validation_results"][0]["returncode"], 0)

    def test_step_review_uses_separator_fixed_grep_when_plan_grep_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Curl notes validation")
            (workspace / "CURL_NOTES.md").write_text(
                "Use curl --data-binary @payload.json for complex JSON.\n",
                encoding="utf-8",
            )
            accepted_command = ["bash", "-lc", "grep -q -- '--data-binary @' CURL_NOTES.md"]
            step = {
                "id": "S1",
                "title": "Create CURL_NOTES.md",
                "description": "Create curl notes.",
                "depends_on": [],
                "acceptance_criteria": ["CURL_NOTES.md explains --data-binary @file."],
                "validation_commands": [["bash", "-lc", "grep -q '--data-binary @' CURL_NOTES.md"]],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])

            review = agent._step_review_pass(
                step,
                2,
                {
                    "written": ["CURL_NOTES.md"],
                    "commands": [{
                        "command": accepted_command,
                        "returncode": 0,
                        "expected_returncode": 0,
                        "returncode_matches_expected": True,
                        "timed_out": False,
                        "stdout": "",
                        "stderr": "",
                    }],
                    "raw": {"test_evidence": ["separator-fixed grep validation passed"]},
                },
                "hard_pushback",
            )

            self.assertEqual(review["status"], "resolved")
            self.assertEqual(review["deterministic_evidence_findings"], [])
            evidence = review["feedback_tool_evidence"]
            self.assertEqual(evidence["validation_results"][0]["returncode"], 126)
            self.assertEqual(evidence["accepted_validation_results"][0]["returncode"], 0)

    def test_plan_failure_superseded_by_arg_separator_accepted_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            plan_result = {
                "command": ["bash", "-lc", "python slugify.py '---' | grep -q '^$'"],
                "returncode": 1,
                "expected_returncode": 0,
                "stderr": "usage: slugify.py [-h] text\nslugify.py: error: the following arguments are required: text\n",
                "stdout": "",
                "timed_out": False,
            }
            accepted_results = [{
                "command": ["bash", "-lc", "python slugify.py -- '---' | grep -q '^$'"],
                "returncode": 0,
                "expected_returncode": 0,
                "timed_out": False,
            }]

            self.assertTrue(agent._plan_failure_is_superseded_by_accepted_validation(plan_result, accepted_results))

    def test_plan_failure_superseded_by_grep_separator_accepted_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            plan_result = {
                "command": ["bash", "-lc", "grep -q '--data-binary @' CURL_NOTES.md"],
                "returncode": 2,
                "expected_returncode": 0,
                "stderr": "grep: unrecognized option '--data-binary @'\n",
                "stdout": "",
                "timed_out": False,
            }
            accepted_results = [{
                "command": ["bash", "-lc", "grep -q -- '--data-binary @' CURL_NOTES.md"],
                "returncode": 0,
                "expected_returncode": 0,
                "timed_out": False,
            }]

            self.assertTrue(agent._plan_failure_is_superseded_by_accepted_validation(plan_result, accepted_results))

    def test_blocked_plan_grep_superseded_by_separator_accepted_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            plan_result = {
                "command": ["bash", "-lc", "grep -q '--data-binary @' CURL_NOTES.md"],
                "returncode": 126,
                "expected_returncode": 0,
                "stderr": (
                    "Tool call blocked before execution by verification step: "
                    "The command is missing the '--' separator for grep, which will cause it to fail."
                ),
                "stdout": "",
                "timed_out": False,
            }
            accepted_results = [{
                "command": ["bash", "-lc", "grep -q -- '--data-binary @' CURL_NOTES.md"],
                "returncode": 0,
                "expected_returncode": 0,
                "timed_out": False,
            }]

            self.assertTrue(agent._plan_failure_is_superseded_by_accepted_validation(plan_result, accepted_results))

    def test_tool_call_verifier_blocks_malformed_bash_script_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            findings = agent._deterministic_tool_call_findings([
                [
                    "bash",
                    "-lc",
                    (
                        "if [ -f CURL_NOTES.md ]; then echo ok; else exit 1; fi && "
                        "grep -q -- '--data-binary @' CURL_NOTES.md; else exit 1; fi"
                    ),
                ]
            ])

            reasons = "\n".join(str(item.get("reason", "")) for item in findings)
            self.assertIn("static parse check", reasons)

    def test_blocked_stale_plan_validation_superseded_by_accepted_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            plan_result = {
                "command": [
                    "bash",
                    "-lc",
                    "TEMP_DIR=$(mktemp -d); TARGET=\"$TEMP_DIR/target\"; LOG=\"$TEMP_DIR/log\"; "
                    "touch \"$TARGET\"; echo 'trigger' > \"$TARGET\"; "
                    "./watch_and_react.sh \"$TARGET\" 'trigger' 0 5 \"$LOG\" && grep 'trigger' \"$LOG\"",
                ],
                "returncode": 126,
                "expected_returncode": 0,
                "timed_out": False,
                "blocked_by_tool_verifier": True,
                "stdout": "",
                "stderr": (
                    "Tool call blocked before execution by verification step: "
                    "the reviewer-owned validation command is stale or misaligned with the current artifact."
                ),
            }
            accepted_results = [{
                "command": [
                    "bash",
                    "-lc",
                    "TEMP_DIR=$(mktemp -d); TARGET=\"$TEMP_DIR/target\"; LOG=\"$TEMP_DIR/log\"; "
                    "touch \"$TARGET\"; echo 'trigger' > \"$TARGET\"; "
                    "./watch_and_react.sh \"$TARGET\" 'trigger' 0 5 \"$LOG\" && "
                    "grep 'Pattern matched in' \"$LOG\"",
                ],
                "returncode": 0,
                "expected_returncode": 0,
                "timed_out": False,
            }]

            self.assertTrue(agent._plan_failure_is_superseded_by_accepted_validation(plan_result, accepted_results))

            step = {
                "id": "S1",
                "title": "Create watch script",
                "description": "Create a script and validate it.",
                "depends_on": [],
                "acceptance_criteria": ["script logs when the watched pattern appears"],
                "validation_commands": [plan_result["command"]],
                "status": "pending",
            }
            findings = agent._evidence_findings(
                step,
                {"commands": []},
                {
                    "validation_results": [plan_result],
                    "accepted_validation_results": accepted_results,
                    "workspace_files": [],
                    "git": {"meaningful_changed_paths": ["watch_and_react.sh"]},
                },
            )

            self.assertEqual(findings, [])

    def test_plan_failure_superseded_by_doc_heading_punctuation_grep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            plan_result = {
                "command": ["bash", "-lc", "test -s README.md && grep -q 'Usage:' README.md && grep -q 'Tests' README.md"],
                "returncode": 1,
                "expected_returncode": 0,
                "stderr": "",
                "stdout": "",
                "timed_out": False,
            }
            accepted_results = [{
                "command": ["bash", "-lc", "grep -q 'Usage' README.md && grep -q 'Tests' README.md"],
                "returncode": 0,
                "expected_returncode": 0,
                "timed_out": False,
            }]

            self.assertTrue(agent._plan_failure_is_superseded_by_accepted_validation(plan_result, accepted_results))

    def test_doc_heading_punctuation_supersession_is_limited_to_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)

            plan_result = {
                "command": ["bash", "-lc", "grep -q 'Mode:' config.json"],
                "returncode": 1,
                "expected_returncode": 0,
                "stderr": "",
                "stdout": "",
                "timed_out": False,
            }
            accepted_results = [{
                "command": ["bash", "-lc", "grep -q 'Mode' config.json"],
                "returncode": 0,
                "expected_returncode": 0,
                "timed_out": False,
            }]

            self.assertFalse(agent._plan_failure_is_superseded_by_accepted_validation(plan_result, accepted_results))

    def test_resolved_step_adopts_accepted_validation_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Adopt repaired validation")
            accepted_command = [
                "python",
                "-c",
                "from pathlib import Path; assert Path('answer.txt').read_text().strip() == 'done'; print('accepted ok')",
            ]
            step = {
                "id": "T6",
                "title": "Create answer",
                "description": "Write answer.txt and validate it.",
                "depends_on": [],
                "acceptance_criteria": ["answer.txt contains done."],
                "validation_commands": [["python", "-m", "unittest", "missing_plan_tests.py"]],
                "status": "resolved",
            }
            agent.plan_steps = [step]

            agent._adopt_accepted_validation_commands_for_step(
                step,
                {
                    "implementation": {
                        "commands": [{
                            "command": accepted_command,
                            "returncode": 0,
                            "expected_returncode": 0,
                            "timed_out": False,
                            "stdout": "accepted ok\n",
                            "stderr": "",
                        }],
                    }
                },
            )

            self.assertEqual(step["validation_commands"], [accepted_command])

    def test_expected_returncode_command_is_accepted_validation_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()

            commands = agent._accepted_validation_commands_from_implementation({
                "commands": [{
                    "command": ["python", "slugify.py"],
                    "returncode": 2,
                    "expected_returncode": 2,
                    "returncode_matches_expected": True,
                    "timed_out": False,
                    "stdout": "",
                    "stderr": "usage: slugify.py [-h] text\n",
                }],
            })

            self.assertEqual(commands, [{"cmd": ["python", "slugify.py"], "expected_returncode": 2}])

    def test_failed_validation_summary_preserves_concrete_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Failing validation evidence")
            (workspace / "test_example.py").write_text(
                "import unittest\n\n"
                "class ExampleTests(unittest.TestCase):\n"
                "    def test_placeholder(self):\n"
                "        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            step = {
                "id": "T4",
                "title": "Create tests",
                "description": "Create and run tests.",
                "depends_on": [],
                "acceptance_criteria": ["Tests pass."],
                "validation_commands": [[
                    "python",
                    "-c",
                    "import sys; sys.stderr.write('regex mismatch detail\\n'); sys.exit(1)",
                ]],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])

            review = agent._step_review_pass(
                step,
                1,
                {
                    "written": ["test_example.py"],
                    "commands": [],
                    "raw": {"test_evidence": ["failing validation reproduced"]},
                },
                "hard_pushback",
            )

            self.assertEqual(review["status"], "needs_rework")
            self.assertIn("First finding: Feedback validation command returned 1", review["summary"])
            self.assertIn("regex mismatch detail", review["summary"])
            self.assertIn("regex mismatch detail", "\n".join(review["required_changes"]))

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

    def test_step_review_rejects_executable_deliverable_without_executable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Slug CLI")
            step = {
                "id": "S1",
                "title": "Implement slugify.py and tests",
                "description": "Create the Python CLI and test suite.",
                "depends_on": [],
                "acceptance_criteria": [
                    "`slugify.py` exists and is executable.",
                    "`python -m unittest discover` passes.",
                    "CLI exits with code 2 when no argument is provided.",
                ],
                "validation_commands": [
                    ["python", "-m", "unittest", "discover"],
                    {"cmd": ["python", "slugify.py"], "expected_returncode": 2},
                ],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])
            write_files(workspace, [{
                "path": "slugify.py",
                "content": (
                    "#!/usr/bin/env python3\n"
                    "import argparse\n"
                    "parser = argparse.ArgumentParser()\n"
                    "parser.add_argument('text')\n"
                    "args = parser.parse_args()\n"
                    "print(args.text.lower())\n"
                ),
            }])
            (workspace / "test_slugify.py").write_text(
                "import subprocess, sys, unittest\n\n"
                "class SlugTests(unittest.TestCase):\n"
                "    def test_cli(self):\n"
                "        result = subprocess.run([sys.executable, 'slugify.py', 'HELLO'], capture_output=True, text=True)\n"
                "        self.assertEqual(result.returncode, 0)\n"
                "        self.assertEqual(result.stdout.strip(), 'hello')\n",
                encoding="utf-8",
            )

            review = agent._step_review_pass(
                step,
                1,
                {
                    "written": ["slugify.py", "test_slugify.py"],
                    "commands": [],
                    "raw": {"test_evidence": ["unit tests and no-arg CLI check passed"]},
                },
                "hard_pushback",
            )

            self.assertEqual(review["status"], "needs_rework")
            text = "\n".join(review["deterministic_evidence_findings"])
            self.assertIn("`slugify.py` is required to be executable", text)
            self.assertIn("test -x ./slugify.py", text)

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

    def test_final_review_rejects_executable_deliverable_without_executable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Executable final evidence")
            step = {
                "id": "S1",
                "title": "Create monitor script",
                "description": "Write monitor_disk.sh.",
                "depends_on": [],
                "acceptance_criteria": ["`monitor_disk.sh` exists and is executable."],
                "validation_commands": [["bash", "-lc", "test -f monitor_disk.sh"]],
                "status": "resolved",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])
            write_files(workspace, [{
                "path": "monitor_disk.sh",
                "content": "#!/bin/sh\nprintf '%s\\n' ok\n",
            }])

            review = agent._final_project_review(
                1,
                [{"step_id": "S1", "status": "resolved", "attempts": [{"implementation": {"commands": []}}]}],
            )

            self.assertEqual(review["status"], "needs_rework")
            self.assertIn("monitor_disk.sh", "\n".join(review["deterministic_evidence_findings"]))
            self.assertIn("direct `./monitor_disk.sh`", "\n".join(review["deterministic_evidence_findings"]))

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

    def test_final_review_rescue_updates_current_plan_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            step_result = {"step_id": "S1", "status": "skipped_with_note", "attempts": []}
            agent.plan_steps = [{
                "id": "S1",
                "title": "Create artifact",
                "description": "Create and verify artifact.",
                "depends_on": [],
                "acceptance_criteria": ["artifact exists"],
                "validation_commands": [["test", "-f", "artifact.txt"]],
                "status": "skipped_with_note",
            }]
            review = {
                "feedback_tool_evidence": {
                    "step_validations": [{
                        "step_id": "S1",
                        "validation_results": [{
                            "command": ["test", "-f", "artifact.txt"],
                            "returncode": 0,
                            "expected_returncode": 0,
                            "returncode_matches_expected": True,
                            "timed_out": False,
                        }],
                    }]
                }
            }

            agent._apply_final_review_rescues([step_result], review)

            self.assertEqual(step_result["status"], "resolved")
            self.assertEqual(step_result["historical_status"], "skipped_with_note")
            self.assertTrue(step_result["resolved_by_final_review"])
            self.assertEqual(agent.plan_steps[0]["status"], "resolved")
            self.assertEqual(agent.plan_steps[0]["historical_status"], "skipped_with_note")

    def test_final_review_records_effective_deterministic_override_for_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(
                root,
                workspace,
                feedback_responses=[json.dumps({
                    "status": "resolved",
                    "needs_rework": False,
                    "summary": "Raw model accepted missing proof.",
                    "required_changes": [],
                    "verification_evidence": [],
                })],
            )
            agent.initialize()
            agent.requirements = base_requirements("Missing final proof")
            step = {
                "id": "S1",
                "title": "Create artifact",
                "description": "Create required artifact.",
                "depends_on": [],
                "acceptance_criteria": ["artifact.txt exists"],
                "validation_commands": [["test", "-f", "artifact.txt"]],
                "status": "resolved",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])

            review = agent._final_project_review(
                1,
                [{"step_id": "S1", "status": "resolved", "attempts": [{"implementation": {"commands": []}}]}],
            )
            control = latest_control_state(agent.conversation.turns)

            self.assertEqual(review["status"], "needs_rework")
            self.assertIn("status=needs_rework", control)
            self.assertIn("deterministic evidence", control.lower())

    def test_final_review_records_bounded_fallback_for_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.conversation.append("user", "FEEDBACK_AGENT_REQUEST:\nFINAL_PROJECT_REVIEW_PHASE\n{}")

            def fake_final_project_review(self, attempt, step_results):
                return {
                    "status": "needs_rework",
                    "needs_rework": True,
                    "summary": "Still missing proof.",
                    "required_changes": ["Add proof."],
                }

            def fake_final_correction_pass(self, attempt, review):
                return {"plan_note": "No usable correction."}

            agent._final_project_review = types.MethodType(fake_final_project_review, agent)
            agent._final_correction_pass = types.MethodType(fake_final_correction_pass, agent)

            result = agent._final_review_phase([])
            control = latest_control_state(agent.conversation.turns)

            self.assertEqual(result["status"], "cannot_resolve")
            self.assertIn("status=cannot_resolve", control)
            self.assertIn("bounded", control.lower())

    def test_final_review_rechecks_workspace_after_final_correction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            review_attempts: list[int] = []
            correction_attempts: list[int] = []

            def fake_final_project_review(self, attempt, step_results):
                review_attempts.append(attempt)
                if attempt == 1:
                    return {
                        "status": "needs_rework",
                        "needs_rework": True,
                        "summary": "README example is stale.",
                        "required_changes": ["Correct README example."],
                    }
                return {
                    "status": "resolved",
                    "needs_rework": False,
                    "summary": "Final correction evidence is now consistent.",
                    "required_changes": [],
                }

            def fake_final_correction_pass(self, attempt, review):
                correction_attempts.append(attempt)
                return {"plan_note": "Corrected README example.", "commands": []}

            agent._final_project_review = types.MethodType(fake_final_project_review, agent)
            agent._final_correction_pass = types.MethodType(fake_final_correction_pass, agent)

            result = agent._final_review_phase([])

            self.assertEqual(result["status"], "resolved")
            self.assertEqual(review_attempts, [1, 2])
            self.assertEqual(correction_attempts, [1])

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

    def test_implementation_loop_keeps_skipped_step_as_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.initialize()
            agent.requirements = base_requirements("Skipped implementation should not resolve")
            step = {
                "id": "S1",
                "title": "Do risky work",
                "description": "A step that cannot be proven.",
                "depends_on": [],
                "acceptance_criteria": ["proof exists"],
                "validation_commands": [],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])
            agent._implementation_pass = types.MethodType(lambda self, current, attempt: {"commands": []}, agent)
            agent._step_review_pass = types.MethodType(
                lambda self, current, attempt, implementation, review_mode: {
                    "status": "skipped_with_note",
                    "summary": "Compromise was explicitly recorded.",
                    "required_changes": [],
                },
                agent,
            )

            result = agent._implementation_loop_for_step(step)

            self.assertEqual(result["status"], "skipped_with_note")
            self.assertEqual(step["status"], "skipped_with_note")
            self.assertIsNone(agent._next_pending_step())

    def test_run_stops_before_implementation_when_plan_validation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent._web_research_phase = types.MethodType(lambda self: {"status": "skipped"}, agent)
            agent._analysis_phase = types.MethodType(lambda self, **kwargs: {"status": "resolved"}, agent)
            agent._requirements_refinement_phase = types.MethodType(lambda self, **kwargs: {"status": "resolved"}, agent)
            agent._plan_validation_phase = types.MethodType(
                lambda self: {
                    "status": "cannot_resolve",
                    "resolution": {"note": "Plan validation command stayed malformed."},
                    "iterations": [],
                },
                agent,
            )
            agent._implementation_loop_for_step = types.MethodType(
                lambda self, step: (_ for _ in ()).throw(AssertionError("implementation should not run")),
                agent,
            )
            agent._final_review_phase = types.MethodType(
                lambda self, steps: (_ for _ in ()).throw(AssertionError("final review should not run")),
                agent,
            )

            summary = agent.run()

            self.assertEqual(summary["final_status"], "cannot_resolve")
            self.assertEqual(summary["steps"][0]["step_id"], "plan_phase")
            self.assertIn("malformed", summary["steps"][0]["last_review_summary"])

    def test_run_blocks_dependent_step_after_failed_prerequisite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent._web_research_phase = types.MethodType(lambda self: {"status": "skipped"}, agent)
            agent._analysis_phase = types.MethodType(lambda self, **kwargs: {"status": "resolved"}, agent)
            agent._requirements_refinement_phase = types.MethodType(lambda self, **kwargs: {"status": "resolved"}, agent)

            def plan_validation(self: FeedbackLoopAgent) -> dict[str, Any]:
                self.requirements = base_requirements("Dependency gating")
                self.plan_steps = [
                    {
                        "id": "S1",
                        "title": "Prerequisite",
                        "description": "Create the core artifact.",
                        "depends_on": [],
                        "acceptance_criteria": ["core exists"],
                        "validation_commands": [["test", "-f", "core.txt"]],
                        "status": "pending",
                    },
                    {
                        "id": "S2",
                        "title": "Dependent validation",
                        "description": "Validate the core artifact.",
                        "depends_on": ["S1"],
                        "acceptance_criteria": ["validation passed"],
                        "validation_commands": [["test", "-f", "validation.txt"]],
                        "status": "pending",
                    },
                ]
                self._write_requirements_doc()
                self._write_plan_doc()
                return {"status": "resolved", "iterations": []}

            calls: list[str] = []

            def implementation_loop(self: FeedbackLoopAgent, step: dict[str, Any]) -> dict[str, Any]:
                calls.append(str(step["id"]))
                if step["id"] == "S1":
                    step["status"] = "cannot_resolve"
                    return {"step_id": "S1", "status": "cannot_resolve", "attempts": []}
                raise AssertionError("dependent step should not run after failed prerequisite")

            agent._plan_validation_phase = types.MethodType(plan_validation, agent)
            agent._implementation_loop_for_step = types.MethodType(implementation_loop, agent)
            agent._final_review_phase = types.MethodType(
                lambda self, steps: {"status": "cannot_resolve", "summary": "failed", "iterations": []},
                agent,
            )
            agent._approach_review_phase = types.MethodType(
                lambda self, attempt, steps, final: {"status": "resolved", "summary": "no retry"},
                agent,
            )

            summary = agent.run()

            self.assertEqual(calls, ["S1"])
            self.assertEqual(summary["steps"][1]["step_id"], "S2")
            self.assertEqual(summary["steps"][1]["status"], "cannot_resolve")
            self.assertEqual(summary["steps"][1]["blocked_by_dependency"]["dependency"], "S1")

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

    def test_compact_plan_preserves_normal_acceptance_criteria_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.plan_steps = [{
                "id": "S1",
                "title": "Multi-criterion step",
                "status": "pending",
                "acceptance_criteria": [
                    "criterion one",
                    "criterion two",
                    "criterion three",
                    "criterion four",
                    "criterion five",
                ],
                "validation_commands": [["python", "-m", "pytest"]],
            }]

            compact = agent._compact_plan_for_prompt()

            self.assertEqual(compact[0]["acceptance_criteria"], [
                "criterion one",
                "criterion two",
                "criterion three",
                "criterion four",
                "criterion five",
            ])
            self.assertEqual(compact[0]["acceptance_criteria_total"], 5)
            self.assertEqual(compact[0]["acceptance_criteria_omitted_count"], 0)

    def test_compact_plan_marks_omitted_acceptance_criteria(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agent = load_test_agent(root, workspace)
            agent.plan_steps = [{
                "id": "S1",
                "title": "Oversized criteria step",
                "status": "pending",
                "acceptance_criteria": [f"criterion {i} " + ("x" * 900) for i in range(20)],
                "validation_commands": [],
            }]

            compact = agent._compact_plan_for_prompt()

            self.assertLess(len(compact[0]["acceptance_criteria"]), 20)
            self.assertEqual(
                compact[0]["acceptance_criteria_total"],
                len(compact[0]["acceptance_criteria"]) + compact[0]["acceptance_criteria_omitted_count"],
            )
            self.assertGreater(compact[0]["acceptance_criteria_omitted_count"], 0)
            self.assertIn("acceptance criterion truncated", compact[0]["acceptance_criteria"][0])


if __name__ == "__main__":
    unittest.main()
