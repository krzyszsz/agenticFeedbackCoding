from __future__ import annotations

import contextlib
import functools
import http.server
import json
from pathlib import Path
import socketserver
import subprocess
import tempfile
import threading
import unittest

from feedback_agent.agent import FeedbackLoopAgent
from feedback_agent.config import load_config
from feedback_agent.llm import MockClient
from feedback_agent.workspace import run_commands, write_plan_doc


class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


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
            "name": "mock",
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
        "loop": {"max_iterations": 3, "stop_when_review_clean": True},
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


class FeedbackLoopAgentTests(unittest.TestCase):
    def test_command_timeout_can_be_overridden_per_command_and_clamped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = run_commands(
                root,
                [
                    {
                        "cmd": ["python", "-c", "print('long command shape accepted')"],
                        "timeout_seconds": 999,
                    }
                ],
                timeout_seconds=1,
                max_timeout_seconds=7,
            )

            self.assertEqual(results[0]["returncode"], 0)
            self.assertEqual(results[0]["timeout_seconds"], 7)
            self.assertIn("long command shape accepted", results[0]["stdout"])

    def test_mock_feedback_loop_has_distinct_phases_and_per_step_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            config_path = write_config(root, workspace, "mock tracker", "Build and test a task tracker.")
            cfg = load_config(config_path, repo_root=root)
            summary = FeedbackLoopAgent(cfg, mock=True).run()

            self.assertEqual(summary["requirements_refinement"]["status"], "resolved")
            self.assertEqual(summary["plan_validation"]["status"], "resolved")
            self.assertEqual(summary["final_status"], "resolved")
            self.assertGreaterEqual(len(summary["steps"]), 2)
            self.assertTrue(all(step["status"] == "resolved" for step in summary["steps"]))
            self.assertTrue(any(len(step["attempts"]) >= 2 for step in summary["steps"]))
            self.assertTrue((workspace / "REQUIREMENTS.md").exists())
            self.assertTrue((workspace / "PLAN.md").exists())
            self.assertTrue((workspace / "ARCHITECTURE.md").exists())
            self.assertTrue((workspace / "task_tracker.py").exists())
            self.assertTrue((workspace / "test_task_tracker.py").exists())
            self.assertTrue((workspace / ".agent_state" / "conversation.md").exists())
            transcript = (workspace / ".agent_state" / "conversation.jsonl").read_text()
            self.assertIn("IMPLEMENTATION_AGENT_REQUEST", transcript)
            self.assertIn("IMPLEMENTATION_AGENT_RESPONSE", transcript)
            self.assertIn("FEEDBACK_AGENT_REQUEST", transcript)
            self.assertIn("FEEDBACK_AGENT_RESPONSE", transcript)
            self.assertIn("planning_confirmation", transcript)
            self.assertIn("FINAL_PROJECT_REVIEW_PHASE", transcript)

    def test_mock_website_scenario_builds_browser_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "website"
            config_path = write_config(root, workspace, "mock website", "Build a tiny static website with a browser clicker game.")
            cfg = load_config(config_path, repo_root=root)
            summary = FeedbackLoopAgent(cfg, mock=True).run()

            self.assertEqual(summary["final_status"], "resolved")
            for name in ["index.html", "about.html", "game.html", "style.css", "app.js"]:
                self.assertTrue((workspace / name).exists(), name)
            self.assertIn("id=\"increment\"", (workspace / "game.html").read_text())
            self.assertIn("addEventListener", (workspace / "app.js").read_text())

    def test_mock_non_development_city_collection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "cities"
            config_path = write_config(root, workspace, "mock cities", "Use web scraping style workflow to collect images of big cities from Wikipedia.")
            cfg = load_config(config_path, repo_root=root)
            summary = FeedbackLoopAgent(cfg, mock=True).run()

            self.assertEqual(summary["final_status"], "resolved")
            manifest = json.loads((workspace / "city_image_manifest.json").read_text())
            self.assertGreaterEqual(len(manifest["cities"]), 4)
            self.assertTrue((workspace / "scripts" / "collect_city_images.py").exists())
            self.assertTrue((workspace / "collection_status.txt").exists())

    def test_platformer_review_rejects_failed_playwright_command(self) -> None:
        client = MockClient()
        prompt = json.dumps({
            "phase": "STEP_REVIEW_PHASE",
            "step": {"id": "S2", "title": "Add controllable movement and savepoints"},
            "requirements": {"project_summary": "platformer savepoint browser game"},
            "implementation": {
                "commands": [
                    {"command": ["python", "scripts/playwright_game_check.py"], "returncode": 1, "timed_out": False}
                ]
            },
        })

        review = json.loads(client.chat([{"role": "user", "content": prompt}]))

        self.assertEqual(review["status"], "needs_rework")
        self.assertIn("validation command failed", review["summary"])

    def test_feedback_review_runs_plan_validation_even_without_implementation_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            config_path = write_config(root, workspace, "tool evidence", "Build a tiny checked artifact.")
            cfg = load_config(config_path, repo_root=root)
            agent = FeedbackLoopAgent(cfg, mock=True)
            agent.initialize()
            agent.requirements = {
                "project_summary": "Tool evidence smoke test",
                "refined_requirements": ["Feedback must verify evidence independently."],
            }
            step = {
                "id": "T1",
                "title": "Create checked artifact",
                "description": "Write ok.txt and validate it.",
                "depends_on": [],
                "acceptance_criteria": ["ok.txt exists and contains pass"],
                "validation_commands": [[
                    "python",
                    "-c",
                    "from pathlib import Path; assert Path('ok.txt').read_text().strip() == 'pass'; print('feedback tool evidence ok')",
                ]],
                "status": "pending",
            }
            agent.plan_steps = [step]
            write_plan_doc(workspace, agent.requirements, agent.plan_steps, [])
            (workspace / "ok.txt").write_text("pass\n", encoding="utf-8")

            review = agent._step_review_pass(
                step,
                1,
                {"written": ["ok.txt"], "commands": [], "raw": {"test_evidence": []}},
                "hard_pushback",
            )

            self.assertEqual(review["status"], "resolved")
            self.assertEqual(review["deterministic_evidence_findings"], [])
            evidence = review["feedback_tool_evidence"]
            self.assertEqual(evidence["validation_results"][0]["returncode"], 0)
            self.assertIn("ok.txt", {item["path"] for item in evidence["workspace_files"]})

    def test_feedback_review_rejects_failing_feedback_tool_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            config_path = write_config(root, workspace, "tool evidence failure", "Build a tiny checked artifact.")
            cfg = load_config(config_path, repo_root=root)
            agent = FeedbackLoopAgent(cfg, mock=True)
            agent.initialize()
            agent.requirements = {
                "project_summary": "Tool evidence failure smoke test",
                "refined_requirements": ["Feedback must reject failing reviewer-owned validation."],
            }
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
            self.assertTrue(review["needs_rework"])
            self.assertIn("Feedback validation command failed", "\n".join(review["required_changes"]))
            self.assertNotEqual(review["feedback_tool_evidence"]["validation_results"][0]["returncode"], 0)

    def test_final_review_reruns_plan_validation_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            config_path = write_config(root, workspace, "final tool evidence", "Build a tiny checked artifact.")
            cfg = load_config(config_path, repo_root=root)
            agent = FeedbackLoopAgent(cfg, mock=True)
            agent.initialize()
            agent.requirements = {
                "project_summary": "Final review tool evidence smoke test",
                "refined_requirements": ["Final review must rerun plan validations."],
            }
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

    def test_web_research_is_fetched_and_used_in_structure_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = root / "site"
            site.mkdir()
            (site / "research.html").write_text(
                "<html><head><title>Harness Research Fixture</title></head>"
                "<body><main>CITATION_MARKER_ALPHA: use a small adapter boundary, "
                "write deterministic tests, and cite this source in architecture notes.</main></body></html>",
                encoding="utf-8",
            )
            with local_http_server(site) as base_url:
                workspace = root / "workspace"
                url = f"{base_url}/research.html"
                config_path = write_config(
                    root,
                    workspace,
                    "mock researched tracker",
                    f"Research this source before planning and building the tiny task tracker: {url}",
                )
                cfg = load_config(config_path, repo_root=root)
                summary = FeedbackLoopAgent(cfg, mock=True).run()

            self.assertEqual(summary["final_status"], "resolved")
            self.assertEqual(summary["web_research"]["status"], "completed")
            self.assertTrue(summary["web_research"]["requested"])
            self.assertEqual(summary["web_research"]["targets"][0]["status"], "ok")
            research_doc = (workspace / "RESEARCH.md").read_text(encoding="utf-8")
            architecture = (workspace / "ARCHITECTURE.md").read_text(encoding="utf-8")
            transcript = (workspace / ".agent_state" / "conversation.jsonl").read_text(encoding="utf-8")
            self.assertIn(url, research_doc)
            self.assertIn(url, architecture)
            self.assertIn("CITATION_MARKER_ALPHA", architecture)
            self.assertIn("WEB_RESEARCH_TOOL_RESULT", transcript)
            self.assertIn("WEB_RESEARCH_USAGE_REQUIREMENT", transcript)
            first_step_review = summary["steps"][0]["attempts"][-1]["review"]
            self.assertEqual(first_step_review["deterministic_evidence_findings"], [])

    def test_research_usage_gate_rejects_ignored_research(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            config_path = write_config(root, workspace, "ignored research", "Research this source: http://example.test/research")
            cfg = load_config(config_path, repo_root=root)
            agent = FeedbackLoopAgent(cfg, mock=True)
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
            agent.requirements = {
                "project_summary": "Research must be used",
                "refined_requirements": ["Generated architecture must cite researched source URLs."],
            }
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

    def test_git_commits_each_accepted_plan_step_and_final_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            config_path = write_config(root, workspace, "mock tracker git", "Build and test a task tracker.")
            cfg = load_config(config_path, repo_root=root)
            summary = FeedbackLoopAgent(cfg, mock=True).run()

            self.assertEqual(summary["final_status"], "resolved")
            self.assertTrue((workspace / ".git").exists())
            status = subprocess.run(
                ["git", "-C", str(workspace), "status", "--short"],
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
            log = subprocess.run(
                ["git", "-C", str(workspace), "log", "--oneline", "--reverse"],
                text=True,
                capture_output=True,
                check=True,
            ).stdout

            self.assertEqual(status, "")
            self.assertIn("harness baseline: requirements and validated plan", log)
            self.assertIn("S1: Research required patterns and plan project structure", log)
            self.assertIn("S2: Implement task tracker core", log)
            self.assertIn("S3: Add persistence tests and documentation", log)
            self.assertIn("final review: accepted project state", log)
            self.assertTrue(all("git_commit" in step["attempts"][-1] for step in summary["steps"]))

    def test_git_diff_gate_rejects_no_change_attempt_then_accepts_fix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            config_path = write_config(
                root,
                workspace,
                "mock empty-diff",
                "Build the empty-diff no-change marker-file project.",
            )
            cfg = load_config(config_path, repo_root=root)
            summary = FeedbackLoopAgent(cfg, mock=True).run()

            self.assertEqual(summary["final_status"], "resolved")
            s2 = next(step for step in summary["steps"] if step["step_id"] == "S2")
            self.assertGreaterEqual(len(s2["attempts"]), 2)
            first_review = s2["attempts"][0]["review"]
            self.assertEqual(first_review["status"], "needs_rework")
            self.assertIn("Git working tree has no implementation changes", "\n".join(first_review["required_changes"]))
            self.assertTrue((workspace / "marker.txt").exists())
            self.assertEqual((workspace / "marker.txt").read_text().strip(), "done")

    def test_git_policy_can_leave_final_changes_uncommitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            config_path = write_config(root, workspace, "mock tracker soft reset", "Build and test a task tracker.")
            data = json.loads(config_path.read_text(encoding="utf-8"))
            data["git_policy"]["leave_final_changes_uncommitted"] = True
            config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            cfg = load_config(config_path, repo_root=root)
            summary = FeedbackLoopAgent(cfg, mock=True).run()

            self.assertEqual(summary["final_status"], "resolved")
            self.assertTrue(summary["git"]["finalize"]["left_uncommitted"])
            status = subprocess.run(
                ["git", "-C", str(workspace), "status", "--short"],
                text=True,
                capture_output=True,
                check=True,
            ).stdout
            self.assertIn("ARCHITECTURE.md", status)
            self.assertIn("task_tracker.py", status)


if __name__ == "__main__":
    unittest.main()
