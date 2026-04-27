from __future__ import annotations

from collections import defaultdict
import json
import re
import urllib.error
import urllib.request

from .config import ModelConfig


class OpenAICompatClient:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int | None = None, temperature: float | None = None) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": self.cfg.max_tokens if max_tokens is None else max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.cfg.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.api_key}",
            },
            method="POST",
        )
        request_timeout = None if self.cfg.request_timeout_seconds <= 0 else self.cfg.request_timeout_seconds
        try:
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"model request failed: {exc}") from exc
        if body.get("error"):
            raise RuntimeError(f"model returned error: {body['error']}")
        msg = body["choices"][0]["message"]
        return str(msg.get("content") or msg.get("reasoning_content") or "")


class MockClient:
    """Deterministic local stand-in used to test the agent workflow itself."""

    def __init__(self) -> None:
        self.requirements_calls = 0
        self.requirements_review_calls = 0
        self.plan_validation_calls = 0
        self.plan_refinement_calls = 0
        self.step_attempts: defaultdict[str, int] = defaultdict(int)
        self.step_review_attempts: defaultdict[str, int] = defaultdict(int)

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int | None = None, temperature: float | None = None) -> str:
        last = messages[-1]["content"]
        scenario = self._scenario(messages)
        if "REQUIREMENTS_REVIEW_PHASE" in last:
            return self._requirements_review()
        if "REQUIREMENTS_REFINEMENT_PHASE" in last:
            return self._requirements(scenario)
        if "PLAN_VALIDATION_PHASE" in last:
            return self._plan_validation()
        if "PLAN_REFINEMENT_PHASE" in last:
            return self._plan_refinement(scenario)
        if "STEP_REVIEW_PHASE" in last:
            return self._step_review(last)
        if "IMPLEMENT_PLAN_STEP_PHASE" in last:
            return self._implementation(last, scenario)
        if "FINAL_PROJECT_REVIEW_PHASE" in last:
            return self._final_review(last)
        if "FINAL_PROJECT_CORRECTION_PHASE" in last:
            return self._final_correction()
        if "Compact this coding-agent conversation" in last:
            return "Mock compacted memory: keep refined requirements, validated plan, and unresolved review items."
        return json.dumps({
            "needs_rework": False,
            "status": "resolved",
            "summary": "Default mock response.",
            "required_changes": [],
        })

    def _scenario(self, messages: list[dict[str, str]]) -> str:
        text = "\n".join(m.get("content", "") for m in messages).lower()
        if "platformer" in text or "mario" in text or "savepoint" in text:
            return "platformer"
        if "empty-diff" in text or "empty diff" in text or "no-change" in text:
            return "emptydiff"
        if "city" in text or "wikipedia" in text:
            return "cities"
        if "static website" in text or "multi-page browser demo" in text or "clicker" in text:
            return "website"
        return "tracker"

    def _planning_confirmation(self, scenario: str) -> dict:
        return {
            "is_feasible": True,
            "is_clear": True,
            "is_verifiable": True,
            "verification_strategy": (
                "Execute each ordered step separately, run its validation command, "
                "inspect the generated files/reports, and reject the step if evidence "
                "does not match its acceptance criteria."
            ),
            "remaining_risks": [
                f"Mock {scenario} validation is deterministic; real-model runs may need extra retries for ambiguous outputs."
            ],
        }

    def _quality_requirement(self) -> str:
        return (
            "Use a maintainable structure with clear modules/files, automated tests, "
            "and concise documentation unless the user explicitly overrides that expectation."
        )

    def _structure_step(self) -> dict:
        return {
            "id": "S1",
            "title": "Research required patterns and plan project structure",
            "description": (
                "Research needed patterns/knowledge, choose the project structure/architecture, "
                "cite any fetched source URLs, and rewrite or confirm the remaining plan order before feature implementation."
            ),
            "depends_on": [],
            "acceptance_criteria": [
                "ARCHITECTURE.md records relevant patterns or knowledge requirements",
                "ARCHITECTURE.md defines the project structure and confirms whether the remaining plan order changed",
                "When web research evidence exists, ARCHITECTURE.md cites and applies the researched source URLs",
            ],
            "validation_commands": [["python", "-c", "from pathlib import Path; text=Path('ARCHITECTURE.md').read_text(); assert 'Structure' in text and 'Plan order' in text; print('architecture plan ok')"]],
        }

    def _requirements(self, scenario: str) -> str:
        self.requirements_calls += 1
        if self.requirements_calls == 1:
            return json.dumps({
                "project_summary": "Initial refinement found the goal but still has gaps around validation and deliverables.",
                "refined_requirements": ["Produce a working deliverable inside the workspace."],
                "open_questions": [
                    {"question": "How should success be verified?", "resolution_strategy": "ask_or_assume", "decision": "Need explicit validation commands."},
                    {"question": "What should be skipped if external access fails?", "resolution_strategy": "defer", "decision": "Need fallback policy."},
                ],
                "planning_confirmation": {
                    "is_feasible": False,
                    "is_clear": False,
                    "is_verifiable": False,
                    "verification_strategy": "",
                    "remaining_risks": ["Plan is intentionally too broad in the first mock pass."],
                },
                "plan": [
                    {"id": "S1", "title": "Build everything", "description": "Too broad initial plan.", "depends_on": [], "acceptance_criteria": ["Something exists"], "validation_commands": []}
                ],
            })
        if scenario == "emptydiff":
            return json.dumps({
                "project_summary": "Build a tiny marker-file deliverable that intentionally starts with a no-change attempt.",
                "refined_requirements": [
                    self._quality_requirement(),
                    "Create marker.txt with the exact text done so the feedback agent can detect real file changes.",
                    "Provide validation that checks the marker file content.",
                ],
                "assumptions": [
                    "The first implementation attempt intentionally makes no project changes to exercise git-diff feedback."
                ],
                "planning_confirmation": self._planning_confirmation(scenario),
                "plan": [
                    self._structure_step(),
                    {
                        "id": "S2",
                        "title": "Create checked marker file",
                        "description": "Write marker.txt and validate its exact content.",
                        "depends_on": ["S1"],
                        "acceptance_criteria": ["marker.txt exists", "marker.txt contains done"],
                        "validation_commands": [[
                            "python",
                            "-c",
                            "from pathlib import Path; assert Path('marker.txt').read_text().strip() == 'done'; print('marker ok')",
                        ]],
                    },
                ],
            })
        if scenario == "cities":
            return json.dumps({
                "project_summary": "Collect a small reproducible city-image manifest, using Wikipedia-style metadata with an offline fixture fallback.",
                "refined_requirements": [
                    self._quality_requirement(),
                    "Choose several globally recognizable large cities without requiring user-supplied city names.",
                    "Produce a manifest describing image source URLs, local filenames, and attribution notes.",
                    "Provide a validation script that checks manifest shape and minimum city count.",
                    "If live web access is unavailable, record that fixture data was used instead of pretending the scrape succeeded.",
                ],
                "assumptions": [
                    "Mock harness uses fixture data so tests are deterministic; real runs may replace it with live Wikipedia/API calls.",
                    "The output is a collection manifest, not a polished gallery website.",
                ],
                "planning_confirmation": self._planning_confirmation(scenario),
                "plan": [
                    self._structure_step(),
                    {"id": "S2", "title": "Create city-image collection manifest", "description": "Select cities and write deterministic manifest/notes.", "depends_on": ["S1"], "acceptance_criteria": ["At least four cities", "Each item has city, country, page, image_url, and local_filename"], "validation_commands": [["python", "scripts/validate_manifest.py"]]},
                    {"id": "S3", "title": "Add reproducible collection script", "description": "Add a script showing the live/fixture collection flow.", "depends_on": ["S2"], "acceptance_criteria": ["Script can run in fixture mode", "README explains live vs fixture mode"], "validation_commands": [["python", "scripts/collect_city_images.py", "--fixture"]]},
                ],
            })
        if scenario == "platformer":
            return json.dumps({
                "project_summary": "Build a deterministic browser platformer with external text levels, data-scripted actors, savepoints, and Playwright/screenshot-friendly validation.",
                "refined_requirements": [
                    self._quality_requirement(),
                    "The player controls a small 2D platform character using keyboard input that Playwright can send deterministically.",
                    "At least one level must be stored in an external text file, not hardcoded only in JavaScript.",
                    "Enemies and NPC friends must be loaded from data so behavior/dialogue is scriptable without changing the engine.",
                    "The game must expose savepoints and visible state markers so a screenshot or VLM review can identify player, friend, enemy, goal, and savepoint state.",
                    "Validation must launch the game in Chromium via Playwright, press controls, save screenshots, and produce a JSON report.",
                ],
                "assumptions": [
                    "This smoke test uses a small DOM/CSS platformer rather than a full canvas engine so screenshots are easy for a VLM to inspect.",
                    "Visual-model analysis is represented by screenshot-friendly markers and a deterministic Playwright report; a real VLM can be plugged into the report later.",
                ],
                "planning_confirmation": self._planning_confirmation(scenario),
                "plan": [
                    self._structure_step(),
                    {"id": "S2", "title": "Scaffold external-level platformer", "description": "Create HTML/CSS/JS skeleton plus an external text level file and first validation script.", "depends_on": ["S1"], "acceptance_criteria": ["Level text file exists", "Player/savepoint/goal are visible with test IDs", "Playwright can load the page and capture a screenshot"], "validation_commands": [["python", "scripts/playwright_game_check.py"]]},
                    {"id": "S3", "title": "Add controllable movement and savepoints", "description": "Implement deterministic keyboard control, collision-ish movement, savepoint activation, and state export for tests.", "depends_on": ["S2"], "acceptance_criteria": ["Arrow keys move the player", "Savepoint can be activated", "Validation report records player movement and savepoint status"], "validation_commands": [["python", "scripts/playwright_game_check.py"]]},
                    {"id": "S4", "title": "Add data-scripted NPC friends and enemies", "description": "Load actor script data and render friend/enemy markers with inspectable behavior labels.", "depends_on": ["S3"], "acceptance_criteria": ["NPC friend and enemy are loaded from JSON", "Screenshot includes actor labels", "Playwright report includes actor count and final screenshot path"], "validation_commands": [["python", "scripts/playwright_game_check.py"]]},
                ],
            })
        if scenario == "website":
            return json.dumps({
                "project_summary": "Build a tiny multi-page browser demo with visible navigation and a simple interaction.",
                "refined_requirements": [
                    self._quality_requirement(),
                    "Create a static website with at least three linked pages.",
                    "Include a small browser-game or interaction that can be validated without a full browser.",
                    "Keep all assets local and easy to inspect.",
                    "Provide a deterministic validation script for structure and interaction hooks.",
                ],
                "assumptions": [
                    "The smoke harness validates DOM/text structure; full Playwright screenshots are optional for real-model runs.",
                    "No external npm dependencies are needed for the demo.",
                ],
                "planning_confirmation": self._planning_confirmation(scenario),
                "plan": [
                    self._structure_step(),
                    {"id": "S2", "title": "Scaffold static site", "description": "Create pages, shared stylesheet, and navigation.", "depends_on": ["S1"], "acceptance_criteria": ["index/about/game pages exist", "Navigation links all pages", "Shared stylesheet is referenced"], "validation_commands": [["python", "scripts/check_structure.py"]]},
                    {"id": "S3", "title": "Add browser interaction", "description": "Add a tiny clicker game and validation hook.", "depends_on": ["S2"], "acceptance_criteria": ["Game button exists", "JavaScript increments score", "Validation script checks required IDs"], "validation_commands": [["python", "scripts/check_site.py"]]},
                ],
            })
        return json.dumps({
            "project_summary": "Build a tiny task tracker with persistence and tests.",
            "refined_requirements": [
                self._quality_requirement(),
                "Support adding, completing, listing, saving, and loading tasks.",
                "Reject blank task titles.",
                "Provide unit tests for JSON round-trip behavior.",
            ],
            "assumptions": ["Command-line polish is secondary to small testable core behavior."],
            "planning_confirmation": self._planning_confirmation(scenario),
            "plan": [
                self._structure_step(),
                {"id": "S2", "title": "Implement task tracker core", "description": "Create module with state operations.", "depends_on": ["S1"], "acceptance_criteria": ["Core methods exist", "Blank titles fail"], "validation_commands": [["python", "-m", "py_compile", "task_tracker.py"]]},
                {"id": "S3", "title": "Add persistence tests and documentation", "description": "Add unittest coverage for save/load and concise README documentation.", "depends_on": ["S2"], "acceptance_criteria": ["Round-trip test passes", "README documents usage"], "validation_commands": [["python", "-m", "unittest", "-v"]]},
            ],
        })

    def _requirements_review(self) -> str:
        self.requirements_review_calls += 1
        if self.requirements_review_calls == 1:
            return json.dumps({
                "status": "needs_rework",
                "needs_rework": True,
                "summary": "Requirements are too broad and do not define validation commands or fallback assumptions.",
                "required_changes": ["Add acceptance criteria", "Add validation commands", "Resolve external-access assumptions"],
            })
        return json.dumps({
            "status": "resolved",
            "needs_rework": False,
            "summary": "Requirements are explicit enough to plan and validate.",
            "required_changes": [],
        })

    def _plan_validation(self) -> str:
        self.plan_validation_calls += 1
        if self.plan_validation_calls == 1:
            return json.dumps({
                "status": "needs_plan_change",
                "needs_rework": True,
                "summary": "The plan must be checked for distinct ordered tasks before implementation; first version is accepted only after refinement.",
                "required_changes": ["Make each step independently verifiable", "Make dependencies explicit"],
            })
        return json.dumps({
            "status": "resolved",
            "needs_rework": False,
            "summary": "Plan has distinct, ordered, verifiable steps.",
            "required_changes": [],
        })

    def _plan_refinement(self, scenario: str) -> str:
        self.plan_refinement_calls += 1
        # Reuse the already clean requirements payload shape as the refined plan source.
        return self._requirements(scenario)

    def _step_id_from_prompt(self, text: str) -> str:
        explicit = re.search(r"step_id=([A-Za-z0-9_.:-]+)", text)
        if explicit:
            return explicit.group(1)
        match = re.search(r'"id"\s*:\s*"([^"]+)"', text)
        return match.group(1) if match else "S1"

    def _is_structure_step(self, prompt: str, step_id: str | None = None) -> bool:
        if step_id is not None:
            return step_id == "S1"
        text = prompt.lower()
        current = text.split("current step:", 1)[-1] if "current step:" in text else text
        return "research required patterns" in current or "plan project structure" in current

    def _structure_implementation(self, scenario: str, prompt: str) -> str:
        source_urls = []
        for url in re.findall(r'"url"\s*:\s*"([^"]+)"', prompt):
            if url.startswith("http") and url not in source_urls:
                source_urls.append(url)
        marker = "CITATION_MARKER_ALPHA" if "CITATION_MARKER_ALPHA" in prompt else "source evidence"
        research_note = ""
        if source_urls:
            research_note = (
                "\n## Web Research Applied\n\n"
                f"- Source URL: {source_urls[0]}\n"
                f"- Applied finding: {marker}; this source is cited here so the feedback agent can verify "
                "that fetched research affected the architecture plan.\n"
            )
        architecture = f"""# Architecture Notes

## Research Notes

- Scenario: {scenario}.
- Use a small, testable design with explicit validation commands.
- Keep generated files simple enough for the feedback agent to inspect.
{research_note}

## Structure

- Source files live at the project root unless a script or data directory is clearer.
- Validation lives in `scripts/` or unittest files.
- Documentation lives in `README.md`.

## Plan order

The remaining plan order is still appropriate after this structure pass.
"""
        command = "from pathlib import Path; text=Path('ARCHITECTURE.md').read_text(); assert 'Structure' in text and 'Plan order' in text; print('architecture plan ok')"
        if source_urls:
            command = f"from pathlib import Path; text=Path('ARCHITECTURE.md').read_text(); assert 'Structure' in text and 'Plan order' in text and {source_urls[0]!r} in text; print('architecture plan ok')"
        return json.dumps({
            "plan_note": "Researched required patterns/knowledge and recorded the project structure before feature work.",
            "files": [{"path": "ARCHITECTURE.md", "content": architecture}],
            "commands": [["python", "-c", command]],
            "test_evidence": ["ARCHITECTURE.md existence and required headings verified by Python command."],
        })

    def _implementation(self, prompt: str, scenario: str) -> str:
        step_id = self._step_id_from_prompt(prompt)
        self.step_attempts[step_id] += 1
        attempt = self.step_attempts[step_id]
        if self._is_structure_step(prompt, step_id):
            return self._structure_implementation(scenario, prompt)
        if scenario == "emptydiff":
            return self._emptydiff_implementation(step_id, attempt)
        if scenario == "cities":
            return self._city_implementation(step_id, attempt)
        if scenario == "platformer":
            return self._platformer_implementation(step_id, attempt)
        if scenario == "website":
            return self._website_implementation(step_id, attempt)
        return self._tracker_implementation(step_id, attempt)

    def _step_review(self, prompt: str) -> str:
        step_id = self._step_id_from_prompt(prompt)
        self.step_review_attempts[step_id] += 1
        attempt = self.step_review_attempts[step_id]
        if self._is_structure_step(prompt, step_id):
            return json.dumps({
                "status": "resolved",
                "needs_rework": False,
                "summary": "Structure/research step has ARCHITECTURE.md evidence and a passing validation command.",
                "required_changes": [],
                "verification_evidence": ["ARCHITECTURE.md validation command passed"],
            })
        if "platformer" in prompt.lower() or "savepoint" in prompt.lower():
            if re.search(r'"returncode"\s*:\s*(?!0\b)\d+', prompt) or re.search(r'"timed_out"\s*:\s*true', prompt, re.IGNORECASE):
                return json.dumps({
                    "status": "needs_rework",
                    "needs_rework": True,
                    "summary": f"Playwright/VL feedback for {step_id}: the validation command failed, so this step cannot be accepted yet. Fix the browser game or validation script and rerun until Chromium produces screenshots and a JSON report.",
                    "required_changes": [
                        "Treat non-zero Playwright validation as a blocking failure",
                        "Fix JavaScript/runtime errors before marking the step resolved",
                        "Keep the screenshot and validation JSON as evidence only after the command exits successfully",
                    ],
                })
            if step_id == "S2" and attempt == 1:
                return json.dumps({
                    "status": "needs_rework",
                    "needs_rework": True,
                    "summary": "Playwright/VL feedback: the first platformer draft is not controllable or visually inspectable enough. The review needs deterministic keyboard controls, visible test IDs, screenshot output, and a validation report before later gameplay steps can be trusted.",
                    "required_changes": [
                        "Add a Playwright validation command that launches Chromium and captures screenshots",
                        "Expose player/savepoint/goal markers with data-testid attributes",
                        "Write a JSON validation report with movement and screenshot evidence",
                    ],
                })
            return json.dumps({
                "status": "resolved",
                "needs_rework": False,
                "summary": f"Playwright/VL feedback for {step_id}: the validation command produced screenshot/report evidence and the game exposes deterministic controls plus visible markers for screenshot review.",
                "required_changes": [],
            })
        if step_id == "S2" and attempt == 1:
            return json.dumps({
                "status": "needs_rework",
                "needs_rework": True,
                "summary": "First step attempt is incomplete; fix the missing validation target before continuing.",
                "required_changes": ["Add the missing file or behavior checked by validation"],
            })
        return json.dumps({
            "status": "resolved",
            "needs_rework": False,
            "summary": f"{step_id} satisfies its acceptance criteria and validation commands passed or were appropriately documented.",
            "required_changes": [],
        })

    def _emptydiff_implementation(self, step_id: str, attempt: int) -> str:
        if step_id == "S2" and attempt == 1:
            return json.dumps({
                "plan_note": "Intentionally produced no project file changes so git-diff review can reject the attempt.",
                "files": [],
                "commands": [],
                "test_evidence": [],
            })
        return json.dumps({
            "plan_note": "Created the checked marker file after git-diff feedback requested a real change.",
            "files": [{"path": "marker.txt", "content": "done\n"}],
            "commands": [[
                "python",
                "-c",
                "from pathlib import Path; assert Path('marker.txt').read_text().strip() == 'done'; print('marker ok')",
            ]],
            "test_evidence": ["marker.txt content validated by Python command."],
        })

    def _tracker_implementation(self, step_id: str, attempt: int) -> str:
        if step_id == "S2" and attempt == 1:
            return json.dumps({
                "plan_note": "Started task tracker core but left persistence for the next pass.",
                "files": [{"path": "task_tracker.py", "content": "class TaskTracker:\n    def __init__(self):\n        self.tasks = []\n\n    def add(self, title):\n        self.tasks.append({'title': title, 'done': False})\n"}],
                "commands": [["python", "-m", "py_compile", "task_tracker.py"]],
            })
        files = []
        if step_id == "S2":
            files.append({"path": "task_tracker.py", "content": "import json\nfrom pathlib import Path\n\nclass TaskTracker:\n    def __init__(self):\n        self.tasks = []\n\n    def add(self, title):\n        if not title.strip():\n            raise ValueError('title is required')\n        self.tasks.append({'title': title, 'done': False})\n\n    def done(self, index):\n        self.tasks[index]['done'] = True\n\n    def list(self):\n        return list(self.tasks)\n\n    def save(self, path):\n        Path(path).write_text(json.dumps(self.tasks, indent=2), encoding='utf-8')\n\n    def load(self, path):\n        self.tasks = json.loads(Path(path).read_text(encoding='utf-8'))\n"})
            commands = [["python", "-m", "py_compile", "task_tracker.py"]]
        else:
            files.append({"path": "test_task_tracker.py", "content": "import tempfile\nimport unittest\nfrom task_tracker import TaskTracker\n\nclass TaskTrackerTests(unittest.TestCase):\n    def test_save_load_roundtrip(self):\n        tracker = TaskTracker()\n        tracker.add('write docs')\n        tracker.done(0)\n        with tempfile.NamedTemporaryFile() as f:\n            tracker.save(f.name)\n            loaded = TaskTracker()\n            loaded.load(f.name)\n        self.assertEqual(loaded.tasks, [{'title': 'write docs', 'done': True}])\n\n    def test_blank_title_rejected(self):\n        with self.assertRaises(ValueError):\n            TaskTracker().add('   ')\n\nif __name__ == '__main__':\n    unittest.main()\n"})
            files.append({"path": "README.md", "content": "# Task Tracker\n\nUse `TaskTracker.add`, `done`, `list`, `save`, and `load` for a tiny JSON-backed task list.\n"})
            commands = [["python", "-m", "unittest", "-v"]]
        return json.dumps({"plan_note": f"Implemented {step_id} for task tracker.", "files": files, "commands": commands})

    def _website_implementation(self, step_id: str, attempt: int) -> str:
        structure_checker = """from pathlib import Path\nroot = Path('.')\nfor name in ['index.html', 'about.html', 'game.html', 'style.css']:\n    assert (root / name).exists(), f'missing {name}'\nindex = (root / 'index.html').read_text()\nabout = (root / 'about.html').read_text()\ngame = (root / 'game.html').read_text()\nfor text in [index, about, game]:\n    assert 'href=\"index.html\"' in text\n    assert 'href=\"about.html\"' in text\n    assert 'href=\"game.html\"' in text\nassert 'stylesheet' in index and 'style.css' in index\nprint('site structure ok')\n"""
        checker = """from pathlib import Path\nroot = Path('.')\nfor name in ['index.html', 'about.html', 'game.html', 'style.css', 'app.js']:\n    assert (root / name).exists(), f'missing {name}'\nindex = (root / 'index.html').read_text()\ngame = (root / 'game.html').read_text()\napp = (root / 'app.js').read_text()\nassert 'href=\"about.html\"' in index\nassert 'href=\"game.html\"' in index\nassert 'id=\"score\"' in game\nassert 'id=\"increment\"' in game\nassert 'addEventListener' in app and 'score' in app\nprint('site structure ok')\n"""
        if step_id == "S2" and attempt == 1:
            return json.dumps({
                "plan_note": "Created first static page draft; validation should catch missing pages.",
                "files": [
                    {"path": "index.html", "content": "<!doctype html><title>Demo</title><link rel=\"stylesheet\" href=\"style.css\"><h1>Demo Site</h1><a href=\"about.html\">About</a><a href=\"game.html\">Game</a>"},
                    {"path": "style.css", "content": "body{font-family:sans-serif;margin:2rem;background:#f6efe3;color:#1f2933}"},
                    {"path": "scripts/check_structure.py", "content": structure_checker},
                ],
                "commands": [["python", "scripts/check_structure.py"]],
            })
        if step_id == "S2":
            files = [
                {"path": "index.html", "content": "<!doctype html><html><head><title>Demo</title><link rel=\"stylesheet\" href=\"style.css\"></head><body><nav><a href=\"index.html\">Home</a><a href=\"about.html\">About</a><a href=\"game.html\">Game</a></nav><main><h1>Demo Site</h1><p>A small generated website.</p></main></body></html>"},
                {"path": "about.html", "content": "<!doctype html><html><head><title>About</title><link rel=\"stylesheet\" href=\"style.css\"></head><body><nav><a href=\"index.html\">Home</a><a href=\"about.html\">About</a><a href=\"game.html\">Game</a></nav><h1>About</h1><p>Built by the phased feedback-loop harness.</p></body></html>"},
                {"path": "game.html", "content": "<!doctype html><html><head><title>Game</title><link rel=\"stylesheet\" href=\"style.css\"></head><body><nav><a href=\"index.html\">Home</a><a href=\"about.html\">About</a><a href=\"game.html\">Game</a></nav><h1>Button Game</h1><p>The interaction arrives in the next plan step.</p></body></html>"},
                {"path": "style.css", "content": "body{font-family:sans-serif;margin:2rem;background:#f6efe3;color:#1f2933}nav{display:flex;gap:1rem;margin-bottom:2rem}button{font-size:1.1rem;padding:.7rem 1rem}"},
                {"path": "scripts/check_structure.py", "content": structure_checker},
            ]
            return json.dumps({"plan_note": "Scaffolded static pages, navigation, and shared stylesheet.", "files": files, "commands": [["python", "scripts/check_structure.py"]]})
        files = [
            {"path": "game.html", "content": "<!doctype html><html><head><title>Game</title><link rel=\"stylesheet\" href=\"style.css\"></head><body><nav><a href=\"index.html\">Home</a><a href=\"about.html\">About</a><a href=\"game.html\">Game</a></nav><h1>Button Game</h1><p>Score: <span id=\"score\">0</span></p><button id=\"increment\">Add point</button><script src=\"app.js\"></script></body></html>"},
            {"path": "app.js", "content": "const score = document.getElementById('score');\nconst button = document.getElementById('increment');\nif (button && score) { button.addEventListener('click', () => { score.textContent = String(Number(score.textContent) + 1); }); }\n"},
            {"path": "scripts/check_site.py", "content": checker},
        ]
        return json.dumps({"plan_note": "Added the clicker interaction and validation hook.", "files": files, "commands": [["python", "scripts/check_site.py"]]})

    def _platformer_implementation(self, step_id: str, attempt: int) -> str:
        level = """############################
#..........................#
#..P....S........E....N..F.#
#............####..........#
############################
"""
        actors = json.dumps({
            "actors": [
                {"id": "enemy-0", "kind": "enemy", "marker": "E", "script": "patrol:left-right", "label": "enemy: patrol"},
                {"id": "npc-friend", "kind": "friend", "marker": "N", "script": "wave:helpful", "label": "friend: checkpoint hint", "dialogue": "Save at the blue flag, then reach the star."},
            ]
        }, indent=2)
        html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Text Level Platformer</title>
  <link rel="icon" href="data:,">
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <main>
    <h1>Text Level Platformer</h1>
    <p id="instructions">Controls: Arrow keys move, Space jumps, S activates the savepoint.</p>
    <section id="hud" data-testid="hud">Loading...</section>
    <section id="game" data-testid="game" aria-label="platformer game board"></section>
    <pre id="vl-notes" data-testid="vl-notes">Visual markers: blue player, cyan savepoint, red enemy, green friend, gold goal.</pre>
  </main>
  <script src="game.js"></script>
</body>
</html>
"""
        css = """body { margin: 0; font-family: sans-serif; background: #101827; color: #eef5ff; }
main { width: min(940px, calc(100vw - 24px)); margin: 0 auto; padding: 18px; box-sizing: border-box; }
#game { position: relative; width: 896px; height: 160px; overflow: hidden; border: 4px solid #d7e7ff; background: linear-gradient(#193b68, #172033); box-shadow: 0 16px 40px #0008; }
.tile { position: absolute; width: 32px; height: 32px; box-sizing: border-box; }
.wall { background: #6b4f32; border: 1px solid #a77a47; }
.entity { position: absolute; width: 28px; height: 28px; border: 2px solid #fff; display: grid; place-items: center; font-size: 11px; font-weight: 700; box-sizing: border-box; }
#player { background: #2f80ed; border-radius: 8px 8px 4px 4px; z-index: 5; }
.savepoint { background: #00c2ff; border-radius: 50% 50% 4px 4px; }
.savepoint.active { background: #7cffd4; box-shadow: 0 0 16px #7cffd4; }
.goal { background: #ffd166; color: #251a00; clip-path: polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%); }
.enemy { background: #ef476f; border-radius: 50%; }
.friend { background: #06d6a0; border-radius: 40% 40% 30% 30%; }
.label { position: absolute; top: -18px; left: -8px; white-space: nowrap; color: #fff; background: #000a; padding: 1px 4px; border-radius: 4px; font-size: 10px; }
#hud { margin: 8px 0; padding: 8px 10px; background: #0008; border: 1px solid #5d7290; }
#vl-notes { background: #0b1220; color: #b5f7d3; padding: 8px; }
"""
        game_js = """const TILE = 32;
const board = document.getElementById('game');
const hud = document.getElementById('hud');
const state = { player: { x: 0, y: 0 }, walls: [], savepoints: [], goal: null, actors: [], savepointActivated: false, lastMessage: 'loading' };

function cellToPx(col, row) { return { x: col * TILE + 2, y: row * TILE + 2 }; }
function place(el, x, y) { el.style.left = `${x}px`; el.style.top = `${y}px`; }
function entity(kind, id, x, y, label) {
  const el = document.createElement('div');
  el.className = `entity ${kind}`;
  el.id = id;
  el.dataset.testid = id;
  el.dataset.kind = kind;
  el.textContent = kind === 'friend' ? 'NPC' : kind === 'enemy' ? 'BAD' : kind === 'goal' ? 'GO' : 'S';
  const tag = document.createElement('span');
  tag.className = 'label';
  tag.textContent = label || id;
  el.appendChild(tag);
  place(el, x, y);
  board.appendChild(el);
  return el;
}
function updateHud() {
  hud.textContent = `player=(${Math.round(state.player.x)},${Math.round(state.player.y)}) savepoint=${state.savepointActivated ? 'active' : 'inactive'} actors=${state.actors.length} message=${state.lastMessage}`;
}
function updatePlayer() {
  const el = document.getElementById('player');
  if (el) place(el, state.player.x, state.player.y);
  updateHud();
}
function nearestSavepoint() {
  return state.savepoints.find(sp => Math.abs(sp.x - state.player.x) < 180 && Math.abs(sp.y - state.player.y) < 80) || state.savepoints[0];
}
function activateSavepoint() {
  const sp = nearestSavepoint();
  if (!sp) return;
  state.savepointActivated = true;
  state.lastMessage = `savepoint:${sp.id}`;
  document.getElementById(sp.id)?.classList.add('active');
  updateHud();
}
function move(dx, dy) {
  state.player.x = Math.max(34, Math.min(820, state.player.x + dx));
  state.player.y = Math.max(34, Math.min(112, state.player.y + dy));
  updatePlayer();
}

document.addEventListener('keydown', event => {
  if (event.code === 'ArrowRight') move(24, 0);
  if (event.code === 'ArrowLeft') move(-24, 0);
  if (event.code === 'Space') { move(0, -32); setTimeout(() => move(0, 32), 120); }
  if (event.code === 'KeyS') activateSavepoint();
});

window.__gameState = () => ({
  player: { ...state.player },
  savepointActivated: state.savepointActivated,
  savepoints: state.savepoints.map(sp => ({ id: sp.id, x: sp.x, y: sp.y })),
  actors: {
    total: state.actors.length,
    friend: state.actors.filter(a => a.kind === 'friend').length,
    enemy: state.actors.filter(a => a.kind === 'enemy').length,
    labels: state.actors.map(a => a.label),
  },
  message: state.lastMessage,
});

async function loadGame() {
  const [levelText, actorData] = await Promise.all([
    fetch('levels/level1.txt').then(r => r.text()),
    fetch('actors/level1.json').then(r => r.json()),
  ]);
  const scripts = new Map(actorData.actors.map(actor => [actor.marker, actor]));
  levelText.trim().split('\\n').forEach((line, row) => {
    [...line].forEach((char, col) => {
      const { x, y } = cellToPx(col, row);
      if (char === '#') {
        const tile = document.createElement('div');
        tile.className = 'tile wall';
        tile.dataset.testid = `wall-${col}-${row}`;
        place(tile, x - 2, y - 2);
        board.appendChild(tile);
        state.walls.push({ x, y });
      }
      if (char === 'P') {
        state.player = { x, y };
        entity('player', 'player', x, y, 'player');
      }
      if (char === 'S') {
        const id = `savepoint-${state.savepoints.length}`;
        entity('savepoint', id, x, y, 'savepoint');
        state.savepoints.push({ id, x, y });
      }
      if (char === 'F') {
        state.goal = { x, y };
        entity('goal', 'goal', x, y, 'goal');
      }
      if (scripts.has(char)) {
        const script = scripts.get(char);
        entity(script.kind, script.id, x, y, script.label);
        state.actors.push({ ...script, x, y });
      }
    });
  });
  state.lastMessage = 'ready';
  document.body.dataset.ready = 'true';
  updatePlayer();
}
loadGame().catch(error => { hud.textContent = `load failed: ${error}`; throw error; });
"""
        validator = """from __future__ import annotations

import functools
import http.server
import json
from pathlib import Path
import socketserver
import threading

from playwright.sync_api import sync_playwright

root = Path('.').resolve()
out_dir = root / 'out' / 'playwright'
out_dir.mkdir(parents=True, exist_ok=True)

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

with ReusableTCPServer(('127.0.0.1', 0), handler) as httpd:
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    with sync_playwright() as p:
        page_errors = []
        browser = p.chromium.launch(
            executable_path='/usr/bin/chromium',
            headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu'],
        )
        page = browser.new_page(viewport={'width': 960, 'height': 540})
        page.on('pageerror', lambda exc: page_errors.append(str(exc)))
        page.on(
            'console',
            lambda msg: page_errors.append(f'{msg.type}: {msg.text}')
            if msg.type in {'error', 'warning'} and 'favicon' not in msg.text.lower()
            else None,
        )
        page.goto(f'http://127.0.0.1:{port}/index.html')
        page.wait_for_function("document.body.dataset.ready === 'true'", timeout=10000)
        if page_errors:
            raise AssertionError({'browser_errors': page_errors})
        page.wait_for_selector('[data-testid="player"]', timeout=10000)
        page.screenshot(path=str(out_dir / 'platformer_initial.png'))
        initial = page.evaluate('window.__gameState()')
        page.locator('body').click()
        for key in ['ArrowRight'] * 14 + ['Space'] + ['ArrowRight'] * 8 + ['KeyS']:
            page.keyboard.press(key)
            page.wait_for_timeout(35)
        final = page.evaluate('window.__gameState()')
        page.screenshot(path=str(out_dir / 'platformer_final.png'))
        browser.close()
    httpd.shutdown()

report = {
    'url': f'http://127.0.0.1:{port}/index.html',
    'initial': initial,
    'final': final,
    'player_moved_right': final['player']['x'] > initial['player']['x'] + 120,
    'savepoint_activated': final['savepointActivated'],
    'has_friend': final['actors']['friend'] >= 1,
    'has_enemy': final['actors']['enemy'] >= 1,
    'screenshot_initial': str(out_dir / 'platformer_initial.png'),
    'screenshot_final': str(out_dir / 'platformer_final.png'),
    'vl_review_hint': 'Screenshot should show blue player, cyan/green savepoint, red enemy, green NPC friend, gold goal, and HUD text.',
}
(root / 'out' / 'playwright' / 'platformer_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
assert report['player_moved_right'], report
assert report['savepoint_activated'], report
assert report['has_friend'] and report['has_enemy'], report
print(json.dumps(report, indent=2))
"""
        readme = """# Text Level Platformer

A small deterministic browser platformer generated by the phased feedback-loop harness.

- `levels/level1.txt` defines the map.
- `actors/level1.json` defines scriptable enemy/NPC friend behavior labels.
- `scripts/playwright_game_check.py` launches Chromium, presses controls, captures screenshots, and writes a validation report.

Controls: Arrow keys move, Space jumps, S activates a savepoint.
"""
        if step_id == "S2" and attempt == 1:
            return json.dumps({
                "plan_note": "Created an intentionally incomplete platformer draft so feedback can require browser/screenshot validation.",
                "files": [
                    {"path": "index.html", "content": "<!doctype html><title>Draft Platformer</title><h1>Draft</h1><p>No controls yet.</p>"},
                    {"path": "levels/level1.txt", "content": level},
                ],
                "commands": [["python", "-c", "from pathlib import Path; assert Path('levels/level1.txt').exists(); print('draft level exists')"]],
            })
        files = [
            {"path": "index.html", "content": html},
            {"path": "style.css", "content": css},
            {"path": "game.js", "content": game_js},
            {"path": "levels/level1.txt", "content": level},
            {"path": "actors/level1.json", "content": actors},
            {"path": "scripts/playwright_game_check.py", "content": validator},
            {"path": "README.md", "content": readme},
        ]
        if step_id == "S3":
            files.append({
                "path": "evidence/movement_savepoints.md",
                "content": "# Movement And Savepoint Evidence\n\nS3 adds/validates deterministic keyboard movement, savepoint activation, and exported state for Playwright checks.\n",
            })
        if step_id == "S4":
            files.append({
                "path": "evidence/scripted_actors.md",
                "content": "# Scripted Actor Evidence\n\nS4 confirms NPC friend and enemy actor data are loaded from `actors/level1.json` and exposed in the Playwright validation report.\n",
            })
        return json.dumps({
            "plan_note": f"Implemented {step_id} platformer deliverable with Playwright screenshot validation.",
            "files": files,
            "commands": [["python", "scripts/playwright_game_check.py"]],
        })

    def _city_implementation(self, step_id: str, attempt: int) -> str:
        manifest = json.dumps({
            "mode": "fixture",
            "cities": [
                {"city": "Tokyo", "country": "Japan", "page": "Tokyo", "image_url": "https://upload.wikimedia.org/wikipedia/commons/1/1b/Tokyo_Montage_2015.jpg", "local_filename": "tokyo.jpg"},
                {"city": "New York City", "country": "United States", "page": "New York City", "image_url": "https://upload.wikimedia.org/wikipedia/commons/4/47/New_york_times_square-terabass.jpg", "local_filename": "new_york_city.jpg"},
                {"city": "London", "country": "United Kingdom", "page": "London", "image_url": "https://upload.wikimedia.org/wikipedia/commons/c/cd/London_Montage_L.jpg", "local_filename": "london.jpg"},
                {"city": "Shanghai", "country": "China", "page": "Shanghai", "image_url": "https://upload.wikimedia.org/wikipedia/commons/5/5c/Shanghai_montage.png", "local_filename": "shanghai.png"},
            ]
        }, indent=2)
        validator = """import json\nfrom pathlib import Path\ndata = json.loads(Path('city_image_manifest.json').read_text())\nassert len(data['cities']) >= 4\nfor item in data['cities']:\n    for key in ['city', 'country', 'page', 'image_url', 'local_filename']:\n        assert item.get(key), f'missing {key}'\nprint('manifest ok')\n"""
        collector = """import argparse\nfrom pathlib import Path\n\nparser = argparse.ArgumentParser()\nparser.add_argument('--fixture', action='store_true')\nargs = parser.parse_args()\nif not args.fixture:\n    raise SystemExit('Live download intentionally disabled in this deterministic smoke test; rerun with --fixture or extend with Wikipedia API calls.')\nPath('collection_status.txt').write_text('fixture collection completed\\n', encoding='utf-8')\nprint('fixture collection completed')\n"""
        if step_id == "S2" and attempt == 1:
            return json.dumps({
                "plan_note": "Created initial manifest but intentionally left it too small for review to catch.",
                "files": [
                    {"path": "city_image_manifest.json", "content": json.dumps({"mode": "fixture", "cities": []}, indent=2)},
                    {"path": "scripts/validate_manifest.py", "content": validator},
                ],
                "commands": [["python", "scripts/validate_manifest.py"]],
            })
        files = [
            {"path": "city_image_manifest.json", "content": manifest},
            {"path": "scripts/validate_manifest.py", "content": validator},
            {"path": "scripts/collect_city_images.py", "content": collector},
            {"path": "README.md", "content": "# City Image Collection\n\nThis deterministic smoke run records Wikipedia image candidates for several large cities. Live download is intentionally disabled in mock mode; use the fixture command for repeatable validation.\n"},
        ]
        commands = [["python", "scripts/validate_manifest.py"]] if step_id == "S2" else [["python", "scripts/collect_city_images.py", "--fixture"]]
        return json.dumps({"plan_note": f"Implemented {step_id} city collection deliverable.", "files": files, "commands": commands})

    def _final_review(self, prompt: str) -> str:
        if re.search(r'"deterministic_evidence_findings"\s*:\s*\[\s*\]', prompt):
            return json.dumps({
                "status": "resolved",
                "needs_rework": False,
                "summary": "Whole-project review found no deterministic evidence gaps; requirements, files, and validation evidence are consistent.",
                "required_changes": [],
                "verification_evidence": ["All step command evidence was checked in the final review payload."],
            })
        return json.dumps({
            "status": "needs_rework",
            "needs_rework": True,
            "summary": "Whole-project review found unresolved deterministic evidence gaps.",
            "required_changes": ["Fix the evidence gaps listed in deterministic_evidence_findings."],
            "verification_evidence": [],
        })

    def _final_correction(self) -> str:
        return json.dumps({
            "plan_note": "Final correction pass recorded evidence limitations; mock scenario normally resolves before this.",
            "files": [],
            "commands": [],
            "test_evidence": [],
        })
