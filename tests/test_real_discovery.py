"""Load the plugin through Hermes's real discovery under a temp HERMES_HOME and fire the hook the
way the tool dispatcher does. Needs a hermes-agent checkout on PYTHONPATH."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
hermes_plugins = pytest.importorskip("hermes_cli.plugins")


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    (h / "plugins").mkdir(parents=True)
    os.symlink(ROOT, h / "plugins" / "details-summary")
    (h / "config.yaml").write_text("plugins:\n  enabled: [details-summary]\nagent:\n  reasoning_effort: high\n")
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return h


def test_real_discovery_blocks_and_renders_prompt(home, tmp_path):
    manager = hermes_plugins.PluginManager()
    manager.discover_and_load()
    assert "pre_tool_call" in manager._hooks, "hook did not register"
    (tmp_path / "body.md").write_text("Adds a thing.")
    results = manager.invoke_hook(
        "pre_tool_call", tool_name="terminal",
        args={"command": "gh pr create --body-file body.md", "workdir": str(tmp_path)})
    blocks = [r for r in results if isinstance(r, dict) and r.get("action") == "block"]
    assert blocks and "No <details>" in blocks[0]["message"]

    sections = manager.render_system_prompt_sections({"model": "anthropic/claude-opus-5.5"})
    text = next(s.content for s in sections if s.id == "details-summary")
    assert "Claude Opus 5.5 High" in text
