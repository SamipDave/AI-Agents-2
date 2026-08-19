from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.validate_agents import validate_repository

VALID_METADATA = """\
title: example-agent
description: A small validation fixture
author: contributor
language: python
framework: langgraph
tags: [testing, example]
industry: general
difficulty: beginner
llm: none
entrypoint: agent.py
requirements: requirements.txt
"""


class ValidateAgentsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        (self.root / "agents").mkdir()

    def create_agent(self, name: str = "01-example-agent", metadata: str = VALID_METADATA) -> Path:
        agent = self.root / "agents" / name
        agent.mkdir()
        (agent / "README.md").write_text("# Example Agent\n", encoding="utf-8")
        (agent / ".env.example").write_text("API_KEY=your-key-here\n", encoding="utf-8")
        (agent / "metadata.yaml").write_text(metadata, encoding="utf-8")
        (agent / "agent.py").write_text("print('hello')\n", encoding="utf-8")
        (agent / "requirements.txt").write_text("PyYAML==6.0.2\n", encoding="utf-8")
        return agent

    def test_valid_agent_passes(self) -> None:
        self.create_agent()

        self.assertEqual(validate_repository(self.root), [])

    def test_missing_metadata_field_is_reported_once(self) -> None:
        self.create_agent(metadata=VALID_METADATA.replace("author: contributor\n", ""))

        errors = validate_repository(self.root)

        self.assertEqual(sum("author" in error for error in errors), 1)

    def test_declared_file_cannot_escape_agent_directory(self) -> None:
        metadata = VALID_METADATA.replace("entrypoint: agent.py", "entrypoint: ../outside.py")
        self.create_agent(metadata=metadata)
        (self.root / "agents" / "outside.py").write_text("", encoding="utf-8")

        errors = validate_repository(self.root)

        self.assertTrue(any("must stay inside" in error for error in errors))

    def test_numeric_prefixes_must_be_unique(self) -> None:
        self.create_agent("01-first-agent")
        self.create_agent("001-second-agent")

        errors = validate_repository(self.root)

        self.assertTrue(any("numeric prefix 1 is already used" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
