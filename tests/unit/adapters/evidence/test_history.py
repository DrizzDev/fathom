from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fathom.adapters.evidence.history import HistoryEvidenceSource
from fathom.core.exceptions import ScriptExportError
from fathom.core.services.generation.assembler import EvidenceAssembler
from fathom.core.services.generation.classifier import LauncherClassifier
from fathom.core.services.generation.distiller import Distiller
from fathom.core.services.generation.normalizer import RunTraceNormalizer
from fathom.schemas.flow import RunObjective


class StubPathManager:
    """
    Returns a fixed history directory for any session.
    """

    def __init__(self, *, directory: Path) -> None:
        """
        Hold the directory to return.
        """

        self.__directory = directory

    def get_history_directory(self, *, session_id: str) -> Path:
        """
        Return the held directory regardless of session.
        """

        return self.__directory


class HistoryEvidenceSourceTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover reading persisted run history into an evidence aggregate.
    """

    def __objective(self) -> RunObjective:
        """
        Build a representative run objective.
        """

        return RunObjective(intent="open and verify", goal="home visible", package="com.example")

    def __source(self, *, directory: Path) -> HistoryEvidenceSource:
        """
        Build a source wired to the workflow trace under the given directory.
        """

        return HistoryEvidenceSource(
            path_manager=StubPathManager(directory=directory),
            distiller=Distiller(),
            normalizer=RunTraceNormalizer(classifier=LauncherClassifier()),
            assembler=EvidenceAssembler(),
        )

    async def test_reads_history_into_evidence(self) -> None:
        """
        Persisted step records and the objective become a populated evidence aggregate.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload = {
                "workflow_id": "run-1",
                "history": [
                    {
                        "duration": 5,
                        "target": "App",
                        "success": True,
                        "timestamp": 123,
                        "step_number": 0,
                        "action_type": "tap",
                        "event_type": "action",
                        "screen_changed": True,
                        "is_app_launcher": True,
                    },
                    {
                        "duration": 2,
                        "success": True,
                        "step_number": 1,
                        "target": "Login",
                        "action_type": "tap",
                        "event_type": "action",
                        "is_conditional": True,
                        "screen_changed": False,
                        "condition": "Overlay is visible",
                    },
                ],
            }
            (directory / "history__workflow.json").write_text(json.dumps(payload))
            source = self.__source(directory=directory)

            evidence = await source.read(run="run-1", objective=self.__objective())

            self.assertEqual(len(evidence.steps), 2)
            self.assertEqual(evidence.package, "com.example")
            self.assertEqual(evidence.intent, "open and verify")

            self.assertEqual(evidence.steps[0].action, "tap")
            self.assertEqual(evidence.steps[1].guard.condition, "Overlay is visible")

    async def test_missing_history_raises(self) -> None:
        """
        A run with no persisted history fails explicitly.
        """

        with TemporaryDirectory() as temporary:
            source = self.__source(directory=Path(temporary))

            with self.assertRaises(ScriptExportError):
                await source.read(run="missing", objective=self.__objective())

    async def test_malformed_history_raises(self) -> None:
        """
        A history file without a valid records list fails fast instead of becoming an empty run.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "history__workflow.json").write_text(json.dumps({"history": "nope"}))
            source = self.__source(directory=directory)

            with self.assertRaises(ScriptExportError):
                await source.read(run="run-1", objective=self.__objective())
