from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fathom.adapters.evidence.history import HistoryEvidenceSource
from fathom.constants.flow import AssertionSource, CheckKind
from fathom.constants.generation import COMPLETION_ASSERTIONS_FILENAME
from fathom.core.exceptions import ScriptExportError
from fathom.core.services.generation.assembler import EvidenceAssembler
from fathom.core.services.generation.classifier import LauncherClassifier
from fathom.core.services.generation.distiller import Distiller
from fathom.core.services.generation.normalizer import RunTraceNormalizer
from fathom.schemas.artifacts import ScreenArtifact, ScreenArtifactBundle, StepArtifacts
from fathom.schemas.flow import CompletionAssertion, RunObjective


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
            (directory / "history__execution.json").write_text(json.dumps(payload))
            source = self.__source(directory=directory)

            evidence = await source.read(execution_id="run-1", objective=self.__objective())

            self.assertEqual(len(evidence.steps), 2)
            self.assertEqual(evidence.package, "com.example")
            self.assertEqual(evidence.intent, "open and verify")

            self.assertEqual(evidence.steps[0].action, "tap")
            self.assertEqual(evidence.steps[1].guard.condition, "Overlay is visible")

    async def test_preserves_step_artifacts_from_history_record(self) -> None:
        """
        Step artifacts persisted in history remain attached to the assembled evidence step.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            screenshot = str(directory / "screenshot.png")
            artifacts = StepArtifacts(
                screen=ScreenArtifactBundle(before=ScreenArtifact(uri=screenshot))
            )
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
                        "artifacts": artifacts.model_dump(mode="json"),
                    },
                ],
            }
            (directory / "history__execution.json").write_text(json.dumps(payload))
            source = self.__source(directory=directory)

            evidence = await source.read(execution_id="run-1", objective=self.__objective())

            self.assertIsNotNone(evidence.steps[0].artifacts)
            assert evidence.steps[0].artifacts is not None
            self.assertIsNotNone(evidence.steps[0].artifacts.screen)
            assert evidence.steps[0].artifacts.screen is not None
            self.assertIsNotNone(evidence.steps[0].artifacts.screen.before)
            assert evidence.steps[0].artifacts.screen.before is not None
            self.assertEqual(evidence.steps[0].artifacts.screen.before.uri, screenshot)

    async def test_loads_completion_assertions_as_terminal_evidence(self) -> None:
        """
        Persisted verifier assertions are loaded as completed run evidence.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload = {
                "execution_id": "run-1",
                "history": [
                    {
                        "duration": 5,
                        "target": "Buy Now",
                        "success": True,
                        "timestamp": 123,
                        "step_number": 0,
                        "action_type": "tap",
                        "event_type": "action",
                        "screen_changed": True,
                    },
                ],
            }
            assertion = CompletionAssertion(
                id="terminal.login",
                kind=CheckKind.VISIBLE,
                subject="Phone Number input field",
                source=AssertionSource.VERIFICATION,
            )
            (directory / "history__execution.json").write_text(json.dumps(payload))
            (directory / COMPLETION_ASSERTIONS_FILENAME).write_text(
                json.dumps([assertion.model_dump(mode="json")])
            )
            source = self.__source(directory=directory)

            evidence = await source.read(execution_id="run-1", objective=self.__objective())

            self.assertFalse(evidence.partial)
            self.assertEqual(len(evidence.assertions), 1)
            self.assertEqual(evidence.assertions[0].subject, "Phone Number input field")

    async def test_missing_history_raises(self) -> None:
        """
        A run with no persisted history fails explicitly.
        """

        with TemporaryDirectory() as temporary:
            source = self.__source(directory=Path(temporary))

            with self.assertRaises(ScriptExportError):
                await source.read(execution_id="missing", objective=self.__objective())

    async def test_malformed_history_raises(self) -> None:
        """
        A history file without a valid records list fails fast instead of becoming an empty run.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "history__execution.json").write_text(json.dumps({"history": "nope"}))
            source = self.__source(directory=directory)

            with self.assertRaises(ScriptExportError):
                await source.read(execution_id="run-1", objective=self.__objective())
