from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple

from fathom.constants import ActionType
from fathom.constants.execution import LAUNCHER_PACKAGES
from fathom.constants.flow import AssertionSource, CheckKind, IssueCode
from fathom.constants.generation import (
    BASELINE_METADATA_FILENAME,
    BASELINE_SCRIPT_FILENAME,
    COMPLETION_ASSERTIONS_FILENAME,
    ScriptArtifactMode,
    ScriptArtifactScope,
    ScriptSource,
    ScriptStatus,
)
from fathom.core.services.history import HistoryService
from fathom.interfaces.script import ScriptRefresher
from fathom.schemas.actions import Action
from fathom.schemas.flow import CompletionAssertion, Issue, RunObjective
from fathom.schemas.generation import ScriptFileMetadata
from fathom.schemas.steps import Step, StepGoal, StepResult


class StubPathManager:
    """
    Resolves every session's history directory to one fixed temporary directory.
    """

    def __init__(self, *, directory: Path) -> None:
        """
        Hold the directory to return.
        """

        self.__directory = directory

    def get_history_directory(self, *, session_id: str) -> Path:
        """
        Return the fixed directory.
        """

        return self.__directory


class RecordingRefresher(ScriptRefresher):
    """
    Test refresher that records schedule calls and drain count.
    """

    def __init__(self) -> None:
        """
        Start with no scheduled refreshes.
        """

        self.scheduled: List[Tuple[str, RunObjective]] = []
        self.drains = 0

    def schedule(self, *, execution_id: str, objective: RunObjective) -> None:
        """
        Record the scheduled execution and objective.
        """

        self.scheduled.append((execution_id, objective))

    async def drain(self) -> None:
        """
        Record that finalization drained pending refreshes.
        """

        self.drains += 1


class HistoryWorkflowTraceTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover the single ordered workflow trace written across packages by HistoryService.
    """

    def __launcher(self) -> str:
        """
        Return a representative launcher package from the canonical set.
        """

        return sorted(LAUNCHER_PACKAGES)[0]

    def __service(
        self,
        *,
        directory: Path,
        refresher: Optional[ScriptRefresher] = None,
        artifact_mode: ScriptArtifactMode = ScriptArtifactMode.NORMAL,
    ) -> HistoryService:
        """
        Build a storage-free HistoryService whose history directory is the given path.
        """

        return HistoryService(
            execution_id="wf-1",
            package_name="com.app.one",
            path_manager=StubPathManager(directory=directory),
            refresher=refresher,
            artifact_mode=artifact_mode,
        )

    def __result(self, *, number: int) -> StepResult:
        """
        Build a minimal successful step result with the given step number.
        """

        return StepResult(
            step=Step(
                action=Action(action_type=ActionType.TAP, rationale="reason"),
                screen_hash="pre",
                step_number=number,
            ),
            success=True,
            pre_hash="pre",
            post_hash="post",
            screen_changed=True,
            duration=0,
        )

    def __trace(self, *, directory: Path) -> List[Dict[str, Any]]:
        """
        Read the workflow trace's ordered records.
        """

        payload: Dict[str, Any] = json.loads((directory / "history__execution.json").read_text())
        records: List[Dict[str, Any]] = payload["history"]
        return records

    async def test_appends_every_step_in_order_across_packages(self) -> None:
        """
        Steps saved under different packages all land in one ordered workflow trace.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            service = self.__service(directory=directory)

            await service.save_step(
                self.__result(number=0),
                package_name="com.shopping.supply",
                execution_activity=self.__launcher(),
            )
            await service.save_step(
                self.__result(number=1),
                package_name="com.shopping.supply",
                execution_activity="com.shopping.supply",
            )
            await service.save_step(
                self.__result(number=2),
                package_name="com.google.android.gms",
                execution_activity="com.google.android.gms",
            )

            self.assertEqual(
                [record["step_number"] for record in self.__trace(directory=directory)], [0, 1, 2]
            )

    async def test_preserves_execution_activity(self) -> None:
        """
        The pre-action execution activity is preserved on the workflow record.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            service = self.__service(directory=directory)

            await service.save_step(
                self.__result(number=0),
                package_name="com.shopping.supply",
                execution_activity=self.__launcher(),
            )

            self.assertEqual(
                self.__trace(directory=directory)[0]["execution_activity"], self.__launcher()
            )

    async def test_persists_goal_context_on_workflow_record(self) -> None:
        """
        Saved intent steps persist compact sub-goal context for script authoring episodes.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            service = self.__service(directory=directory)

            await service.save_step(
                self.__result(number=0),
                package_name="com.shopping.supply",
                execution_activity="com.shopping.supply",
                goal=StepGoal(
                    index=2,
                    description="Check whether customer rating is >= 4.2",
                    directive="validate",
                ),
            )

            goal = self.__trace(directory=directory)[0]["goal"]
            self.assertEqual(goal["index"], 2)
            self.assertEqual(goal["description"], "Check whether customer rating is >= 4.2")
            self.assertEqual(goal["directive"], "validate")

    async def test_corrupt_trace_is_backed_up_not_silently_restarted(self) -> None:
        """
        A corrupt workflow trace is preserved as a backup rather than silently discarded.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "history__execution.json").write_text("{ not valid json")
            service = self.__service(directory=directory)

            await service.save_step(
                self.__result(number=0),
                package_name="com.shopping.supply",
                execution_activity="com.shopping.supply",
            )

            backups = list(directory.glob("history__execution.corrupt.*.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(
                [record["step_number"] for record in self.__trace(directory=directory)], [0]
            )

    async def test_save_step_schedules_baseline_refresh_for_workflow_trace(self) -> None:
        """
        A saved step with an intent schedules one baseline refresh against the workflow package.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            refresher = RecordingRefresher()
            service = self.__service(directory=directory, refresher=refresher)

            await service.save_step(
                self.__result(number=0),
                intent="open and verify",
                package_name="com.shopping.supply",
                execution_activity="com.shopping.supply",
            )

            self.assertEqual(len(refresher.scheduled), 1)
            execution_id, objective = refresher.scheduled[0]
            self.assertEqual(execution_id, "wf-1")
            self.assertEqual(objective.intent, "open and verify")
            self.assertEqual(objective.package, "com.shopping.supply")

    async def test_flush_drains_baseline_refresher(self) -> None:
        """
        Flushing pending history work drains the baseline refresher before finalization reads artifacts.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            refresher = RecordingRefresher()
            service = self.__service(directory=directory, refresher=refresher)

            await service.flush_pending_operations()

            self.assertEqual(refresher.drains, 1)

    async def test_completion_assertions_schedule_fresh_baseline_refresh(self) -> None:
        """
        Terminal assertions mutate script evidence, so they refresh the baseline before finalization.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            refresher = RecordingRefresher()
            service = self.__service(directory=directory, refresher=refresher)
            assertion = CompletionAssertion(
                id="terminal.login",
                kind=CheckKind.VISIBLE,
                subject="Phone Number input field",
                source=AssertionSource.VERIFICATION,
            )

            await service.save_step(
                self.__result(number=0),
                intent="open and verify login screen",
                package_name="com.shopping.supply",
                execution_activity="com.shopping.supply",
            )
            service.save_completion_assertions(assertions=(assertion,))

            self.assertEqual(len(refresher.scheduled), 2)
            self.assertEqual(refresher.scheduled[1][0], "wf-1")
            self.assertEqual(refresher.scheduled[1][1].intent, "open and verify login screen")
            self.assertEqual(refresher.scheduled[1][1].package, "com.shopping.supply")

    async def test_normal_mode_writes_only_workflow_trace_for_step_history(self) -> None:
        """
        Normal artifact mode suppresses per-package JSON/YAML debug histories.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            service = self.__service(directory=directory)

            await service.save_step(
                self.__result(number=0),
                package_name="com.shopping.supply",
                execution_activity="com.shopping.supply",
            )

            self.assertTrue((directory / "history__execution.json").exists())
            self.assertFalse((directory / "history__com.shopping.supply.json").exists())
            self.assertFalse((directory / "history__com.shopping.supply.yaml").exists())

    async def test_debug_mode_writes_per_package_step_history(self) -> None:
        """
        Debug artifact mode preserves per-package JSON/YAML histories.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            service = self.__service(
                directory=directory,
                artifact_mode=ScriptArtifactMode.DEBUG,
            )

            await service.save_step(
                self.__result(number=0),
                package_name="com.shopping.supply",
                execution_activity="com.shopping.supply",
            )

            self.assertTrue((directory / "history__execution.json").exists())
            self.assertTrue((directory / "history__com.shopping.supply.json").exists())
            self.assertTrue((directory / "history__com.shopping.supply.yaml").exists())


class HistoryFinalizationTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover the guarantee-path finalization: quality without exporter, and baseline read/promotion or typed failure.
    """

    PACKAGE = "com.app.one"

    def __service(
        self,
        *,
        directory: Path,
        artifact_mode: ScriptArtifactMode = ScriptArtifactMode.NORMAL,
    ) -> HistoryService:
        """
        Build a generation-free, refresher-free HistoryService over the given directory.
        """

        return HistoryService(
            execution_id="wf-1",
            package_name=self.PACKAGE,
            path_manager=StubPathManager(directory=directory),
            artifact_mode=artifact_mode,
        )

    def __write_baseline(
        self, *, directory: Path, metadata: ScriptFileMetadata, text: Optional[str]
    ) -> None:
        """
        Write an execution-scoped baseline artifact.
        """

        def __scoped(filename: str) -> Path:
            stem, _, ext = filename.rpartition(".")
            return directory / f"{stem}__{ScriptArtifactScope.EXECUTION.value}.{ext}"

        __scoped(BASELINE_METADATA_FILENAME).write_text(metadata.model_dump_json())
        if text is not None:
            __scoped(BASELINE_SCRIPT_FILENAME).write_text(text)

    def test_completion_assertions_are_execution_sidecar_not_scoped_artifact(self) -> None:
        """
        Completion assertions are written at the path read by HistoryEvidenceSource.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            service = self.__service(directory=directory)
            assertion = CompletionAssertion(
                id="terminal.login",
                kind=CheckKind.VISIBLE,
                subject="Phone Number input field",
                source=AssertionSource.VERIFICATION,
            )

            service.save_completion_assertions(assertions=(assertion,))

            self.assertTrue((directory / COMPLETION_ASSERTIONS_FILENAME).exists())
            self.assertFalse((directory / "completion.assertions__execution.json").exists())

    async def test_baseline_outcome_promotes_generated_to_canonical_script(self) -> None:
        """
        A generated baseline is returned, promoted, and cleaned up in normal mode.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            service = self.__service(directory=directory)
            self.__write_baseline(
                directory=directory,
                metadata=ScriptFileMetadata(
                    source=ScriptSource.BASELINE, status=ScriptStatus.GENERATED
                ),
                text="OPEN_APP: com.app.one\nTap on Search box",
            )

            artifact = await service.read_baseline_outcome(step_number=2)

            self.assertIs(artifact.metadata.status, ScriptStatus.GENERATED)
            self.assertEqual(artifact.text, "OPEN_APP: com.app.one\nTap on Search box")
            self.assertEqual(
                (directory / "script__execution.txt").read_text(),
                "OPEN_APP: com.app.one\nTap on Search box",
            )
            metadata = ScriptFileMetadata.model_validate_json(
                (directory / "script.meta__execution.json").read_text()
            )

            self.assertIs(metadata.source, ScriptSource.BASELINE)
            self.assertFalse(
                (directory / f"script.baseline__{ScriptArtifactScope.EXECUTION.value}.txt").exists()
            )
            self.assertFalse(
                (
                    directory / f"script.baseline.meta__{ScriptArtifactScope.EXECUTION.value}.json"
                ).exists()
            )

    async def test_baseline_peek_reads_without_promoting_or_cleaning(self) -> None:
        """
        Authoring may read the baseline scaffold without consuming the finalization handoff.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            service = self.__service(directory=directory)
            self.__write_baseline(
                directory=directory,
                metadata=ScriptFileMetadata(
                    source=ScriptSource.BASELINE, status=ScriptStatus.GENERATED
                ),
                text="OPEN_APP: com.app.one\nTap on Search box",
            )

            artifact = await service.peek_baseline_outcome()

            self.assertIs(artifact.metadata.status, ScriptStatus.GENERATED)
            self.assertEqual(artifact.text, "OPEN_APP: com.app.one\nTap on Search box")
            self.assertFalse((directory / "script__execution.txt").exists())
            self.assertTrue(
                (directory / f"script.baseline__{ScriptArtifactScope.EXECUTION.value}.txt").exists()
            )
            self.assertTrue(
                (
                    directory / f"script.baseline.meta__{ScriptArtifactScope.EXECUTION.value}.json"
                ).exists()
            )

    async def test_generated_baseline_metadata_without_script_text_fails_with_diagnostic(
        self,
    ) -> None:
        """
        GENERATED baseline metadata with no script text is a typed failure, not a diagnostics-free artifact.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            service = self.__service(directory=directory)
            self.__write_baseline(
                directory=directory,
                metadata=ScriptFileMetadata(
                    source=ScriptSource.BASELINE, status=ScriptStatus.GENERATED
                ),
                text=None,
            )

            artifact = await service.read_baseline_outcome(step_number=2)

            self.assertIs(artifact.metadata.status, ScriptStatus.FAILED)
            self.assertEqual(artifact.metadata.issues[0].code, IssueCode.BASELINE_UNAVAILABLE)
            self.assertFalse((directory / "script__execution.txt").exists())

    async def test_generated_baseline_with_unreadable_script_text_fails_with_diagnostic(
        self,
    ) -> None:
        """
        GENERATED baseline metadata with unreadable script text is a typed failure, not an exception.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            service = self.__service(directory=directory)
            self.__write_baseline(
                directory=directory,
                metadata=ScriptFileMetadata(
                    source=ScriptSource.BASELINE, status=ScriptStatus.GENERATED
                ),
                text=None,
            )
            (directory / f"script.baseline__{ScriptArtifactScope.EXECUTION.value}.txt").mkdir()

            with self.assertLogs(HistoryService.__module__, level="INFO") as captured:
                artifact = await service.read_baseline_outcome(step_number=2)

            events = [getattr(record, "event", None) for record in captured.records]
            self.assertIn("script.baseline.read.failed_text", events)
            self.assertIs(artifact.metadata.status, ScriptStatus.FAILED)
            self.assertEqual(artifact.metadata.issues[0].code, IssueCode.BASELINE_UNAVAILABLE)
            self.assertFalse((directory / "script__execution.txt").exists())

    async def test_debug_mode_keeps_baseline_after_promotion(self) -> None:
        """
        Debug artifact mode keeps baseline handoff files after promotion.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            service = self.__service(directory=directory, artifact_mode=ScriptArtifactMode.DEBUG)
            self.__write_baseline(
                directory=directory,
                metadata=ScriptFileMetadata(
                    source=ScriptSource.BASELINE, status=ScriptStatus.GENERATED
                ),
                text="OPEN_APP: com.app.one\nTap on Search box",
            )

            artifact = await service.read_baseline_outcome(step_number=2)

            self.assertIs(artifact.metadata.status, ScriptStatus.GENERATED)
            self.assertTrue(
                (directory / f"script.baseline__{ScriptArtifactScope.EXECUTION.value}.txt").exists()
            )
            self.assertTrue(
                (
                    directory / f"script.baseline.meta__{ScriptArtifactScope.EXECUTION.value}.json"
                ).exists()
            )

    async def test_baseline_outcome_failed_when_artifact_missing(self) -> None:
        """
        A missing baseline yields a FAILED outcome carrying the BASELINE_UNAVAILABLE diagnostic.
        """

        with TemporaryDirectory() as temporary:
            service = self.__service(directory=Path(temporary))

            artifact = await service.read_baseline_outcome(step_number=2)

            self.assertIs(artifact.metadata.status, ScriptStatus.FAILED)
            self.assertIsNone(artifact.text)
            self.assertEqual(artifact.metadata.issues[0].code, IssueCode.BASELINE_UNAVAILABLE)

    async def test_baseline_outcome_ignores_package_scoped_artifact(self) -> None:
        """
        Finalization reads the execution-scoped baseline, never the terminal package artifact.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            service = self.__service(directory=directory)
            metadata = ScriptFileMetadata(
                source=ScriptSource.BASELINE, status=ScriptStatus.GENERATED
            )
            (directory / f"script.baseline.meta__{self.PACKAGE}.json").write_text(
                metadata.model_dump_json()
            )
            (directory / f"script.baseline__{self.PACKAGE}.txt").write_text(
                "OPEN_APP: com.google.android.gms\nTap on Wrong package"
            )

            artifact = await service.read_baseline_outcome(step_number=2)

            self.assertIs(artifact.metadata.status, ScriptStatus.FAILED)
            self.assertEqual(artifact.metadata.issues[0].code, IssueCode.BASELINE_UNAVAILABLE)
            self.assertFalse((directory / "script__execution.txt").exists())

    async def test_baseline_outcome_passes_through_failed_metadata(self) -> None:
        """
        A persisted FAILED baseline is surfaced as-is, never promoted to a canonical script.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            service = self.__service(directory=directory)
            self.__write_baseline(
                directory=directory,
                metadata=ScriptFileMetadata(
                    source=ScriptSource.BASELINE,
                    status=ScriptStatus.FAILED,
                    issues=(Issue(code=IssueCode.EMPTY_SCRIPT, message="no scriptable step"),),
                ),
                text=None,
            )

            artifact = await service.read_baseline_outcome(step_number=2)

            self.assertIs(artifact.metadata.status, ScriptStatus.FAILED)
            self.assertEqual(artifact.metadata.issues[0].code, IssueCode.EMPTY_SCRIPT)
            self.assertFalse((directory / "script__execution.txt").exists())

    async def test_unreadable_baseline_metadata_logs_failed_metadata(self) -> None:
        """
        A corrupt baseline metadata sidecar logs failed_metadata and yields an unavailable failure.
        """

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            service = self.__service(directory=directory)
            (
                directory / f"script.baseline.meta__{ScriptArtifactScope.EXECUTION.value}.json"
            ).write_text("{ not json")

            with self.assertLogs(HistoryService.__module__, level="INFO") as captured:
                artifact = await service.read_baseline_outcome(step_number=2)

            events = [getattr(record, "event", None) for record in captured.records]
            self.assertIn("script.baseline.read.failed_metadata", events)
            self.assertIs(artifact.metadata.status, ScriptStatus.FAILED)
            self.assertEqual(artifact.metadata.issues[0].code, IssueCode.BASELINE_UNAVAILABLE)
