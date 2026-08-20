from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, List, Optional
from unittest import mock

from fathom.adapters.dialect.drizz.factory import DrizzDialectFactory
from fathom.adapters.script.refresher import BaselineRefresher
from fathom.constants.flow import LaunchProvenance
from fathom.constants.generation import (
    BASELINE_METADATA_FILENAME,
    BASELINE_SCRIPT_FILENAME,
    ScriptArtifactScope,
    ScriptSource,
    ScriptStatus,
    SkipReason,
)
from fathom.core.dialect.policy import Policy
from fathom.core.services.generation.baseline import BaselineScriptService
from fathom.core.services.generation.projector import DeterministicFlowGenerator
from fathom.interfaces.evidence import EvidenceSource
from fathom.schemas.flow import (
    Evidence,
    EvidenceStep,
    RunObjective,
    StepLaunch,
    StepOutcome,
    StepTarget,
)

PACKAGE: str = "com.example.delivery"


class _RecordingSource(EvidenceSource):
    """
    Evidence source that records every read and either returns canned evidence or raises.
    """

    def __init__(
        self, *, evidence: Optional[Evidence] = None, error: Optional[Exception] = None
    ) -> None:
        self.__error = error
        self.__evidence = evidence
        self.reads: List[str] = []

    async def read(self, *, execution_id: str, objective: RunObjective) -> Evidence:
        _ = objective

        self.reads.append(execution_id)
        if self.__error is not None:
            raise self.__error

        assert self.__evidence is not None
        return self.__evidence


class _StubPaths:
    """
    History path resolver that returns a single fixed directory for every run.
    """

    def __init__(self, *, directory: Path) -> None:
        self.__directory = directory

    def get_history_directory(self, *, session_id: str) -> Path:
        _ = session_id

        return self.__directory


class _FailingPaths:
    """
    History path resolver that raises to exercise best-effort persistence.
    """

    def get_history_directory(self, *, session_id: str) -> Path:
        """
        Raise instead of resolving a directory.
        """

        _ = session_id
        raise RuntimeError("history directory unavailable")


class BaselineRefresherTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the background refresher: atomic persistence, exception safety, and burst coalescing.
    """

    @staticmethod
    def __evidence() -> Evidence:
        """
        Build a clean three-step run that projects to a renderable baseline.
        """

        return Evidence(
            intent="order a burger",
            goal="burger added to cart",
            package=PACKAGE,
            steps=(
                EvidenceStep(
                    index=0,
                    event="launch",
                    action="launch",
                    launch=StepLaunch(
                        package=PACKAGE,
                        source_steps=(0,),
                        provenance=LaunchProvenance.LAUNCHER_TRANSITION,
                    ),
                ),
                EvidenceStep(
                    index=1,
                    action="tap",
                    event="action",
                    outcome=StepOutcome(success=True),
                    target=StepTarget(export="Search box"),
                ),
                EvidenceStep(
                    index=2,
                    action="complete",
                    event="validation",
                    outcome=StepOutcome(success=True),
                    target=StepTarget(export="Cart screen"),
                ),
            ),
        )

    def __refresher(self, *, source: EvidenceSource, directory: Path) -> BaselineRefresher:
        """
        Wire a refresher against the real baseline pipeline and a fixed history directory.
        """

        return BaselineRefresher(
            source=source,
            baseline=BaselineScriptService(
                policy=Policy(),
                generator=DeterministicFlowGenerator(),
                dialect=DrizzDialectFactory().create(),
            ),
            path_manager=_StubPaths(directory=directory),
        )

    @staticmethod
    def __objective() -> RunObjective:
        return RunObjective(intent="order a burger", package=PACKAGE)

    @staticmethod
    def __scoped(*, directory: Path, filename: str) -> Path:
        stem, _, ext = filename.rpartition(".")
        return directory / f"{stem}__{ScriptArtifactScope.EXECUTION.value}.{ext}"

    async def test_schedule_then_drain_writes_generated_artifact(self) -> None:
        """
        A scheduled refresh persists both the script and a generated metadata sidecar, leaving no temp file.
        """

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            refresher = self.__refresher(
                source=_RecordingSource(evidence=self.__evidence()), directory=directory
            )

            refresher.schedule(execution_id="run-1", objective=self.__objective())
            await refresher.drain()

            script = self.__scoped(directory=directory, filename=BASELINE_SCRIPT_FILENAME)
            metadata = self.__scoped(directory=directory, filename=BASELINE_METADATA_FILENAME)

            self.assertTrue(script.exists())
            self.assertTrue(script.read_text().startswith("OPEN_APP"))

            payload = json.loads(metadata.read_text())
            self.assertEqual(payload["status"], ScriptStatus.GENERATED.value)
            self.assertEqual(payload["source"], ScriptSource.BASELINE.value)

            self.assertEqual(list(directory.glob("*.tmp")), [])

    async def test_source_failure_writes_failed_metadata_without_raising(self) -> None:
        """
        An exploding evidence source yields a failed sidecar, no script text, and never raises into the caller.
        """

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            refresher = self.__refresher(
                directory=directory,
                source=_RecordingSource(error=RuntimeError("evidence read blew up")),
            )

            refresher.schedule(execution_id="run-1", objective=self.__objective())
            await refresher.drain()

            script = self.__scoped(directory=directory, filename=BASELINE_SCRIPT_FILENAME)
            metadata = self.__scoped(directory=directory, filename=BASELINE_METADATA_FILENAME)

            self.assertFalse(script.exists())
            payload = json.loads(metadata.read_text())

            self.assertTrue(payload["issues"])
            self.assertEqual(payload["status"], ScriptStatus.FAILED.value)

    async def test_failed_refresh_removes_stale_generated_script(self) -> None:
        """
        A failed refresh removes stale baseline text so FAILED metadata cannot sit beside old script.
        """

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            script = self.__scoped(directory=directory, filename=BASELINE_SCRIPT_FILENAME)
            metadata = self.__scoped(directory=directory, filename=BASELINE_METADATA_FILENAME)

            generated = self.__refresher(
                directory=directory,
                source=_RecordingSource(evidence=self.__evidence()),
            )
            generated.schedule(execution_id="run-1", objective=self.__objective())
            await generated.drain()
            self.assertTrue(script.exists())

            failed = self.__refresher(
                directory=directory,
                source=_RecordingSource(error=RuntimeError("evidence read blew up")),
            )
            failed.schedule(execution_id="run-1", objective=self.__objective())
            await failed.drain()

            self.assertFalse(script.exists())
            payload = json.loads(metadata.read_text())
            self.assertEqual(payload["status"], ScriptStatus.FAILED.value)

    async def test_persist_path_failure_is_logged_without_raising(self) -> None:
        """
        A path-resolution failure is caught by the best-effort persistence boundary.
        """

        refresher = BaselineRefresher(
            source=_RecordingSource(evidence=self.__evidence()),
            baseline=BaselineScriptService(
                policy=Policy(),
                generator=DeterministicFlowGenerator(),
                dialect=DrizzDialectFactory().create(),
            ),
            path_manager=_FailingPaths(),
        )

        with self.assertLogs(BaselineRefresher.__module__, level="WARNING") as captured:
            refresher.schedule(execution_id="run-1", objective=self.__objective())
            await refresher.drain()

        self.assertIn("script.baseline.refresh.persist_failed", self.__events(captured.records))

    async def test_synchronous_burst_coalesces_to_single_refresh(self) -> None:
        """
        Many schedules issued before the loop yields collapse into one trailing, latest-wins refresh.
        """

        with tempfile.TemporaryDirectory() as raw:
            source = _RecordingSource(evidence=self.__evidence())
            refresher = self.__refresher(source=source, directory=Path(raw))

            for _ in range(5):
                refresher.schedule(execution_id="run-1", objective=self.__objective())

            await refresher.drain()

            self.assertEqual(source.reads, ["run-1"])

    async def test_drain_recovers_refresh_stranded_without_a_loop(self) -> None:
        """
        A refresh scheduled while no loop could accept the task is still run by the next drain.
        """

        with tempfile.TemporaryDirectory() as raw:
            source = _RecordingSource(evidence=self.__evidence())
            refresher = self.__refresher(source=source, directory=Path(raw))

            def __reject(coroutine: object) -> None:
                coroutine.close()  # type: ignore[attr-defined]
                raise RuntimeError("no running event loop")

            with mock.patch(
                "fathom.adapters.script.refresher.asyncio.create_task", side_effect=__reject
            ):
                refresher.schedule(execution_id="run-1", objective=self.__objective())

            self.assertEqual(source.reads, [])

            await refresher.drain()

            self.assertEqual(source.reads, ["run-1"])

    async def test_schedule_after_drain_runs_again(self) -> None:
        """
        A schedule issued after a completed refresh starts a fresh worker rather than staying idle.
        """

        with tempfile.TemporaryDirectory() as raw:
            source = _RecordingSource(evidence=self.__evidence())
            refresher = self.__refresher(source=source, directory=Path(raw))

            refresher.schedule(execution_id="run-1", objective=self.__objective())
            await refresher.drain()

            refresher.schedule(execution_id="run-2", objective=self.__objective())
            await refresher.drain()

            self.assertEqual(source.reads, ["run-1", "run-2"])

    @staticmethod
    def __evidence_with_dropped_step() -> Evidence:
        """
        Build a run whose failed tap the projector must drop, leaving a generated baseline.
        """

        return Evidence(
            intent="order a burger",
            goal="burger added to cart",
            package=PACKAGE,
            steps=(
                EvidenceStep(
                    index=0,
                    event="launch",
                    action="launch",
                    launch=StepLaunch(
                        package=PACKAGE,
                        source_steps=(0,),
                        provenance=LaunchProvenance.LAUNCHER_TRANSITION,
                    ),
                ),
                EvidenceStep(
                    index=1,
                    action="tap",
                    event="action",
                    outcome=StepOutcome(success=False),
                    target=StepTarget(export="Search box"),
                ),
                EvidenceStep(
                    index=2,
                    event="validation",
                    action="complete",
                    outcome=StepOutcome(success=True),
                    target=StepTarget(export="Cart screen"),
                ),
            ),
        )

    @staticmethod
    def __events(captured: List[Any]) -> List[Any]:
        """
        Extract the structured event identifiers from captured log records.
        """

        return [getattr(record, "event", None) for record in captured]

    async def test_source_failure_logs_failed_and_persisted(self) -> None:
        """
        A source exception logs the refresh failure and the failed-metadata persistence.
        """

        with tempfile.TemporaryDirectory() as raw:
            refresher = self.__refresher(
                source=_RecordingSource(error=RuntimeError("evidence read blew up")),
                directory=Path(raw),
            )

            with self.assertLogs(BaselineRefresher.__module__, level="INFO") as captured:
                refresher.schedule(execution_id="run-1", objective=self.__objective())
                await refresher.drain()

            events = self.__events(captured.records)
            self.assertIn("script.baseline.refresh.failed", events)
            self.assertIn("script.baseline.refresh.persisted", events)

    async def test_dropped_steps_log_skipped_reasons(self) -> None:
        """
        A generated baseline logs the projector's dropped-step reasons grouped by reason.
        """

        with tempfile.TemporaryDirectory() as raw:
            refresher = self.__refresher(
                source=_RecordingSource(evidence=self.__evidence_with_dropped_step()),
                directory=Path(raw),
            )

            with self.assertLogs(BaselineRefresher.__module__, level="INFO") as captured:
                refresher.schedule(execution_id="run-1", objective=self.__objective())
                await refresher.drain()

            generated = next(
                record
                for record in captured.records
                if getattr(record, "event", None) == "script.baseline.refresh.generated"
            )
            self.assertGreaterEqual(generated.__dict__["script.skipped_count"], 1)
            self.assertIn(SkipReason.FAILED.value, generated.__dict__["script.skipped_reasons"])
