from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Tuple

from fathom.adapters.dialect.drizz.factory import DrizzDialectFactory
from fathom.adapters.evidence.history import HistoryEvidenceSource
from fathom.constants.flow import CheckKind, IssueCode, LaunchProvenance
from fathom.core.dialect.policy import Policy
from fathom.core.services.generation.assembler import EvidenceAssembler
from fathom.core.services.generation.classifier import LauncherClassifier
from fathom.core.services.generation.distiller import Distiller
from fathom.core.services.generation.normalizer import RunTraceNormalizer
from fathom.schemas.flow import (
    Check,
    CheckNode,
    Evidence,
    Flow,
    FlowNode,
    LaunchNode,
    RunObjective,
)


class StubPathManager:
    """
    Resolves every session to one fixed history fixture directory.
    """

    def __init__(self, *, directory: Path) -> None:
        """
        Hold the fixture directory to return.
        """

        self.__directory = directory

    def get_history_directory(self, *, session_id: str) -> Path:
        """
        Return the held directory regardless of session.
        """

        return self.__directory


class WorkflowGenerationRegressionTest(unittest.IsolatedAsyncioTestCase):
    """
    Regression for run 78a8dcbf, which once generated a launcher-only partial script.
    """

    def setUp(self) -> None:
        """
        Build the workflow-scoped evidence source, the policy, and the Drizz dialect.
        """

        workflow_trace = self.__workflow_trace()
        if not workflow_trace.exists():
            self.skipTest(f"History fixture absent (gitignored): {workflow_trace}")

        self.__temporary = TemporaryDirectory()
        self.addCleanup(self.__temporary.cleanup)
        self.__fixture_directory = Path(self.__temporary.name)
        self.__execution_trace().write_text(workflow_trace.read_text())

        self.__source = HistoryEvidenceSource(
            path_manager=StubPathManager(directory=self.__fixture()),
            distiller=Distiller(),
            normalizer=RunTraceNormalizer(classifier=LauncherClassifier()),
            assembler=EvidenceAssembler(),
        )
        self.__policy = Policy()
        self.__dialect = DrizzDialectFactory().create()

    def __fixture(self) -> Path:
        """
        Return the execution-scoped fixture directory.
        """

        return self.__fixture_directory

    @staticmethod
    def __workflow_trace() -> Path:
        """
        Return the committed 78a8dcbf workflow-trace fixture.
        """

        return Path("assets/history/2026-06-23/78a8dcbf/history__workflow.json")

    def __execution_trace(self) -> Path:
        """
        Return the execution-scoped trace path consumed by HistoryEvidenceSource.
        """

        return self.__fixture_directory / "history__execution.json"

    def __objective(self) -> RunObjective:
        """
        Build the recorded run's objective.
        """

        return RunObjective(
            intent="open meesho and verify", goal="cart visible", package="com.meesho.supply"
        )

    async def __evidence(self) -> Evidence:
        """
        Read the run's workflow trace into evidence.
        """

        return await self.__source.read(execution_id="78a8dcbf", objective=self.__objective())

    def __flow(self, *, evidence: Evidence) -> Flow:
        """
        Build the flow the generator must produce: each launch marker plus a grounded terminal.
        """

        launches: Tuple[FlowNode, ...] = tuple(
            LaunchNode(
                package=step.launch.package,
                provenance=step.launch.provenance,
                source_steps=step.launch.source_steps,
            )
            for step in evidence.steps
            if step.launch is not None
        )
        validation = next(step.index for step in evidence.steps if step.event == "validation")
        terminal = CheckNode(
            checks=(Check(kind=CheckKind.VISIBLE, subject="cart"),), source_steps=(validation,)
        )

        return Flow(
            intent=evidence.intent,
            package=evidence.package,
            nodes=(*launches, terminal),
            partial=evidence.partial,
        )

    async def test_run_launches_the_real_app_not_the_launcher(self) -> None:
        """
        The launcher-mediated run yields a single grounded Meesho launch and stays non-partial.
        """

        evidence = await self.__evidence()
        launches = [step.launch for step in evidence.steps if step.launch is not None]

        self.assertFalse(evidence.partial)
        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0].package, "com.meesho.supply")
        self.assertEqual(launches[0].provenance, LaunchProvenance.LAUNCHER_TRANSITION)

    async def test_system_steps_stay_in_flow_in_order(self) -> None:
        """
        The in-flow Google Mobile Services step is preserved in recorded order, not split out.
        """

        evidence = await self.__evidence()
        indices = [step.index for step in evidence.steps if step.launch is None]

        self.assertEqual(indices, [1, 2, 3, 4, 5, 6, 7, 8, 11])

    async def test_rendered_script_opens_meesho_and_passes_policy(self) -> None:
        """
        The flow renders to a Meesho-first script with no launcher launch and no fidelity issues.
        """

        evidence = await self.__evidence()
        flow = self.__flow(evidence=evidence)

        report = self.__policy.evaluate(flow=flow, evidence=evidence)
        launch_codes = {
            IssueCode.MISSING_LAUNCH,
            IssueCode.STRAY_LAUNCH,
            IssueCode.LAUNCH_MISMATCH,
            IssueCode.UNGROUNDED_LAUNCH,
        }
        raised = {issue.code for issue in report.issues}
        script = self.__dialect.renderer.render(flow=flow)

        self.assertEqual(raised & launch_codes, set())
        self.assertEqual(script.splitlines()[0], "OPEN_APP: com.meesho.supply")
        self.assertNotIn("nexuslauncher", script)

    def __warm_start_record(
        self, *, number: int, event: str, action: str, export: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build one workflow-trace record for a run that started already inside the app.
        """

        record = {
            "duration": 1,
            "success": True,
            "target": "Element",
            "screen_changed": True,
            "step_number": number,
            "event_type": event,
            "action_type": action,
            "execution_activity": "com.warm.app",
        }
        if export is not None:
            record["export_target"] = export

        return record

    async def test_warm_start_run_generates_without_index_collision(self) -> None:
        """
        A run that begins already inside the app assembles, gates, and renders a warm-start launch.
        """

        records = (
            self.__warm_start_record(number=0, event="action", action="tap"),
            self.__warm_start_record(number=1, event="action", action="tap"),
            self.__warm_start_record(
                number=2, event="validation", action="complete", export="cart"
            ),
        )

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "history__execution.json").write_text(
                json.dumps({"workflow_id": "warm", "history": list(records)})
            )
            source = HistoryEvidenceSource(
                path_manager=StubPathManager(directory=directory),
                distiller=Distiller(),
                normalizer=RunTraceNormalizer(classifier=LauncherClassifier()),
                assembler=EvidenceAssembler(),
            )

            evidence = await source.read(
                execution_id="warm",
                objective=RunObjective(intent="i", goal="g", package="workflow"),
            )

        launches = [step.launch for step in evidence.steps if step.launch is not None]
        flow = self.__flow(evidence=evidence)
        report = self.__policy.evaluate(flow=flow, evidence=evidence)
        script = self.__dialect.renderer.render(flow=flow)

        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0].provenance, LaunchProvenance.SYNTHETIC_WARM_START)
        self.assertEqual(launches[0].source_steps, ())
        self.assertEqual(evidence.package, "com.warm.app")
        self.assertEqual(tuple(report.issues), ())
        self.assertEqual(script.splitlines()[0], "OPEN_APP: com.warm.app")
