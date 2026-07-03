from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Tuple

from fathom.adapters.dialect.drizz.factory import DrizzDialectFactory
from fathom.constants.flow import LaunchProvenance, NodeKind
from fathom.core.dialect.policy import Policy
from fathom.core.services.generation.assembler import EvidenceAssembler
from fathom.core.services.generation.classifier import LauncherClassifier
from fathom.core.services.generation.distiller import Distiller
from fathom.core.services.generation.normalizer import RunTraceNormalizer
from fathom.core.services.generation.projector import DeterministicFlowGenerator
from fathom.schemas.flow import Evidence, Issue, LaunchNode
from fathom.schemas.generation import ProjectionReport
from fathom.schemas.steps import StepHistory

FIXTURE_DIR = Path(__file__).resolve().parents[4] / "fixtures" / "generation"


class ProjectorRcaFixtureTest(unittest.IsolatedAsyncioTestCase):
    """
    Drives real recorded RCA histories through distill -> normalize -> assemble -> project -> policy -> render -> check.
    """

    def __evidence(self, *, fixture: str, intent: str, goal: str, package: str) -> Evidence:
        """
        Build evidence from a recorded history fixture via the production generation pipeline.
        """

        raw = json.loads((FIXTURE_DIR / fixture).read_text())
        records = StepHistory.model_validate(raw).history
        distillation = Distiller().distill(records=records)
        trace = RunTraceNormalizer(classifier=LauncherClassifier()).normalize(
            records=distillation.records
        )

        return EvidenceAssembler().assemble(
            trace=trace,
            intent=intent,
            goal=goal,
            package=package,
            partial=distillation.partial,
            discarded=distillation.discarded,
            reason=distillation.reason,
        )

    def __render(
        self, *, evidence: Evidence
    ) -> Tuple[ProjectionReport, Tuple[Issue, ...], str, Tuple[Issue, ...]]:
        """
        Project the evidence and return the report, policy issues, rendered text, and checker issues.
        """

        report = DeterministicFlowGenerator().project(evidence=evidence)
        policy_evidence = evidence.model_copy(update={"partial": report.flow.partial})
        policy = Policy().evaluate(flow=report.flow, evidence=policy_evidence)
        dialect = DrizzDialectFactory().create()
        text = dialect.renderer.render(flow=report.flow)
        check = dialect.checker.check(text=text)

        return report, policy.issues, text, check.issues

    async def test_swiggy_history_projects_to_clean_grounded_script(self) -> None:
        """
        The Swiggy RCA run projects to a launch-led, partial, policy- and syntax-clean script.
        """

        evidence = self.__evidence(
            fixture="swiggy_history.json",
            intent="order a burger on swiggy",
            goal="burger added to cart",
            package="in.swiggy.android",
        )
        report, policy_issues, text, check_issues = self.__render(evidence=evidence)
        launch = report.flow.nodes[0]
        self.assertIsInstance(launch, LaunchNode)
        assert isinstance(launch, LaunchNode)

        self.assertEqual(policy_issues, ())
        self.assertEqual(check_issues, ())
        self.assertIs(launch.kind, NodeKind.LAUNCH)
        self.assertEqual(launch.package, "in.swiggy.android")
        self.assertIs(launch.provenance, LaunchProvenance.LAUNCHER_TRANSITION)
        self.assertEqual(launch.source_steps, (0,))
        self.assertTrue(report.flow.partial)
        self.assertIsNot(report.flow.nodes[-1].kind, NodeKind.CHECK)
        self.assertTrue(text.startswith("OPEN_APP"))
        self.assertIn("in.swiggy.android", text.splitlines()[0])
        self.assertNotIn("Validate", text)

    async def test_radioplayer_history_projects_to_partial_grounded_script(self) -> None:
        """
        The Radioplayer RCA run projects without fabricating an unstructured validation.
        """

        evidence = self.__evidence(
            fixture="radioplayer_history.json",
            intent="open radioplayer and reach the homepage",
            goal="homepage visible",
            package="com.radioplayer.mobile",
        )
        report, policy_issues, text, check_issues = self.__render(evidence=evidence)

        self.assertEqual(policy_issues, ())
        self.assertEqual(check_issues, ())
        self.assertTrue(text.startswith("OPEN_APP"))
        self.assertIn("com.radioplayer.mobile", text.splitlines()[0])
        self.assertTrue(report.flow.partial)
        self.assertIsNot(report.flow.nodes[-1].kind, NodeKind.CHECK)
