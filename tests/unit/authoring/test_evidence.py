from __future__ import annotations

import unittest

from fathom.authoring.evidence import AuthoringEvidenceBuilder
from fathom.constants.authoring import AuthoringArtifactKind, AuthoringArtifactRole
from fathom.schemas.artifacts import ScreenArtifact, ScreenArtifactBundle, StepArtifacts
from fathom.schemas.authoring import AuthoringBaseline
from fathom.schemas.flow import Evidence, EvidenceStep, Flow
from fathom.schemas.steps import StepGoal


class AuthoringEvidenceBuilderTest(unittest.TestCase):
    """
    Cover deterministic conversion from normalized Fathom evidence to authoring evidence views.
    """

    def setUp(self) -> None:
        """
        Build the adapter under test.
        """

        self.builder = AuthoringEvidenceBuilder()

    def test_run_evidence_reuses_existing_evidence_and_groups_by_goal(self) -> None:
        """
        Run evidence must carry excluded source evidence and derive prompt-facing steps and episodes.
        """

        goal = StepGoal(index=2, description="find product", directive="scroll")
        evidence = Evidence(
            intent="find soap",
            goal="find soap",
            package="com.example",
            steps=(
                EvidenceStep(action="scroll", event="action", goal=goal, index=4),
                EvidenceStep(action="tap", event="action", goal=goal, index=5),
            ),
        )

        result = self.builder.build_run(evidence=evidence)
        assert result.run is not None

        self.assertIs(result.run.source, evidence)
        self.assertEqual(result.run.run.intent, "find soap")
        self.assertEqual(len(result.run.steps), 2)
        self.assertEqual(result.run.steps[0].command.action, "scroll")
        self.assertEqual(len(result.run.episodes), 1)
        self.assertEqual(result.run.episodes[0].goal, goal)
        self.assertEqual(result.run.episodes[0].steps, (4, 5))

    def test_run_evidence_collects_run_and_step_artifact_references(self) -> None:
        """
        Run evidence must represent multiple artifact references without loading artifact bytes.
        """

        artifacts = StepArtifacts(
            screen=ScreenArtifactBundle(
                before=ScreenArtifact(uri="history://run/step-1-before.png"),
                after=ScreenArtifact(uri="history://run/step-1-after.png"),
                annotated=ScreenArtifact(uri="history://run/step-1-annotated.png"),
                traces=(ScreenArtifact(uri="history://run/step-1-trace.png"),),
            )
        )
        evidence = Evidence(
            intent="find soap",
            goal="find soap",
            package="com.example",
            artifacts=("history://run/log.txt",),
            steps=(
                EvidenceStep(
                    action="tap",
                    event="action",
                    index=1,
                    screenshot="history://run/step-1-after.png",
                    artifacts=artifacts,
                ),
            ),
        )

        result = self.builder.build_run(evidence=evidence)
        assert result.run is not None

        self.assertEqual(len(result.run.artifacts), 5)
        self.assertEqual(result.run.artifacts[0].kind, AuthoringArtifactKind.TEXT)
        self.assertEqual(result.run.artifacts[0].role, AuthoringArtifactRole.LOG)
        self.assertEqual(result.run.artifacts[1].kind, AuthoringArtifactKind.IMAGE)
        self.assertEqual(result.run.artifacts[1].role, AuthoringArtifactRole.BEFORE)
        self.assertEqual(result.run.artifacts[1].step_index, 1)
        self.assertEqual(result.run.artifacts[2].role, AuthoringArtifactRole.AFTER)
        self.assertEqual(result.run.artifacts[3].role, AuthoringArtifactRole.ANNOTATED)
        self.assertEqual(result.run.artifacts[4].kind, AuthoringArtifactKind.TRACE)

    def test_run_evidence_carries_baseline_scaffold_when_available(self) -> None:
        """
        Run evidence must expose the deterministic baseline scaffold to final authoring.
        """

        evidence = Evidence(intent="find soap", goal="find soap", package="com.example")
        baseline = AuthoringBaseline(
            content="OPEN_APP: com.example\nTap on Search input field",
            partial=True,
            reason="No completion assertion was recorded.",
        )

        result = self.builder.build_run(evidence=evidence, baseline=baseline)
        assert result.run is not None

        self.assertEqual(result.run.baseline, baseline)

    def test_step_evidence_reuses_existing_evidence_and_filters_artifacts(self) -> None:
        """
        Step evidence must carry excluded source evidence and only selected-step artifact refs.
        """

        evidence = Evidence(
            intent="find soap",
            goal="find soap",
            package="com.example",
            steps=(
                EvidenceStep(
                    action="tap",
                    event="action",
                    index=1,
                    screenshot="history://run/step-1-after.png",
                ),
                EvidenceStep(
                    action="tap",
                    event="action",
                    index=2,
                    screenshot="history://run/step-2-after.png",
                ),
            ),
        )

        result = self.builder.build_step(evidence=evidence, step_index=2)
        assert result.step is not None

        self.assertIs(result.step.source, evidence)
        self.assertEqual(result.step.step_index, 2)
        self.assertEqual(result.step.step.index, 2)
        self.assertEqual(len(result.step.artifacts), 1)
        self.assertEqual(result.step.artifacts[0].uri, "history://run/step-2-after.png")

    def test_step_evidence_rejects_unknown_step(self) -> None:
        """
        Step evidence must fail fast when the selected step does not exist.
        """

        evidence = Evidence(intent="find soap", goal="find soap", package="com.example")

        with self.assertRaises(ValueError):
            self.builder.build_step(evidence=evidence, step_index=9)

    def test_repair_evidence_accepts_script_flow_review_and_optional_evidence(self) -> None:
        """
        Repair evidence must support external repair with optional execution evidence.
        """

        flow = Flow(intent="find soap", package="com.example")
        result = self.builder.build_repair(script="Tap on Search", flow=flow)
        assert result.repair is not None

        self.assertEqual(result.repair.script, "Tap on Search")
        self.assertEqual(result.repair.flow, flow)
        self.assertIsNone(result.repair.review)
        self.assertIsNone(result.repair.source)
        self.assertEqual(result.repair.artifacts, ())
