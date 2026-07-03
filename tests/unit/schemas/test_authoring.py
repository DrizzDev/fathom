from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.constants.authoring import (
    AuthoringArtifactKind,
    AuthoringArtifactRole,
    AuthoringKind,
    AuthoringStatus,
)
from fathom.constants.dialect import DialectName
from fathom.constants.flow import LaunchProvenance
from fathom.schemas.authoring import (
    AuthoringArtifact,
    AuthoringArtifactReference,
    AuthoringConfiguration,
    AuthoringEvidence,
    AuthoringResponse,
    AuthoringTask,
    PromptEvidence,
    RepairAuthoringEvidence,
    RunAuthoringEvidence,
    StepAuthoringEvidence,
)
from fathom.schemas.flow import (
    Evidence,
    EvidenceStep,
    StepCapture,
    StepLaunch,
    StepTarget,
)
from fathom.schemas.steps import StepGoal


class AuthoringEvidenceTest(unittest.TestCase):
    """
    Cover authoring evidence boundary validation.
    """

    def test_requires_exactly_one_view(self) -> None:
        """
        AuthoringEvidence must carry exactly one task-specific view.
        """

        evidence = Evidence(
            intent="open app",
            goal="open app",
            package="com.example",
            steps=(EvidenceStep(action="tap", event="action", index=0),),
        )

        with self.assertRaises(ValidationError):
            AuthoringEvidence()

        with self.assertRaises(ValidationError):
            AuthoringEvidence(
                run=RunAuthoringEvidence(evidence=evidence),
                step=StepAuthoringEvidence(evidence=evidence, step_index=0),
            )

    def test_step_evidence_requires_selected_step(self) -> None:
        """
        StepAuthoringEvidence must reject a missing selected step.
        """

        evidence = Evidence(intent="open app", goal="open app", package="com.example")

        with self.assertRaises(ValidationError):
            StepAuthoringEvidence(evidence=evidence, step_index=3)

    def test_authoring_task_carries_dialect_and_evidence(self) -> None:
        """
        AuthoringTask must carry target dialect and typed evidence.
        """

        evidence = Evidence(
            intent="open app",
            goal="open app",
            package="com.example",
            steps=(EvidenceStep(action="tap", event="action", index=0),),
        )
        task = AuthoringTask(
            evidence=AuthoringEvidence(run=RunAuthoringEvidence(evidence=evidence)),
            intent="open app",
            kind=AuthoringKind.RUN,
            workflow_id="workflow-1",
            step_number=1,
        )

        self.assertIs(task.dialect, DialectName.DRIZZ)
        assert task.evidence.run is not None
        self.assertIs(task.evidence.run.evidence, evidence)

    def test_authoring_task_rejects_kind_evidence_mismatch(self) -> None:
        """
        AuthoringTask must reject a task kind that does not match the evidence view.
        """

        evidence = Evidence(
            intent="open app",
            goal="open app",
            package="com.example",
            steps=(EvidenceStep(action="tap", event="action", index=0),),
        )

        with self.assertRaises(ValidationError):
            AuthoringTask(
                evidence=AuthoringEvidence(run=RunAuthoringEvidence(evidence=evidence)),
                intent="open app",
                kind=AuthoringKind.STEP,
                workflow_id="workflow-1",
                step_number=1,
            )

    def test_repair_evidence_requires_input(self) -> None:
        """
        RepairAuthoringEvidence must fail fast when no repair input is supplied.
        """

        with self.assertRaises(ValidationError):
            RepairAuthoringEvidence()

    def test_artifact_reference_supports_step_scoped_refs(self) -> None:
        """
        Artifact references must represent step-scoped assets without embedding bytes.
        """

        reference = AuthoringArtifactReference(
            kind=AuthoringArtifactKind.IMAGE,
            role=AuthoringArtifactRole.AFTER,
            uri="history://workflow/step-1.png",
            mime="image/png",
            step_index=1,
        )

        self.assertEqual(reference.uri, "history://workflow/step-1.png")
        self.assertEqual(reference.step_index, 1)

    def test_authoring_response_exposes_artifact_as_script_text(self) -> None:
        """
        AuthoringResponse stores an artifact and exposes script text as a derived view.
        """

        response = AuthoringResponse(
            status=AuthoringStatus.GENERATED,
            artifact=AuthoringArtifact(
                dialect=DialectName.DRIZZ,
                kind=AuthoringArtifactKind.TEXT,
                content="Tap on Search field",
            ),
        )

        self.assertTrue(response.has_script)
        self.assertEqual(response.script, "Tap on Search field")

    def test_authoring_configuration_owns_attempt_budget(self) -> None:
        """
        AuthoringConfiguration must own retry budget validation.
        """

        configuration = AuthoringConfiguration(attempts=2)

        self.assertEqual(configuration.attempts, 2)

        with self.assertRaises(ValidationError):
            AuthoringConfiguration(attempts=0)


class PromptEvidenceTest(unittest.TestCase):
    """
    Pins the compact evidence packet sent to an authoring prompt.
    """

    def test_packet_keeps_authoring_truth_and_drops_noisy_references(self) -> None:
        """
        The packet carries command evidence but omits artifact and screenshot references.
        """

        evidence = Evidence(
            goal="product visible",
            package="com.example",
            intent="store price",
            artifacts=("blob://large",),
            steps=(
                EvidenceStep(
                    index=0,
                    event="launch",
                    action="launch",
                    screenshot="screenshot://ignored",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                        source_steps=(0,),
                    ),
                ),
                EvidenceStep(
                    index=1,
                    event="action",
                    action="store",
                    goal=StepGoal(index=1, description="Store price", directive="store"),
                    target=StepTarget(export="Price label", element="text"),
                    capture=StepCapture(
                        name="item_price",
                        subject="price",
                        success=True,
                        value="₹86",
                    ),
                ),
            ),
        )

        packet = PromptEvidence.from_evidence(evidence=evidence)
        payload = packet.model_dump_json(exclude_none=True)

        self.assertEqual(packet.run.package, "com.example")
        self.assertEqual(len(packet.episodes), 2)
        self.assertIsNotNone(packet.episodes[1].goal)
        self.assertIsNotNone(packet.episodes[1].steps[0].capture)
        assert packet.episodes[1].steps[0].capture is not None

        self.assertEqual(packet.episodes[1].steps[0].capture.name, "item_price")
        self.assertIsNotNone(packet.episodes[1].steps[0].target)
        assert packet.episodes[1].steps[0].target is not None
        self.assertEqual(packet.episodes[1].steps[0].target.element, "text")
        self.assertIn("₹86", payload)
        self.assertIn("element", payload)
        self.assertIn("source_steps", payload)
        self.assertNotIn("blob://large", payload)
        self.assertNotIn("screenshot://ignored", payload)

    def test_packet_groups_contiguous_steps_by_goal(self) -> None:
        """
        Prompt evidence groups repeated attempts under the same sub-goal episode.
        """

        goal = StepGoal(index=2, description="Check rating", directive="validate")
        evidence = Evidence(
            goal="login visible",
            package="com.example",
            intent="find product",
            steps=(
                EvidenceStep(index=5, event="action", action="swipe_up", goal=goal),
                EvidenceStep(index=6, event="action", action="swipe_up", goal=goal),
                EvidenceStep(index=7, event="action", action="tap", goal=goal),
            ),
        )

        packet = PromptEvidence.from_evidence(evidence=evidence)

        self.assertEqual(len(packet.episodes), 1)
        self.assertIsNotNone(packet.episodes[0].goal)
        assert packet.episodes[0].goal is not None
        self.assertEqual(packet.episodes[0].goal.description, "Check rating")
        self.assertEqual([step.step_id for step in packet.episodes[0].steps], [5, 6, 7])

    def test_packet_does_not_group_steps_without_goal_context(self) -> None:
        """
        Missing goal context does not imply multiple steps share one authoring purpose.
        """

        evidence = Evidence(
            goal="login visible",
            package="com.example",
            intent="find product",
            steps=(
                EvidenceStep(index=1, event="action", action="tap"),
                EvidenceStep(index=2, event="action", action="tap"),
            ),
        )

        packet = PromptEvidence.from_evidence(evidence=evidence)

        self.assertEqual(len(packet.episodes), 2)
        self.assertIsNone(packet.episodes[0].goal)
        self.assertIsNone(packet.episodes[1].goal)
        self.assertEqual(packet.episodes[0].steps[0].step_id, 1)
        self.assertEqual(packet.episodes[1].steps[0].step_id, 2)
