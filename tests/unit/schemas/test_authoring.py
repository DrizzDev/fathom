from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.authoring.evidence import AuthoringEvidenceBuilder
from fathom.constants.authoring import (
    AuthoringArtifactKind,
    AuthoringArtifactRole,
    AuthoringKind,
    AuthoringStatus,
)
from fathom.constants.dialect import DialectName
from fathom.schemas.authoring import (
    AuthoringArtifact,
    AuthoringArtifactReference,
    AuthoringConfiguration,
    AuthoringEvidence,
    AuthoringResponse,
    AuthoringTask,
    RepairAuthoringEvidence,
)
from fathom.schemas.flow import (
    Evidence,
    EvidenceStep,
)


class AuthoringEvidenceTest(unittest.TestCase):
    """
    Cover authoring evidence boundary validation.
    """

    def test_requires_exactly_one_view(self) -> None:
        """
        AuthoringEvidence must carry exactly one task-specific view.
        """

        with self.assertRaises(ValidationError):
            AuthoringEvidence()

    def test_rejects_multiple_views(self) -> None:
        """
        AuthoringEvidence rejects packets that carry more than one task view.
        """

        evidence = Evidence(
            intent="open app",
            goal="open app",
            package="com.example",
            steps=(EvidenceStep(action="tap", event="action", index=0),),
        )
        builder = AuthoringEvidenceBuilder()

        with self.assertRaises(ValidationError):
            AuthoringEvidence(
                run=builder.build_run(evidence=evidence).run,
                step=builder.build_step(evidence=evidence, step_index=0).step,
            )

    def test_step_evidence_requires_selected_step(self) -> None:
        """
        StepAuthoringEvidence must reject a missing selected step.
        """

        evidence = Evidence(intent="open app", goal="open app", package="com.example")

        with self.assertRaises(ValueError):
            AuthoringEvidenceBuilder().build_step(evidence=evidence, step_index=3)

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
            evidence=AuthoringEvidenceBuilder().build_run(evidence=evidence),
            intent="open app",
            kind=AuthoringKind.RUN,
            execution_id="execution-1",
            step_number=1,
        )

        self.assertIs(task.dialect, DialectName.DRIZZ)
        assert task.evidence.run is not None
        self.assertIs(task.evidence.run.source, evidence)

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
                evidence=AuthoringEvidenceBuilder().build_run(evidence=evidence),
                intent="open app",
                kind=AuthoringKind.STEP,
                execution_id="execution-1",
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

    def test_authoring_response_rejects_whitespace_as_script(self) -> None:
        """
        AuthoringResponse must not treat whitespace content as a usable script.
        """

        response = AuthoringResponse(
            status=AuthoringStatus.GENERATED,
            artifact=AuthoringArtifact(
                dialect=DialectName.DRIZZ,
                kind=AuthoringArtifactKind.TEXT,
                content="   ",
            ),
        )

        self.assertFalse(response.has_script)

    def test_authoring_configuration_owns_attempt_budget(self) -> None:
        """
        AuthoringConfiguration must own retry budget validation.
        """

        configuration = AuthoringConfiguration(attempts=2)

        self.assertEqual(configuration.attempts, 2)

        with self.assertRaises(ValidationError):
            AuthoringConfiguration(attempts=0)
