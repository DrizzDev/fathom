from __future__ import annotations

import unittest
from typing import Tuple

from fathom.authoring.agent import AuthoringAgent
from fathom.authoring.application import AuthoringRunner, StepAuthoringScheduler
from fathom.authoring.evidence import AuthoringEvidenceBuilder
from fathom.constants.authoring import (
    AuthoringArtifactKind,
    AuthoringKind,
    AuthoringMode,
    AuthoringStatus,
)
from fathom.interfaces.authoring import AuthoringDraftStore, AuthoringPort
from fathom.interfaces.evidence import EvidenceSource
from fathom.schemas.authoring import (
    AuthoringArtifact,
    AuthoringConfiguration,
    AuthoringResponse,
    AuthoringTask,
    StepAuthoringConfiguration,
)
from fathom.schemas.authoring.draft import AuthoringDraft
from fathom.schemas.flow import Evidence, EvidenceStep, RunObjective


class StubEvidenceSource(EvidenceSource):
    """
    Evidence source returning one configured evidence aggregate.
    """

    def __init__(self, *, evidence: Evidence) -> None:
        """
        Store the evidence returned to the scheduler.
        """

        self.__evidence = evidence

    async def read(self, *, run: str, objective: RunObjective) -> Evidence:
        """
        Return the configured evidence aggregate.
        """

        _ = (run, objective)
        return self.__evidence


class StubAuthoring(AuthoringPort):
    """
    Authoring source returning one generated step artifact.
    """

    async def author(self, *, task: AuthoringTask) -> AuthoringResponse:
        """
        Return a generated artifact for the scheduled step task.
        """

        return AuthoringResponse(
            status=AuthoringStatus.GENERATED,
            artifact=AuthoringArtifact(
                dialect=task.dialect,
                content="Tap on Search field",
                kind=AuthoringArtifactKind.TEXT,
            ),
        )


class RecordingDraftStore(AuthoringDraftStore):
    """
    In-memory draft store for scheduler tests.
    """

    def __init__(self) -> None:
        """
        Start with no recorded drafts.
        """

        self.drafts: Tuple[AuthoringDraft, ...] = ()

    async def save(self, *, draft: AuthoringDraft) -> None:
        """
        Record one draft.
        """

        self.drafts = self.drafts + (draft,)

    async def list(self, *, workflow_id: str) -> Tuple[AuthoringDraft, ...]:
        """
        Return recorded drafts for the requested workflow.
        """

        return tuple(draft for draft in self.drafts if draft.workflow_id == workflow_id)


class StepAuthoringSchedulerTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover optional per-step authoring scheduling.
    """

    async def test_schedule_step_authors_and_persists_draft(self) -> None:
        """
        Scheduler turns one persisted step into one saved authoring draft.
        """

        evidence = Evidence(
            intent="tap search",
            goal="search focused",
            package="com.example",
            steps=(EvidenceStep(action="tap", event="action", index=2),),
        )
        drafts = RecordingDraftStore()
        scheduler = StepAuthoringScheduler(
            builder=AuthoringEvidenceBuilder(),
            source=StubEvidenceSource(evidence=evidence),
            runner=AuthoringRunner(
                agent=AuthoringAgent(),
                configuration=AuthoringConfiguration(
                    step=StepAuthoringConfiguration(mode=AuthoringMode.SYNC)
                ),
            ),
            drafts=drafts,
            author=StubAuthoring(),
        )

        scheduler.schedule_step(
            workflow_id="workflow-1",
            step_index=2,
            objective=RunObjective(
                intent="tap search",
                goal="search focused",
                package="com.example",
            ),
        )
        await scheduler.drain()

        self.assertEqual(len(drafts.drafts), 1)
        self.assertTrue(drafts.drafts[0].generated)
        self.assertEqual(drafts.drafts[0].step_index, 2)
        self.assertIs(drafts.drafts[0].kind, AuthoringKind.STEP)
