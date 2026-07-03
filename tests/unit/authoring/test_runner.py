from __future__ import annotations

import unittest
from typing import Optional

from fathom.authoring.agent import AuthoringAgent
from fathom.authoring.application.runner import AuthoringRunner
from fathom.authoring.evidence import AuthoringEvidenceBuilder
from fathom.constants.authoring import (
    AuthoringArtifactKind,
    AuthoringKind,
    AuthoringMode,
    AuthoringStatus,
)
from fathom.constants.dialect import DialectName
from fathom.schemas.authoring import (
    AuthoringArtifact,
    AuthoringConfiguration,
    AuthoringEvidence,
    AuthoringResponse,
    AuthoringTask,
    RepairAuthoringEvidence,
    RunConfiguration,
    StepAuthoringConfiguration,
)
from fathom.schemas.flow import Evidence, EvidenceStep


class StubAuthoringPort:
    """
    Test port that records authoring calls.
    """

    def __init__(self, *, text: str) -> None:
        """
        Store the script returned by author.
        """

        self.calls = 0
        self.__text = text
        self.task: Optional[AuthoringTask] = None

    async def author(self, *, task: AuthoringTask) -> AuthoringResponse:
        """
        Return the configured script and record call arguments.
        """

        self.calls += 1
        self.task = task

        return AuthoringResponse(
            status=AuthoringStatus.GENERATED,
            artifact=AuthoringArtifact(
                dialect=DialectName.DRIZZ,
                kind=AuthoringArtifactKind.TEXT,
                content=self.__text,
            ),
        )


class AuthoringRunnerTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover configurable authoring.
    """

    @staticmethod
    def __evidence() -> Evidence:
        """
        Build minimal normalized execution evidence for authoring tasks.
        """

        return Evidence(intent="open app", goal="open app", package="com.example")

    @classmethod
    def __run_evidence(cls) -> AuthoringEvidence:
        """
        Build run authoring evidence.
        """

        return AuthoringEvidenceBuilder().build_run(evidence=cls.__evidence())

    @staticmethod
    def __step_evidence() -> AuthoringEvidence:
        """
        Build step authoring evidence with the selected step present.
        """

        evidence = Evidence(
            intent="open app",
            goal="open app",
            package="com.example",
            steps=(EvidenceStep(action="tap", event="action", index=3),),
        )
        return AuthoringEvidenceBuilder().build_step(evidence=evidence, step_index=3)

    @staticmethod
    def __repair_evidence() -> AuthoringEvidence:
        """
        Build repair authoring evidence.
        """

        return AuthoringEvidence(repair=RepairAuthoringEvidence(script="Tap on Search"))

    async def test_enabled_run_authoring_calls_source(self) -> None:
        """
        Enabled run authoring delegates to the single AuthoringAgent.
        """

        authoring = StubAuthoringPort(text="OPEN_APP: com.example")
        runner = AuthoringRunner(
            agent=AuthoringAgent(),
            configuration=AuthoringConfiguration(),
        )
        task = AuthoringTask(
            evidence=self.__run_evidence(),
            intent="open app",
            kind=AuthoringKind.RUN,
            workflow_id="workflow-1",
            step_number=3,
        )

        result = await runner.author(
            author=authoring,
            task=task,
        )

        self.assertIs(result.status, AuthoringStatus.GENERATED)
        self.assertEqual(result.script, "OPEN_APP: com.example")

        self.assertEqual(authoring.calls, 1)
        self.assertEqual(authoring.task, task)

    async def test_disabled_run_authoring_skips_without_calling_source(self) -> None:
        """
        Disabled run authoring returns an explicit skip and lets baseline fallback handle output.
        """

        authoring = StubAuthoringPort(text="OPEN_APP: com.example")
        runner = AuthoringRunner(
            agent=AuthoringAgent(),
            configuration=AuthoringConfiguration(run=RunConfiguration(enabled=False)),
        )
        task = AuthoringTask(
            evidence=self.__run_evidence(),
            intent="open app",
            kind=AuthoringKind.RUN,
            workflow_id="workflow-1",
            step_number=3,
        )

        result = await runner.author(
            author=authoring,
            task=task,
        )

        self.assertEqual(authoring.calls, 0)
        self.assertIs(result.status, AuthoringStatus.SKIPPED)

    async def test_disabled_step_authoring_skips_without_calling_source(self) -> None:
        """
        Disabled per-step authoring returns an explicit skip and does not call the source.
        """

        authoring = StubAuthoringPort(text="Tap on Search")
        runner = AuthoringRunner(
            agent=AuthoringAgent(),
            configuration=AuthoringConfiguration(),
        )
        task = AuthoringTask(
            evidence=self.__step_evidence(),
            intent="open app",
            kind=AuthoringKind.STEP,
            workflow_id="workflow-1",
            step_number=3,
        )

        result = await runner.author(author=authoring, task=task)

        self.assertEqual(authoring.calls, 0)
        self.assertIs(result.status, AuthoringStatus.SKIPPED)

    async def test_enabled_step_authoring_delegates_to_single_agent(self) -> None:
        """
        Enabled per-step authoring uses the same AuthoringAgent with a STEP task.
        """

        authoring = StubAuthoringPort(text="Tap on Search")
        runner = AuthoringRunner(
            agent=AuthoringAgent(),
            configuration=AuthoringConfiguration(
                step=StepAuthoringConfiguration(mode=AuthoringMode.SYNC)
            ),
        )
        task = AuthoringTask(
            evidence=self.__step_evidence(),
            intent="open app",
            kind=AuthoringKind.STEP,
            workflow_id="workflow-1",
            step_number=3,
        )

        result = await runner.author(author=authoring, task=task)

        self.assertEqual(authoring.calls, 1)
        self.assertEqual(authoring.task, task)
        self.assertIs(result.status, AuthoringStatus.GENERATED)

    async def test_repair_authoring_delegates_to_single_agent(self) -> None:
        """
        Repair authoring uses the same AuthoringAgent without the run or step switches.
        """

        authoring = StubAuthoringPort(text="Tap on Search")
        runner = AuthoringRunner(
            agent=AuthoringAgent(),
            configuration=AuthoringConfiguration(
                run=RunConfiguration(enabled=False),
                step=StepAuthoringConfiguration(mode=AuthoringMode.DISABLED),
            ),
        )
        task = AuthoringTask(
            evidence=self.__repair_evidence(),
            intent="repair script",
            kind=AuthoringKind.REPAIR,
            workflow_id="workflow-1",
            step_number=3,
        )

        result = await runner.author(author=authoring, task=task)

        self.assertEqual(authoring.calls, 1)
        self.assertEqual(authoring.task, task)
        self.assertIs(result.status, AuthoringStatus.GENERATED)
