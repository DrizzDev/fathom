from __future__ import annotations

import unittest
from typing import Any, Dict, Optional, Sequence, Tuple

from fathom.adapters.dialect.drizz.factory import DrizzDialectFactory
from fathom.authoring.application.request import AuthoringRequestBuilder
from fathom.authoring.application.reviewer import AuthoringReviewer
from fathom.authoring.evidence import AuthoringEvidenceBuilder
from fathom.constants.authoring import AuthoringKind, AuthoringStatus
from fathom.constants.flow import AssertionSource, CheckKind, IssueCode
from fathom.core.dialect.policy import Policy
from fathom.core.services.authoring import AuthoringService
from fathom.interfaces.llm import LLMPort, PromptPart
from fathom.schemas.authoring import AuthoringTask
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.flow import (
    Check,
    CheckNode,
    CompletionAssertion,
    Evidence,
    EvidenceStep,
    Flow,
    Selector,
    StepTarget,
    TapNode,
    TargetClaim,
)
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.results import GenerateResult


class FakeAuthoringLlm(LLMPort):
    """
    LLM test double that returns a configured Flow payload.
    """

    def __init__(self, *, flow: Flow, retries: Tuple[Flow, ...] = ()) -> None:
        """
        Store the Flow to emit.
        """

        self.calls = 0
        self.prompt: Optional[Sequence[PromptPart]] = None
        self.system_instruction: Optional[str] = None
        self.structured_output: Optional[StructuredOutput] = None
        self.__flows = (flow,) + retries

    @property
    def model_name(self) -> str:
        """
        Return the fake model name.
        """

        return "fake-authoring-model"

    async def generate(
        self,
        *,
        use_cache: bool,
        prompt: Sequence[PromptPart],
        tools: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        conversation_history: Optional[Sequence[ConversationTurn]] = None,
        structured_output: Optional[StructuredOutput] = None,
    ) -> GenerateResult:
        """
        Record the request and return the configured Flow as structured JSON.
        """

        _ = (use_cache, tools, conversation_history)
        self.calls += 1
        self.prompt = prompt
        self.system_instruction = system_instruction
        self.structured_output = structured_output
        index = min(self.calls - 1, len(self.__flows) - 1)
        return GenerateResult(
            content=self.__flows[index].model_dump_json(),
            metrics={"prompt_tokens": 10, "completion_tokens": 5},
        )

    async def cleanup(self) -> None:
        """
        Release no resources.
        """

        return None


class NoopAuthoringArtifactProvider:
    """
    Test artifact provider that attaches no external files.
    """

    def build(self, *, task: AuthoringTask) -> Tuple[PromptPart, ...]:
        """
        Return no artifact prompt parts for the task.
        """

        _ = task
        return ()


class AuthoringServiceTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover LLM-backed authoring service behavior.
    """

    @staticmethod
    def __task() -> AuthoringTask:
        """
        Build a STEP task so the service can author a single command fragment.
        """

        evidence = Evidence(
            intent="tap search",
            goal="search focused",
            package="com.example",
            steps=(EvidenceStep(action="tap", event="action", index=1),),
        )
        return AuthoringTask(
            kind=AuthoringKind.STEP,
            intent="tap search",
            step_number=1,
            execution_id="execution-1",
            evidence=AuthoringEvidenceBuilder().build_step(evidence=evidence, step_index=1),
        )

    async def test_authors_step_script_through_llm_and_dialect(self) -> None:
        """
        AuthoringService must prompt the model and return checked dialect text.
        """

        flow = Flow(
            intent="tap search",
            package="com.example",
            nodes=(
                TapNode(
                    source_steps=(1,),
                    selector=Selector(text="Search field"),
                ),
            ),
        )
        llm = FakeAuthoringLlm(flow=flow)
        dialect = DrizzDialectFactory().create()
        service = AuthoringService(
            llm=llm,
            use_cache=True,
            attempts=3,
            reviewer=AuthoringReviewer(policy=Policy(), dialect=dialect),
            requests=AuthoringRequestBuilder(artifacts=NoopAuthoringArtifactProvider()),
        )

        response = await service.author(task=self.__task())

        self.assertIs(response.status, AuthoringStatus.GENERATED)
        self.assertEqual(response.script, "Tap on Search field")
        self.assertEqual(llm.calls, 1)
        self.assertIsNotNone(llm.system_instruction)
        self.assertIsNotNone(llm.prompt)
        assert llm.structured_output is not None
        self.assertIs(llm.structured_output.payload, Flow)

    async def test_retries_until_terminal_validation_cites_completion_assertion(self) -> None:
        """
        A final Validate must be grounded by verifier assertions, not invented from a normal step.
        """

        assertion = CompletionAssertion(
            id="completion-1",
            kind=CheckKind.VISIBLE,
            subject="Phone number input field",
            source=AssertionSource.VERIFICATION,
        )
        evidence = Evidence(
            intent="reach login",
            goal="login screen appears",
            package="com.example",
            assertions=(assertion,),
            steps=(EvidenceStep(action="tap", event="action", index=2),),
        )
        task = AuthoringTask(
            kind=AuthoringKind.RUN,
            intent="reach login",
            step_number=2,
            execution_id="execution-1",
            evidence=AuthoringEvidenceBuilder().build_run(evidence=evidence),
        )
        invented = Flow(
            intent="reach login",
            package="com.example",
            nodes=(
                CheckNode(
                    source_steps=(2,),
                    checks=(Check(kind=CheckKind.VISIBLE, subject="Buy Now button"),),
                ),
            ),
        )
        grounded = Flow(
            intent="reach login",
            package="com.example",
            nodes=(
                CheckNode(
                    source_steps=(2,),
                    assertion_ids=("completion-1",),
                    checks=(Check(kind=CheckKind.VISIBLE, subject="Phone number input field"),),
                ),
            ),
        )
        llm = FakeAuthoringLlm(flow=invented, retries=(grounded,))
        dialect = DrizzDialectFactory().create()
        service = AuthoringService(
            llm=llm,
            use_cache=True,
            attempts=3,
            reviewer=AuthoringReviewer(policy=Policy(), dialect=dialect),
            requests=AuthoringRequestBuilder(artifacts=NoopAuthoringArtifactProvider()),
        )

        response = await service.author(task=task)

        self.assertIs(response.status, AuthoringStatus.GENERATED)
        self.assertEqual(response.script, "Validate Phone number input field is visible")
        self.assertEqual(llm.calls, 2)
        self.assertIsNotNone(response.artifact)
        assert response.artifact is not None
        self.assertEqual(
            response.artifact.lineage[0].verified_by,
            ("execution", "completion_assertion"),
        )

    async def test_partial_evidence_must_return_partial_flow_with_executed_commands(self) -> None:
        """
        Incomplete execution may publish useful commands only when the authored Flow is partial.
        """

        evidence = Evidence(
            intent="open menu",
            goal="menu visible",
            package="com.example",
            partial=True,
            reason="Execution stopped before the goal was verified.",
            steps=(
                EvidenceStep(
                    action="tap",
                    event="action",
                    index=1,
                    target=StepTarget(export="Menu button"),
                ),
            ),
        )
        task = AuthoringTask(
            kind=AuthoringKind.RUN,
            intent="open menu",
            step_number=1,
            execution_id="execution-1",
            evidence=AuthoringEvidenceBuilder().build_run(evidence=evidence),
        )
        incomplete = Flow(
            intent="open menu",
            package="com.example",
            nodes=(
                TapNode(
                    source_steps=(1,),
                    selector=Selector(text="Menu button"),
                ),
            ),
        )
        partial = incomplete.model_copy(update={"partial": True})
        llm = FakeAuthoringLlm(flow=incomplete, retries=(partial,))
        dialect = DrizzDialectFactory().create()
        service = AuthoringService(
            llm=llm,
            use_cache=True,
            attempts=3,
            reviewer=AuthoringReviewer(policy=Policy(), dialect=dialect),
            requests=AuthoringRequestBuilder(artifacts=NoopAuthoringArtifactProvider()),
        )

        response = await service.author(task=task)

        self.assertIs(response.status, AuthoringStatus.GENERATED)
        self.assertEqual(response.script, "Tap on Menu button")
        self.assertEqual(llm.calls, 2)

    async def test_unconfirmed_target_claim_is_returned_as_advisory(self) -> None:
        """
        Weak target provenance is metadata, not a blocking authoring failure.
        """

        evidence = Evidence(
            intent="select product",
            goal="product selected",
            package="com.example",
            partial=True,
            steps=(
                EvidenceStep(
                    action="tap",
                    event="action",
                    index=1,
                    target=StepTarget(
                        name="Magic Soap 3 Pack",
                        claim=TargetClaim(text="Magic Soap 3 Pack", verified=False),
                    ),
                ),
            ),
        )
        task = AuthoringTask(
            kind=AuthoringKind.RUN,
            intent="select product",
            step_number=1,
            execution_id="execution-1",
            evidence=AuthoringEvidenceBuilder().build_run(evidence=evidence),
        )
        flow = Flow(
            intent="select product",
            package="com.example",
            partial=True,
            nodes=(
                TapNode(
                    source_steps=(1,),
                    selector=Selector(text="Magic Soap 3 Pack"),
                ),
            ),
        )
        llm = FakeAuthoringLlm(flow=flow)
        dialect = DrizzDialectFactory().create()
        service = AuthoringService(
            llm=llm,
            use_cache=True,
            attempts=3,
            reviewer=AuthoringReviewer(policy=Policy(), dialect=dialect),
            requests=AuthoringRequestBuilder(artifacts=NoopAuthoringArtifactProvider()),
        )

        response = await service.author(task=task)

        self.assertIs(response.status, AuthoringStatus.GENERATED)
        self.assertIsNotNone(response.artifact)
        assert response.artifact is not None
        self.assertEqual(response.script, "Tap on Magic Soap 3 Pack")
        self.assertEqual(
            [issue.code for issue in response.artifact.advisories],
            [IssueCode.UNCONFIRMED_TARGET_CLAIM],
        )
        self.assertEqual(len(response.artifact.lineage), 1)
        self.assertEqual(response.artifact.lineage[0].source_steps, (1,))
        self.assertEqual(response.artifact.lineage[0].verified_by, ("execution",))
        self.assertTrue(response.artifact.lineage[0].screen_authored)
