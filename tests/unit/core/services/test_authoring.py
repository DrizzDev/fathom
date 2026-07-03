from __future__ import annotations

import unittest
from typing import Any, Dict, Optional, Sequence

from fathom.adapters.dialect.drizz.factory import DrizzDialectFactory
from fathom.authoring.evidence import AuthoringEvidenceBuilder
from fathom.constants.authoring import AuthoringKind, AuthoringStatus
from fathom.core.dialect.policy import Policy
from fathom.core.services.authoring import AuthoringService
from fathom.interfaces.llm import LLMPort, PromptPart
from fathom.schemas.authoring import AuthoringTask
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.flow import Evidence, EvidenceStep, Flow, Selector, TapNode
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.results import GenerateResult


class FakeAuthoringLlm(LLMPort):
    """
    LLM test double that returns a configured Flow payload.
    """

    def __init__(self, *, flow: Flow) -> None:
        """
        Store the Flow to emit.
        """

        self.calls = 0
        self.prompt: Optional[Sequence[PromptPart]] = None
        self.system_instruction: Optional[str] = None
        self.structured_output: Optional[StructuredOutput] = None
        self.__flow = flow

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
        return GenerateResult(
            content=self.__flow.model_dump_json(),
            metrics={"prompt_tokens": 10, "completion_tokens": 5},
        )

    async def cleanup(self) -> None:
        """
        Release no resources.
        """

        return None


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
            workflow_id="workflow-1",
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
        service = AuthoringService(
            llm=llm,
            policy=Policy(),
            use_cache=True,
            attempts=3,
            dialect=DrizzDialectFactory().create(),
        )

        response = await service.author(task=self.__task())

        self.assertIs(response.status, AuthoringStatus.GENERATED)
        self.assertEqual(response.script, "Tap on Search field")
        self.assertEqual(llm.calls, 1)
        self.assertIsNotNone(llm.system_instruction)
        self.assertIsNotNone(llm.prompt)
        assert llm.structured_output is not None
        self.assertIs(llm.structured_output.payload, Flow)
