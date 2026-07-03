from __future__ import annotations

import unittest
from typing import Any, Dict, Optional, Sequence

from fathom.constants.flow import CheckKind
from fathom.core.exceptions import LanguageComplianceError
from fathom.core.prompts.generation import FlowPromptBuilder
from fathom.core.services.generation.llm import LlmFlowGenerator
from fathom.interfaces.llm import LLMPort, PromptPart
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.flow import (
    Check,
    CheckNode,
    Evidence,
    Flow,
    LaunchNode,
    Selector,
    TapNode,
)
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.results import GenerateResult


class CannedLLM(LLMPort):
    """
    Returns fixed content and captures the structured-output contract and prompt it received.
    """

    def __init__(self, *, content: str) -> None:
        """
        Hold the content to return.
        """

        self.__content = content
        self.last_prompt: Sequence[PromptPart] = ()
        self.last_output: Optional[StructuredOutput] = None

    @property
    def model_name(self) -> str:
        """
        Return a static model name.
        """

        return "canned"

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
        Capture the request and return the canned content.
        """

        _ = tools, use_cache
        _ = system_instruction, conversation_history

        self.last_prompt = prompt
        self.last_output = structured_output
        return GenerateResult(content=self.__content)

    async def cleanup(self) -> None:
        """
        Release nothing.
        """

        return


class LlmFlowGeneratorTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover flow generation and boundary validation of the LLM response.
    """

    def setUp(self) -> None:
        """
        Build a shared minimal evidence aggregate.
        """

        self.__evidence = Evidence(intent="t", goal="g", package="com.example")

    def __flow(self) -> Flow:
        """
        Build a representative flow to round-trip through the generator.
        """

        return Flow(
            intent="t",
            package="com.example",
            nodes=(
                LaunchNode(package="com.example", source_steps=(0,)),
                TapNode(selector=Selector(text="Login"), source_steps=(1,)),
                CheckNode(
                    checks=(Check(kind=CheckKind.VISIBLE, subject="home"),), source_steps=(2,)
                ),
            ),
        )

    async def test_parses_flow_and_requests_flow_schema(self) -> None:
        """
        A conforming response is parsed into a Flow and the Flow schema was requested.
        """

        flow = self.__flow()
        llm = CannedLLM(content=flow.model_dump_json())
        generator = LlmFlowGenerator(llm=llm, prompt=FlowPromptBuilder(), use_cache=False)

        result = await generator.generate(evidence=self.__evidence)

        self.assertEqual(result, flow)
        self.assertIsNotNone(llm.last_output)

        assert llm.last_output is not None

        self.assertTrue(llm.last_prompt)
        self.assertIs(llm.last_output.payload, Flow)

    async def test_non_conforming_response_raises(self) -> None:
        """
        A response that is not a valid Flow fails explicitly.
        """

        llm = CannedLLM(content="{}")
        generator = LlmFlowGenerator(llm=llm, prompt=FlowPromptBuilder(), use_cache=False)

        with self.assertRaises(LanguageComplianceError):
            await generator.generate(evidence=self.__evidence)
