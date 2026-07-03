from __future__ import annotations

import unittest
from typing import Any, Dict, Optional, Sequence

from fathom.adapters.dialect.drizz.factory import DrizzDialectFactory
from fathom.constants.flow import CheckKind, LaunchProvenance
from fathom.core.dialect.policy import Policy
from fathom.core.prompts.generation import FlowPromptBuilder
from fathom.core.services.generation.binder import LaunchBinder
from fathom.core.services.generation.llm import LlmFlowGenerator
from fathom.core.services.generation.service import ScriptGenerationService
from fathom.interfaces.evidence import EvidenceSource
from fathom.interfaces.llm import LLMPort, PromptPart
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.flow import (
    Check,
    CheckNode,
    Evidence,
    EvidenceStep,
    Flow,
    LaunchNode,
    RunObjective,
    Selector,
    StepLaunch,
    StepTarget,
    TapNode,
)
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.results import GenerateResult


class StubEvidenceSource(EvidenceSource):
    """
    Returns a fixed evidence aggregate.
    """

    def __init__(self, *, evidence: Evidence) -> None:
        """
        Hold the evidence to return.
        """

        self.__evidence = evidence

    async def read(self, *, run: str, objective: RunObjective) -> Evidence:
        """
        Return the held evidence.
        """

        return self.__evidence


class CannedLLM(LLMPort):
    """
    Returns fixed content for any request.
    """

    def __init__(self, *, content: str) -> None:
        """
        Hold the content to return.
        """

        self.__content = content

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
        Return the canned content.
        """

        return GenerateResult(content=self.__content)

    async def cleanup(self) -> None:
        """
        Release nothing.
        """

        return


class GenerationPipelineTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover the composed pipeline (real generator, policy, dialect) over a canned LLM flow.
    """

    async def test_composed_pipeline_generates_drizz(self) -> None:
        """
        A canned conforming flow is gated by the real policy and rendered to Drizz.
        """

        evidence = Evidence(
            intent="open and verify",
            goal="home visible",
            package="com.example",
            steps=(
                EvidenceStep(
                    index=0,
                    event="launch",
                    action="launch",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                    ),
                ),
                EvidenceStep(
                    index=1,
                    event="action",
                    action="tap",
                    target=StepTarget(export="Login"),
                ),
                EvidenceStep(
                    index=2,
                    event="validation",
                    action="complete",
                    target=StepTarget(export="home"),
                ),
            ),
        )
        flow = Flow(
            intent="open and verify",
            package="com.example",
            nodes=(
                LaunchNode(package="com.example", source_steps=()),
                TapNode(selector=Selector(text="Login"), source_steps=(1,)),
                CheckNode(
                    checks=(Check(kind=CheckKind.VISIBLE, subject="home"),), source_steps=(2,)
                ),
            ),
        )
        service = ScriptGenerationService(
            evidence=StubEvidenceSource(evidence=evidence),
            generator=LlmFlowGenerator(
                llm=CannedLLM(content=flow.model_dump_json()),
                prompt=FlowPromptBuilder(),
                use_cache=False,
            ),
            policy=Policy(),
            dialect=DrizzDialectFactory().create(),
            binder=LaunchBinder(),
        )

        objective = RunObjective(
            intent="open and verify", goal="home visible", package="com.example"
        )
        result = await service.generate(run="run-x", objective=objective)

        self.assertEqual(result.attempts, 1)
        self.assertIn("OPEN_APP: com.example", result.text)
