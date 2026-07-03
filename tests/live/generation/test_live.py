from __future__ import annotations

import pytest

from fathom.adapters.dialect.drizz.factory import DrizzDialectFactory
from fathom.core.dialect.policy import Policy
from fathom.core.prompts.generation import FlowPromptBuilder
from fathom.core.services.generation.llm import LlmFlowGenerator
from fathom.interfaces.llm import LLMPort
from fathom.schemas.flow import Evidence, EvidenceStep, StepGuard, StepTarget, StepWait

pytestmark = pytest.mark.release


class TestLiveFlowGeneration:
    """
    Cover real-LLM structured-output generation of a Flow, reproduced from a recorded run.
    """

    def __evidence(self) -> Evidence:
        """
        Build evidence mirroring a real Posh run.
        """

        return Evidence(
            package="com.posh.mobile",
            goal="The login screen is displayed",
            intent="Open Posh, choose New York, open the first trending event, reach checkout",
            steps=(
                EvidenceStep(
                    index=1,
                    event="action",
                    action="wait",
                    wait=StepWait(subject="Home screen with location dropdown", pattern="splash"),
                    guard=StepGuard(
                        conditional=True,
                        kind="transient",
                        condition="Home screen with location dropdown is visible",
                    ),
                ),
                EvidenceStep(
                    index=2,
                    action="tap",
                    event="action",
                    target=StepTarget(export="Nearby dropdown"),
                ),
                EvidenceStep(
                    index=3, event="action", action="tap", target=StepTarget(export="New York")
                ),
                EvidenceStep(
                    index=4,
                    action="tap",
                    event="action",
                    target=StepTarget(export="first trending card", positional=True),
                ),
                EvidenceStep(
                    index=5,
                    action="tap",
                    event="action",
                    target=StepTarget(export="Buy tickets button"),
                ),
                EvidenceStep(
                    index=6,
                    action="complete",
                    event="validation",
                    target=StepTarget(export="login screen"),
                ),
            ),
        )

    async def test_llm_emits_schema_valid_flow(self, llm: LLMPort) -> None:
        """
        The LLM constrains its output to the Flow schema, and the flow renders and gates clean.
        """

        generator = LlmFlowGenerator(llm=llm, prompt=FlowPromptBuilder(), use_cache=False)
        flow = await generator.generate(evidence=self.__evidence())

        assert flow.nodes

        dialect = DrizzDialectFactory().create()
        text = dialect.renderer.render(flow=flow)

        assert text.strip()
        assert dialect.checker.check(text=text).issues == ()
        assert Policy().evaluate(flow=flow, evidence=self.__evidence()).issues == ()
