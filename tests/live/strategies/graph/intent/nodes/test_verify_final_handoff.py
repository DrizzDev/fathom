from __future__ import annotations

import io
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image, ImageDraw
from tests.builders import SubGoalFixtures

from fathom.constants import ActionType
from fathom.constants.state import CommonStateKey, IntentStateKey, VerifyMode
from fathom.core.agent.state import AgentState
from fathom.interfaces.llm import LLMPort, PromptPart
from fathom.schemas.actions import Action
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.results import GenerateResult, PlanResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.strategies.graph.intent.nodes.record import RecordNode
from fathom.strategies.graph.intent.nodes.verify import VerifyNode

pytestmark = pytest.mark.release


class _RecordingLlm(LLMPort):
    """
    Wrap the live LLM and retain raw verifier responses for test output.
    """

    def __init__(self, *, llm: LLMPort) -> None:
        """
        Bind the live LLM adapter.
        """

        self.__llm = llm
        self.responses: List[str] = []

    @property
    def model_name(self) -> str:
        """
        Return the wrapped model name.
        """

        return self.__llm.model_name

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
        Generate through the wrapped LLM and store the raw content.
        """

        result = await self.__llm.generate(
            use_cache=use_cache,
            prompt=prompt,
            tools=tools,
            system_instruction=system_instruction,
            conversation_history=conversation_history,
            structured_output=structured_output,
        )
        self.responses.append(result.content)
        return result

    async def cleanup(self) -> None:
        """
        The owning fixture cleans up the wrapped live LLM.
        """

        return


def _print_llm_responses(*, responses: List[str]) -> None:
    """
    Emit raw verifier LLM responses when pytest is run with ``-s``.
    """

    for index, response in enumerate(responses, start=1):
        print(f"\n[VERIFY_LLM_RESPONSE_{index}]\n{response}\n")


class SalarySeModalFixture:
    """
    Builds a visual replay of the SalarySe address-confirmation blocker.
    """

    @staticmethod
    def capture() -> ScreenCapture:
        """
        Render a modal screenshot where the requested final address is not yet accepted.
        """

        image = Image.new("RGB", (402, 874), "#f5f5f5")
        draw = ImageDraw.Draw(image)

        draw.rectangle((0, 0, 402, 874), fill="#f3f3f3")
        draw.text((30, 60), "Selected Address", fill="#666666")
        draw.text((30, 90), "SalarySe office", fill="#111111")
        draw.rectangle((0, 0, 402, 874), fill="#000000")

        draw.rectangle((36, 280, 366, 560), fill="#ffffff", outline="#222222", width=2)
        draw.text((70, 315), "Switch delivery address?", fill="#111111")
        draw.text((70, 360), "You are currently ordering food.", fill="#333333")
        draw.text((70, 390), "Confirm this change before continuing.", fill="#333333")
        draw.rectangle((70, 455, 332, 515), fill="#fc8019")
        draw.text((132, 476), "Yes, continue", fill="#ffffff")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return ScreenCapture(
            width=402,
            height=874,
            image=buffer.getvalue(),
            timestamp=int(time.time() * 1000),
            activity="in.swiggy.android/.HomeActivity",
        )

    @staticmethod
    def completed_capture() -> ScreenCapture:
        """
        Render the post-correction screen where the SalarySe address is selected and no modal remains.
        """

        image = Image.new("RGB", (402, 874), "#ffffff")
        draw = ImageDraw.Draw(image)

        draw.rectangle((0, 0, 402, 96), fill="#fc8019")
        draw.text((24, 36), "Selected Address", fill="#ffffff")
        draw.rectangle((24, 140, 378, 260), fill="#f7fff7", outline="#178a3b", width=3)
        draw.text((48, 170), "SalarySe office", fill="#111111")
        draw.text((48, 205), "Delivery address selected", fill="#178a3b")
        draw.text((48, 330), "Restaurants near SalarySe office", fill="#111111")
        draw.rectangle((48, 385, 354, 455), fill="#f1f1f1", outline="#dddddd")
        draw.text((72, 410), "Food delivery home screen", fill="#333333")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return ScreenCapture(
            width=402,
            height=874,
            image=buffer.getvalue(),
            timestamp=int(time.time() * 1000),
            activity="in.swiggy.android/.HomeActivity",
        )


class HsrProdAssetFixture:
    """
    Loads the downloaded 6f72ec86 Swiggy screenshots for live verifier replay.
    """

    ASSET_ROOT = Path(
        "/Users/aman/Desktop/Drizz/fathom/debug/"
        "6f72ec86-5952-4850-add9-631b4858c6aa/assets/screenshot/"
        "6f72ec86-5952-4850-add9-631b4858c6aa"
    )
    MODAL_SCREENSHOT = ASSET_ROOT / "step-005__screenshot__2026-06-11T06-28-10Z-692.png"
    COMPLETED_SCREENSHOT = ASSET_ROOT / "step-006__screenshot__2026-06-11T06-28-25Z-609.png"

    @classmethod
    def is_available(cls) -> bool:
        """
        Return whether the downloaded prod replay screenshots are present locally.
        """

        return cls.MODAL_SCREENSHOT.exists() and cls.COMPLETED_SCREENSHOT.exists()

    @classmethod
    def captures(cls) -> List[ScreenCapture]:
        """
        Return modal and completed captures from the downloaded prod artifacts.
        """

        return [
            cls.__capture(path=cls.MODAL_SCREENSHOT, timestamp=1781159290692),
            cls.__capture(path=cls.COMPLETED_SCREENSHOT, timestamp=1781159305609),
        ]

    @staticmethod
    def __capture(*, path: Path, timestamp: int) -> ScreenCapture:
        """
        Build a ScreenCapture from a downloaded screenshot path.
        """

        image = Image.open(path)
        try:
            width, height = image.size
        finally:
            image.close()

        return ScreenCapture(
            width=width,
            height=height,
            image=path.read_bytes(),
            timestamp=timestamp,
            activity="in.swiggy.android",
        )


class _Perception:
    """
    Live verifier perception double backed by synthetic replay screenshots.
    """

    def __init__(self, *, captures: List[ScreenCapture]) -> None:
        """
        Bind the ordered capture sequence used by VERIFY turns.
        """

        self.__captures = list(captures)

    async def perceive(self, **_: object) -> ScreenCapture:
        """
        Return the next replay capture.
        """

        if len(self.__captures) > 1:
            return self.__captures.pop(0)

        return self.__captures[0]


class _ContextManager:
    """
    Captures verifier feedback emitted by the live verifier.
    """

    def __init__(self) -> None:
        """
        Initialise feedback capture.
        """

        self.feedback: List[str] = []
        self.cleared = False
        self.commits: List[str] = []

    def get_user_guidance(self) -> List[object]:
        """
        Return no operator guidance for this replay.
        """

        return []

    def get_full_context(self) -> Dict[str, object]:
        """
        Return empty trace context.
        """

        return {"trace": []}

    async def commit(self, *, observation: str, action: Action, thought: str) -> None:
        """
        Capture the corrective action committed by RECORD.
        """

        _ = action, thought
        self.commits.append(observation)

    async def inject_verifier_feedback(self, *, feedback: str, step: int | None = None) -> None:
        """
        Capture verifier feedback for assertions.
        """

        _ = step
        self.feedback.append(feedback)

    def clear_verifier_feedback(self) -> None:
        """
        Capture feedback clearing after accepted verification.
        """

        self.cleared = True


class _Persistence:
    """
    Persistence double for the live verifier node.
    """

    def restore(self, *, state: Dict[object, object]) -> None:
        """
        Keep live replay state in memory.
        """

        _ = state

    def persist(self, *, result: Dict[object, object]) -> None:
        """
        Drop persisted graph patches.
        """

        _ = result


class _Provider:
    """
    Provider fixture binding real LLM and replay dependencies to VerifyNode.
    """

    def __init__(
        self,
        *,
        llm: LLMPort,
        agent_state: AgentState,
        captures: Optional[List[ScreenCapture]] = None,
        intent: str = "change the address to salary-se office",
        workflow_id: str = "live-salary-se-final-verify",
    ) -> None:
        """
        Bind the verifier provider surface.
        """

        self.persistence = _Persistence()
        self.context_manager = _ContextManager()
        self.completion = SimpleNamespace(evaluate=AsyncMock(return_value=None))

        self.context = SimpleNamespace(
            llm=llm,
            max_steps=20,
            artifact_pipeline=None,
            agent_state=agent_state,
            perception=_Perception(
                captures=captures
                or [SalarySeModalFixture.capture(), SalarySeModalFixture.completed_capture()]
            ),
            context_manager=self.context_manager,
            auditor=SimpleNamespace(log_step=MagicMock()),
            workflow_id=workflow_id,
            intent=intent,
            memory=SimpleNamespace(store_experience=AsyncMock()),
            telemetry=SimpleNamespace(info=AsyncMock(), warning=AsyncMock(), error=AsyncMock()),
        )
        self.persistence.should_skip_launcher = MagicMock(return_value=False)  # type: ignore[attr-defined]
        self.persistence.enqueue_history = MagicMock()  # type: ignore[attr-defined]

    async def is_cancelled(self) -> bool:
        """
        Keep the live replay on the normal non-cancelled path.
        """

        return False


class TestSalarySeFinalVerifyLive:
    """
    Live Gemini replay for the final-subgoal VERIFY rejection case.
    """

    @staticmethod
    def __agent_state() -> AgentState:
        """
        Build AgentState with the final SalarySe sub-goal active.
        """

        agent_state = AgentState(
            intent="change the address to salary-se office",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        agent_state.update_screen(
            screen=ScreenState(
                timestamp=1,
                visual_hash="1" * 16,
                activity_hash="a" * 16,
                activity="in.swiggy.android/.HomeActivity",
            )
        )
        agent_state.set_sub_goals(
            [
                SubGoalFixtures.make(index=0, description="Tap address selector"),
                SubGoalFixtures.make(index=1, description="Confirm SalarySe office address"),
            ]
        )
        agent_state.advance_current_sub_goal()
        return agent_state

    @staticmethod
    def __hsr_agent_state() -> AgentState:
        """
        Build AgentState at the 6f72ec86 final-confirmation handoff.
        """

        agent_state = AgentState(
            intent="Change the address to HSR Layout",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        agent_state.update_screen(
            screen=ScreenState(
                timestamp=1,
                visual_hash="1" * 16,
                activity_hash="a" * 16,
                activity="in.swiggy.android",
            )
        )
        agent_state.set_sub_goals(
            [
                SubGoalFixtures.make(index=0, description="Tap on the current address or change address option"),
                SubGoalFixtures.make(index=1, description="Type HSR Layout into the address search field"),
                SubGoalFixtures.make(index=2, description="Tap HSR Layout from the search results"),
                SubGoalFixtures.make(index=3, description="Tap the button to confirm or save the address change"),
            ]
        )
        for _ in range(3):
            agent_state.advance_current_sub_goal()
        return agent_state

    @staticmethod
    def __corrective_step_result() -> StepResult:
        """
        Build the corrective action that dismisses the modal after verifier rejection.
        """

        action = Action(
            confidence=1.0,
            action_type=ActionType.TAP,
            target="Yes, continue",
            rationale="Dismiss the confirmation modal before validating the selected address",
        )
        return StepResult(
            success=True,
            duration=25,
            pre_hash="modal",
            post_hash="selected",
            screen_changed=True,
            step=Step(action=action, step_number=1, screen_hash="modal"),
        )

    @staticmethod
    def __completed_screen_state() -> ScreenState:
        """
        Build the post-correction screen state consumed by RECORD.
        """

        return ScreenState(
            timestamp=2,
            visual_hash="2" * 16,
            activity_hash="a" * 16,
            activity="in.swiggy.android/.HomeActivity",
        )

    @staticmethod
    def __hsr_corrective_step_result() -> StepResult:
        """
        Build the corrective HSR modal action from the prod run.
        """

        action = Action(
            confidence=1.0,
            action_type=ActionType.TAP,
            target="Yes, continue with this location",
            rationale="Tap the confirmation modal action before validating HSR Layout",
        )
        return StepResult(
            success=True,
            duration=6749,
            pre_hash="f75dfff77f7ff7df",
            post_hash="ae41df2b4e7fb7df",
            screen_changed=True,
            step=Step(action=action, step_number=6, screen_hash="f75dfff77f7ff7df"),
        )

    def __record_state(self) -> Dict[object, object]:
        """
        Build the graph patch that RECORD sees after the corrective modal action.
        """

        return {
            CommonStateKey.IS_NEW_SCREEN: True,
            CommonStateKey.SCREEN_STATE: self.__completed_screen_state(),
            IntentStateKey.POST_ACTIVITY: "in.swiggy.android/.HomeActivity",
            CommonStateKey.STEP_RESULT: self.__corrective_step_result(),
            IntentStateKey.PLAN: PlanResult(
                step=None,
                is_complete=True,
                reason="SalarySe office address is selected",
            ),
        }

    def __hsr_record_state(self) -> Dict[object, object]:
        """
        Build the graph patch that RECORD sees after tapping the HSR confirmation action.
        """

        return {
            CommonStateKey.IS_NEW_SCREEN: True,
            CommonStateKey.SCREEN_STATE: ScreenState(
                timestamp=2,
                visual_hash="ae41df2b4e7fb7df",
                activity_hash="a" * 16,
                activity="in.swiggy.android",
            ),
            IntentStateKey.POST_ACTIVITY: "in.swiggy.android",
            CommonStateKey.STEP_RESULT: self.__hsr_corrective_step_result(),
            IntentStateKey.PLAN: PlanResult(
                step=None,
                is_complete=True,
                reason="HSR Layout is selected in the delivery address header",
            ),
        }

    async def test_modal_blocker_rejection_keeps_final_subgoal_active(self, llm: LLMPort) -> None:
        """
        Real verifier should reject final completion while the confirmation modal is still visible.
        """

        agent_state = self.__agent_state()
        recording_llm = _RecordingLlm(llm=llm)
        provider = _Provider(llm=recording_llm, agent_state=agent_state)
        node = VerifyNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(
            state={IntentStateKey.VERIFY_MODE: VerifyMode.PENDING_FINAL_COMMIT.value}
        )  # type: ignore[arg-type]

        assert result[IntentStateKey.VERIFY_MODE] is None
        assert result[CommonStateKey.IS_COMPLETE] is False
        assert result[IntentStateKey.SHOULD_RETRY] is True

        assert agent_state.is_complete is False
        assert provider.context_manager.feedback
        assert agent_state.current_sub_goal_index == 1
        _print_llm_responses(responses=recording_llm.responses)

    async def test_modal_rejection_then_corrective_record_completes_intent(
        self, llm: LLMPort
    ) -> None:
        """
        Real verifier should reject the modal, then accept after the corrective action is recorded.
        """

        agent_state = self.__agent_state()
        recording_llm = _RecordingLlm(llm=llm)
        provider = _Provider(llm=recording_llm, agent_state=agent_state)
        verify = VerifyNode(provider=provider)  # type: ignore[arg-type]
        record = RecordNode(provider=provider)  # type: ignore[arg-type]

        rejected = await verify.run(
            state={IntentStateKey.VERIFY_MODE: VerifyMode.PENDING_FINAL_COMMIT.value}
        )  # type: ignore[arg-type]

        assert rejected[CommonStateKey.IS_COMPLETE] is False
        assert rejected[IntentStateKey.SHOULD_RETRY] is True
        assert agent_state.is_complete is False
        assert provider.context_manager.feedback

        pending_again = await record.run(state=self.__record_state())  # type: ignore[arg-type]

        assert pending_again[CommonStateKey.IS_COMPLETE] is True
        assert pending_again[IntentStateKey.SHOULD_RETRY] is False
        assert pending_again[IntentStateKey.VERIFY_MODE] == VerifyMode.PENDING_FINAL_COMMIT.value
        assert agent_state.is_complete is False
        assert agent_state.verification_loop is None

        accepted = await verify.run(state=pending_again)  # type: ignore[arg-type]

        assert accepted[CommonStateKey.IS_COMPLETE] is True
        assert accepted[IntentStateKey.SHOULD_RETRY] is False
        assert accepted[IntentStateKey.VERIFY_MODE] is None
        assert agent_state.is_complete is True
        assert agent_state.all_sub_goals_complete() is True
        assert provider.context_manager.cleared is True
        _print_llm_responses(responses=recording_llm.responses)

    async def test_hsr_prod_artifact_replay_rejects_then_completes(self, llm: LLMPort) -> None:
        """
        Live Gemini replay of 6f72ec86 using the downloaded modal and post-correction screenshots.
        """

        if not HsrProdAssetFixture.is_available():
            pytest.skip("6f72ec86 downloaded screenshots are not available in debug assets")

        agent_state = self.__hsr_agent_state()
        recording_llm = _RecordingLlm(llm=llm)
        provider = _Provider(
            llm=recording_llm,
            agent_state=agent_state,
            captures=HsrProdAssetFixture.captures(),
            intent="Change the address to HSR Layout",
            workflow_id="live-6f72ec86-hsr-final-verify",
        )
        verify = VerifyNode(provider=provider)  # type: ignore[arg-type]
        record = RecordNode(provider=provider)  # type: ignore[arg-type]

        rejected = await verify.run(
            state={IntentStateKey.VERIFY_MODE: VerifyMode.PENDING_FINAL_COMMIT.value}
        )  # type: ignore[arg-type]

        assert rejected[CommonStateKey.IS_COMPLETE] is False
        assert rejected[IntentStateKey.SHOULD_RETRY] is True
        assert agent_state.current_sub_goal_index == 3
        assert provider.context_manager.feedback

        pending_again = await record.run(state=self.__hsr_record_state())  # type: ignore[arg-type]

        assert pending_again[CommonStateKey.IS_COMPLETE] is True
        assert pending_again[IntentStateKey.SHOULD_RETRY] is False
        assert pending_again[IntentStateKey.VERIFY_MODE] == VerifyMode.PENDING_FINAL_COMMIT.value
        assert agent_state.current_sub_goal_index == 3
        assert agent_state.is_complete is False

        accepted = await verify.run(state=pending_again)  # type: ignore[arg-type]

        assert accepted[CommonStateKey.IS_COMPLETE] is True
        assert accepted[IntentStateKey.SHOULD_RETRY] is False
        assert accepted[IntentStateKey.VERIFY_MODE] is None
        assert agent_state.is_complete is True
        assert agent_state.all_sub_goals_complete() is True
        assert provider.context_manager.cleared is True
        _print_llm_responses(responses=recording_llm.responses)
