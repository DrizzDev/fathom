from __future__ import annotations

import io
import time
from typing import Any, Dict, List

import pytest
from PIL import Image, ImageDraw

from fathom.constants.retries import RetryBranch, RetryKind
from fathom.constants.tools import BASE_TOOLS
from fathom.core.agent.state import AgentState
from fathom.core.context.manager import ContextManager
from fathom.core.prompts.rejection import RepeatedFailureRejectionPromptBuilder
from fathom.core.services.vision import VisionService
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.conversation import ConversationTurn, TurnPart
from fathom.schemas.retries import RetryLimits
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.tools import AllowedTools

pytestmark = pytest.mark.release


class _NoopTelemetry(TelemetryPort):
    """
    Telemetry sink for live planner-retry tests.
    """

    async def debug(self, message: str, **context: Any) -> None:
        """
        Drop debug telemetry.
        """

        _ = message, context

    async def info(self, message: str, **context: Any) -> None:
        """
        Drop info telemetry.
        """

        _ = message, context

    async def warning(self, message: str, **context: Any) -> None:
        """
        Drop warning telemetry.
        """

        _ = message, context

    async def error(self, message: str, **context: Any) -> None:
        """
        Drop error telemetry.
        """

        _ = message, context

    async def exception(
        self,
        message: str,
        *,
        exception: BaseException | None = None,
        **context: Any,
    ) -> None:
        """
        Drop exception telemetry.
        """

        _ = message, exception, context


class SettingsScreenFixture:
    """
    Synthetic settings screen with two distinct, unrelated actions so the LLM has a real second choice.
    """

    @staticmethod
    def capture() -> ScreenCapture:
        """
        Render a synthetic settings screen offering two unrelated controls.
        """

        image = Image.new("RGB", (402, 874), "white")

        draw = ImageDraw.Draw(image)
        draw.text((150, 60), "Settings", fill="black")

        # Button 1: "Logout" — the action we'll mark as already-failed.
        draw.rectangle((38, 200, 364, 260), outline="#cc2222", width=2)
        draw.text((170, 222), "Logout", fill="#cc2222")

        # Button 2: "Edit profile" — the unrelated alternative.
        draw.rectangle((38, 300, 364, 360), outline="#1167ff", width=2)
        draw.text((150, 322), "Edit profile", fill="#1167ff")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return ScreenCapture(
            width=402,
            height=874,
            image=buffer.getvalue(),
            activity="com.example/.Settings",
            timestamp=int(time.time() * 1000),
        )

    @staticmethod
    def elements() -> Dict[str, Dict[str, str]]:
        """
        Manifest matching the two buttons on the synthetic screen.
        """

        return {
            "1": {
                "label": "Logout",
                "bounds": "[38,200][364,260]",
                "type": "XCUIElementTypeButton",
            },
            "2": {
                "label": "Edit profile",
                "bounds": "[38,300][364,360]",
                "type": "XCUIElementTypeButton",
            },
        }


class TestPlannerRejectionFeedbackLive:
    """
    Verifies against a real Gemini that ``RepeatedFailureRejectionPromptBuilder`` actually steers the model away from the rejected action on the next turn.
    """

    @staticmethod
    def __capabilities() -> RuntimeCapabilities:
        """
        Build non-interactive capabilities; live tests dispatch without HITL.
        """

        return RuntimeCapabilities(hitl=HITLCapability(enabled=False))

    @staticmethod
    def __tools() -> AllowedTools:
        """
        Canonical base tools used by the live planner-retry tests.
        """

        return AllowedTools(names=BASE_TOOLS)

    @staticmethod
    def __rejection_history(*, rejection_reason: str) -> List[ConversationTurn]:
        """
        Build a minimal two-turn rejection_history: model proposed Tap Logout, system rejected with the prompt.
        """

        model_turn = ConversationTurn(
            role="model",
            parts=[TurnPart.from_function_call(name="execute_ui", args={"action": "tap Logout"})],
        )
        user_turn = ConversationTurn(
            role="user",
            parts=[TurnPart.from_text(text=rejection_reason)],
        )
        return [model_turn, user_turn]

    async def test_rejected_action_is_not_re_emitted(
        self,
        llm: LLMPort,
        memory_port_stub: MemoryPort,
    ) -> None:
        """
        When the prior turn rejected ``Tap Logout`` with the production rejection sentence, Gemini must propose a different action (Edit profile) on the next turn; otherwise the rejection_history feedback loop is broken.
        """

        rejection_reason = RepeatedFailureRejectionPromptBuilder.build(
            interactive=False,
            action_descriptor="Tap on Logout",
        )
        history = self.__rejection_history(rejection_reason=rejection_reason)

        service = VisionService(
            llm=llm,
            use_cache=False,
            memory=memory_port_stub,
            telemetry=_NoopTelemetry(),
            session_id="live-rejection-loop",
            capabilities=self.__capabilities(),
        )
        context_manager = ContextManager(
            memory=memory_port_stub,
            workflow_id="live-rejection-loop",
        )

        analysis = await service.analyze(
            use_xml=True,
            screen_width=402,
            screen_height=874,
            tools=self.__tools(),
            visual_hash="abcdef1234567890",
            context_manager=context_manager,
            prior_rejection_history=history,
            intent="Open the profile edit screen.",
            capture=SettingsScreenFixture.capture(),
            elements=SettingsScreenFixture.elements(),
        )

        # The agent should NOT re-emit Tap Logout — that's what the rejection_history forbids.
        target = (analysis.action.target or "").casefold()
        descriptor = analysis.action.natural_language_target or ""

        self.assertNotIn("logout", descriptor.casefold())
        self.assertNotIn("logout", target, msg=f"LLM re-emitted rejected action: {target!r}")

    async def test_silent_rejection_loop_bounded_against_real_llm(
        self,
        llm: LLMPort,
        memory_port_stub: MemoryPort,
    ) -> None:
        """
        Even if the LLM ignores the rejection_history (best-effort signal only), the AgentState planner-retry budget must terminate the workflow at ``cap`` iterations — this is the production-defense backstop that bounded PFTXN's runaway loop.
        """

        state = AgentState(
            max_steps=10,
            retries=RetryLimits(planner=5),
            intent="Open the profile edit screen.",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        rejection_reason = RepeatedFailureRejectionPromptBuilder.build(
            interactive=False,
            action_descriptor="Tap on Logout",
        )
        history = self.__rejection_history(rejection_reason=rejection_reason)

        service = VisionService(
            llm=llm,
            use_cache=False,
            memory=memory_port_stub,
            telemetry=_NoopTelemetry(),
            capabilities=self.__capabilities(),
            session_id="live-rejection-budget",
        )
        context_manager = ContextManager(
            memory=memory_port_stub,
            workflow_id="live-rejection-budget",
        )

        for _ in range(state.retries.planner.cap):
            await service.analyze(
                use_xml=True,
                screen_width=402,
                screen_height=874,
                tools=self.__tools(),
                visual_hash="abcdef1234567890",
                context_manager=context_manager,
                prior_rejection_history=history,
                intent="Open the profile edit screen.",
                capture=SettingsScreenFixture.capture(),
                elements=SettingsScreenFixture.elements(),
            )
            state.tick_planner_retry(
                action="Tap on Logout",
                kind=RetryKind.SILENT_REJECTION,
                branch=RetryBranch.SHOULD_AVOID_ACTION,
            )

        self.assertEqual(state.step_count, 0)
        self.assertTrue(state.retries.planner.exhausted)

    def assertNotIn(self, member: object, container: object, msg: object = None) -> None:
        """
        Mirror unittest's assertNotIn so this fixture class can run under plain pytest collection.
        """

        if member in container:  # type: ignore[operator]
            raise AssertionError(msg or f"{member!r} unexpectedly found in {container!r}")

    def assertTrue(self, expression: object, msg: object = None) -> None:
        """
        Mirror unittest's assertTrue.
        """

        if not expression:
            raise AssertionError(msg or f"{expression!r} is not truthy")

    def assertEqual(self, first: object, second: object, msg: object = None) -> None:
        """
        Mirror unittest's assertEqual.
        """

        if first != second:
            raise AssertionError(msg or f"{first!r} != {second!r}")
