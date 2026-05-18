from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import AsyncMock

from fathom.constants import ActionType
from fathom.core.services.hitl import HITLService
from fathom.schemas.actions import Action
from fathom.schemas.steps import Step
from fathom.strategies.graph.intent.nodes.hitl import Hitl


class _FakeHitlService(HITLService):
    """
    :class:`HITLService` test double driving pause/resume and ASK_USER.

    The real :class:`HITLService` reaches Redis-or-equivalent. This
    double records every method call so the test can assert exact
    sequencing — pause → resume → drain — without standing up a
    transport.
    """

    def __init__(
        self,
        *,
        pause_requested: bool = False,
        injected_contexts: Optional[List[str]] = None,
        ask_response: str = "human answer",
    ) -> None:
        """
        Pre-seed the double with pause state, queued injected contexts
        (drained on resume), and the response :meth:`ask` returns.
        """

        # Intentionally skip super().__init__() — the real service holds
        # a Redis client we cannot construct here. The test only drives
        # the public coroutine surface.
        self.__pause_requested = pause_requested
        self.__injected = list(injected_contexts or [])
        self.__ask_response = ask_response
        self.resume_calls: int = 0
        self.consume_calls: int = 0
        self.ask_calls: List[str] = []

    async def is_pause_requested(self) -> bool:
        """
        Single source of truth for the paused state. Tests flip this
        directly via the constructor argument.
        """

        return self.__pause_requested

    async def wait_for_resume(self) -> None:
        """
        Mark the resume as observed and clear the pause flag so any
        subsequent ``is_pause_requested`` call returns ``False``.
        """

        self.resume_calls += 1
        self.__pause_requested = False

    async def has_injected_context(self) -> bool:
        """
        Yield True until the queued context list is drained.
        """

        return bool(self.__injected)

    async def peek_next_context(self) -> Optional[str]:
        """
        Peek without consuming — :class:`Hitl.__drain_context` decides
        whether to consume each entry.
        """

        return self.__injected[0] if self.__injected else None

    async def consume_context(self) -> None:
        """
        Drop the head of the injected-context queue and tick the
        consume counter for the test to assert against.
        """

        if self.__injected:
            self.__injected.pop(0)
            self.consume_calls += 1

    async def ask(self, *, prompt: str, step: int) -> str:
        """
        Record the prompt for the test to assert against and return the
        preconfigured human response.
        """

        _ = step
        self.ask_calls.append(prompt)
        return self.__ask_response

    async def check_signal(self) -> str:
        """
        Cancellation signal stays clean for these tests.
        """

        return ""


class HitlPromptTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins :meth:`Hitl.prompt` pause / resume / drain orchestration.

    ``prompt`` is the gate ANALYZE calls before planning. It must
    short-circuit when the HITL service is not present or no pause was
    requested, and otherwise wait for resume then drain every queued
    injected context as user guidance.
    """

    @staticmethod
    def __context(*, hitl: object, step_count: int = 0) -> SimpleNamespace:
        """
        :class:`GraphContext` stub exposing the four attributes
        :meth:`Hitl.prompt` and its drain helper actually consume.
        """

        return SimpleNamespace(
            hitl=hitl,
            workflow_id="run-test",
            agent_state=SimpleNamespace(
                step_count=step_count,
                record_hitl_intervention=lambda: None,
            ),
            context_manager=SimpleNamespace(
                inject_user_guidance=AsyncMock(),
            ),
        )

    async def test_no_hitl_service_short_circuits(self) -> None:
        """
        If the context's ``hitl`` is not an :class:`HITLService`, the
        prompt returns immediately without inspecting any other state.
        """

        helper = Hitl(context=self.__context(hitl=object()))  # type: ignore[arg-type]

        # Must not raise even though the placeholder hitl has none of
        # the service methods.
        await helper.prompt(step=0)

    async def test_no_pause_requested_short_circuits(self) -> None:
        """
        If the service reports no pause, the prompt must not wait for
        resume or drain any injected context.
        """

        fake = _FakeHitlService(pause_requested=False)
        helper = Hitl(context=self.__context(hitl=fake))  # type: ignore[arg-type]

        await helper.prompt(step=0)

        self.assertEqual(fake.resume_calls, 0)
        self.assertEqual(fake.consume_calls, 0)

    async def test_paused_run_waits_for_resume_and_drains_contexts(self) -> None:
        """
        A paused run must wait for resume and then drain every queued
        injected context, calling ``inject_user_guidance`` once per
        entry. The consume count must equal the queue length.
        """

        fake = _FakeHitlService(
            pause_requested=True,
            injected_contexts=["first hint", "second hint"],
        )
        context_manager = SimpleNamespace(inject_user_guidance=AsyncMock())
        recorded_interventions: List[bool] = []
        agent_state = SimpleNamespace(
            step_count=0,
            record_hitl_intervention=lambda: recorded_interventions.append(True),
        )
        ctx = SimpleNamespace(
            hitl=fake,
            workflow_id="run-test",
            agent_state=agent_state,
            context_manager=context_manager,
        )
        helper = Hitl(context=ctx)  # type: ignore[arg-type]

        await helper.prompt(step=0)

        self.assertEqual(fake.resume_calls, 1)
        self.assertEqual(fake.consume_calls, 2)
        self.assertEqual(context_manager.inject_user_guidance.await_count, 2)
        # Each consumed context records one HITL intervention so the
        # realignment budget tracks them.
        self.assertEqual(len(recorded_interventions), 2)


class HitlAskTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins :meth:`Hitl.ask` ASK_USER turn handling.

    ``ask`` runs when EXECUTE encounters an ASK_USER action: it sends
    the action's text (or the default prompt) to the HITL service,
    routes the response as user guidance, records the intervention,
    and returns a synthetic successful :class:`ExecutionResult`.
    """

    @staticmethod
    def __action(*, text: Optional[str]) -> Action:
        """
        :class:`Action` fixture parameterised on the question text so
        the default-prompt fall-back can be driven by passing ``None``.
        """

        return Action(
            action_type=ActionType.ASK_USER,
            target="x",
            text=text,
            rationale="t",
            confidence=1.0,
        )

    @staticmethod
    def __step(*, action: Action) -> Step:
        """
        :class:`Step` wrapping the supplied action. Step number and
        condition are arbitrary placeholders.
        """

        return Step(
            action=action,
            event_type="action",
            condition="ask",
            screen_hash="0" * 16,
            step_number=1,
        )

    @staticmethod
    def __context(*, hitl: _FakeHitlService) -> SimpleNamespace:
        """
        Context stub exposing the four attributes :meth:`Hitl.ask`
        actually consumes.
        """

        interventions: List[bool] = []
        return SimpleNamespace(
            hitl=hitl,
            workflow_id="run-test",
            agent_state=SimpleNamespace(
                step_count=0,
                record_hitl_intervention=lambda: interventions.append(True),
            ),
            context_manager=SimpleNamespace(
                inject_user_guidance=AsyncMock(),
            ),
            _interventions=interventions,
        )

    async def test_ask_with_action_text_uses_action_text_as_prompt(self) -> None:
        """
        When the action carries explicit text, that text becomes the
        prompt sent to the HITL service — not the default fall-back.
        """

        fake = _FakeHitlService(ask_response="ok")
        ctx = self.__context(hitl=fake)
        helper = Hitl(context=ctx)  # type: ignore[arg-type]

        result = await helper.ask(
            step=self.__step(action=self.__action(text="What size?")),
            start_time=0.0,
        )

        self.assertEqual(fake.ask_calls, ["What size?"])
        self.assertTrue(result.success)

    async def test_ask_without_action_text_uses_default_prompt(self) -> None:
        """
        An action without text falls back to :data:`HITL_DEFAULT_PROMPT`
        so the human still sees a meaningful question.
        """

        fake = _FakeHitlService(ask_response="ok")
        helper = Hitl(context=self.__context(hitl=fake))  # type: ignore[arg-type]

        await helper.ask(
            step=self.__step(action=self.__action(text=None)),
            start_time=0.0,
        )

        self.assertEqual(len(fake.ask_calls), 1)
        self.assertTrue(fake.ask_calls[0])  # non-empty default prompt

    async def test_ask_injects_response_as_user_guidance(self) -> None:
        """
        The HITL response must be routed through
        :meth:`context_manager.inject_user_guidance` so the next
        ANALYZE turn sees it as authoritative guidance.
        """

        fake = _FakeHitlService(ask_response="Medium")
        ctx = self.__context(hitl=fake)
        helper = Hitl(context=ctx)  # type: ignore[arg-type]

        await helper.ask(
            step=self.__step(action=self.__action(text="Size?")),
            start_time=0.0,
        )

        ctx.context_manager.inject_user_guidance.assert_awaited_once()
        kwargs = ctx.context_manager.inject_user_guidance.await_args.kwargs
        self.assertEqual(kwargs["guidance"], "Medium")

    async def test_ask_records_hitl_intervention_for_realignment_budget(self) -> None:
        """
        Every ASK_USER turn must tick the HITL-intervention counter so
        the realignment budget tracks human turns alongside model turns.
        """

        fake = _FakeHitlService(ask_response="ok")
        ctx = self.__context(hitl=fake)
        helper = Hitl(context=ctx)  # type: ignore[arg-type]

        await helper.ask(
            step=self.__step(action=self.__action(text="x")),
            start_time=0.0,
        )

        self.assertEqual(len(ctx._interventions), 1)
