from __future__ import annotations

import asyncio
import time
import unittest
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import AsyncMock

from tests.builders.agent import AgentFixtures

from fathom.constants import ActionType
from fathom.constants.agent import DirectiveKind
from fathom.constants.state import CompletionReason
from fathom.core.exceptions import WorkflowCancelledError
from fathom.core.services.hitl import HITLService
from fathom.interfaces.abort import AbortDetectorPort
from fathom.schemas.abort import AbortDecision
from fathom.schemas.actions import Action
from fathom.schemas.steps import Step
from fathom.strategies.graph.intent.nodes.hitl import Hitl


class _StubAborter(AbortDetectorPort):
    """
    Configurable test double for the abort detector port.
    """

    def __init__(
        self,
        *,
        aborted: bool = False,
        confidence: float = 0.0,
        fallback: bool = False,
    ) -> None:
        """
        Bind the scripted decision returned from every call to :meth:`aborted`.
        """

        self.__decision = AbortDecision(aborted=aborted, confidence=confidence, fallback=fallback)
        self.calls: List[str] = []
        self.warmup_calls: int = 0

    async def aborted(self, *, response: str) -> AbortDecision:
        """
        Record the inspected response and return the scripted decision.
        """

        self.calls.append(response)
        return self.__decision

    async def warmup(self) -> None:
        """
        Record the warmup invocation; no underlying resource to prime.
        """

        self.warmup_calls += 1


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


class _BlockingHitlService(_FakeHitlService):
    """
    ASK_USER double that blocks until cancelled by the test subject.
    """

    def __init__(self) -> None:
        super().__init__(pause_requested=True)
        self.ask_cancelled = False
        self.resume_cancelled = False

    async def ask(self, *, prompt: str, step: int) -> str:
        """
        Block forever unless the helper cancels the in-flight ask task.
        """

        _ = prompt, step
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.ask_cancelled = True
            raise

    async def wait_for_resume(self) -> None:
        """
        Block forever unless the helper cancels the in-flight resume task.
        """

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.resume_cancelled = True
            raise


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
            is_cancelled=False,
            agent_state=SimpleNamespace(
                step_count=step_count,
                record_hitl_intervention=lambda: None,
            ),
            context_manager=AgentFixtures.context_manager(),
        )

    async def test_no_hitl_service_short_circuits(self) -> None:
        """
        If the context's ``hitl`` is not an :class:`HITLService`, the
        prompt returns immediately without inspecting any other state.
        """

        helper = Hitl(context=self.__context(hitl=object()), aborter=_StubAborter())  # type: ignore[arg-type]

        # Must not raise even though the placeholder hitl has none of
        # the service methods.
        await helper.prompt(step=0)

    async def test_no_pause_requested_short_circuits(self) -> None:
        """
        If the service reports no pause, the prompt must not wait for
        resume or drain any injected context.
        """

        fake = _FakeHitlService(pause_requested=False)
        helper = Hitl(context=self.__context(hitl=fake), aborter=_StubAborter())  # type: ignore[arg-type]

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
            is_cancelled=False,
            agent_state=agent_state,
            context_manager=context_manager,
        )
        helper = Hitl(context=ctx, aborter=_StubAborter())  # type: ignore[arg-type]

        await helper.prompt(step=0)

        self.assertEqual(fake.resume_calls, 1)
        self.assertEqual(fake.consume_calls, 2)
        self.assertEqual(context_manager.inject_user_guidance.await_count, 2)
        # Each consumed context records one HITL intervention so the
        # realignment budget tracks them.
        self.assertEqual(len(recorded_interventions), 2)

    async def test_paused_run_stops_waiting_when_context_is_cancelled(self) -> None:
        """
        Ctrl-C cancellation must interrupt an in-flight pause/resume wait.
        """

        fake = _BlockingHitlService()
        ctx = self.__context(hitl=fake)
        helper = Hitl(context=ctx, aborter=_StubAborter())  # type: ignore[arg-type]

        async def _cancel_soon() -> None:
            await asyncio.sleep(0.15)
            ctx.is_cancelled = True

        asyncio.create_task(_cancel_soon())

        with self.assertRaises(WorkflowCancelledError):
            await helper.prompt(step=0)

        self.assertTrue(fake.resume_cancelled)
        ctx.context_manager.inject_user_guidance.assert_not_awaited()


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
        directives: List[SimpleNamespace] = []

        def __set_directive(*, kind, source_text, target_descriptor=None, ttl_turns=2):
            """
            Stand-in for ``context.set_active_directive`` that records each call.
            """

            directive = SimpleNamespace(
                kind=kind,
                ttl_turns=ttl_turns,
                source_text=source_text,
                target_descriptor=target_descriptor,
            )
            directives.append(directive)

            return directive

        return SimpleNamespace(
            hitl=hitl,
            is_cancelled=False,
            workflow_id="run-test",
            agent_state=SimpleNamespace(
                step_count=0,
                set_operator_directive=__set_directive,
                record_hitl_intervention=lambda: interventions.append(True),
            ),
            context_manager=AgentFixtures.context_manager(),
            _directives=directives,
            _interventions=interventions,
        )

    async def test_ask_with_action_text_uses_action_text_as_prompt(self) -> None:
        """
        When the action carries explicit text, that text becomes the
        prompt sent to the HITL service — not the default fall-back.
        """

        fake = _FakeHitlService(ask_response="ok")
        ctx = self.__context(hitl=fake)
        helper = Hitl(context=ctx, aborter=_StubAborter())  # type: ignore[arg-type]

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
        helper = Hitl(context=self.__context(hitl=fake), aborter=_StubAborter())  # type: ignore[arg-type]

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
        helper = Hitl(context=ctx, aborter=_StubAborter())  # type: ignore[arg-type]

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
        helper = Hitl(context=ctx, aborter=_StubAborter())  # type: ignore[arg-type]

        await helper.ask(
            step=self.__step(action=self.__action(text="x")),
            start_time=0.0,
        )

        self.assertEqual(len(ctx._interventions), 1)

    async def test_ask_stops_waiting_when_context_is_cancelled(self) -> None:
        """
        Ctrl-C cancellation must interrupt an in-flight ASK_USER wait.
        """

        fake = _BlockingHitlService()
        ctx = self.__context(hitl=fake)
        helper = Hitl(context=ctx, aborter=_StubAborter())  # type: ignore[arg-type]

        async def _cancel_soon() -> None:
            await asyncio.sleep(0.15)
            ctx.is_cancelled = True

        asyncio.create_task(_cancel_soon())

        with self.assertRaises(WorkflowCancelledError):
            await helper.ask(
                step=self.__step(action=self.__action(text="How should I proceed?")),
                start_time=0.0,
            )

        self.assertTrue(fake.ask_cancelled)
        ctx.context_manager.inject_user_guidance.assert_not_awaited()


class _SpyContext(SimpleNamespace):
    """
    Cancel-tracking :class:`GraphContext` double for the abort-routing tests.
    """

    def __init__(self, *, hitl: HITLService, workflow_id: str = "wf-test") -> None:
        """
        Build the spy with the bare-minimum surface the HITL ASK_USER path touches.
        """

        self.cancel_calls = 0
        self.directive_kinds: List[DirectiveKind] = []

        def record_directive(*, kind, source_text, target_descriptor=None, ttl_turns=2):
            """
            Capture every directive recorded by Hitl for the cancel-vs-record assertions.
            """

            _ = source_text
            self.directive_kinds.append(kind)
            return SimpleNamespace(
                kind=kind,
                ttl_turns=ttl_turns,
                target_descriptor=target_descriptor,
            )

        super().__init__(
            hitl=hitl,
            workflow_id=workflow_id,
            is_cancelled=False,
            agent_state=SimpleNamespace(
                step_count=3,
                record_hitl_intervention=lambda: None,
                set_operator_directive=record_directive,
            ),
            context_manager=AgentFixtures.context_manager(),
        )

    def cancel(self) -> None:
        """
        Track the cancel invocation and flip the cancellation flag.
        """

        self.cancel_calls += 1
        self.is_cancelled = True


def _ask_user_step() -> Step:
    """
    Build a minimal ASK_USER step fixture for the cancellation tests.
    """

    return Step(
        step_number=1,
        screen_hash="hash",
        action=Action(
            target="user",
            confidence=1.0,
            rationale="ask",
            action_type=ActionType.ASK_USER,
            text="Are you sure?",
        ),
    )


class HitlAbortCancellationTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins :meth:`Hitl.ask` cancel-on-abort behaviour driven by the injected detector.
    """

    async def test_abort_decision_cancels_context(self) -> None:
        """
        An aborted decision must invoke ``context.cancel`` exactly once.
        """

        service = _FakeHitlService(ask_response="close the execution")
        aborter = _StubAborter(aborted=True, confidence=0.95)
        context = _SpyContext(hitl=service)
        helper = Hitl(context=context, aborter=aborter)  # type: ignore[arg-type]

        with self.assertRaises(WorkflowCancelledError):
            await helper.ask(step=_ask_user_step(), start_time=time.time())

        self.assertEqual(context.cancel_calls, 1)
        self.assertTrue(context.is_cancelled)
        self.assertEqual(aborter.calls, ["close the execution"])

    async def test_abort_decision_raises_with_operator_aborted_reason(self) -> None:
        """
        The raised :class:`WorkflowCancelledError` carries the OPERATOR_ABORTED reason.
        """

        service = _FakeHitlService(ask_response="please stop the run")
        aborter = _StubAborter(aborted=True, confidence=0.9)
        context = _SpyContext(hitl=service)
        helper = Hitl(context=context, aborter=aborter)  # type: ignore[arg-type]

        with self.assertRaises(WorkflowCancelledError) as recorded:
            await helper.ask(step=_ask_user_step(), start_time=time.time())

        self.assertEqual(
            recorded.exception.reason,
            CompletionReason.OPERATOR_ABORTED.value,
        )
        self.assertEqual(recorded.exception.workflow_id, "wf-test")

    async def test_abort_decision_skips_directive_recording(self) -> None:
        """
        Abort must bypass the operator-directive recorder entirely.
        """

        service = _FakeHitlService(ask_response="cancel the workflow")
        aborter = _StubAborter(aborted=True)
        context = _SpyContext(hitl=service)
        helper = Hitl(context=context, aborter=aborter)  # type: ignore[arg-type]

        with self.assertRaises(WorkflowCancelledError):
            await helper.ask(step=_ask_user_step(), start_time=time.time())

        self.assertEqual(context.directive_kinds, [])


class HitlNonAbortResponseTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins :meth:`Hitl.ask` non-abort routing through the existing directive recorder.
    """

    async def test_non_abort_decision_runs_directive_recording(self) -> None:
        """
        A non-abort decision must allow the directive recorder to fire normally.
        """

        service = _FakeHitlService(ask_response="go to settings")
        aborter = _StubAborter(aborted=False)
        context = _SpyContext(hitl=service)
        helper = Hitl(context=context, aborter=aborter)  # type: ignore[arg-type]

        result = await helper.ask(step=_ask_user_step(), start_time=time.time())

        self.assertTrue(result.success)
        self.assertEqual(context.cancel_calls, 0)
        self.assertFalse(context.is_cancelled)

    async def test_retry_action_response_records_retry_directive(self) -> None:
        """
        A response with a retry prefix records DirectiveKind.RETRY_ACTION.
        """

        service = _FakeHitlService(ask_response="tap on Close button")
        aborter = _StubAborter(aborted=False)
        context = _SpyContext(hitl=service)
        helper = Hitl(context=context, aborter=aborter)  # type: ignore[arg-type]

        await helper.ask(step=_ask_user_step(), start_time=time.time())

        self.assertEqual(context.directive_kinds, [DirectiveKind.RETRY_ACTION])
        self.assertEqual(context.cancel_calls, 0)

    async def test_freeform_response_records_freeform_directive(self) -> None:
        """
        Free-form text records DirectiveKind.FREE_FORM without triggering cancellation.
        """

        service = _FakeHitlService(ask_response="the close X is in the top right corner")
        aborter = _StubAborter(aborted=False)
        context = _SpyContext(hitl=service)
        helper = Hitl(context=context, aborter=aborter)  # type: ignore[arg-type]

        await helper.ask(step=_ask_user_step(), start_time=time.time())

        self.assertEqual(context.directive_kinds, [DirectiveKind.FREE_FORM])
        self.assertEqual(context.cancel_calls, 0)
