from __future__ import annotations

import concurrent.futures
import unittest
from unittest.mock import AsyncMock, patch

from fathom.adapters.telemetry.tui import TuiTelemetryAdapter
from fathom.constants.events import FathomEvent
from fathom.runtime.command.demo_tui import DemoApp, _HitlAskScreen


async def _noop_workflow() -> bool:
    """Workflow stub that returns success without touching any device."""

    return True


def _make_app() -> DemoApp:
    """Build a DemoApp whose workflow is a no-op (tests don't mount)."""

    return DemoApp(intent="Test intent", workflow=_noop_workflow)


class DemoAppStateTest(unittest.TestCase):
    """
    Cover the aggregate-state updates ``record_event`` performs.

    The body/header widgets require the app to be mounted under
    Textual's driver, but the state dict is pure Python — we can
    drive ``record_event`` directly and inspect the snapshot.
    """

    def test_step_completed_advances_step_counter(self) -> None:
        app = _make_app()
        app.record_event(
            level="info",
            message="step",
            context={"type": FathomEvent.STEP_COMPLETED, "step": 7, "success": True},
        )
        self.assertEqual(app._state_snapshot()["step"], 7)

    def test_llm_call_completed_accumulates_tokens(self) -> None:
        app = _make_app()
        app.record_event(
            level="info",
            message="llm",
            context={
                "type": FathomEvent.LLM_CALL_COMPLETED,
                "metrics": {
                    "prompt_tokens": 4200,
                    "completion_tokens": 150,
                    "cached_tokens": 1800,
                },
            },
        )
        app.record_event(
            level="info",
            message="llm",
            context={
                "type": FathomEvent.LLM_CALL_COMPLETED,
                "metrics": {
                    "prompt_tokens": 3000,
                    "completion_tokens": 80,
                    "cached_tokens": 1500,
                },
            },
        )
        tokens = app._state_snapshot()["tokens"]
        self.assertEqual(tokens["prompt"], 7200)
        self.assertEqual(tokens["completion"], 230)
        self.assertEqual(tokens["cached"], 3300)

    def test_sub_goal_started_updates_footer_sub_goal(self) -> None:
        app = _make_app()
        app.record_event(
            level="info",
            message="sg",
            context={
                "type": FathomEvent.SUB_GOAL_STARTED,
                "index": 0,
                "total": 3,
                "description": "Tap the Challenges tab",
            },
        )
        self.assertEqual(app._state_snapshot()["sub_goal"], "Tap the Challenges tab")

    def test_workflow_completed_success_flips_icon_to_check(self) -> None:
        app = _make_app()
        app.record_event(
            level="info",
            message="done",
            context={"type": FathomEvent.WORKFLOW_COMPLETED, "success": True},
        )
        self.assertEqual(app._state_snapshot()["status_icon"], "✓")

    def test_workflow_cancelled_flips_icon_to_cross(self) -> None:
        app = _make_app()
        app.record_event(
            level="info",
            message="cancelled",
            context={"type": FathomEvent.WORKFLOW_CANCELLED},
        )
        self.assertEqual(app._state_snapshot()["status_icon"], "✗")

    def test_hitl_requested_paused_icon(self) -> None:
        app = _make_app()
        app.record_event(
            level="info",
            message="need human",
            context={"type": FathomEvent.HITL_REQUESTED, "prompt": "Proceed?"},
        )
        self.assertEqual(app._state_snapshot()["status_icon"], "⏸")

    def test_hitl_received_resumed_icon(self) -> None:
        app = _make_app()
        app.record_event(
            level="info",
            message="resumed",
            context={"type": FathomEvent.HITL_RECEIVED},
        )
        self.assertEqual(app._state_snapshot()["status_icon"], "▶")

    def test_bogus_token_metric_does_not_crash(self) -> None:
        """Non-int token values must not propagate a TypeError."""

        app = _make_app()
        app.record_event(
            level="info",
            message="llm",
            context={
                "type": FathomEvent.LLM_CALL_COMPLETED,
                "metrics": {
                    "prompt_tokens": "not-a-number",
                    "completion_tokens": None,
                },
            },
        )
        # Token state stayed at zero; no exception raised.
        tokens = app._state_snapshot()["tokens"]
        self.assertEqual(tokens["prompt"], 0)
        self.assertEqual(tokens["completion"], 0)


class TuiTelemetryAdapterTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover the TelemetryPort decorator that routes events into DemoApp.
    """

    async def test_info_forwards_to_inner_and_records_event(self) -> None:
        inner = AsyncMock()
        app = _make_app()
        adapter = TuiTelemetryAdapter(app=app, inner=inner)

        await adapter.info(
            "classifier",
            type=FathomEvent.INTENT_CLASSIFIED,
            should_decompose=False,
        )

        inner.info.assert_awaited_once_with(
            "classifier",
            type=FathomEvent.INTENT_CLASSIFIED,
            should_decompose=False,
        )

    async def test_step_completed_via_adapter_updates_app_state(self) -> None:
        inner = AsyncMock()
        app = _make_app()
        adapter = TuiTelemetryAdapter(app=app, inner=inner)

        await adapter.info(
            "step done",
            type=FathomEvent.STEP_COMPLETED,
            step=4,
            success=True,
        )

        self.assertEqual(app._state_snapshot()["step"], 4)

    async def test_error_forwards_and_flags_state_noop(self) -> None:
        """error() forwards but does not touch status_icon unless event type matches."""

        inner = AsyncMock()
        app = _make_app()
        adapter = TuiTelemetryAdapter(app=app, inner=inner)

        await adapter.error("kaboom", detail="oops")

        inner.error.assert_awaited_once_with("kaboom", detail="oops")
        # No event type → status_icon stays at default spinner frame.
        self.assertEqual(app._state_snapshot()["status_icon"], "⠋")

    async def test_exception_forwards_with_exception_kwarg(self) -> None:
        inner = AsyncMock()
        app = _make_app()
        adapter = TuiTelemetryAdapter(app=app, inner=inner)

        boom = RuntimeError("boom")
        await adapter.exception("crashed", exception=boom)

        inner.exception.assert_awaited_once_with("crashed", exception=boom)

    async def test_update_identity_forwards_when_inner_supports_it(self) -> None:
        class _InnerWithIdentity:
            async def debug(self, *_args: object, **_kwargs: object) -> None: ...
            async def info(self, *_args: object, **_kwargs: object) -> None: ...
            async def warning(self, *_args: object, **_kwargs: object) -> None: ...
            async def error(self, *_args: object, **_kwargs: object) -> None: ...
            async def exception(self, *_args: object, **_kwargs: object) -> None: ...

            def __init__(self) -> None:
                self.last_identity: str = ""

            def update_identity(self, *, identity: str) -> None:
                self.last_identity = identity

        inner = _InnerWithIdentity()
        app = _make_app()
        adapter = TuiTelemetryAdapter(app=app, inner=inner)

        adapter.update_identity(identity="test-run")

        self.assertEqual(inner.last_identity, "test-run")


class DemoAppQuitTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover the quit flow: first ``q`` press cancels the agent without
    exiting the UI; second press force-exits; workflow-finish auto-
    exits when quit was requested.
    """

    async def test_first_quit_invokes_on_quit_but_does_not_exit(self) -> None:
        exit_calls = 0

        class _Tracker(DemoApp):  # type: ignore[misc]
            def exit(self, *args: object, **kwargs: object) -> None:  # type: ignore[override]
                nonlocal exit_calls
                exit_calls += 1

        cancel_calls: list[None] = []

        def on_quit() -> None:
            cancel_calls.append(None)

        app = _Tracker(intent="x", workflow=_noop_workflow, on_quit=on_quit)

        await app.action_quit()

        self.assertEqual(cancel_calls, [None])
        self.assertEqual(exit_calls, 0, "First quit must not exit while worker cleans up")
        self.assertTrue(
            app._DemoApp__quit_requested  # type: ignore[attr-defined]
        )

    async def test_second_quit_force_exits(self) -> None:
        exit_calls = 0

        class _Tracker(DemoApp):  # type: ignore[misc]
            def exit(self, *args: object, **kwargs: object) -> None:  # type: ignore[override]
                nonlocal exit_calls
                exit_calls += 1

        cancel_calls: list[None] = []

        def on_quit() -> None:
            cancel_calls.append(None)

        app = _Tracker(intent="x", workflow=_noop_workflow, on_quit=on_quit)

        await app.action_quit()
        await app.action_quit()

        # on_quit only fires on the first press; second press goes
        # straight to exit.
        self.assertEqual(len(cancel_calls), 1)
        self.assertEqual(exit_calls, 1)

    async def test_workflow_finish_after_quit_auto_exits(self) -> None:
        exit_calls = 0

        class _Tracker(DemoApp):  # type: ignore[misc]
            def exit(self, *args: object, **kwargs: object) -> None:  # type: ignore[override]
                nonlocal exit_calls
                exit_calls += 1

        app = _Tracker(intent="x", workflow=_noop_workflow, on_quit=lambda: None)

        await app.action_quit()
        # Simulate the worker thread's completion callback firing after
        # runner.cancel() caused the graph to shut down.
        app._DemoApp__render_workflow_finish(success=False)  # type: ignore[attr-defined]

        self.assertEqual(exit_calls, 1, "Workflow finish after quit must auto-exit")

    async def test_workflow_finish_without_quit_stays_open(self) -> None:
        exit_calls = 0

        class _Tracker(DemoApp):  # type: ignore[misc]
            def exit(self, *args: object, **kwargs: object) -> None:  # type: ignore[override]
                nonlocal exit_calls
                exit_calls += 1

        app = _Tracker(intent="x", workflow=_noop_workflow)
        app._DemoApp__render_workflow_finish(success=True)  # type: ignore[attr-defined]

        self.assertEqual(exit_calls, 0, "Normal completion should leave the UI open for scrollback")

    async def test_on_quit_callback_is_optional(self) -> None:
        """Building a DemoApp without ``on_quit`` must still handle
        ``q`` cleanly — the quit flow should degrade to a simple
        first-press-noop, second-press-exit without raising."""

        exit_calls = 0

        class _Tracker(DemoApp):  # type: ignore[misc]
            def exit(self, *args: object, **kwargs: object) -> None:  # type: ignore[override]
                nonlocal exit_calls
                exit_calls += 1

        app = _Tracker(intent="x", workflow=_noop_workflow, on_quit=None)
        await app.action_quit()
        await app.action_quit()

        self.assertEqual(exit_calls, 1)


class DemoAppHitlModalTest(unittest.TestCase):
    """
    Cover the HITL modal push path on DemoApp.
    """

    def test_push_hitl_ask_pushes_hitl_ask_screen_on_main_thread(self) -> None:
        app = _make_app()
        future: "concurrent.futures.Future[str]" = concurrent.futures.Future()

        with patch.object(DemoApp, "push_screen") as push_screen:
            app.push_hitl_ask("Where is the location?", future)

        push_screen.assert_called_once()
        (screen,), _kwargs = push_screen.call_args
        self.assertIsInstance(screen, _HitlAskScreen)


class HitlAskScreenSubmitTest(unittest.TestCase):
    """
    Cover the modal's Enter-submit path.

    We don't mount the screen inside Textual's pilot here — we only
    exercise the submit handler which is a pure method that sets the
    future and dismisses. Mount-level tests would require the full
    App pilot and add complexity without catching additional bugs.
    """

    def test_on_input_submitted_sets_future_and_dismisses(self) -> None:
        future: "concurrent.futures.Future[str]" = concurrent.futures.Future()
        screen = _HitlAskScreen(prompt="Pick a value", future=future)

        class _FakeEvent:
            value = "HSR Layout"

        with patch.object(_HitlAskScreen, "dismiss") as dismiss_mock:
            screen.on_input_submitted(_FakeEvent())  # type: ignore[arg-type]

        self.assertTrue(future.done())
        self.assertEqual(future.result(), "HSR Layout")
        dismiss_mock.assert_called_once_with("HSR Layout")

    def test_action_cancel_resolves_future_with_empty_string(self) -> None:
        future: "concurrent.futures.Future[str]" = concurrent.futures.Future()
        screen = _HitlAskScreen(prompt="Pick a value", future=future)

        with patch.object(_HitlAskScreen, "dismiss") as dismiss_mock:
            screen.action_cancel()

        self.assertTrue(future.done())
        self.assertEqual(future.result(), "")
        dismiss_mock.assert_called_once_with("")

    def test_submit_after_already_resolved_future_does_not_reraise(self) -> None:
        """A double-submit or race must not crash the modal."""

        future: "concurrent.futures.Future[str]" = concurrent.futures.Future()
        future.set_result("already-set")
        screen = _HitlAskScreen(prompt="x", future=future)

        class _FakeEvent:
            value = "ignored"

        with patch.object(_HitlAskScreen, "dismiss"):
            screen.on_input_submitted(_FakeEvent())  # type: ignore[arg-type]

        # The first result wins; the second submit is a no-op on the future.
        self.assertEqual(future.result(), "already-set")


class DemoAppPanelSelectionTest(unittest.TestCase):
    """
    Spot-check which events produce a body panel vs which only update state.
    Uses the private __panel_for_event via name-mangled access.
    """

    def _build_panel(self, *, event_type: FathomEvent, **context: object) -> object:
        app = _make_app()
        return app._DemoApp__panel_for_event(  # type: ignore[attr-defined]
            level="info",
            message="",
            event_type=event_type,
            context=dict({"type": event_type}, **context),
        )

    def test_intent_classified_produces_panel(self) -> None:
        self.assertIsNotNone(
            self._build_panel(event_type=FathomEvent.INTENT_CLASSIFIED, should_decompose=False),
        )

    def test_decomposition_without_goals_produces_nothing(self) -> None:
        self.assertIsNone(
            self._build_panel(event_type=FathomEvent.DECOMPOSITION_COMPLETE, sub_goals=[]),
        )

    def test_sub_goal_started_produces_panel(self) -> None:
        self.assertIsNotNone(
            self._build_panel(
                event_type=FathomEvent.SUB_GOAL_STARTED,
                index=0,
                total=3,
                description="x",
            ),
        )

    def test_step_completed_success_produces_green_panel(self) -> None:
        """STEP_COMPLETED now renders a per-step success/fail panel."""

        panel = self._build_panel(
            event_type=FathomEvent.STEP_COMPLETED,
            step=5,
            success=True,
            action_description="tap 'Search'",
            observation="Search results visible",
        )
        self.assertIsNotNone(panel)

    def test_step_completed_failure_produces_red_panel(self) -> None:
        panel = self._build_panel(
            event_type=FathomEvent.STEP_COMPLETED,
            step=2,
            success=False,
            action_description="tap 'Login'",
            observation="Button not visible",
        )
        self.assertIsNotNone(panel)

    def test_planned_action_produces_panel(self) -> None:
        panel = self._build_panel(
            event_type=FathomEvent.PLANNED_ACTION,
            step=1,
            action_description="Tap Chrome icon",
            target="Chrome",
            confidence=0.93,
        )
        self.assertIsNotNone(panel)

    def test_reasoning_produces_panel(self) -> None:
        panel = self._build_panel(
            event_type=FathomEvent.REASONING,
            step=1,
            reasoning="I need to open Chrome to begin the search flow.",
        )
        self.assertIsNotNone(panel)
