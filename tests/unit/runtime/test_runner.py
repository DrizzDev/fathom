from __future__ import annotations

import unittest
from typing import List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

from fathom.constants.events import FathomEvent
from fathom.constants.exploration import DEFAULT_EXPLORATION_INTENT, STEP_TIME_BUDGET
from fathom.constants.qualification import QualificationLabel, RationaleCategory
from fathom.constants.state import CompletionReason
from fathom.interfaces.qualifier import IntentQualifierPort
from fathom.runtime.runner import FathomRunner
from fathom.schemas.qualification import QualificationVerdict, Rationale
from fathom.schemas.results import ActionResult


class BlockingQualifier(IntentQualifierPort):
    """
    Stub qualifier that always returns a blocking verdict with a custom message.
    """

    def __init__(self, *, message: str = "blocked-by-test") -> None:
        """
        Initialize with the rejection message the stub should attach to its verdict.
        """

        self.__message = message
        self.calls: List[str] = []

    async def qualify(self, *, intent: str) -> QualificationVerdict:
        """
        Record the call and return a high-confidence NOT_EXECUTABLE verdict.
        """

        self.calls.append(intent)
        return QualificationVerdict(
            label=QualificationLabel.NOT_EXECUTABLE,
            confidence=0.99,
            rationale=Rationale(
                category=RationaleCategory.INFORMATIONAL,
                reasoning="test-rejection",
            ),
            message=self.__message,
        )


class PassingQualifier(IntentQualifierPort):
    """
    Stub qualifier that always passes.
    """

    def __init__(self) -> None:
        """
        Initialize the call-recording stub with an empty history.
        """

        self.calls: List[str] = []

    async def qualify(self, *, intent: str) -> QualificationVerdict:
        """
        Record the call and return an EXECUTABLE verdict.
        """

        self.calls.append(intent)
        return QualificationVerdict(
            label=QualificationLabel.EXECUTABLE,
            confidence=0.95,
            rationale=Rationale(category=RationaleCategory.UI_TASK, reasoning="ok"),
        )


class RunnerHarness:
    """
    Factory for constructing a FathomRunner wired with mocked ports for gate-path tests.
    """

    @staticmethod
    def build(*, qualifier: IntentQualifierPort) -> Tuple[FathomRunner, MagicMock]:
        """
        Build a runner with mocked dependencies and return both the runner and the telemetry mock.
        """

        telemetry = MagicMock()
        telemetry.info = AsyncMock()
        telemetry.warning = AsyncMock()
        telemetry.error = AsyncMock()
        telemetry.debug = AsyncMock()

        device = MagicMock()
        device.configuration = MagicMock(identifier="test-device")
        device.get_current_package = AsyncMock(return_value="com.example.test")

        runner = FathomRunner(
            llm=MagicMock(),
            device=device,
            perception=MagicMock(),
            memory=MagicMock(),
            signal=MagicMock(),
            storage=MagicMock(),
            knowledge=MagicMock(),
            telemetry=telemetry,
            summarizer=MagicMock(),
            qualifier=qualifier,
            path_manager=MagicMock(),
        )
        return runner, telemetry


class RunnerQualifierGateTest(unittest.IsolatedAsyncioTestCase):
    """
    Runner must short-circuit on a blocking verdict and emit BOTH the new
    INTENT_REJECTED event (for clients that switch on the verdict) AND the
    legacy WORKFLOW_COMPLETED event (so existing terminal-event consumers
    still see the workflow as finalized).
    """

    async def test_blocking_verdict_short_circuits_with_completed_result(self) -> None:
        """
        Block must skip ContextManager and IntentStrategy and produce a completed result.
        """

        qualifier = BlockingQualifier(message="custom-rejection-message")
        runner, telemetry = RunnerHarness.build(qualifier=qualifier)

        with (
            patch("fathom.runtime.runner.ContextManager") as context_manager_cls,
            patch("fathom.runtime.runner.IntentStrategy") as strategy_cls,
        ):
            result = await runner.run_intent(intent="who founded google?")

        context_manager_cls.assert_not_called()
        strategy_cls.assert_not_called()
        self.assertEqual(qualifier.calls, ["who founded google?"])

        self.assertEqual(result.status, "completed")
        self.assertFalse(result.success)
        self.assertEqual(result.completion_reason, CompletionReason.NOT_EXECUTABLE.value)
        self.assertIsNone(result.error)
        self.assertEqual(result.intent, "who founded google?")
        self.assertEqual(result.steps_taken, 0)

    async def test_blocking_verdict_never_touches_device(self) -> None:
        """
        Regression check: a rejected intent must not trigger device.get_current_package
        or any other device call. The qualifier sits in front of the device interaction.
        """

        qualifier = BlockingQualifier()
        runner, _ = RunnerHarness.build(qualifier=qualifier)

        with (
            patch("fathom.runtime.runner.ContextManager"),
            patch("fathom.runtime.runner.IntentStrategy"),
        ):
            await runner.run_intent(intent="+")

        runner.device.get_current_package.assert_not_called()  # type: ignore[attr-defined]

    async def test_blocking_verdict_emits_intent_rejected_with_full_payload(
        self,
    ) -> None:
        """
        Rejection must emit exactly one qualifier-typed event: INTENT_REJECTED
        with the full verdict and user-facing message. INTENT_QUALIFIED must
        never fire on the reject path.
        """

        qualifier = BlockingQualifier(message="custom-rejection-message")
        runner, telemetry = RunnerHarness.build(qualifier=qualifier)

        with (
            patch("fathom.runtime.runner.ContextManager"),
            patch("fathom.runtime.runner.IntentStrategy"),
        ):
            await runner.run_intent(intent="who founded google?")

        qualifier_typed_calls = [
            call
            for call in telemetry.info.call_args_list + telemetry.warning.call_args_list
            if call.kwargs.get("type")
            in {FathomEvent.INTENT_QUALIFIED, FathomEvent.INTENT_REJECTED}
        ]
        self.assertEqual(len(qualifier_typed_calls), 1)

        rejection = qualifier_typed_calls[0]
        self.assertEqual(rejection.kwargs["type"], FathomEvent.INTENT_REJECTED)
        self.assertEqual(rejection.args[0], "custom-rejection-message")
        self.assertEqual(rejection.kwargs["label"], QualificationLabel.NOT_EXECUTABLE.value)
        self.assertEqual(rejection.kwargs["confidence"], 0.99)
        self.assertEqual(
            rejection.kwargs["rationale"]["category"], RationaleCategory.INFORMATIONAL.value
        )

    async def test_blocking_verdict_dual_emits_workflow_completed_for_legacy_consumers(
        self,
    ) -> None:
        """Backward-compat: rejection must also emit WORKFLOW_COMPLETED so legacy consumers (Genymotion, Temporal activity result handlers) that key off the terminal event still get a completion signal."""

        qualifier = BlockingQualifier(message="custom-rejection-message")
        runner, telemetry = RunnerHarness.build(qualifier=qualifier)

        with (
            patch("fathom.runtime.runner.ContextManager"),
            patch("fathom.runtime.runner.IntentStrategy"),
        ):
            await runner.run_intent(intent="who founded google?")

        terminal_calls = [
            call
            for call in telemetry.info.call_args_list
            if call.kwargs.get("type") == FathomEvent.WORKFLOW_COMPLETED
        ]
        self.assertEqual(
            len(terminal_calls),
            1,
            msg="rejection path must emit exactly one WORKFLOW_COMPLETED event",
        )

        terminal = terminal_calls[0]
        self.assertEqual(terminal.kwargs["success"], False)
        self.assertEqual(terminal.kwargs["steps_taken"], 0)
        self.assertIn("duration", terminal.kwargs)
        self.assertGreaterEqual(terminal.kwargs["duration"], 0.0)

    async def test_passing_verdict_proceeds_to_strategy(self) -> None:
        """
        Allow path must construct ContextManager and IntentStrategy and run execute().
        """

        qualifier = PassingQualifier()
        runner, telemetry = RunnerHarness.build(qualifier=qualifier)

        strategy_instance = MagicMock()
        strategy_instance.execute = AsyncMock(
            return_value=MagicMock(success=True, is_cancelled=False, error=None, duration=10)
        )
        strategy_instance.get_progress = MagicMock(
            return_value={"step_count": 1, "completion_reason": "Completed successfully"}
        )
        strategy_instance.get_subgoal_execution_audit = MagicMock(return_value=([], [], 0))
        strategy_instance.get_metrics = MagicMock(return_value=None)
        strategy_instance.completion_reason = "Completed successfully"
        strategy_instance.step_results = []

        memory_summary = {"screens": [], "total_screens": 0, "experience_count": 0}

        with (
            patch("fathom.runtime.runner.ContextManager") as context_manager_cls,
            patch("fathom.runtime.runner.IntentStrategy", return_value=strategy_instance),
            patch.object(
                FathomRunner,
                "_FathomRunner__get_memory_summary",
                AsyncMock(return_value=memory_summary),
            ),
        ):
            result = await runner.run_intent(intent="Search for McPuff")

        context_manager_cls.assert_called_once()
        strategy_instance.execute.assert_awaited_once()
        self.assertEqual(qualifier.calls, ["Search for McPuff"])

        self.assertEqual(result.status, "completed")
        self.assertTrue(result.success)
        self.assertEqual(result.completion_reason, "Completed successfully")

        # Allow path emits exactly one qualifier event: INTENT_QUALIFIED.
        qualifier_typed_calls = [
            call
            for call in telemetry.info.call_args_list + telemetry.warning.call_args_list
            if call.kwargs.get("type")
            in {FathomEvent.INTENT_QUALIFIED, FathomEvent.INTENT_REJECTED}
        ]
        self.assertEqual(len(qualifier_typed_calls), 1)
        self.assertEqual(qualifier_typed_calls[0].kwargs["type"], FathomEvent.INTENT_QUALIFIED)
        self.assertEqual(
            qualifier_typed_calls[0].kwargs["label"], QualificationLabel.EXECUTABLE.value
        )


class RunnerOwnedResourcesCleanupTest(unittest.IsolatedAsyncioTestCase):
    """
    Runner takes optional ownership of LLM resources passed by the builder's
    .with_assembly() path. Those resources must be drained in cleanup() so
    SDK callers don't have to track them themselves.
    """

    async def test_cleanup_drains_each_owned_resource(self) -> None:
        """
        Every entry in owned_resources must have cleanup() awaited exactly once.
        """

        cleanup_order: list[str] = []

        planner_llm = MagicMock()
        planner_llm.cleanup = AsyncMock(side_effect=lambda: cleanup_order.append("planner"))

        owned_a = MagicMock()
        owned_a.cleanup = AsyncMock(side_effect=lambda: cleanup_order.append("owned_a"))

        owned_b = MagicMock()
        owned_b.cleanup = AsyncMock(side_effect=lambda: cleanup_order.append("owned_b"))

        runner = FathomRunner(
            llm=planner_llm,
            device=MagicMock(),
            perception=MagicMock(),
            memory=MagicMock(),
            signal=MagicMock(),
            storage=MagicMock(),
            knowledge=MagicMock(),
            telemetry=MagicMock(),
            summarizer=MagicMock(),
            qualifier=MagicMock(),
            path_manager=MagicMock(),
            owned_resources=[owned_a, owned_b],
        )

        await runner.cleanup()

        planner_llm.cleanup.assert_awaited_once_with()
        owned_a.cleanup.assert_awaited_once_with()
        owned_b.cleanup.assert_awaited_once_with()
        # Planner cleaned first, then owned resources in registration order.
        self.assertEqual(cleanup_order, ["planner", "owned_a", "owned_b"])

    async def test_cleanup_isolates_owned_resource_failures(self) -> None:
        """
        A failure on one owned resource cleanup must not skip the others —
        the per-resource try/except in runner.cleanup must isolate them.
        """

        planner_llm = MagicMock()
        planner_llm.cleanup = AsyncMock()

        good_first = MagicMock()
        good_first.cleanup = AsyncMock()

        bad = MagicMock()
        bad.cleanup = AsyncMock(side_effect=RuntimeError("kaboom"))

        good_last = MagicMock()
        good_last.cleanup = AsyncMock()

        runner = FathomRunner(
            llm=planner_llm,
            device=MagicMock(),
            perception=MagicMock(),
            memory=MagicMock(),
            signal=MagicMock(),
            storage=MagicMock(),
            knowledge=MagicMock(),
            telemetry=MagicMock(),
            summarizer=MagicMock(),
            qualifier=MagicMock(),
            path_manager=MagicMock(),
            owned_resources=[good_first, bad, good_last],
        )

        await runner.cleanup()

        good_first.cleanup.assert_awaited_once_with()
        bad.cleanup.assert_awaited_once_with()
        good_last.cleanup.assert_awaited_once_with()


class RunnerWorkflowCancelledEmitTest(unittest.IsolatedAsyncioTestCase):
    """
    Runner must emit WORKFLOW_CANCELLED (not WORKFLOW_COMPLETED) when the
    strategy returns ``is_cancelled=True`` and stamp the OPERATOR_ABORTED completion reason on the published terminal event.
    """

    async def test_cancelled_run_emits_workflow_cancelled_event(self) -> None:
        """
        ``execution_result.is_cancelled=True`` routes the terminal event to WORKFLOW_CANCELLED.
        """

        qualifier = PassingQualifier()
        runner, telemetry = RunnerHarness.build(qualifier=qualifier)

        strategy_instance = MagicMock()
        strategy_instance.execute = AsyncMock(
            return_value=MagicMock(
                error=None,
                duration=42,
                success=False,
                is_cancelled=True,
            )
        )
        strategy_instance.get_progress = MagicMock(
            return_value={
                "step_count": 9,
                "completion_reason": CompletionReason.OPERATOR_ABORTED.value,
            }
        )

        strategy_instance.step_results = []
        strategy_instance.get_metrics = MagicMock(return_value=None)
        strategy_instance.completion_reason = CompletionReason.OPERATOR_ABORTED.value
        strategy_instance.get_subgoal_execution_audit = MagicMock(return_value=([], [], 0))

        with (
            patch("fathom.runtime.runner.ContextManager"),
            patch("fathom.runtime.runner.IntentStrategy", return_value=strategy_instance),
            patch.object(
                FathomRunner,
                "_FathomRunner__get_memory_summary",
                AsyncMock(return_value={}),
            ),
        ):
            result = await runner.run_intent(intent="Stop me anytime")

        cancelled_calls = [
            call
            for call in telemetry.info.call_args_list
            if call.kwargs.get("type") == FathomEvent.WORKFLOW_CANCELLED
        ]
        completed_calls = [
            call
            for call in telemetry.info.call_args_list
            if call.kwargs.get("type") == FathomEvent.WORKFLOW_COMPLETED
        ]

        self.assertEqual(len(cancelled_calls), 1)
        self.assertEqual(len(completed_calls), 0)

        terminal = cancelled_calls[0]
        self.assertEqual(terminal.kwargs["success"], False)
        self.assertEqual(terminal.kwargs["steps_taken"], 9)
        self.assertEqual(
            terminal.kwargs["completion_reason"],
            CompletionReason.CANCELLED.value,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.completion_reason, CompletionReason.CANCELLED.value)

    async def test_successful_run_still_emits_workflow_completed_event(self) -> None:
        """
        Regression guard: a normal completion must keep emitting WORKFLOW_COMPLETED.
        """

        qualifier = PassingQualifier()
        runner, telemetry = RunnerHarness.build(qualifier=qualifier)

        strategy_instance = MagicMock()
        strategy_instance.execute = AsyncMock(
            return_value=MagicMock(
                error=None,
                success=True,
                duration=100,
                is_cancelled=False,
            )
        )
        strategy_instance.get_progress = MagicMock(
            return_value={"step_count": 4, "completion_reason": CompletionReason.SUCCESS.value}
        )

        strategy_instance.step_results = []
        strategy_instance.get_metrics = MagicMock(return_value=None)
        strategy_instance.completion_reason = CompletionReason.SUCCESS.value
        strategy_instance.get_subgoal_execution_audit = MagicMock(return_value=([], [], 0))

        with (
            patch("fathom.runtime.runner.ContextManager"),
            patch("fathom.runtime.runner.IntentStrategy", return_value=strategy_instance),
            patch.object(
                FathomRunner,
                "_FathomRunner__get_memory_summary",
                AsyncMock(return_value={}),
            ),
        ):
            await runner.run_intent(intent="Search for biryani")

        cancelled_calls = [
            call
            for call in telemetry.info.call_args_list
            if call.kwargs.get("type") == FathomEvent.WORKFLOW_CANCELLED
        ]
        completed_calls = [
            call
            for call in telemetry.info.call_args_list
            if call.kwargs.get("type") == FathomEvent.WORKFLOW_COMPLETED
        ]

        self.assertEqual(len(cancelled_calls), 0)
        self.assertEqual(len(completed_calls), 1)

    async def test_failed_run_emits_workflow_failed_event(self) -> None:
        """
        Failed strategy outcomes must not be announced as WORKFLOW_COMPLETED.
        """

        qualifier = PassingQualifier()
        runner, telemetry = RunnerHarness.build(qualifier=qualifier)

        strategy_instance = MagicMock()
        strategy_instance.execute = AsyncMock(
            return_value=MagicMock(
                error="Planner retry budget exhausted",
                success=False,
                duration=100,
                is_cancelled=False,
            )
        )
        strategy_instance.get_progress = MagicMock(
            return_value={
                "step_count": 7,
                "completion_reason": CompletionReason.RETRY_BUDGET_EXHAUSTED.value,
            }
        )

        strategy_instance.step_results = []
        strategy_instance.get_metrics = MagicMock(return_value=None)
        strategy_instance.completion_reason = CompletionReason.RETRY_BUDGET_EXHAUSTED.value
        strategy_instance.get_subgoal_execution_audit = MagicMock(return_value=([], [], 0))

        with (
            patch("fathom.runtime.runner.ContextManager"),
            patch("fathom.runtime.runner.IntentStrategy", return_value=strategy_instance),
            patch.object(
                FathomRunner,
                "_FathomRunner__get_memory_summary",
                AsyncMock(return_value={}),
            ),
        ):
            result = await runner.run_intent(intent="Search for biryani")

        failed_calls = [
            call
            for call in telemetry.info.call_args_list
            if call.kwargs.get("type") == FathomEvent.WORKFLOW_FAILED
        ]
        completed_calls = [
            call
            for call in telemetry.info.call_args_list
            if call.kwargs.get("type") == FathomEvent.WORKFLOW_COMPLETED
        ]

        self.assertEqual(len(failed_calls), 1)
        self.assertEqual(len(completed_calls), 0)
        self.assertFalse(result.success)

        terminal = failed_calls[0]
        self.assertEqual(terminal.args[0], "Run failed: Planner retry budget exhausted")
        self.assertEqual(terminal.kwargs["success"], False)
        self.assertEqual(terminal.kwargs["steps_taken"], 7)
        self.assertEqual(
            terminal.kwargs["completion_reason"],
            CompletionReason.RETRY_BUDGET_EXHAUSTED.value,
        )


class RunnerExplorationLaunchTest(unittest.IsolatedAsyncioTestCase):
    """
    run_exploration launches an explicitly requested package before exploring.
    """

    @staticmethod
    def __strategy() -> MagicMock:
        """
        Build a mock exploration strategy that completes immediately.
        """

        strategy = MagicMock()
        strategy.execute = AsyncMock(
            return_value=MagicMock(success=True, error=None, duration=1, is_cancelled=False)
        )
        strategy.get_progress = MagicMock(return_value={"stats": {}, "steps": 0})
        strategy.graph = MagicMock(nodes={})
        return strategy

    @staticmethod
    def __on_launcher_then_app() -> Tuple[AsyncMock, AsyncMock, MagicMock]:
        """
        Device/perception/hash mocks for: launcher, then app focuses while the
        launcher is still painted, then the app renders and settles.
        """

        get_current_package = AsyncMock(
            side_effect=[
                "com.android.launcher",  # pre-launch fingerprint
                "ai.hangjam.app",  # poll 1: focused, launcher still drawn
                "ai.hangjam.app",  # poll 2: app rendering
                "ai.hangjam.app",  # poll 3: app settled
            ]
        )
        capture = AsyncMock(return_value=MagicMock(image=b"frame"))
        hash_engine = MagicMock(
            hash=MagicMock(
                side_effect=[
                    "ffffffff00000000",  # pre-launch: launcher
                    "ffffffff00000000",  # poll 1: still launcher
                    "00000000ffffffff",  # poll 2: app (changed, not yet settled)
                    "00000000ffffffff",  # poll 3: app settled
                ]
            )
        )
        return get_current_package, capture, hash_engine

    async def test_explicit_package_is_launched(self) -> None:
        """
        A provided package is launched to the foreground before exploration starts.
        """

        runner, _ = RunnerHarness.build(qualifier=PassingQualifier())
        runner.device.launch_package = AsyncMock(  # type: ignore[attr-defined]
            return_value=ActionResult(success=True, duration=1)
        )
        runner.device.get_current_package = AsyncMock(  # type: ignore[attr-defined]
            return_value="ai.hangjam.app"
        )
        runner.perception.capture = AsyncMock(  # type: ignore[attr-defined]
            return_value=MagicMock(image=b"frame")
        )
        runner._FathomRunner__visual_hash_engine = MagicMock(  # type: ignore[attr-defined]
            hash=MagicMock(return_value="aaaaaaaaaaaaaaaa")
        )

        with (
            patch("fathom.runtime.runner.ExplorationStrategy", return_value=self.__strategy()),
            patch("fathom.runtime.runner.stability_wait", AsyncMock()),
            patch.object(FathomRunner, "_FathomRunner__export_graph", AsyncMock(return_value={})),
            patch.object(FathomRunner, "_FathomRunner__write_artifacts", AsyncMock()),
        ):
            await runner.run_exploration(
                max_steps=1, request_id="wf", package_name="ai.hangjam.app"
            )

        runner.device.launch_package.assert_awaited_once_with(package_name="ai.hangjam.app")

    async def test_first_capture_waits_until_app_renders_past_launcher(self) -> None:
        """
        Polling continues while the launcher is still painted and returns only once
        the on-package screen has changed away from it and settled, so the first
        capture lands on the app rather than the launcher it was launched from.
        """

        runner, _ = RunnerHarness.build(qualifier=PassingQualifier())
        runner.device.launch_package = AsyncMock(  # type: ignore[attr-defined]
            return_value=ActionResult(success=True, duration=1)
        )
        get_current_package, capture, hash_engine = self.__on_launcher_then_app()
        runner.device.get_current_package = get_current_package  # type: ignore[attr-defined]
        runner.perception.capture = capture  # type: ignore[attr-defined]
        runner._FathomRunner__visual_hash_engine = hash_engine  # type: ignore[attr-defined]

        with (
            patch("fathom.runtime.runner.ExplorationStrategy", return_value=self.__strategy()),
            patch("fathom.runtime.runner.stability_wait", AsyncMock()),
            patch.object(FathomRunner, "_FathomRunner__export_graph", AsyncMock(return_value={})),
            patch.object(FathomRunner, "_FathomRunner__write_artifacts", AsyncMock()),
        ):
            await runner.run_exploration(
                max_steps=1, request_id="wf", package_name="ai.hangjam.app"
            )

        # Pre-launch capture plus three polls, returning on the settled app frame.
        self.assertEqual(capture.await_count, 4)
        self.assertEqual(
            runner.device.get_current_package.await_count,  # type: ignore[attr-defined]
            4,
        )

    async def test_absent_package_falls_back_to_foreground(self) -> None:
        """
        Without a package, the foreground application is queried and not launched.
        """

        runner, _ = RunnerHarness.build(qualifier=PassingQualifier())
        runner.device.launch_package = AsyncMock()  # type: ignore[attr-defined]

        with (
            patch("fathom.runtime.runner.ExplorationStrategy", return_value=self.__strategy()),
            patch.object(FathomRunner, "_FathomRunner__export_graph", AsyncMock(return_value={})),
            patch.object(FathomRunner, "_FathomRunner__write_artifacts", AsyncMock()),
        ):
            await runner.run_exploration(max_steps=1, request_id="wf")

        runner.device.get_current_package.assert_awaited_once()
        runner.device.launch_package.assert_not_called()

    async def test_timeout_scales_with_step_budget(self) -> None:
        """
        The wall-clock timeout is derived from the step budget, so a large step
        budget is honoured instead of being cut short by the configured timeout.
        """

        runner, _ = RunnerHarness.build(qualifier=PassingQualifier())

        with (
            patch(
                "fathom.runtime.runner.ExplorationStrategy", return_value=self.__strategy()
            ) as strategy_cls,
            patch("fathom.runtime.runner.ContextManager"),
            patch.object(FathomRunner, "_FathomRunner__export_graph", AsyncMock(return_value={})),
            patch.object(FathomRunner, "_FathomRunner__write_artifacts", AsyncMock()),
        ):
            await runner.run_exploration(max_steps=200, request_id="wf")

        kwargs = strategy_cls.call_args.kwargs
        self.assertEqual(kwargs["max_steps"], 200)
        self.assertEqual(kwargs["timeout"], float(200 * STEP_TIME_BUDGET))

    async def test_focus_intent_forwarded_to_strategy(self) -> None:
        """
        A provided intent reaches both the strategy and the context roadmap.
        """

        runner, _ = RunnerHarness.build(qualifier=PassingQualifier())

        with (
            patch(
                "fathom.runtime.runner.ExplorationStrategy", return_value=self.__strategy()
            ) as strategy_cls,
            patch("fathom.runtime.runner.ContextManager") as context_manager_cls,
            patch.object(FathomRunner, "_FathomRunner__export_graph", AsyncMock(return_value={})),
            patch.object(FathomRunner, "_FathomRunner__write_artifacts", AsyncMock()),
        ):
            await runner.run_exploration(
                max_steps=1, request_id="wf", intent="Focus on the checkout flow"
            )

        self.assertEqual(strategy_cls.call_args.kwargs["intent"], "Focus on the checkout flow")
        context_manager_cls.return_value.set_roadmap.assert_called_once_with(
            intent="Focus on the checkout flow"
        )

    async def test_absent_intent_defaults_to_constant(self) -> None:
        """
        Without an intent, the strategy and roadmap fall back to the default goal.
        """

        runner, _ = RunnerHarness.build(qualifier=PassingQualifier())

        with (
            patch(
                "fathom.runtime.runner.ExplorationStrategy", return_value=self.__strategy()
            ) as strategy_cls,
            patch("fathom.runtime.runner.ContextManager") as context_manager_cls,
            patch.object(FathomRunner, "_FathomRunner__export_graph", AsyncMock(return_value={})),
            patch.object(FathomRunner, "_FathomRunner__write_artifacts", AsyncMock()),
        ):
            await runner.run_exploration(max_steps=1, request_id="wf")

        self.assertEqual(strategy_cls.call_args.kwargs["intent"], DEFAULT_EXPLORATION_INTENT)
        context_manager_cls.return_value.set_roadmap.assert_called_once_with(
            intent=DEFAULT_EXPLORATION_INTENT
        )


class IsRenderedAppScreenTest(unittest.TestCase):
    """The launch wait accepts only a settled on-package screen that is not the launcher."""

    LAUNCHER = "ffffffff00000000"
    APP = "00000000ffffffff"

    @staticmethod
    def __decide(*, current: str, previous: object, launcher: object) -> bool:
        return FathomRunner._FathomRunner__is_rendered_app_screen(  # type: ignore[attr-defined]
            current=current, previous=previous, launcher=launcher
        )

    def test_first_capture_is_never_rendered(self) -> None:
        self.assertFalse(self.__decide(current=self.LAUNCHER, previous=None, launcher=None))

    def test_unsettled_pair_is_not_rendered(self) -> None:
        self.assertFalse(self.__decide(current=self.APP, previous=self.LAUNCHER, launcher=None))

    def test_settled_without_a_launcher_frame_is_rendered(self) -> None:
        self.assertTrue(self.__decide(current=self.APP, previous=self.APP, launcher=None))

    def test_settled_on_the_launcher_is_refused(self) -> None:
        self.assertFalse(
            self.__decide(current=self.LAUNCHER, previous=self.LAUNCHER, launcher=self.LAUNCHER)
        )

    def test_settled_away_from_the_launcher_is_rendered(self) -> None:
        self.assertTrue(self.__decide(current=self.APP, previous=self.APP, launcher=self.LAUNCHER))


if __name__ == "__main__":
    unittest.main()
