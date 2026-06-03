from __future__ import annotations

import unittest
from typing import List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

from fathom.constants.events import FathomEvent
from fathom.constants.qualification import QualificationLabel, RationaleCategory
from fathom.constants.state import CompletionReason
from fathom.interfaces.qualifier import IntentQualifierPort
from fathom.runtime.runner import FathomRunner
from fathom.schemas.qualification import QualificationVerdict, Rationale


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
        """
        Backward-compat: rejection must also emit WORKFLOW_COMPLETED so legacy
        consumers (Genymotion, Temporal activity result handlers) that key off
        the terminal event still get a completion signal. Payload mirrors the
        success-path terminal event shape: success=False, steps_taken=0,
        duration present.
        """

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


if __name__ == "__main__":
    unittest.main()
