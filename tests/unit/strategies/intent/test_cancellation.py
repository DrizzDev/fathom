from __future__ import annotations

import time
import unittest
from typing import Any, Callable, Optional, cast
from unittest.mock import AsyncMock, MagicMock

from fathom.constants.events import FathomEvent
from fathom.constants.state import CompletionReason, RunOutcome
from fathom.strategies.intent import IntentStrategy


class IntentStrategyResultAssemblerOutcomeTest(unittest.TestCase):
    """
    Pins :class:`IntentStrategy.__ResultAssembler.assemble` for every :class:`RunOutcome`.
    """

    @staticmethod
    def __build_assembler() -> Any:
        """
        Construct an assembler bound to a synthetic workflow id and start time.
        """

        return IntentStrategy._IntentStrategy__ResultAssembler(  # type: ignore[attr-defined]
            workflow_id="workflow__1",
            started_at=time.time(),
        )

    @staticmethod
    def __build_agent_state(*, is_complete: bool = False) -> Any:
        """
        Build a minimal agent-state stub for the assembler's success branch.
        """

        return MagicMock(completion_reason=None, is_complete=is_complete)

    def test_completed_outcome_assembles_normal_result(self) -> None:
        """
        :attr:`RunOutcome.COMPLETED` runs the strict resolution path.
        """

        assembler = self.__build_assembler()

        result = assembler.assemble(
            final_state=None,
            run_outcome=RunOutcome.COMPLETED,
            agent_state=self.__build_agent_state(is_complete=True),
        )

        self.assertIsNone(result.error)
        self.assertFalse(result.is_cancelled)

    def test_failed_outcome_assembles_failed_result(self) -> None:
        """
        :attr:`RunOutcome.FAILED` returns a failed, non-cancelled execution result.
        """

        assembler = self.__build_assembler()

        result = assembler.assemble(
            final_state=None,
            run_outcome=RunOutcome.FAILED,
            agent_state=self.__build_agent_state(),
        )

        self.assertFalse(result.success)
        self.assertFalse(result.is_cancelled)
        self.assertEqual(result.error, "executor failed before terminal state")
        self.assertEqual(assembler.completion_reason, CompletionReason.FAILED.value)

    def test_cancelled_outcome_marks_is_cancelled_and_reason(self) -> None:
        """
        :attr:`RunOutcome.CANCELLED` stamps OPERATOR_ABORTED on the assembler and
        flags ``is_cancelled`` on the result so downstream consumers can branch.
        """

        assembler = self.__build_assembler()

        result = assembler.assemble(
            final_state=None,
            run_outcome=RunOutcome.CANCELLED,
            agent_state=self.__build_agent_state(),
        )

        self.assertFalse(result.success)
        self.assertTrue(result.is_cancelled)
        self.assertIsNone(result.error)
        self.assertEqual(assembler.completion_reason, CompletionReason.OPERATOR_ABORTED.value)

    def test_failed_outcome_does_not_set_is_cancelled(self) -> None:
        """
        A FAILED outcome must not be conflated with cancellation; downstream
        consumers branch on these signals independently.
        """

        assembler = self.__build_assembler()

        result = assembler.assemble(
            final_state=None,
            run_outcome=RunOutcome.FAILED,
            agent_state=self.__build_agent_state(),
        )

        self.assertFalse(result.is_cancelled)


class IntentStrategyScriptGeneratedRunOutcomeTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins :meth:`IntentStrategy.__emit_script_generated_event` propagation of run_outcome.
    """

    @staticmethod
    def __build_strategy(*, step_count: int = 4) -> Any:
        """
        Build a bare IntentStrategy with stubs the SCRIPT_GENERATED emit reads.
        """

        strategy = object.__new__(IntentStrategy)

        telemetry = MagicMock()
        telemetry.info = AsyncMock()

        graph_context = MagicMock()
        graph_context.telemetry = telemetry
        graph_context.agent_state = MagicMock(step_count=step_count)

        strategy.__setattr__("_IntentStrategy__workflow_id", "workflow__1")
        strategy.__setattr__("_IntentStrategy__graph_context", graph_context)

        return strategy, telemetry

    @staticmethod
    async def __invoke_emit(
        *,
        strategy: Any,
        run_outcome: RunOutcome,
        script_data: Optional[str],
    ) -> None:
        """
        Call the private SCRIPT_GENERATED emit on the strategy under test.
        """

        emit = cast(
            "Callable[..., Any]",
            strategy.__getattribute__("_IntentStrategy__emit_script_generated_event"),
        )
        await emit(script_data=script_data, run_outcome=run_outcome)

    async def test_completed_outcome_tags_telemetry_event(self) -> None:
        """
        On COMPLETED runs the SCRIPT_GENERATED event carries ``run_outcome='completed'``.
        """

        strategy, telemetry = self.__build_strategy(step_count=5)

        await self.__invoke_emit(
            strategy=strategy,
            run_outcome=RunOutcome.COMPLETED,
            script_data="open swiggy\nsearch biryani",
        )

        call = telemetry.info.call_args

        self.assertFalse(call.kwargs["is_empty"])
        self.assertEqual(call.kwargs["type"], FathomEvent.SCRIPT_GENERATED)
        self.assertEqual(call.kwargs["run_outcome"], RunOutcome.COMPLETED.value)

    async def test_cancelled_outcome_emits_partial_script_with_outcome_tag(self) -> None:
        """
        On CANCELLED runs the SCRIPT_GENERATED event still emits but tags ``run_outcome='cancelled'``.
        """

        strategy, telemetry = self.__build_strategy(step_count=12)
        await self.__invoke_emit(
            strategy=strategy,
            script_data="open app\ntap play",
            run_outcome=RunOutcome.CANCELLED,
        )

        call = telemetry.info.call_args

        self.assertEqual(call.kwargs["step"], 12)
        self.assertFalse(call.kwargs["is_empty"])
        self.assertEqual(call.kwargs["type"], FathomEvent.SCRIPT_GENERATED)
        self.assertEqual(call.kwargs["run_outcome"], RunOutcome.CANCELLED.value)

    async def test_cancelled_outcome_with_empty_script_still_emits(self) -> None:
        """
        An empty partial script on cancellation still emits a terminal event with is_empty=True.
        """

        strategy, telemetry = self.__build_strategy(step_count=0)

        await self.__invoke_emit(
            script_data=None,
            strategy=strategy,
            run_outcome=RunOutcome.CANCELLED,
        )

        call = telemetry.info.call_args

        self.assertTrue(call.kwargs["is_empty"])
        self.assertEqual(call.kwargs["type"], FathomEvent.SCRIPT_GENERATED)
        self.assertEqual(call.kwargs["run_outcome"], RunOutcome.CANCELLED.value)
