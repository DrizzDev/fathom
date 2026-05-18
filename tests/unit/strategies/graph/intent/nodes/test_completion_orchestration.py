from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

from fathom.constants import ActionType
from fathom.constants.state import CommonStateKey, IntentStateKey, PlanMetadataKey
from fathom.core.recovery import RecoveryTrigger
from fathom.schemas.actions import Action
from fathom.schemas.completion import CompletionVerdict
from fathom.schemas.observation import KeyboardObservation, ScreenObservation
from fathom.schemas.outcomes import ActionOutcome, OutcomeStatus
from fathom.schemas.results import AnalysisResult, PlanResult
from fathom.schemas.screens import ScreenCapture, ScreenHashBundle
from fathom.schemas.steps import Step, StepResult
from fathom.schemas.subgoal import SubGoal
from fathom.schemas.tasks import ExecutionTaskState, TaskStatus
from fathom.strategies.graph.intent.nodes.completion import SubGoalEvaluator


def _action() -> Action:
    """
    Minimal :class:`Action` used by step-level fixtures.
    """

    return Action(
        action_type=ActionType.TAP,
        target="Continue",
        rationale="t",
        confidence=1.0,
    )


def _step_result(*, success: bool = True, screen_changed: bool = True) -> StepResult:
    """
    Step-result fixture parameterised on success and screen-change so
    the failed-step short-circuit and the validation-bypass branches
    can be exercised independently.
    """

    return StepResult(
        step=Step(
            action=_action(),
            event_type="action",
            condition="x",
            screen_hash="0" * 16,
            step_number=1,
        ),
        success=success,
        pre_hash="0" * 16,
        post_hash="1" * 16,
        screen_changed=screen_changed,
        duration=10,
        generalized_target="Continue",
        is_positional=False,
    )


def _analysis(*, task_status: TaskStatus = TaskStatus.MET) -> AnalysisResult:
    """
    Analysis-result fixture wrapping the action so :meth:`evaluate`'s
    ``__analysis_from`` extractor returns a real value, and exposing
    a configurable ``task_status``.
    """

    return AnalysisResult(
        action=_action(),
        reasoning="r",
        screen_description="s",
        task_status=task_status,
    )


def _plan(*, analysis: AnalysisResult) -> PlanResult:
    """
    Plan-result fixture carrying the analysis under the canonical
    metadata key the evaluator reads.
    """

    return PlanResult(
        step=Step(
            action=_action(),
            event_type="action",
            condition="x",
            screen_hash="0" * 16,
            step_number=1,
        ),
        metadata={PlanMetadataKey.ANALYSIS.value: analysis},
        should_retry=False,
        is_complete=False,
        reason="t",
    )


def _capture() -> ScreenCapture:
    """
    Screen-capture fixture used by :meth:`recover_if_stuck` tests.
    """

    return ScreenCapture(
        width=100,
        height=200,
        activity="app",
        image=b"PNG",
        timestamp=0,
    )


def _observation() -> ScreenObservation:
    """
    Observation placeholder needed by :class:`ActionOutcome`.
    """

    return ScreenObservation(
        activity="app",
        elements=(),
        hashes=ScreenHashBundle(
            visual_hash="0" * 16,
            xml_hash="a" * 16,
            interaction_hash="b" * 16,
        ),
        keyboard=KeyboardObservation(visible=False),
    )


def _outcome(*, status: OutcomeStatus = OutcomeStatus.EFFECTIVE) -> ActionOutcome:
    """
    Action-outcome fixture parameterised on status so the floor branches
    (EFFECTIVE clears, NO_EFFECT blocks) can be driven.
    """

    return ActionOutcome(
        status=status,
        reason="t",
        action=_action(),
        before=_observation(),
    )


def _verdict(
    *, complete: bool, next_state: ExecutionTaskState = ExecutionTaskState.SUCCEEDED
) -> CompletionVerdict:
    """
    Completion-verdict fixture driving the post-floor advancement path.
    """

    return CompletionVerdict(
        complete=complete,
        next_state=next_state,
        reason="t",
        missing=(),
    )


def _signal(*, claim_verified: bool = True, action_effective: bool = True) -> SimpleNamespace:
    """
    Reasoner signal stub. The evaluator reads ``claim_verified`` to
    decide whether to override the task status before consulting the
    completion service.
    """

    return SimpleNamespace(
        claim_verified=claim_verified,
        action_effective=action_effective,
    )


class _StubContext:
    """
    :class:`GraphContext` test double exposing only the surface
    :class:`SubGoalEvaluator` actually consumes.

    Provides knobs to configure the agent state (sub-goals, stuck flag,
    no-progress count, over-budget flag), the reasoner verdict, and the
    completion-service verdict so each test can pin one specific branch.
    """

    def __init__(
        self,
        *,
        sub_goals: Optional[List[SubGoal]] = None,
        is_stuck: bool = False,
        no_progress_count: int = 0,
        over_budget: bool = False,
        reasoner_signal: Optional[SimpleNamespace] = None,
        verdict: Optional[CompletionVerdict] = None,
        has_more: bool = True,
    ) -> None:
        """
        Initialise the stub with the test's per-branch configuration.
        Tracks every state-mutating call so the test can assert that
        the evaluator advanced the sub-goal index when expected.
        """

        active = sub_goals[0] if sub_goals else None
        self.advance_calls: int = 0
        self.completion_calls: int = 0
        self.agent_state = SimpleNamespace(
            step_count=0,
            sub_goal_list=list(sub_goals or []),
            has_sub_goals=lambda: bool(sub_goals),
            get_current_sub_goal=lambda: active,
            get_sub_goal_progress=lambda: (0, len(sub_goals or [])),
            mark_current_sub_goal_complete=self.__advance,
            mark_complete=self.__mark_complete,
            is_stuck=is_stuck,
            consecutive_no_progress_count=no_progress_count,
            current_sub_goal_over_budget=over_budget,
            current_sub_goal_action_count=3,
            last_delta_score=None,
        )
        self.reasoner = SimpleNamespace(
            analyze_subgoal_completion=MagicMock(
                return_value=reasoner_signal or _signal(),
            ),
        )
        self.completion_service = SimpleNamespace(
            evaluate=MagicMock(return_value=verdict or _verdict(complete=True)),
        )
        self.event_emitter = SimpleNamespace(emit=AsyncMock())
        self.workflow_id = "run-test"
        self.completion_reason: Optional[str] = None
        self.__has_more = has_more

    def __advance(self, *, completion_signal: Any) -> bool:
        """
        Tick the advance counter and return the preconfigured has-more
        flag so the test can drive the "more sub-goals" vs "all done"
        branches.
        """

        _ = completion_signal
        self.advance_calls += 1
        return self.__has_more

    def __mark_complete(self, *, reason: str) -> None:
        """
        Capture the final completion reason for the all-done branch
        assertion.
        """

        self.completion_calls += 1
        self.completion_reason = reason


class SubGoalEvaluatorEvaluateTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins :meth:`SubGoalEvaluator.evaluate` orchestration branches.

    Tests cover the four routing decisions: no active sub-goal,
    failed step, floor-blocks-advancement, and floor-clears with
    has-more vs all-done. The completion service and reasoner are
    stubbed so the evaluator's own decision logic is what's under test.
    """

    @staticmethod
    def __dispatcher() -> SimpleNamespace:
        """
        Recovery dispatcher stub. :meth:`evaluate` does not call into
        recovery, so the stub is intentionally empty.
        """

        return SimpleNamespace()

    async def test_returns_none_when_no_active_sub_goal(self) -> None:
        """
        With no sub-goals on the agent state, ``evaluate`` short-
        circuits — there is nothing to advance.
        """

        ctx = _StubContext(sub_goals=None)
        evaluator = SubGoalEvaluator(context=ctx, recovery=self.__dispatcher())  # type: ignore[arg-type]

        result = await evaluator.evaluate(
            plan=_plan(analysis=_analysis()),
            step_result=_step_result(),
            accumulated=[],
            outcome=_outcome(),
        )

        self.assertIsNone(result)

    async def test_returns_none_on_failed_step(self) -> None:
        """
        A failed step never advances a sub-goal; the evaluator returns
        ``None`` and never consults the reasoner or completion service.
        """

        sub_goals = [SubGoal(index=0, description="Open the app")]
        ctx = _StubContext(sub_goals=sub_goals)
        evaluator = SubGoalEvaluator(context=ctx, recovery=self.__dispatcher())  # type: ignore[arg-type]

        result = await evaluator.evaluate(
            plan=_plan(analysis=_analysis()),
            step_result=_step_result(success=False),
            accumulated=[],
            outcome=_outcome(),
        )

        self.assertIsNone(result)
        ctx.reasoner.analyze_subgoal_completion.assert_not_called()

    async def test_returns_none_when_plan_has_no_analysis_metadata(self) -> None:
        """
        Without an :class:`AnalysisResult` in plan metadata the evaluator
        cannot make a decision; it must return ``None``.
        """

        sub_goals = [SubGoal(index=0, description="Open the app")]
        ctx = _StubContext(sub_goals=sub_goals)
        evaluator = SubGoalEvaluator(context=ctx, recovery=self.__dispatcher())  # type: ignore[arg-type]

        bare_plan = PlanResult(
            step=Step(
                action=_action(),
                event_type="action",
                condition="x",
                screen_hash="0" * 16,
                step_number=1,
            ),
            metadata={},
            should_retry=False,
            is_complete=False,
            reason="t",
        )

        result = await evaluator.evaluate(
            plan=bare_plan,
            step_result=_step_result(),
            accumulated=[],
            outcome=_outcome(),
        )

        self.assertIsNone(result)

    async def test_floor_blocks_advancement_returns_none(self) -> None:
        """
        When the floor blocks (NOT_MET + EFFECTIVE outcome), the
        verdict is overridden to ``complete=False`` and the evaluator
        returns ``None`` regardless of the reasoner's claim.
        """

        sub_goals = [SubGoal(index=0, description="Open the app")]
        ctx = _StubContext(
            sub_goals=sub_goals,
            verdict=_verdict(complete=True),
        )
        evaluator = SubGoalEvaluator(context=ctx, recovery=self.__dispatcher())  # type: ignore[arg-type]

        result = await evaluator.evaluate(
            plan=_plan(analysis=_analysis(task_status=TaskStatus.NOT_MET)),
            step_result=_step_result(),
            accumulated=[],
            outcome=_outcome(),
        )

        self.assertIsNone(result)
        self.assertEqual(ctx.advance_calls, 0)

    async def test_floor_clears_with_more_sub_goals_routes_to_retry(self) -> None:
        """
        Floor clears, has-more is true → patch flips ``SHOULD_RETRY``
        and carries the accumulated step results forward.
        """

        sub_goals = [
            SubGoal(index=0, description="Open the app"),
            SubGoal(index=1, description="Tap search"),
        ]
        ctx = _StubContext(
            sub_goals=sub_goals,
            verdict=_verdict(complete=True),
            has_more=True,
        )
        evaluator = SubGoalEvaluator(context=ctx, recovery=self.__dispatcher())  # type: ignore[arg-type]
        accumulated: List[StepResult] = [_step_result()]

        result = await evaluator.evaluate(
            plan=_plan(analysis=_analysis(task_status=TaskStatus.MET)),
            step_result=_step_result(),
            accumulated=accumulated,
            outcome=_outcome(status=OutcomeStatus.EFFECTIVE),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))
        self.assertEqual(result.get(IntentStateKey.STEP_RESULTS), accumulated)
        self.assertEqual(ctx.advance_calls, 1)

    async def test_floor_clears_with_no_more_sub_goals_marks_complete(self) -> None:
        """
        Last sub-goal advancing → the run is marked complete with the
        sequential-completion reason and the patch carries
        ``IS_COMPLETE=True``.
        """

        sub_goals = [SubGoal(index=0, description="Open the app")]
        ctx = _StubContext(
            sub_goals=sub_goals,
            verdict=_verdict(complete=True),
            has_more=False,
        )
        evaluator = SubGoalEvaluator(context=ctx, recovery=self.__dispatcher())  # type: ignore[arg-type]

        result = await evaluator.evaluate(
            plan=_plan(analysis=_analysis(task_status=TaskStatus.MET)),
            step_result=_step_result(),
            accumulated=[],
            outcome=_outcome(status=OutcomeStatus.EFFECTIVE),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(ctx.completion_calls, 1)
        self.assertIn("All sub-goals completed", ctx.completion_reason or "")


class SubGoalEvaluatorRecoverIfStuckTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins :meth:`SubGoalEvaluator.recover_if_stuck` dispatch routing.

    The method consults three independent stuck-signals (loop detector,
    consecutive-no-progress count, sub-goal-over-budget) in that
    priority order and dispatches the matching :class:`RecoveryTrigger`.
    """

    @staticmethod
    def __recovery_stub() -> SimpleNamespace:
        """
        Recovery dispatcher stub recording every ``try_recover`` call
        so the test can assert which trigger fired.
        """

        return SimpleNamespace(
            try_recover=AsyncMock(return_value={"recovered": True}),
        )

    async def test_loop_detected_dispatches_loop_recovery(self) -> None:
        """
        Loop-detector stuck → dispatcher receives
        :attr:`RecoveryTrigger.LOOP_DETECTED`.
        """

        sub_goals = [SubGoal(index=0, description="Open the app")]
        ctx = _StubContext(sub_goals=sub_goals, is_stuck=True)
        recovery = self.__recovery_stub()
        evaluator = SubGoalEvaluator(context=ctx, recovery=recovery)  # type: ignore[arg-type]

        result = await evaluator.recover_if_stuck(
            capture=_capture(),
            step_result=_step_result(),
        )

        self.assertEqual(result, {"recovered": True})
        recovery.try_recover.assert_awaited_once()
        self.assertEqual(
            recovery.try_recover.await_args.kwargs["trigger"],
            RecoveryTrigger.LOOP_DETECTED,
        )

    async def test_consecutive_no_progress_dispatches_no_progress_recovery(self) -> None:
        """
        Loop is clean but the no-progress counter hits the threshold →
        dispatcher receives :attr:`RecoveryTrigger.NO_PROGRESS`.
        """

        sub_goals = [SubGoal(index=0, description="Open the app")]
        ctx = _StubContext(sub_goals=sub_goals, no_progress_count=10)
        recovery = self.__recovery_stub()
        evaluator = SubGoalEvaluator(context=ctx, recovery=recovery)  # type: ignore[arg-type]

        await evaluator.recover_if_stuck(
            capture=_capture(),
            step_result=_step_result(),
        )

        recovery.try_recover.assert_awaited_once()
        self.assertEqual(
            recovery.try_recover.await_args.kwargs["trigger"],
            RecoveryTrigger.NO_PROGRESS,
        )

    async def test_over_budget_dispatches_budget_recovery(self) -> None:
        """
        Loop / no-progress signals clean but the sub-goal is over its
        attempt budget → dispatcher receives
        :attr:`RecoveryTrigger.SUBGOAL_BUDGET_EXCEEDED`.
        """

        sub_goals = [SubGoal(index=0, description="Open the app")]
        ctx = _StubContext(sub_goals=sub_goals, over_budget=True)
        recovery = self.__recovery_stub()
        evaluator = SubGoalEvaluator(context=ctx, recovery=recovery)  # type: ignore[arg-type]

        await evaluator.recover_if_stuck(
            capture=_capture(),
            step_result=_step_result(),
        )

        recovery.try_recover.assert_awaited_once()
        self.assertEqual(
            recovery.try_recover.await_args.kwargs["trigger"],
            RecoveryTrigger.SUBGOAL_BUDGET_EXCEEDED,
        )

    async def test_no_stuck_signal_returns_none(self) -> None:
        """
        Clean state across all three signals → no dispatch, no patch.
        """

        sub_goals = [SubGoal(index=0, description="Open the app")]
        ctx = _StubContext(sub_goals=sub_goals)
        recovery = self.__recovery_stub()
        evaluator = SubGoalEvaluator(context=ctx, recovery=recovery)  # type: ignore[arg-type]

        result = await evaluator.recover_if_stuck(
            capture=_capture(),
            step_result=_step_result(),
        )

        self.assertIsNone(result)
        recovery.try_recover.assert_not_called()
