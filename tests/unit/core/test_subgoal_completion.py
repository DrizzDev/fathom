"""Tests for sub-goal completion signal flow, stuck detection, and guidance clearing."""

from __future__ import annotations

from fathom.core.agent.reasoner import Reasoner
from fathom.core.agent.state import AgentState
from fathom.schemas.actions import Action
from fathom.schemas.reasoning import SubGoalCompletionSignal
from fathom.schemas.results import AnalysisResult
from fathom.schemas.subgoal import SubGoal, SubGoalStatus


def _make_sub_goals(descriptions: list[str]) -> list[SubGoal]:
    return [SubGoal(index=i, description=d) for i, d in enumerate(descriptions)]


def _make_analysis(
    *,
    is_goal_complete: bool = False,
    is_sub_goal_complete: bool = False,
    action_type: str = "tap",
    reasoning: str = "test reasoning",
    subgoal_completion_reason: str = "",
) -> AnalysisResult:
    return AnalysisResult(
        action=Action(
            confidence=0.9,
            rationale=reasoning,
            action_type=action_type,
            target="test_target",
        ),
        alternatives=[],
        reasoning=reasoning,
        is_goal_complete=is_goal_complete,
        is_sub_goal_complete=is_sub_goal_complete,
        subgoal_completion_reason=subgoal_completion_reason or None,
        screen_description="test screen",
    )


class TestReasonerSubGoalCompletion:
    """Test that the reasoner correctly computes sub-goal completion signals."""

    def test_llm_signaled_requires_explicit_flag(self) -> None:
        """COMPLETE action type alone should NOT trigger llm_signaled."""

        reasoner = Reasoner(intent="open app and search")
        analysis = _make_analysis(action_type="complete", is_sub_goal_complete=False)

        signal = reasoner.analyze_subgoal_completion(
            analysis=analysis,
            sub_goal_description="open the app",
        )

        # action_type=complete was removed from llm_signaled check
        assert signal.llm_signaled is False

    def test_llm_signaled_true_when_sub_goal_flag_set(self) -> None:
        reasoner = Reasoner(intent="open app")
        analysis = _make_analysis(is_sub_goal_complete=True)

        signal = reasoner.analyze_subgoal_completion(
            analysis=analysis,
            sub_goal_description="open the app",
        )

        assert signal.llm_signaled is True

    def test_rationale_always_false(self) -> None:
        """Rationale verification is disabled — always False."""

        reasoner = Reasoner(intent="open app")
        analysis = _make_analysis(
            is_sub_goal_complete=True,
            subgoal_completion_reason="App opened successfully",
        )

        signal = reasoner.analyze_subgoal_completion(
            analysis=analysis,
            sub_goal_description="open the app",
        )

        assert signal.rationale_verified is False

    def test_screen_verified_requires_screen_change(self) -> None:
        reasoner = Reasoner(intent="test")
        analysis = _make_analysis(is_sub_goal_complete=True)

        signal = reasoner.analyze_subgoal_completion(
            analysis=analysis,
            sub_goal_description="tap button",
            screen_changed=False,
            pre_screen_hash="aaaa",
            post_screen_hash="aaaa",
        )

        assert signal.screen_verified is False

    def test_two_signal_gate_passes(self) -> None:
        """llm_signaled + effective_action (action + screen) = 2 signals."""

        reasoner = Reasoner(intent="test")
        analysis = _make_analysis(is_sub_goal_complete=True, action_type="tap")

        signal = reasoner.analyze_subgoal_completion(
            analysis=analysis,
            sub_goal_description="tap the button",
            screen_changed=True,
        )

        assert signal.llm_signaled is True
        assert signal.action_executed is True
        assert signal.screen_verified is True
        assert signal.count_signals() == 2
        assert signal.meets_threshold(required_signals=2)

    def test_two_signal_gate_fails_without_screen_change(self) -> None:
        reasoner = Reasoner(intent="test")
        analysis = _make_analysis(is_sub_goal_complete=True, action_type="tap")

        signal = reasoner.analyze_subgoal_completion(
            analysis=analysis,
            sub_goal_description="tap the button",
            screen_changed=False,
        )

        # llm_signaled=True, action_executed=True, but screen_verified=False
        # effective_action = action_executed AND screen_verified = False
        assert signal.count_signals() == 1  # only llm_signaled
        assert not signal.meets_threshold(required_signals=2)


class TestAgentStateSubGoalTracking:
    """Test action count and verify failure tracking."""

    def test_action_count_increments(self) -> None:
        state = AgentState(intent="test", max_steps=50)
        state.set_sub_goals(_make_sub_goals(["step 1"]))

        assert state.sub_goal_action_count == 0
        state.record_sub_goal_action()
        state.record_sub_goal_action()
        assert state.sub_goal_action_count == 2

    def test_verify_failure_increments(self) -> None:
        state = AgentState(intent="test", max_steps=50)
        state.set_sub_goals(_make_sub_goals(["step 1"]))

        assert state.sub_goal_verify_failures == 0
        state.record_verify_failure()
        assert state.sub_goal_verify_failures == 1

    def test_counters_reset_on_subgoal_advance(self) -> None:
        state = AgentState(intent="test", max_steps=50)
        state.set_sub_goals(_make_sub_goals(["step 1", "step 2"]))

        for _ in range(5):
            state.record_sub_goal_action()
        state.record_verify_failure()
        state.record_verify_failure()

        signal = SubGoalCompletionSignal(
            evidence="done", llm_signaled=True, action_executed=True, screen_verified=True
        )
        state.mark_current_sub_goal_complete(completion_signal=signal)

        assert state.sub_goal_action_count == 0
        assert state.sub_goal_verify_failures == 0

    def test_complete_action_does_not_mark_intent_complete(self) -> None:
        """COMPLETE action type in record_step should NOT mark the intent done."""

        state = AgentState(intent="test", max_steps=50)
        state.set_sub_goals(_make_sub_goals(["step 1"]))

        from fathom.schemas.steps import Step, StepResult

        step = Step(
            action=Action(
                confidence=1.0,
                rationale="done",
                action_type="complete",
                target="completion",
            ),
            screen_hash="aabbccdd",
            step_number=1,
        )
        result = StepResult(
            step=step,
            success=True,
            duration=100,
            pre_hash="aabbccdd",
            post_hash="aabbccdd",
            screen_changed=False,
        )

        state.record_step(result=result)
        assert state.is_complete is False


class TestSubGoalListExposure:
    """Test that sub_goals property returns correct data."""

    def test_sub_goals_returns_copy(self) -> None:
        state = AgentState(intent="test", max_steps=50)
        goals = _make_sub_goals(["a", "b", "c"])
        state.set_sub_goals(goals)

        exposed = state.sub_goals
        assert len(exposed) == 3
        assert exposed[0].description == "a"

        # Modifying returned list should not affect internal state
        exposed.pop()
        assert len(state.sub_goals) == 3
