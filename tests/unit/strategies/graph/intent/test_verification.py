from __future__ import annotations

import unittest

from fathom.constants.state import IntentStateKey, VerifyMode
from fathom.core.agent.state import AgentState
from fathom.core.exceptions import InvariantViolation
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.reasoning import SubGoalCompletionSignal
from fathom.schemas.subgoal import SubGoal
from fathom.strategies.graph.intent.verification import VerificationModePolicy


class VerificationModePolicyTest(unittest.TestCase):
    """
    Pins VERIFY mode resolution from graph state and AgentState cursor position.
    """

    def setUp(self) -> None:
        """
        Build the policy under test.
        """

        self.__policy = VerificationModePolicy()

    @staticmethod
    def __caps() -> RuntimeCapabilities:
        """
        Return autonomous capabilities for AgentState fixtures.
        """

        return RuntimeCapabilities(hitl=HITLCapability(enabled=False))

    @staticmethod
    def __signal() -> SubGoalCompletionSignal:
        """
        Return a valid signal for advancing sub-goals.
        """

        return SubGoalCompletionSignal(
            llm_confidence=1.0,
            screen_verified=True,
            action_executed=True,
            flagged_complete=True,
            rationale_verified=True,
            evidence="unit test",
        )

    def test_producer_uses_sub_goal_mode_for_non_final_active_sub_goal(self) -> None:
        """
        Producers stamp SUB_GOAL while there is more sub-goal work after the active one.
        """

        agent_state = AgentState(intent="change address", capabilities=self.__caps())
        agent_state.set_sub_goals(
            [
                SubGoal(index=0, description="Open address selector"),
                SubGoal(index=1, description="Confirm SalarySe address"),
            ]
        )

        mode = self.__policy.mode_for_producer(agent_state=agent_state)

        self.assertIs(mode, VerifyMode.SUB_GOAL)

    def test_producer_uses_pending_final_commit_for_active_final_sub_goal(self) -> None:
        """
        Producers stamp PENDING_FINAL_COMMIT while the terminal sub-goal is still active.
        """

        agent_state = AgentState(intent="change address", capabilities=self.__caps())
        agent_state.set_sub_goals([SubGoal(index=0, description="Confirm SalarySe address")])

        mode = self.__policy.mode_for_producer(agent_state=agent_state)

        self.assertIs(mode, VerifyMode.PENDING_FINAL_COMMIT)

    def test_producer_uses_full_intent_after_all_sub_goals_complete(self) -> None:
        """
        Producers use full-intent mode once no active sub-goal remains.
        """

        agent_state = AgentState(intent="change address", capabilities=self.__caps())
        agent_state.set_sub_goals([SubGoal(index=0, description="Confirm SalarySe address")])
        agent_state.mark_current_sub_goal_complete(completion_signal=self.__signal())

        mode = self.__policy.mode_for_producer(agent_state=agent_state)

        self.assertIs(mode, VerifyMode.FULL_INTENT)

    def test_graph_state_explicit_mode_wins_over_cursor_inference(self) -> None:
        """
        Explicit VERIFY_MODE is the source of truth for new graph transitions.
        """

        agent_state = AgentState(intent="change address", capabilities=self.__caps())

        mode = self.__policy.mode_for_verify(
            state={IntentStateKey.VERIFY_MODE: VerifyMode.PENDING_FINAL_COMMIT.value},
            agent_state=agent_state,
        )

        self.assertIs(mode, VerifyMode.PENDING_FINAL_COMMIT)

    def test_graph_state_without_mode_uses_cursor_inference(self) -> None:
        """
        Old checkpoints without VERIFY_MODE infer mode from the active cursor.
        """

        agent_state = AgentState(intent="change address", capabilities=self.__caps())
        agent_state.set_sub_goals(
            [
                SubGoal(index=0, description="Open address selector"),
                SubGoal(index=1, description="Confirm SalarySe address"),
            ]
        )

        mode = self.__policy.mode_for_verify(state={}, agent_state=agent_state)

        self.assertIs(mode, VerifyMode.SUB_GOAL)

    def test_graph_state_without_mode_uses_full_intent_for_active_final_subgoal(self) -> None:
        """
        Missing VERIFY_MODE must not make an active final sub-goal use sub-goal verification.
        """

        agent_state = AgentState(intent="change address", capabilities=self.__caps())
        agent_state.set_sub_goals([SubGoal(index=0, description="Confirm SalarySe address")])

        mode = self.__policy.mode_for_verify(state={}, agent_state=agent_state)

        self.assertIs(mode, VerifyMode.PENDING_FINAL_COMMIT)

    def test_graph_state_unknown_mode_fails_fast(self) -> None:
        """
        Corrupt VERIFY_MODE values become explicit invariant violations.
        """

        agent_state = AgentState(intent="change address", capabilities=self.__caps())

        with self.assertRaises(InvariantViolation):
            self.__policy.mode_for_verify(
                state={IntentStateKey.VERIFY_MODE: "BAD_MODE"},
                agent_state=agent_state,
            )

    def test_graph_state_non_string_mode_fails_fast(self) -> None:
        """
        Non-string VERIFY_MODE values become explicit invariant violations.
        """

        agent_state = AgentState(intent="change address", capabilities=self.__caps())

        with self.assertRaisesRegex(InvariantViolation, "must be a string"):
            self.__policy.mode_for_verify(
                state={IntentStateKey.VERIFY_MODE: object()},
                agent_state=agent_state,
            )
