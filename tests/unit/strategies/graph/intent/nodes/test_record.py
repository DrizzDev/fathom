from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fathom.constants import ActionType
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey, VerifyMode
from fathom.core.agent.state import AgentState
from fathom.schemas.actions import Action
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.reasoning import SubGoalCompletionSignal
from fathom.schemas.results import PlanResult
from fathom.schemas.screens import ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.schemas.subgoal import SubGoal
from fathom.strategies.graph.intent.nodes.record import RecordNode


class RecordNodeFailureTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins terminal failure behavior for invalid RECORD inputs.
    """

    @staticmethod
    def __provider(*, cancelled: bool = False) -> MagicMock:
        """
        Return the provider surface used by the invalid-state branch.
        """

        provider = MagicMock(name="IntentNodeProvider")
        provider.is_cancelled = AsyncMock(return_value=cancelled)
        provider.context.workflow_id = "run-test"
        provider.persistence.persist = MagicMock()
        return provider

    async def test_cancelled_record_persists_terminal_patch(self) -> None:
        """
        Cancellation at RECORD must be a persisted terminal state with stale retry cleared.
        """

        provider = self.__provider(cancelled=True)
        node = RecordNode(provider=provider)

        result: Any = await node(state={})  # type: ignore[arg-type]

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertFalse(result.get(IntentStateKey.SHOULD_RETRY))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.CANCELLED.value,
        )
        provider.persistence.persist.assert_called_once_with(result=result)

    async def test_missing_step_result_fails_terminally(self) -> None:
        """
        RECORD must not return an empty patch when OBSERVE did not stage a StepResult.
        """

        provider = self.__provider()
        node = RecordNode(provider=provider)

        result: Any = await node(state={})  # type: ignore[arg-type]

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertFalse(result.get(IntentStateKey.SHOULD_RETRY))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.FAILED.value,
        )
        self.assertIn("missing StepResult", result.get(CommonStateKey.FAILURE_DIAGNOSTIC))
        provider.context.agent_state.mark_complete.assert_called_once_with(
            reason=CompletionReason.FAILED.value
        )
        provider.persistence.persist.assert_called_once()


class RecordNodeCompletionRouteTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins completion routing when the planner marks the post-action plan complete.
    """

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

    @staticmethod
    def __step_result() -> StepResult:
        """
        Build a minimal successful tap StepResult.
        """

        action = Action(
            action_type=ActionType.TAP,
            target="Continue",
            rationale="tap continue",
            confidence=1.0,
        )
        return StepResult(
            step=Step(action=action, step_number=0, screen_hash="pre"),
            success=True,
            duration=12,
            screen_changed=True,
            pre_hash="pre",
            post_hash="post",
        )

    @staticmethod
    def __screen() -> ScreenState:
        """
        Build a stable screen state for RecordNode bookkeeping.
        """

        return ScreenState(
            timestamp=0,
            activity="com.test/.MainActivity",
            activity_hash="a" * 16,
            visual_hash="b" * 16,
        )

    def __provider(self, *, agent_state: AgentState) -> MagicMock:
        """
        Return a provider fixture that reaches the PlanResult completion branch.
        """

        provider = MagicMock(name="IntentNodeProvider")
        provider.is_cancelled = AsyncMock(return_value=False)
        provider.context.workflow_id = "run-test"
        provider.context.agent_state = agent_state
        provider.context.telemetry.info = AsyncMock()
        provider.context.telemetry.error = AsyncMock()
        provider.context.telemetry.warning = AsyncMock()
        provider.context.memory.store_experience = AsyncMock()
        provider.context.context_manager.commit = AsyncMock()
        provider.context.context_manager.get_full_context.return_value = {"active_count": 0}
        provider.context.auditor.log_step = MagicMock()
        provider.persistence.persist = MagicMock()
        provider.persistence.enqueue_history = MagicMock()
        provider.persistence.should_skip_launcher.return_value = False
        provider.completion.evaluate = AsyncMock(return_value=None)
        return provider

    async def test_plan_complete_with_active_final_subgoal_routes_pending_verify(self) -> None:
        """
        A planner completion on the active final sub-goal must not mark AgentState complete before VERIFY accepts it.
        """

        agent_state = AgentState(intent="change address", capabilities=self.__caps())
        agent_state.set_sub_goals([SubGoal(index=0, description="Confirm SalarySe address")])
        agent_state.record_complete_deferral()
        agent_state.record_verify_rejection(
            screen=self.__screen(), activity="com.test/.MainActivity"
        )
        provider = self.__provider(agent_state=agent_state)
        node = RecordNode(provider=provider)

        result = await node(
            state={
                CommonStateKey.STEP_RESULT: self.__step_result(),
                CommonStateKey.SCREEN_STATE: self.__screen(),
                CommonStateKey.IS_NEW_SCREEN: False,
                IntentStateKey.POST_ACTIVITY: "com.test/.MainActivity",
                IntentStateKey.PLAN: PlanResult(
                    is_complete=True,
                    reason="Address selected",
                    step=None,
                ),
            }
        )

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertEqual(
            result[IntentStateKey.VERIFY_MODE],
            VerifyMode.PENDING_FINAL_COMMIT.value,
        )
        self.assertFalse(agent_state.is_complete)
        self.assertIsNotNone(agent_state.get_current_sub_goal())
        self.assertIsNone(agent_state.verification_loop)
        self.assertEqual(agent_state.consecutive_complete_deferrals, 0)
        provider.completion.evaluate.assert_not_called()

    async def test_hsr_final_confirmation_replay_routes_pending_verify(self) -> None:
        """
        HSR prod replay: after tapping the final confirmation, RECORD must route to pending final VERIFY without committing the final sub-goal.
        """

        agent_state = AgentState(
            intent="Change the address to HSR Layout",
            capabilities=self.__caps(),
        )
        agent_state.set_sub_goals(
            [
                SubGoal(index=0, description="Tap on the current address or change address"),
                SubGoal(index=1, description="Type HSR Layout into the address search field"),
                SubGoal(index=2, description="Tap HSR Layout from the search results"),
                SubGoal(
                    index=3, description="Tap the button to confirm or save the address change"
                ),
            ]
        )
        for _ in range(3):
            agent_state.mark_current_sub_goal_complete(completion_signal=self.__signal())

        provider = self.__provider(agent_state=agent_state)
        node = RecordNode(provider=provider)

        result = await node(
            state={
                CommonStateKey.STEP_RESULT: self.__step_result(),
                CommonStateKey.SCREEN_STATE: self.__screen(),
                CommonStateKey.IS_NEW_SCREEN: True,
                IntentStateKey.POST_ACTIVITY: "in.swiggy.android",
                IntentStateKey.PLAN: PlanResult(
                    is_complete=True,
                    reason="HSR Layout is selected after tapping Yes, continue with this location",
                    step=None,
                ),
            }
        )

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertFalse(result[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(
            result[IntentStateKey.VERIFY_MODE],
            VerifyMode.PENDING_FINAL_COMMIT.value,
        )
        self.assertFalse(agent_state.is_complete)
        self.assertEqual(agent_state.current_sub_goal_index, 3)
        self.assertFalse(agent_state.all_sub_goals_complete())
        provider.completion.evaluate.assert_not_called()

    async def test_recording_exception_fails_terminally(self) -> None:
        """
        RECORD infrastructure failures must not return a non-terminal patch that loops to GROUND.
        """

        agent_state = AgentState(intent="change address", capabilities=self.__caps())
        agent_state.record_complete_deferral()
        agent_state.record_verify_rejection(
            screen=self.__screen(), activity="com.test/.MainActivity"
        )
        provider = self.__provider(agent_state=agent_state)
        provider.context.memory.store_experience = AsyncMock(
            side_effect=RuntimeError("store broke")
        )
        node = RecordNode(provider=provider)

        result = await node(
            state={
                CommonStateKey.STEP_RESULT: self.__step_result(),
                CommonStateKey.SCREEN_STATE: self.__screen(),
                CommonStateKey.IS_NEW_SCREEN: False,
                IntentStateKey.POST_ACTIVITY: "com.test/.MainActivity",
            }
        )

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertFalse(result[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(result[CommonStateKey.COMPLETION_REASON], CompletionReason.FAILED.value)
        self.assertEqual(agent_state.completion_reason, CompletionReason.FAILED.value)
        self.assertIsNone(agent_state.verification_loop)
        self.assertEqual(agent_state.consecutive_complete_deferrals, 0)

    async def test_plan_complete_without_active_subgoal_preserves_legacy_completion(self) -> None:
        """
        Legacy no-active-subgoal completion still marks AgentState complete immediately.
        """

        agent_state = AgentState(intent="change address", capabilities=self.__caps())
        agent_state.set_sub_goals([SubGoal(index=0, description="Confirm SalarySe address")])
        agent_state.mark_current_sub_goal_complete(completion_signal=self.__signal())
        agent_state.record_complete_deferral()
        provider = self.__provider(agent_state=agent_state)
        node = RecordNode(provider=provider)

        result = await node(
            state={
                CommonStateKey.STEP_RESULT: self.__step_result(),
                CommonStateKey.SCREEN_STATE: self.__screen(),
                CommonStateKey.IS_NEW_SCREEN: False,
                IntentStateKey.POST_ACTIVITY: "com.test/.MainActivity",
                IntentStateKey.PLAN: PlanResult(
                    is_complete=True,
                    reason="Address selected",
                    step=None,
                ),
            }
        )

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertEqual(
            result[IntentStateKey.VERIFY_MODE],
            VerifyMode.FULL_INTENT.value,
        )
        self.assertTrue(agent_state.is_complete)
        self.assertEqual(agent_state.completion_reason, "Address selected")
        self.assertEqual(agent_state.consecutive_complete_deferrals, 0)
        provider.completion.evaluate.assert_not_called()

    async def test_plan_complete_with_blank_reason_normalizes_completion_reason(self) -> None:
        """
        RECORD completion claims must not persist a blank completion reason.
        """

        agent_state = AgentState(intent="change address", capabilities=self.__caps())
        provider = self.__provider(agent_state=agent_state)
        node = RecordNode(provider=provider)

        result = await node(
            state={
                CommonStateKey.STEP_RESULT: self.__step_result(),
                CommonStateKey.SCREEN_STATE: self.__screen(),
                CommonStateKey.IS_NEW_SCREEN: False,
                IntentStateKey.POST_ACTIVITY: "com.test/.MainActivity",
                IntentStateKey.PLAN: PlanResult(
                    is_complete=True,
                    reason="",
                    step=None,
                ),
            }
        )

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertEqual(
            result[CommonStateKey.COMPLETION_REASON],
            CompletionReason.SUCCESS.value,
        )
        self.assertEqual(agent_state.completion_reason, CompletionReason.SUCCESS.value)
