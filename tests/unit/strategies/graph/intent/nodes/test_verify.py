from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, Dict, List

from fathom.constants import ActionType
from fathom.constants.runtime import DEFAULT_VERIFICATION_REJECTION_LIMIT
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey, VerifyMode
from fathom.core.agent.state import AgentState
from fathom.schemas.actions import Action
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.reasoning import SubGoalCompletionSignal
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.schemas.subgoal import SubGoal
from fathom.strategies.graph.intent.nodes.verify import VerifyNode


class _LLM:
    def __init__(self, *, content: str, raises: Exception | None = None) -> None:
        self.content = content
        self.raises = raises
        self.prompts: List[str] = []

    async def generate(self, **kwargs: object) -> SimpleNamespace:
        if self.raises is not None:
            raise self.raises

        prompt = kwargs.get("prompt")
        if isinstance(prompt, list) and prompt:
            self.prompts.append(str(prompt[0]))
        return SimpleNamespace(content=self.content)


class _Perception:
    def __init__(
        self,
        *,
        image: bytes = b"png",
        activity: str = "com.test",
        state: ScreenState | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.__image = image
        self.__activity = activity
        self.__state = state
        self.__raises = raises

    async def perceive(self, **_: object) -> ScreenCapture:
        if self.__raises is not None:
            raise self.__raises

        return ScreenCapture(
            width=100,
            height=200,
            activity=self.__activity,
            image=self.__image,
            timestamp=1,
            state=self.__state,
        )


class _ContextManager:
    def __init__(self) -> None:
        self.feedback: List[str] = []
        self.cleared = False

    def get_user_guidance(self) -> list[object]:
        return []

    def get_full_context(self) -> Dict[str, object]:
        return {"trace": [{"action": {"action_type": "tap", "target": "Continue"}}]}

    async def inject_verifier_feedback(self, *, feedback: str, step: int | None = None) -> None:
        _ = step
        self.feedback.append(feedback)

    def clear_verifier_feedback(self) -> None:
        self.cleared = True


class _Persistence:
    def __init__(self) -> None:
        self.last: Dict[Any, Any] = {}

    def restore(self, *, state: Dict[Any, Any]) -> None:
        _ = state

    def persist(self, *, result: Dict[Any, Any]) -> None:
        self.last = dict(result)


class _Provider:
    def __init__(
        self,
        *,
        agent_state: AgentState,
        llm_content: str,
        llm_error: Exception | None = None,
        capture_image: bytes = b"png",
        capture_activity: str = "com.test",
        capture_state: ScreenState | None = None,
        capture_error: Exception | None = None,
    ) -> None:
        self.context = SimpleNamespace(
            llm=_LLM(content=llm_content, raises=llm_error),
            intent="finish onboarding",
            max_steps=10,
            workflow_id="run-test",
            perception=_Perception(
                image=capture_image,
                activity=capture_activity,
                state=capture_state,
                raises=capture_error,
            ),
            agent_state=agent_state,
            artifact_pipeline=None,
            context_manager=_ContextManager(),
        )
        self.persistence = _Persistence()

    async def is_cancelled(self) -> bool:
        return False


class VerifyNodeSubGoalTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _screen() -> ScreenState:
        """
        Return a stable current screen for verifier-loop accounting.
        """

        return ScreenState(
            activity="com.test",
            activity_hash="a" * 16,
            visual_hash="1" * 16,
            timestamp=1,
        )

    def _agent_state(self) -> AgentState:
        state = AgentState(
            intent="finish onboarding",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        state.update_screen(screen=self._screen())
        state.set_sub_goals(
            [
                SubGoal(index=0, description="Open the app"),
                SubGoal(index=1, description="Reach the Home screen"),
            ]
        )
        state.mark_complete(reason="Sub-goal pending verification")
        return state

    def _final_agent_state(self) -> AgentState:
        state = AgentState(
            intent="change the address to salaryse office",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        state.update_screen(screen=self._screen())
        state.set_sub_goals(
            [
                SubGoal(index=0, description="Tap address selector"),
                SubGoal(index=1, description="Confirm SalarySe office address"),
            ]
        )
        state.mark_current_sub_goal_complete(
            completion_signal=SubGoalCompletionSignal(
                llm_confidence=1.0,
                screen_verified=True,
                action_executed=True,
                flagged_complete=True,
                rationale_verified=True,
                evidence="first step complete",
            )
        )
        return state

    @staticmethod
    def _capture_screen(*, activity: str = "com.test") -> ScreenState:
        """
        Return a stable screen state attached directly to a VERIFY capture.
        """

        return ScreenState(
            activity=activity,
            activity_hash="c" * 16,
            visual_hash="2" * 16,
            timestamp=1,
        )

    @staticmethod
    def _record_validate_step(agent_state: AgentState, *, step_number: int) -> None:
        """
        Record one successful validate-style step between verifier rejections.
        """

        agent_state.record_step(
            result=StepResult(
                step=Step(
                    action=Action(
                        action_type=ActionType.VALIDATE,
                        rationale="claim completion",
                        target="current screen",
                    ),
                    step_number=step_number,
                    screen_hash="b" * 16,
                ),
                success=True,
                pre_hash="b" * 16,
                post_hash="b" * 16,
                screen_changed=False,
                duration=10,
            )
        )

    async def test_subgoal_verification_advances_without_finishing_intent(self) -> None:
        provider = _Provider(
            agent_state=self._agent_state(),
            llm_content='{"is_complete": true, "reason": "App is open"}',
        )
        node = VerifyNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(state={})  # type: ignore[arg-type]

        self.assertFalse(result[CommonStateKey.IS_COMPLETE])
        self.assertTrue(result[IntentStateKey.SHOULD_RETRY])
        self.assertIsNone(result[IntentStateKey.PLAN])
        self.assertIsNone(result[IntentStateKey.PLANNED_STEP])
        self.assertIsNone(result[CommonStateKey.COMPLETION_REASON])
        self.assertEqual(provider.context.agent_state.current_sub_goal_index, 1)
        self.assertFalse(provider.context.agent_state.is_complete)
        self.assertIn("Step: Open the app", provider.context.llm.prompts[0])
        self.assertIsNone(result[IntentStateKey.VERIFY_MODE])

    async def test_subgoal_verification_failure_keeps_same_subgoal(self) -> None:
        provider = _Provider(
            agent_state=self._agent_state(),
            llm_content='{"is_complete": false, "reason": "Still on login"}',
        )
        node = VerifyNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(state={})  # type: ignore[arg-type]

        self.assertFalse(result[CommonStateKey.IS_COMPLETE])
        self.assertTrue(result[IntentStateKey.SHOULD_RETRY])
        self.assertIsNone(result[IntentStateKey.PLAN])
        self.assertIsNone(result[IntentStateKey.PLANNED_STEP])
        self.assertIsNone(result[CommonStateKey.COMPLETION_REASON])
        self.assertEqual(provider.context.agent_state.current_sub_goal_index, 0)
        self.assertEqual(
            provider.context.context_manager.feedback, ["Verification failed: Still on login"]
        )
        self.assertIsNone(result[IntentStateKey.VERIFY_MODE])

    async def test_pending_final_commit_uses_full_intent_prompt_and_commits_on_acceptance(
        self,
    ) -> None:
        provider = _Provider(
            agent_state=self._final_agent_state(),
            llm_content='{"is_complete": true, "reason": "SalarySe office is selected"}',
        )
        node = VerifyNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(
            state={IntentStateKey.VERIFY_MODE: VerifyMode.PENDING_FINAL_COMMIT.value}
        )  # type: ignore[arg-type]

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertIsNone(result[IntentStateKey.VERIFY_MODE])
        self.assertFalse(result[IntentStateKey.SHOULD_RETRY])
        self.assertTrue(provider.context.agent_state.is_complete)
        self.assertTrue(provider.context.agent_state.all_sub_goals_complete())
        self.assertIn("User Intent: finish onboarding", provider.context.llm.prompts[0])
        self.assertNotIn("Step:", provider.context.llm.prompts[0])

    async def test_pending_final_commit_blank_reason_does_not_mark_rationale_verified(
        self,
    ) -> None:
        """
        Empty verifier evidence must not receive a positive rationale signal.
        """

        provider = _Provider(
            agent_state=self._final_agent_state(),
            llm_content='{"is_complete": true, "reason": ""}',
        )
        node = VerifyNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(
            state={IntentStateKey.VERIFY_MODE: VerifyMode.PENDING_FINAL_COMMIT.value}
        )  # type: ignore[arg-type]

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertFalse(result[IntentStateKey.SHOULD_RETRY])
        final_sub_goal = provider.context.agent_state.get_all_sub_goals()[1]
        self.assertTrue(final_sub_goal.is_complete())
        self.assertFalse(final_sub_goal.rationale_verified)
        self.assertEqual(
            result[CommonStateKey.COMPLETION_REASON],
            "Verifier accepted completion without detailed rationale.",
        )

    async def test_pending_final_commit_rejection_keeps_final_subgoal_active(self) -> None:
        provider = _Provider(
            agent_state=self._final_agent_state(),
            llm_content='{"is_complete": false, "reason": "Tap Yes, continue first"}',
        )
        node = VerifyNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(
            state={IntentStateKey.VERIFY_MODE: VerifyMode.PENDING_FINAL_COMMIT.value}
        )  # type: ignore[arg-type]

        self.assertFalse(result[CommonStateKey.IS_COMPLETE])
        self.assertTrue(result[IntentStateKey.SHOULD_RETRY])
        self.assertIsNone(result[IntentStateKey.VERIFY_MODE])
        self.assertEqual(provider.context.agent_state.current_sub_goal_index, 1)
        self.assertFalse(provider.context.agent_state.is_complete)
        self.assertFalse(provider.context.agent_state.all_sub_goals_complete())
        self.assertEqual(
            provider.context.context_manager.feedback,
            ["Verification failed: Tap Yes, continue first"],
        )

    async def test_pending_final_commit_without_active_subgoal_fails_structured(self) -> None:
        provider = _Provider(
            agent_state=self._agent_state(),
            llm_content='{"is_complete": true, "reason": "Done"}',
        )
        provider.context.agent_state.set_current_sub_goal_index(1)
        provider.context.agent_state.mark_current_sub_goal_complete(
            completion_signal=SubGoalCompletionSignal(
                llm_confidence=1.0,
                screen_verified=True,
                action_executed=True,
                flagged_complete=True,
                rationale_verified=True,
                evidence="done",
            )
        )
        node = VerifyNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(
            state={IntentStateKey.VERIFY_MODE: VerifyMode.PENDING_FINAL_COMMIT.value}
        )  # type: ignore[arg-type]

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertIsNone(result[IntentStateKey.VERIFY_MODE])
        self.assertFalse(result[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(result[CommonStateKey.COMPLETION_REASON], CompletionReason.FAILED.value)
        self.assertIn("without an active sub-goal", result[CommonStateKey.FAILURE_DIAGNOSTIC])

    async def test_pending_final_commit_with_non_final_subgoal_fails_structured(self) -> None:
        """
        Pending-final mode is invalid unless the active cursor is on the final sub-goal.
        """

        provider = _Provider(
            agent_state=self._agent_state(),
            llm_content='{"is_complete": true, "reason": "Done"}',
        )
        node = VerifyNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(
            state={IntentStateKey.VERIFY_MODE: VerifyMode.PENDING_FINAL_COMMIT.value}
        )  # type: ignore[arg-type]

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertIsNone(result[IntentStateKey.VERIFY_MODE])
        self.assertFalse(result[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(result[CommonStateKey.COMPLETION_REASON], CompletionReason.FAILED.value)
        self.assertIn("active sub-goal is not final", result[CommonStateKey.FAILURE_DIAGNOSTIC])

    async def test_subgoal_mode_without_active_subgoal_fails_structured(self) -> None:
        """
        SUB_GOAL mode is invalid once the cursor has moved past the sub-goal list.
        """

        agent_state = self._final_agent_state()
        agent_state.mark_current_sub_goal_complete(
            completion_signal=SubGoalCompletionSignal(
                llm_confidence=1.0,
                screen_verified=True,
                action_executed=True,
                flagged_complete=True,
                rationale_verified=True,
                evidence="done",
            )
        )
        provider = _Provider(
            agent_state=agent_state,
            llm_content='{"is_complete": true, "reason": "Done"}',
        )
        node = VerifyNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(state={IntentStateKey.VERIFY_MODE: VerifyMode.SUB_GOAL.value})  # type: ignore[arg-type]

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertIsNone(result[IntentStateKey.VERIFY_MODE])
        self.assertFalse(result[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(result[CommonStateKey.COMPLETION_REASON], CompletionReason.FAILED.value)
        self.assertIn("without an active sub-goal", result[CommonStateKey.FAILURE_DIAGNOSTIC])

    async def test_subgoal_mode_for_active_final_subgoal_fails_structured(self) -> None:
        """
        The final active sub-goal must use full-intent verification before commit.
        """

        provider = _Provider(
            agent_state=self._final_agent_state(),
            llm_content='{"is_complete": true, "reason": "Done"}',
        )
        node = VerifyNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(state={IntentStateKey.VERIFY_MODE: VerifyMode.SUB_GOAL.value})  # type: ignore[arg-type]

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertIsNone(result[IntentStateKey.VERIFY_MODE])
        self.assertFalse(result[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(result[CommonStateKey.COMPLETION_REASON], CompletionReason.FAILED.value)
        self.assertIn("active final sub-goal", result[CommonStateKey.FAILURE_DIAGNOSTIC])

    async def test_full_intent_mode_with_live_subgoals_fails_structured(self) -> None:
        """
        FULL_INTENT mode must not verify a half-finished sub-goal plan.
        """

        provider = _Provider(
            agent_state=self._agent_state(),
            llm_content='{"is_complete": true, "reason": "Done"}',
        )
        node = VerifyNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(state={IntentStateKey.VERIFY_MODE: VerifyMode.FULL_INTENT.value})  # type: ignore[arg-type]

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertIsNone(result[IntentStateKey.VERIFY_MODE])
        self.assertFalse(result[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(result[CommonStateKey.COMPLETION_REASON], CompletionReason.FAILED.value)
        self.assertIn("sub-goals are still active", result[CommonStateKey.FAILURE_DIAGNOSTIC])

    async def test_pending_final_commit_repeated_failure_terminates_frozen_loop(self) -> None:
        """
        Repeated pending-final verifier rejection must terminate as STUCK.
        """

        provider = _Provider(
            agent_state=self._final_agent_state(),
            llm_content='{"is_complete": false, "reason": "Tap Yes, continue first"}',
        )
        node = VerifyNode(provider=provider)  # type: ignore[arg-type]

        result: Dict[Any, Any] = {IntentStateKey.VERIFY_MODE: VerifyMode.PENDING_FINAL_COMMIT.value}
        for _ in range(DEFAULT_VERIFICATION_REJECTION_LIMIT):
            result = await node.run(state=result)  # type: ignore[arg-type]

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertIsNone(result[IntentStateKey.VERIFY_MODE])
        self.assertFalse(result[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(result[CommonStateKey.COMPLETION_REASON], CompletionReason.STUCK.value)
        self.assertEqual(
            provider.context.agent_state.current_sub_goal_index,
            1,
        )

    async def test_pending_final_commit_repeated_failure_survives_recorded_steps(self) -> None:
        """
        Gate-routed verifier rejection loops must terminate even when RECORD increments step_count between VERIFY turns.
        """

        provider = _Provider(
            agent_state=self._final_agent_state(),
            llm_content='{"is_complete": false, "reason": "Tap Yes, continue first"}',
        )
        node = VerifyNode(provider=provider)  # type: ignore[arg-type]

        result: Dict[Any, Any] = {IntentStateKey.VERIFY_MODE: VerifyMode.PENDING_FINAL_COMMIT.value}
        for step_number in range(DEFAULT_VERIFICATION_REJECTION_LIMIT):
            self._record_validate_step(
                agent_state=provider.context.agent_state,
                step_number=step_number,
            )
            result = await node.run(state=result)  # type: ignore[arg-type]

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertIsNone(result[IntentStateKey.VERIFY_MODE])
        self.assertFalse(result[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(result[CommonStateKey.COMPLETION_REASON], CompletionReason.STUCK.value)
        loop_state = provider.context.agent_state.verification_loop
        self.assertIsNotNone(loop_state)
        self.assertEqual(
            loop_state.consecutive_rejections,
            DEFAULT_VERIFICATION_REJECTION_LIMIT,
        )

    async def test_repeated_verification_failure_terminates_frozen_loop(self) -> None:
        provider = _Provider(
            agent_state=self._agent_state(),
            llm_content='{"is_complete": false, "reason": "Still on login"}',
        )
        node = VerifyNode(provider=provider)  # type: ignore[arg-type]

        result: Dict[Any, Any] = {}
        for _ in range(DEFAULT_VERIFICATION_REJECTION_LIMIT):
            result = await node.run(state={})  # type: ignore[arg-type]

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertFalse(result[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(provider.context.agent_state.completion_reason, "Stuck: No progress")

    async def test_repeated_rejection_with_stale_screen_activity_does_not_false_stuck(
        self,
    ) -> None:
        """
        Stored screen state from a prior activity must not be counted as same-screen VERIFY evidence.
        """

        provider = _Provider(
            agent_state=self._final_agent_state(),
            llm_content='{"is_complete": false, "reason": "Still blocked"}',
            capture_activity="com.other",
        )
        node = VerifyNode(provider=provider)  # type: ignore[arg-type]

        result: Dict[Any, Any] = {IntentStateKey.VERIFY_MODE: VerifyMode.PENDING_FINAL_COMMIT.value}
        for _ in range(DEFAULT_VERIFICATION_REJECTION_LIMIT):
            result = await node.run(state=result)  # type: ignore[arg-type]

        self.assertFalse(result[CommonStateKey.IS_COMPLETE])
        self.assertTrue(result[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(
            provider.context.agent_state.verification_loop.consecutive_rejections,
            1,
        )

    async def test_fresh_capture_state_counts_repeated_rejections_despite_stale_agent_screen(
        self,
    ) -> None:
        """
        VERIFY should use the fresh capture state for loop accounting when AgentState screen is stale.
        """

        provider = _Provider(
            agent_state=self._final_agent_state(),
            llm_content='{"is_complete": false, "reason": "Still blocked"}',
            capture_activity="com.other",
            capture_state=self._capture_screen(activity="com.other"),
        )
        node = VerifyNode(provider=provider)  # type: ignore[arg-type]

        result: Dict[Any, Any] = {IntentStateKey.VERIFY_MODE: VerifyMode.PENDING_FINAL_COMMIT.value}
        for _ in range(DEFAULT_VERIFICATION_REJECTION_LIMIT):
            result = await node.run(state=result)  # type: ignore[arg-type]

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertFalse(result[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(result[CommonStateKey.COMPLETION_REASON], CompletionReason.STUCK.value)

    async def test_llm_exception_fails_without_counting_verifier_rejection(self) -> None:
        """
        Verifier transport/parser failures must fail terminally, not consume same-screen rejection budget.
        """

        provider = _Provider(
            agent_state=self._final_agent_state(),
            llm_content="",
            llm_error=RuntimeError("llm unavailable"),
        )
        node = VerifyNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(
            state={IntentStateKey.VERIFY_MODE: VerifyMode.PENDING_FINAL_COMMIT.value}
        )  # type: ignore[arg-type]

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertFalse(result[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(result[CommonStateKey.COMPLETION_REASON], CompletionReason.FAILED.value)
        self.assertIn("llm unavailable", result[CommonStateKey.FAILURE_DIAGNOSTIC])
        self.assertIsNone(provider.context.agent_state.verification_loop)

    async def test_empty_capture_failure_is_persisted(self) -> None:
        """
        Empty verification captures must persist terminal checkpoint state.
        """

        provider = _Provider(
            agent_state=self._agent_state(),
            llm_content='{"is_complete": true, "reason": "Done"}',
            capture_image=b"",
        )
        node = VerifyNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(state={})  # type: ignore[arg-type]

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertFalse(result[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(result[CommonStateKey.COMPLETION_REASON], "Failed")
        self.assertEqual(provider.persistence.last[CommonStateKey.IS_COMPLETE], True)

    async def test_capture_exception_failure_is_persisted(self) -> None:
        """
        Provider capture failures must persist terminal checkpoint state.
        """

        provider = _Provider(
            agent_state=self._agent_state(),
            llm_content='{"is_complete": true, "reason": "Done"}',
            capture_error=RuntimeError("device gone"),
        )
        node = VerifyNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(state={})  # type: ignore[arg-type]

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertFalse(result[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(result[CommonStateKey.COMPLETION_REASON], "Failed")
        self.assertEqual(provider.persistence.last[CommonStateKey.IS_COMPLETE], True)
