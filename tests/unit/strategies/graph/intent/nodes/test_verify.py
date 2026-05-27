from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, Dict, List

from fathom.constants.runtime import DEFAULT_VERIFICATION_REJECTION_LIMIT
from fathom.constants.state import CommonStateKey, IntentStateKey
from fathom.core.agent.state import AgentState
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.subgoal import SubGoal
from fathom.strategies.graph.intent.nodes.verify import VerifyNode


class _LLM:
    def __init__(self, *, content: str) -> None:
        self.content = content
        self.prompts: List[str] = []

    async def generate(self, **kwargs: object) -> SimpleNamespace:
        prompt = kwargs.get("prompt")
        if isinstance(prompt, list) and prompt:
            self.prompts.append(str(prompt[0]))
        return SimpleNamespace(content=self.content)


class _Perception:
    def __init__(self, *, image: bytes = b"png", raises: Exception | None = None) -> None:
        self.__image = image
        self.__raises = raises

    async def perceive(self, **_: object) -> ScreenCapture:
        if self.__raises is not None:
            raise self.__raises

        return ScreenCapture(
            width=100,
            height=200,
            activity="com.test",
            image=self.__image,
            timestamp=1,
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
        capture_image: bytes = b"png",
        capture_error: Exception | None = None,
    ) -> None:
        self.context = SimpleNamespace(
            llm=_LLM(content=llm_content),
            intent="finish onboarding",
            max_steps=10,
            workflow_id="run-test",
            perception=_Perception(image=capture_image, raises=capture_error),
            agent_state=agent_state,
            artifact_pipeline=None,
            context_manager=_ContextManager(),
        )
        self.persistence = _Persistence()

    async def is_cancelled(self) -> bool:
        return False


class VerifyNodeSubGoalTest(unittest.IsolatedAsyncioTestCase):
    def _agent_state(self) -> AgentState:
        state = AgentState(
            intent="finish onboarding",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        state.set_sub_goals(
            [
                SubGoal(index=0, description="Open the app"),
                SubGoal(index=1, description="Reach the Home screen"),
            ]
        )
        state.mark_complete(reason="Sub-goal pending verification")
        return state

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
        self.assertEqual(provider.context.agent_state.completion_reason, "Stuck: No progress")

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
        self.assertEqual(result[CommonStateKey.COMPLETION_REASON], "Failed")
        self.assertEqual(provider.persistence.last[CommonStateKey.IS_COMPLETE], True)
