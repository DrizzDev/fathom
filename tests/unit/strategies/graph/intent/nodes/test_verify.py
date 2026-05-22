from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.strategies.graph.intent.nodes.verify import VerifyNode


class VerifyNodeEarlyExitTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the VERIFY node's cancellation and empty-capture branches and
    the restore-before-work invariant.

    VERIFY is the final stage: it re-captures the screen and asks the LLM
    whether the intent has truly been satisfied. The pins verify two
    early-exit paths plus the load-order invariant — :class:`AgentState`
    must be restored from the checkpoint *before* anything else, so the
    cancellation check sees the latest run state, not the pre-resume one.
    """

    @staticmethod
    def __provider(*, cancelled: bool = False, image: bytes = b"PNG") -> MagicMock:
        """
        Mocked :class:`IntentNodeProvider` exposing the cancellation
        check, persistence hooks, and a capture-returning perception
        port. The ``image`` parameter is overridable so the
        empty-capture path can be driven by passing ``image=b""``.
        """

        provider = MagicMock(name="IntentNodeProvider")
        provider.is_cancelled = AsyncMock(return_value=cancelled)
        provider.context.workflow_id = "run-test"
        provider.persistence.persist = MagicMock()
        provider.persistence.restore = MagicMock()
        provider.context.perception.perceive = AsyncMock(
            return_value=MagicMock(image=image, width=100, height=100, activity="app"),
        )
        return provider

    async def test_cancellation_marks_complete(self) -> None:
        """
        A cancelled run must terminate with :attr:`CompletionReason.CANCELLED`
        without consulting the verifier LLM.
        """

        provider = self.__provider(cancelled=True)
        node = VerifyNode(provider=provider)

        result: Any = await node(state={})  # type: ignore[arg-type]

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.CANCELLED.value,
        )

    async def test_empty_capture_terminates_with_failed_reason(self) -> None:
        """
        An empty post-execution capture means the device surface is gone
        (lost screen, permission revoked, etc.). VERIFY must terminate
        with :attr:`CompletionReason.FAILED` so the run is reported as
        broken rather than silently passing the intent.
        """

        node = VerifyNode(provider=self.__provider(cancelled=False, image=b""))

        result: Any = await node(state={})  # type: ignore[arg-type]

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.FAILED.value,
        )

    async def test_restore_called_before_any_other_work(self) -> None:
        """
        The persistence ``restore`` must be invoked before the
        cancellation check so the cancellation predicate reads the
        latest restored AgentState — not the pre-resume snapshot.
        This pin caught a real regression where a cancelled run would
        continue if the checkpoint was older than the cancel signal.
        """

        provider = self.__provider(cancelled=True)
        node = VerifyNode(provider=provider)

        await node(state={})  # type: ignore[arg-type]

        provider.persistence.restore.assert_called_once()

    async def test_all_sub_goals_done_does_not_force_success_when_verifier_rejects(self) -> None:
        """
        Rejected verification must keep the workflow open unless user guidance
        explicitly requested termination.
        """

        provider = self.__provider(cancelled=False, image=b"PNG")
        provider.context.intent = "Find Ashsa Tiffin"
        provider.context.agent_state.step_count = 4
        provider.context.agent_state.has_sub_goals.return_value = True
        provider.context.agent_state.all_sub_goals_complete.return_value = True
        provider.context.agent_state.reset_completion = MagicMock()
        provider.context.context_manager.get_user_guidance.return_value = []
        provider.context.context_manager.inject_verifier_feedback = AsyncMock()
        provider.context.llm.generate = AsyncMock(
            return_value=MagicMock(
                content='{"is_complete": false, "reason": "Target not visible on screen."}'
            )
        )
        provider.context.artifact_pipeline = None
        provider.recovery.try_recover = AsyncMock(return_value=None)

        node = VerifyNode(provider=provider)
        result: Any = await node(state={})  # type: ignore[arg-type]

        self.assertFalse(result.get(CommonStateKey.IS_COMPLETE))
        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))
        provider.context.agent_state.mark_complete.assert_not_called()
        provider.context.agent_state.reopen_last_completed_sub_goal.assert_called_once()
        provider.context.context_manager.inject_verifier_feedback.assert_awaited_once()

    async def test_user_guidance_does_not_override_verifier_rejection(self) -> None:
        """
        HITL guidance must not convert a rejected verification into success.
        """

        provider = self.__provider(cancelled=False, image=b"PNG")
        provider.context.intent = "Find Millet Express"
        provider.context.agent_state.step_count = 4
        provider.context.agent_state.has_sub_goals.return_value = True
        provider.context.agent_state.all_sub_goals_complete.return_value = True
        provider.context.agent_state.reset_completion = MagicMock()
        provider.context.context_manager.get_user_guidance.return_value = [
            MagicMock(content="Keep going in the same row")
        ]
        provider.context.context_manager.inject_verifier_feedback = AsyncMock()
        provider.context.llm.generate = AsyncMock(
            return_value=MagicMock(
                content='{"is_complete": false, "reason": "Target not visible on screen."}'
            )
        )
        provider.context.artifact_pipeline = None
        provider.recovery.try_recover = AsyncMock(return_value=None)

        node = VerifyNode(provider=provider)
        result: Any = await node(state={})  # type: ignore[arg-type]

        self.assertFalse(result.get(CommonStateKey.IS_COMPLETE))
        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))
        provider.context.agent_state.mark_complete.assert_not_called()
