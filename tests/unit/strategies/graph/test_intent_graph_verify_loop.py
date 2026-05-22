from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

from fathom.constants.graph import NodeName
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.core.agent.state import AgentState
from fathom.schemas.state import VerificationLoopPhase, VerificationLoopState
from fathom.strategies.graph.intent.builder import IntentGraphBuilder
from fathom.strategies.graph.intent.nodes.factory import IntentGraphFactory
from fathom.strategies.graph.state import IntentGraphState


class TestIntentGraphVerifyLoop:
    """
    Seam-level pin for the frozen-step-count VERIFY loop contract.
    """

    def test_frozen_verify_loop_terminates_after_recovery_shot_on_same_epoch(self) -> None:
        """
        A compiled intent graph must not spin forever when VERIFY keeps
        rejecting on the same screen without any newly recorded step.
        """

        agent_state = AgentState(intent="tap sign in", max_steps=10)
        agent_state._AgentState__step_count = 5

        context = MagicMock(name="GraphContext")
        context.agent_state = agent_state
        context.is_cancelled = False
        context.max_steps = 10
        context.recovery.verify_threshold = 2

        verify_calls = {"count": 0}

        def ground(_state: IntentGraphState) -> Dict[str, object]:
            return {}

        def analyze(_state: IntentGraphState) -> Dict[str, object]:
            return {
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.COMPLETION_REASON: "planner thinks done",
                IntentStateKey.SHOULD_RETRY: False,
            }

        def verify(_state: IntentGraphState) -> Dict[str, object]:
            verify_calls["count"] += 1

            loop_state = agent_state.record_verify_rejection(
                screen=None,
                activity="save-account",
            )

            if (
                loop_state.phase is VerificationLoopPhase.RECOVERY_ATTEMPTED
                and loop_state.consecutive_rejections >= context.recovery.verify_threshold
            ):
                agent_state.mark_complete(reason=CompletionReason.STUCK.value)
                return {
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.STUCK.value,
                }

            if loop_state.consecutive_rejections >= context.recovery.verify_threshold:
                agent_state.mark_verify_recovery_attempted()

            agent_state.reset_completion()
            return {
                CommonStateKey.IS_COMPLETE: False,
                IntentStateKey.SHOULD_RETRY: True,
            }

        def passthrough(_state: IntentGraphState) -> Dict[str, object]:
            return {}

        nodes: Dict[str, Any] = {
            NodeName.GROUND: ground,
            NodeName.ANALYZE: analyze,
            NodeName.SUPERVISE: passthrough,
            NodeName.EXECUTE: passthrough,
            NodeName.OBSERVE: passthrough,
            NodeName.RECORD: passthrough,
            NodeName.VERIFY: verify,
        }

        with patch.object(IntentGraphFactory, "build", return_value=nodes):
            graph = IntentGraphBuilder(context=context).build()
            result = graph.invoke({})

        assert result[CommonStateKey.IS_COMPLETE] is True
        assert result[CommonStateKey.COMPLETION_REASON] == CompletionReason.STUCK.value
        assert verify_calls["count"] == 3
        assert agent_state.verification_loop == VerificationLoopState(
            recorded_step_count=5,
            activity="save-account",
            screen=None,
            consecutive_rejections=3,
            phase=VerificationLoopPhase.RECOVERY_ATTEMPTED,
        )
