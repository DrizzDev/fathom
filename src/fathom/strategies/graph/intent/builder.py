from __future__ import annotations

import logging
from typing import List, Optional, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from fathom.constants.graph import NodeName
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.interfaces.graph import GraphBuilder
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.intent.nodes import IntentGraphFactory
from fathom.strategies.graph.state import IntentGraphState

logger = logging.getLogger(__name__)


class IntentGraphBuilder(GraphBuilder):
    """
    Constructs the LangGraph workflow for intent execution.
    """

    __MAX_COMPLETE_DEFERRALS: int = 2

    def __init__(self, context: GraphContext) -> None:
        self.__context = context

    def build(
        self,
        interrupt_before: Optional[List[str]] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> CompiledStateGraph:
        """
        Builds and compiles the graph.
        """

        workflow = StateGraph(IntentGraphState)
        nodes = IntentGraphFactory.build(context=self.__context)

        workflow.add_node(NodeName.GROUND, nodes[NodeName.GROUND])
        workflow.add_node(NodeName.RECORD, nodes[NodeName.RECORD])
        workflow.add_node(NodeName.VERIFY, nodes[NodeName.VERIFY])
        workflow.add_node(NodeName.ANALYZE, nodes[NodeName.ANALYZE])
        workflow.add_node(NodeName.EXECUTE, nodes[NodeName.EXECUTE])
        workflow.add_node(NodeName.OBSERVE, nodes[NodeName.OBSERVE])
        workflow.add_node(NodeName.SUPERVISE, nodes[NodeName.SUPERVISE])

        workflow.set_entry_point(NodeName.GROUND)

        workflow.add_conditional_edges(
            NodeName.GROUND,
            self.__route_after_ground,
            {
                NodeName.END: NodeName.END,
                NodeName.ANALYZE: NodeName.ANALYZE,
            },
        )

        workflow.add_conditional_edges(
            NodeName.ANALYZE,
            self.__route_after_analyze,
            {
                NodeName.END: NodeName.END,
                NodeName.VERIFY: NodeName.VERIFY,
                NodeName.GROUND: NodeName.GROUND,
                NodeName.SUPERVISE: NodeName.SUPERVISE,
            },
        )

        workflow.add_conditional_edges(
            NodeName.SUPERVISE,
            self.__route_after_supervise,
            {
                NodeName.EXECUTE: NodeName.EXECUTE,
                NodeName.RECORD: NodeName.RECORD,
                NodeName.END: NodeName.END,
            },
        )

        workflow.add_edge(NodeName.EXECUTE, NodeName.OBSERVE)
        workflow.add_edge(NodeName.OBSERVE, NodeName.RECORD)

        workflow.add_conditional_edges(
            NodeName.VERIFY,
            self.__route_after_verify,
            {
                NodeName.END: NodeName.END,
                NodeName.GROUND: NodeName.GROUND,
            },
        )

        workflow.add_conditional_edges(
            NodeName.RECORD,
            self.__route_after_record,
            {
                NodeName.END: NodeName.END,
                NodeName.GROUND: NodeName.GROUND,
                NodeName.VERIFY: NodeName.VERIFY,
            },
        )

        return workflow.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)

    def __route_after_ground(self, state: IntentGraphState) -> str:
        """
        Route after ground based on completion status.

        Note: We check context.agent_state instead of state dict because
        LangGraph routing happens before node return values are merged into state.
        """

        # Check AgentState directly (updated by node before return)
        if self.__context.agent_state.is_complete:
            reason = self.__context.agent_state.completion_reason
            logger.info(f"[ROUTING] After GROUND: is_complete=True, reason={reason}")

            # Fatal/terminal reasons should end immediately
            if reason in {
                CompletionReason.FAILED.value,
                CompletionReason.MAX_STEPS.value,
                CompletionReason.CANCELLED.value,
            }:
                logger.info(f"[ROUTING] -> END ({reason})")
                return NodeName.END

        logger.info("[ROUTING] After GROUND: -> ANALYZE")
        return NodeName.ANALYZE

    def __route_after_analyze(self, state: IntentGraphState) -> str:
        """
        Route after analyze based on completion and retry status.
        """

        is_complete = state.get(cast("str", CommonStateKey.IS_COMPLETE))
        should_retry = state.get(cast("str", IntentStateKey.SHOULD_RETRY))
        planned_step = state.get(cast("str", IntentStateKey.PLANNED_STEP))

        logger.info(
            f"[ROUTING] After ANALYZE: is_complete={is_complete}, "
            f"should_retry={should_retry}, has_planned_step={planned_step is not None}, "
            f"planned_step_type={type(planned_step).__name__}"
        )

        # 1. Cancellation overrides everything
        if self.__context.is_cancelled:
            logger.info("[ROUTING] -> END (Cancelled)")
            return NodeName.END

        # 2. Fatal Errors and Completion override retry logic
        if is_complete:
            reason = state.get(cast("str", CommonStateKey.COMPLETION_REASON), "")

            # If it completed due to a fatal error or max steps, do not verify, just end
            if reason in {CompletionReason.FAILED.value, CompletionReason.MAX_STEPS.value}:
                logger.info(f"[ROUTING] -> END (Fatal Error / Max Steps: {reason})")
                return NodeName.END

            # When sub-goals are defined, verification must wait until all sub-goals
            # complete (handled in RECORD node). The planner's overall completion
            # signal is treated as a soft hint — reset and continue working, but
            # only up to ``__MAX_COMPLETE_DEFERRALS`` consecutive times. Beyond
            # that the planner has stably claimed completion for the same screen
            # state; honouring the claim avoids a budget-burning ground-loop and
            # lets the verifier — not the planner — make the final call.
            if (
                self.__context.agent_state.has_sub_goals()
                and not self.__context.agent_state.all_sub_goals_complete()
            ):
                deferrals = self.__context.agent_state.record_complete_deferral()
                if deferrals <= self.__MAX_COMPLETE_DEFERRALS:
                    logger.info(
                        "[ROUTING] -> GROUND "
                        f"(is_complete=True but sub-goals remain; deferral {deferrals}/"
                        f"{self.__MAX_COMPLETE_DEFERRALS}, deferring verification)"
                    )
                    self.__context.agent_state.reset_completion()
                    return NodeName.GROUND

                logger.warning(
                    "[ROUTING] -> VERIFY "
                    f"(is_complete=True repeated {deferrals} times with sub-goals open; "
                    "honouring planner verdict and letting VERIFY adjudicate)"
                )
                self.__context.agent_state.reset_complete_deferrals()
                return NodeName.VERIFY

            # Otherwise, it's a normal goal completion, proceed to verification
            self.__context.agent_state.reset_complete_deferrals()
            logger.info("[ROUTING] -> VERIFY (is_complete=True)")
            return NodeName.VERIFY

        # A non-complete ANALYZE outcome is forward progress; clear any
        # stale complete-deferral streak from previous turns.
        self.__context.agent_state.reset_complete_deferrals()

        # 3. Soft Retries (e.g. missing elements, LLM asked to retry)
        if should_retry:
            logger.info("[ROUTING] -> GROUND (should_retry=True)")
            return NodeName.GROUND

        # 4. Incomplete but missing step -> Error fallback
        if not planned_step:
            logger.info(f"[ROUTING] -> GROUND (no planned_step, value={planned_step})")
            return NodeName.GROUND

        logger.info("[ROUTING] -> SUPERVISE")
        return NodeName.SUPERVISE

    def __route_after_supervise(self, state: IntentGraphState) -> str:
        """
        Route after supervise based on whether the supervisor blocked the action.
        """

        if self.__context.is_cancelled:
            logger.info("[ROUTING] After SUPERVISE -> END (Cancelled)")
            return NodeName.END

        if state.get(cast("str", IntentStateKey.EXECUTION_BLOCKED)):
            logger.info("[ROUTING] After SUPERVISE -> RECORD (blocked)")
            return NodeName.RECORD

        logger.info("[ROUTING] After SUPERVISE -> EXECUTE")
        return NodeName.EXECUTE

    def __route_after_verify(self, state: IntentGraphState) -> str:
        """
        Route after verify based on completion status.
        """

        is_complete = state.get(cast("str", CommonStateKey.IS_COMPLETE))

        logger.info(f"[ROUTING] After VERIFY: is_complete={is_complete}")

        if self.__context.is_cancelled:
            logger.info("[ROUTING] -> END (Cancelled during verify)")
            return NodeName.END

        if is_complete:
            logger.info("[ROUTING] -> END (verification passed)")
            return NodeName.END

        logger.info("[ROUTING] -> GROUND (verification failed)")
        return NodeName.GROUND

    def __route_after_record(self, state: IntentGraphState) -> str:
        """
        Route after record based on completion status.
        """

        if state.get(cast("str", CommonStateKey.IS_COMPLETE)):
            # Check for cancellation
            if self.__context.is_cancelled:
                logger.info("[ROUTING] -> END (Cancelled)")
                return NodeName.END

            # Check if completion was due to Max Steps - if so, fail/end instead of verifying
            reason = state.get(cast("str", CommonStateKey.COMPLETION_REASON))
            if reason == CompletionReason.MAX_STEPS.value:
                logger.info("[ROUTING] -> END (Max steps reached)")
                return NodeName.END

            logger.info("[ROUTING] -> VERIFY (is_complete=True from RECORD)")
            return NodeName.VERIFY

        return NodeName.GROUND
