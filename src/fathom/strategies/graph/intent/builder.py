from __future__ import annotations

import logging
from typing import Dict, List, Optional, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from fathom.constants.graph import NodeName, RouteCause
from fathom.constants.state import (
    TERMINAL_COMPLETION_REASONS,
    CommonStateKey,
    IntentStateKey,
)
from fathom.interfaces.graph import GraphBuilder
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.intent.nodes.factory import IntentGraphFactory
from fathom.strategies.graph.state import IntentGraphState

logger = logging.getLogger(__name__)


class IntentGraphBuilder(GraphBuilder):
    """
    Constructs the LangGraph workflow for intent execution.
    """

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
                NodeName.END: NodeName.END,
                NodeName.GROUND: NodeName.GROUND,
                NodeName.EXECUTE: NodeName.EXECUTE,
            },
        )

        workflow.add_conditional_edges(
            NodeName.EXECUTE,
            self.__route_after_execute,
            {
                NodeName.END: NodeName.END,
                NodeName.GROUND: NodeName.GROUND,
                NodeName.VERIFY: NodeName.VERIFY,
                NodeName.OBSERVE: NodeName.OBSERVE,
            },
        )
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

        Reads the completion fields GROUND returns into graph state, exactly as
        the other routers do. The router must consume the merged node return
        because AgentState restoration is node-scoped — ``GraphStatePersistence.restore``
        runs inside nodes, not before this conditional edge — so ``context.agent_state``
        is not guaranteed to reflect the checkpoint at routing time; the returned patch is.
        """

        if state.get(cast("str", CommonStateKey.IS_COMPLETE)):
            reason = state.get(cast("str", CommonStateKey.COMPLETION_REASON))
            logger.info(f"[ROUTING] After GROUND: is_complete=True, reason={reason}")

            # Fatal/terminal reasons should end immediately
            if reason in TERMINAL_COMPLETION_REASONS:
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
            if reason in TERMINAL_COMPLETION_REASONS:
                logger.info(f"[ROUTING] -> END (Fatal Error / Max Steps: {reason})")
                return NodeName.END

            # Bounded-retry deferral when sub-goals remain is handled inside
            # ANALYZE (so the increment is part of the checkpoint and survives
            # the next ``persistence.restore()``). By the time we reach the
            # router with ``is_complete=True``, either no sub-goals remain or
            # the deferral budget has been exhausted — both cases route to
            # VERIFY, which adjudicates the final outcome.
            logger.info("[ROUTING] -> VERIFY (is_complete=True)")
            return NodeName.VERIFY

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
        Route after supervise based on whether localization produced an executable context.
        """

        if self.__context.is_cancelled:
            logger.info("[ROUTING] After SUPERVISE -> END (Cancelled)")
            return NodeName.END

        if state.get(cast("str", IntentStateKey.SHOULD_RETRY)):
            # SUPERVISE detected incomplete upstream state (missing
            # capture or planned_step) and asked for a re-ground.
            # Routing to GROUND avoids the silent EXECUTE→OBSERVE→RECORD
            # cascade that would otherwise end with the misleading
            # ``record.missing.step_result`` Sentry alert.
            logger.info("[ROUTING] After SUPERVISE -> GROUND (should_retry=True)")
            return NodeName.GROUND

        logger.info("[ROUTING] After SUPERVISE -> EXECUTE")
        return NodeName.EXECUTE

    def __route_after_execute(self, state: IntentGraphState) -> str:
        """
        Route after execute: cancellation > terminal completion > retry > observe.
        """

        if self.__context.is_cancelled:
            return self.__log_execute_route(
                destination=NodeName.END,
                cause=RouteCause.CANCELLED,
            )

        if state.get(cast("str", CommonStateKey.IS_COMPLETE)):
            if (
                reason := str(state.get(cast("str", CommonStateKey.COMPLETION_REASON), ""))
            ) in TERMINAL_COMPLETION_REASONS:
                return self.__log_execute_route(
                    completion_reason=reason,
                    destination=NodeName.END,
                    cause=RouteCause.TERMINAL_COMPLETION,
                )
            return self.__log_execute_route(
                completion_reason=reason,
                destination=NodeName.VERIFY,
                cause=RouteCause.NON_TERMINAL_COMPLETION,
            )

        if state.get(cast("str", IntentStateKey.SHOULD_RETRY)):
            return self.__log_execute_route(
                destination=NodeName.GROUND,
                cause=RouteCause.SHOULD_RETRY,
            )

        return self.__log_execute_route(
            cause=RouteCause.DEFAULT,
            destination=NodeName.OBSERVE,
        )

    @staticmethod
    def __log_execute_route(
        *,
        destination: str,
        cause: RouteCause,
        completion_reason: Optional[str] = None,
    ) -> str:
        """
        Emit the structured route event and return the destination.
        """

        extra: Dict[str, object] = {
            "cause": cause.value,
            "destination": destination,
            "event": "route.after_execute",
        }
        if completion_reason is not None:
            extra["completion.reason"] = completion_reason

        logger.info("Route after execute resolved", extra=extra)
        return destination

    def __route_after_verify(self, state: IntentGraphState) -> str:
        """
        Route after verify based on completion status.
        """

        is_complete = state.get(cast("str", CommonStateKey.IS_COMPLETE))
        reason = state.get(cast("str", CommonStateKey.COMPLETION_REASON))

        logger.info(f"[ROUTING] After VERIFY: is_complete={is_complete}")

        if self.__context.is_cancelled:
            logger.info("[ROUTING] -> END (Cancelled during verify)")
            return NodeName.END

        if is_complete:
            if reason in TERMINAL_COMPLETION_REASONS:
                logger.info(f"[ROUTING] -> END (verification terminal reason: {reason})")
                return NodeName.END
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
            if reason in TERMINAL_COMPLETION_REASONS:
                logger.info(f"[ROUTING] -> END (terminal reason: {reason})")
                return NodeName.END

            logger.info("[ROUTING] -> VERIFY (is_complete=True from RECORD)")
            return NodeName.VERIFY

        return NodeName.GROUND
