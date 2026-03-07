from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Dict, Optional

from langgraph.graph.state import CompiledStateGraph

from fathom.constants.events import FathomEvent
from fathom.constants.state import CommonStateKey, IntentStateKey
from fathom.core.exceptions import WorkflowCancelledError
from fathom.strategies.graph.context import GraphContext

logger = logging.getLogger(__name__)


class GraphExecutor:
    """
    Handles the execution lifecycle of a LangGraph.
    Encapsulates Run-Pause-Resume logic and state synchronization.
    """

    def __init__(
        self,
        thread_id: str,
        context: GraphContext,
        graph: CompiledStateGraph,
        has_interrupts: bool = True,
        invalidate_on_injection: bool = True,
    ) -> None:
        """
        Initialize executor.
        """

        self.__graph = graph
        self.__context = context
        self.__thread_id = thread_id
        self.__has_interrupts = has_interrupts
        self.__invalidate_on_injection = invalidate_on_injection

        self.__replan_count = 0
        self.__config = {"configurable": {"thread_id": self.__thread_id}}

    async def run(self) -> None:
        """
        Executes the graph workflow with HITL support.
        Processes interrupts and resumes until completion or cancellation.
        """

        # For autonomous mode (no interrupts), run graph to completion in one call
        if not self.__has_interrupts:
            logger.info("Executor: Running in autonomous mode (no interrupts)")

            try:
                async for event in self.__graph.astream({}, config=self.__config):
                    if self.__context.is_cancelled:
                        logger.warning("Executor: Workflow cancelled during execution")
                        break
                    # Log node transitions
                    if isinstance(event, dict):
                        for node, _output in event.items():
                            logger.debug(f"Executor: Node '{node}' completed")
            except Exception as exception:
                logger.error(f"Executor: Graph execution failed: {exception}")
                raise
            return

        # Interactive mode with interrupts - use loop for pause/resume
        current_input: Optional[Dict[str, Any]] = {}

        while True:
            # Check cancellation before starting any graph execution
            if self.__context.is_cancelled:
                logger.warning(f"Executor: Workflow {self.__thread_id} cancelled before execution")

                await self.__context.telemetry.info(
                    "Workflow execution cancelled",
                    type=FathomEvent.WORKFLOW_CANCELLED,
                )
                break

            # Race Condition: Run Graph vs Wait for Pause
            # We wrap the graph stream in a task to allow cancellation
            stream_task = asyncio.create_task(self.__stream_graph(current_input))
            pause_task = asyncio.create_task(self.__context.hitl.wait_for_pause())

            done, pending = await asyncio.wait(
                [stream_task, pause_task], return_when=asyncio.FIRST_COMPLETED
            )

            # Case A: Pause Requested
            if pause_task in done:
                logger.info("Executor: Pause signal received during execution")

                # Do not cancel in-flight graph execution. Let current stream cycle
                # finish and handle pause at a safe graph boundary.
                if stream_task not in done:
                    try:
                        await stream_task
                    except Exception as exception:
                        logger.error(f"Executor: Graph stream failed: {exception}")
                        raise

                # Snapshot may already be terminal after stream completion.
                snapshot = await self.__graph.aget_state(self.__config)
                if not snapshot.next:
                    break

                await self.__handle_interrupt(source="manual_pause")
                # Resume loop (with current_input=None to continue from last checkpoint)
                current_input = None
                continue

            # Case B: Graph finished first, stop listening for pause for this cycle.
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            # Case B: Graph Execution Finished (Step or Workflow)
            try:
                await stream_task
            except Exception as exception:
                logger.error(f"Executor: Graph stream failed: {exception}")
                raise

            # Check cancellation after graph execution
            if self.__context.is_cancelled:
                logger.warning(f"Executor: Workflow {self.__thread_id} cancelled after execution")
                await self.__context.telemetry.info(
                    "Workflow execution cancelled",
                    type=FathomEvent.WORKFLOW_CANCELLED,
                )
                break

            # Check Graph State
            snapshot = await self.__graph.aget_state(self.__config)

            # If execution finished (no next node), exit loop
            if not snapshot.next:
                break

            # Handle Interrupt (Breakpoint reached naturally)
            # Check for HITL signals one last time
            await self.__handle_interrupt(source="breakpoint")

            # Check cancellation after interrupt handling
            if self.__context.is_cancelled:
                logger.warning(f"Executor: Workflow {self.__thread_id} cancelled")
                await self.__context.telemetry.info(
                    "Workflow execution cancelled",
                    type=FathomEvent.WORKFLOW_CANCELLED,
                )
                break

            # Resume Execution
            current_input = None

    async def __stream_graph(self, input_val: Optional[Dict[str, Any]]) -> None:
        """
        Wrapper to stream graph events.
        """

        async for event in self.__graph.astream(input_val, config=self.__config):
            if self.__context.is_cancelled:
                logger.warning("Executor: Workflow cancelled during stream")
                break

            # Log node transitions for visibility
            if isinstance(event, dict):
                for node, _output in event.items():
                    logger.debug(f"Executor: Node '{node}' completed")

    async def __handle_interrupt(self, source: str) -> None:
        """
        Processes HITL signals at graph breakpoints.
        """

        logger.debug(f"Executor: Checking signal at interrupt ({source})")

        signal_type = await self.__context.hitl.check_signal()

        if not signal_type:
            return

        logger.info(f"Executor: Pausing execution ({source})")
        await self.__context.telemetry.info(
            "Workflow execution paused",
            type=FathomEvent.WORKFLOW_PAUSED,
        )

        try:
            await self.__context.hitl.wait_for_resume()
        except WorkflowCancelledError:
            logger.info("Executor: Received workflow cancellation while paused")
            self.__context.cancel()
            return

        await self.__context.telemetry.info(
            "Workflow execution resumed",
            type=FathomEvent.WORKFLOW_RESUMED,
        )

        # Process ALL pending contexts in order
        processed_count = 0

        while await self.__context.hitl.has_injected_context():
            context = await self.__context.hitl.peek_next_context()
            if not context:
                break

            processed_count += 1
            logger.info(f"Executor: Processing context {processed_count}: '{context[:50]}...'")

            # Inject into system (resets loop state internally)
            await self.__inject_context(content=context)

            # Explicitly consume
            await self.__context.hitl.consume_context()

        if processed_count > 0:
            logger.info(f"Executor: Processed {processed_count} user contexts")

        logger.info("Executor: Resuming execution")

    async def __inject_context(self, content: str) -> None:
        """
        Injects user guidance into both ContextManager and Graph State.
        """

        current_step = self.__context.agent_state.step_count
        logger.info(f"Executor: Injecting user context at Step {current_step}: '{content}'")

        # Atomic update of budget and loop detector
        self.__context.agent_state.record_hitl_intervention()

        await self.__context.context_manager.inject_user_guidance(
            guidance=content, step=current_step
        )

        update_dict: Dict[str, Any] = {IntentStateKey.INJECTED_CONTEXT: content}

        if self.__invalidate_on_injection:
            # Immediate realignment: Force complete re-evaluation
            self.__replan_count += 1
            remaining = self.__context.realignment.budget - self.__replan_count
            logger.info(
                f"Executor: Re-planning triggered. Budget used: {self.__replan_count}/"
                f"{self.__context.realignment.budget} (Remaining: {remaining})"
            )
            if remaining < 0:
                logger.warning("Executor: Realignment budget exceeded! Proceeding with caution.")

            logger.info("Executor: Invalidating state for immediate realignment")

            # Clear planning and completion state in graph
            update_dict[IntentStateKey.PLAN] = None
            update_dict[IntentStateKey.PLANNED_STEP] = None
            update_dict[CommonStateKey.IS_COMPLETE] = False
            update_dict[IntentStateKey.SHOULD_RETRY] = True
            update_dict[CommonStateKey.COMPLETION_REASON] = None

            logger.info(
                "Executor: Graph routing state reset for fresh start, loop history preserved"
            )
        else:
            # Deferred realignment: Preserve current state, guidance applies to future steps
            logger.info("Executor: Preserving current state (deferred realignment)")

        await self.__graph.aupdate_state(self.__config, update_dict)
