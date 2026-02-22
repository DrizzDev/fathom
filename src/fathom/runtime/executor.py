from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Dict, Optional

from langgraph.graph.state import CompiledStateGraph

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
                break

            # Race Condition: Run Graph vs Wait for Pause
            # We wrap the graph stream in a task to allow cancellation
            stream_task = asyncio.create_task(self.__stream_graph(current_input))
            pause_task = asyncio.create_task(self.__context.signal.wait_for_pause())

            done, pending = await asyncio.wait(
                [stream_task, pause_task], return_when=asyncio.FIRST_COMPLETED
            )

            # Clean up pending tasks immediately
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            # Case A: Pause Requested
            if pause_task in done:
                logger.info("Executor: Pause signal received during execution")
                await self.__handle_interrupt(source="manual_pause")
                # Resume loop (with current_input=None to continue from last checkpoint)
                current_input = None
                continue

            # Case B: Graph Execution Finished (Step or Workflow)
            try:
                await stream_task
            except Exception as exception:
                logger.error(f"Executor: Graph stream failed: {exception}")
                raise

            # Check cancellation after graph execution
            if self.__context.is_cancelled:
                logger.warning(f"Executor: Workflow {self.__thread_id} cancelled after execution")
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
                break

            # Resume Execution
            current_input = None

    async def __stream_graph(self, input_val: Optional[Dict[str, Any]]) -> None:
        """
        Wrapper to stream graph events.
        """

        async for event in self.__graph.astream(input_val, config=self.__config):
            # Log node transitions for visibility
            if isinstance(event, dict):
                for node, _output in event.items():
                    logger.debug(f"Executor: Node '{node}' completed")

    async def __handle_interrupt(self, source: str) -> None:
        """
        Processes HITL signals at graph breakpoints.
        """

        # Log interrupt check for visibility
        logger.debug(f"Executor: Checking signal at interrupt ({source})")

        signal_type = await self.__context.signal.check_signal()

        if not signal_type:
            return

        # Block execution until user resumes
        logger.info(f"Executor: Pausing execution ({source})")
        await self.__context.signal.wait_for_resume()

        # Check for context injection using strict interface
        if injected := await self.__context.signal.get_injected_context():
            await self.__inject_context(content=injected)

        logger.info("Executor: Resuming execution")

    async def __inject_context(self, content: str) -> None:
        """
        Injects user guidance into both ContextManager and Graph State.
        """

        current_step = self.__context.agent_state.step_count
        logger.info(f"Executor: Injecting user context at Step {current_step}: '{content}'")

        # Track budget usage
        if self.__invalidate_on_injection:
            self.__replan_count += 1
            remaining = self.__context.realignment.budget - self.__replan_count
            logger.info(
                f"Executor: Re-planning triggered. Budget used: {self.__replan_count}/{self.__context.realignment.budget} (Remaining: {remaining})"
            )
            if remaining < 0:
                logger.warning("Executor: Realignment budget exceeded! Proceeding with caution.")

        # A. Update ContextManager (Data Source)
        await self.__context.context_manager.inject_user_guidance(
            guidance=content, step=current_step
        )

        # B. Update Graph State
        update_dict: Dict[str, Any] = {"injected_context": content}

        if self.__invalidate_on_injection:
            logger.info("Executor: Invalidating pending plan to force re-planning")
            update_dict["plan"] = None
            update_dict["planned_step"] = None
            update_dict["should_retry"] = True
        else:
            logger.info("Executor: Preserving pending plan (if any)")

        await self.__graph.aupdate_state(self.__config, update_dict)
