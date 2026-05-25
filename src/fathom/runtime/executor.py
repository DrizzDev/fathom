from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any, Coroutine, Dict, Optional, Set

from langgraph.graph.state import CompiledStateGraph

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig

from fathom.constants import SignalType
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

        self.__config: RunnableConfig = {"configurable": {"thread_id": self.__thread_id}}

        self.__active_tasks: Set[asyncio.Task[None]] = set()

    async def run(self) -> None:
        """
        Executes the graph workflow with HITL support.
        Processes interrupts and resumes until completion or cancellation.
        """

        # Validate state consistency before execution
        await self.__validate_state_sync("run_start")

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

        try:
            await self.__run_interactive()
        finally:
            await self.__cancel_active_tasks()

    async def __run_interactive(self) -> None:
        """
        Run the interactive pause/resume execution loop until completion.
        """

        current_input: Optional[Dict[str, Any]] = {}

        while True:
            # Check cancellation before starting any graph execution
            if await self.__stop_for_cancellation(phase="before execution"):
                return

            # Race Condition: Run Graph vs Wait for Pause
            # We wrap the graph stream in a task to allow cancellation.
            stream_task = self.__create_task(
                operation=self.__stream_graph(input_value=current_input)
            )
            pause_task = self.__create_task(operation=self.__context.hitl.wait_for_pause())

            done, pending = await asyncio.wait(
                [stream_task, pause_task], return_when=asyncio.FIRST_COMPLETED
            )

            # Case A: Pause Requested
            if pause_task in done:
                if not (should_continue := await self.__handle_pause(stream_task=stream_task)):
                    return

                if should_continue:
                    current_input = None
                    continue

            # Case B: Graph finished first, stop listening for pause for this cycle.
            await self.__cancel_pending_tasks(tasks=pending)
            await self.__await_stream_task(stream_task=stream_task)

            if await self.__stop_for_cancellation(phase="after execution"):
                return

            # Check Graph State
            if not (await self.__graph.aget_state(self.__config)).next:
                return

            # Handle Interrupt (Breakpoint reached naturally)
            # Check for HITL signals one last time.
            await self.__handle_interrupt(source="breakpoint")

            if await self.__stop_for_cancellation(phase="after interrupt handling"):
                return

            # Resume Execution
            current_input = None

    async def __stream_graph(self, *, input_value: Optional[Dict[str, Any]]) -> None:
        """
        Wrapper to stream graph events.
        """

        async for event in self.__graph.astream(input_value, config=self.__config):
            if self.__context.is_cancelled:
                logger.warning("Executor: Workflow cancelled during stream")
                break

            # Log node transitions for visibility
            if isinstance(event, dict):
                for node, _output in event.items():
                    logger.debug(f"Executor: Node '{node}' completed")

    def __create_task(self, *, operation: Coroutine[object, object, None]) -> asyncio.Task[None]:
        """
        Create a task that is tracked until completion.
        """

        task: asyncio.Task[None] = asyncio.create_task(operation)

        self.__active_tasks.add(task)
        task.add_done_callback(self.__active_tasks.discard)

        return task

    async def __cancel_pending_tasks(self, *, tasks: Set[asyncio.Task[None]]) -> None:
        """
        Cancel and await a set of pending executor-owned tasks.
        """

        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def __await_stream_task(self, *, stream_task: asyncio.Task[None]) -> None:
        """
        Await the graph stream task and surface failures consistently.
        """

        try:
            await stream_task
        except Exception as exception:
            logger.error(f"Executor: Graph stream failed: {exception}")
            raise

    async def __handle_pause(
        self,
        *,
        stream_task: asyncio.Task[None],
    ) -> bool:
        """
        Complete the current stream cycle and process a manual pause.
        """

        logger.info("Executor: Pause signal received during execution")

        signal_type = await self.__context.hitl.check_signal()
        if signal_type == SignalType.CANCELLED.value:
            logger.info("Executor: Cancellation signal received while waiting for pause")
            self.__context.cancel()
            stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stream_task
            await self.__context.telemetry.info(
                "Workflow execution cancelled", type=FathomEvent.WORKFLOW_CANCELLED
            )
            return False

        # Do not cancel in-flight graph execution. Let current stream cycle
        # finish and handle pause at a safe graph boundary.
        if not stream_task.done():
            await self.__await_stream_task(stream_task=stream_task)

        # Snapshot may already be terminal after stream completion.
        if not (await self.__graph.aget_state(self.__config)).next:
            return False

        await self.__handle_interrupt(source="manual_pause")
        return True

    async def __stop_for_cancellation(self, *, phase: str) -> bool:
        """
        Emit cancellation telemetry and signal the caller to stop when cancelled.
        """

        if not self.__context.is_cancelled:
            return False

        logger.warning(f"Executor: Workflow {self.__thread_id} cancelled {phase}")
        await self.__context.telemetry.info(
            "Workflow execution cancelled", type=FathomEvent.WORKFLOW_CANCELLED
        )

        return True

    async def __cancel_active_tasks(self) -> None:
        """
        Cancel and await any executor-owned tasks that are still running.
        """

        active_tasks = tuple(self.__active_tasks)

        for task in active_tasks:
            if not task.done():
                task.cancel()

        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

        self.__active_tasks.clear()

    async def __handle_interrupt(self, source: str) -> None:
        """
        Processes HITL signals at graph breakpoints.
        """

        logger.debug(f"Executor: Checking signal at interrupt ({source})")

        signal_type = await self.__context.hitl.check_signal()

        if not signal_type:
            return

        if signal_type == SignalType.CANCELLED.value:
            logger.info(f"Executor: Cancellation signal received ({source})")
            self.__context.cancel()

            await self.__context.telemetry.info(
                "Workflow execution cancelled", type=FathomEvent.WORKFLOW_CANCELLED
            )
            return

        logger.info(f"Executor: Pausing execution ({source})")
        await self.__context.telemetry.info(
            "Workflow execution paused", type=FathomEvent.WORKFLOW_PAUSED
        )

        try:
            await self.__context.hitl.wait_for_resume()
        except WorkflowCancelledError:
            logger.info("Executor: Received workflow cancellation while paused")
            self.__context.cancel()
            return

        await self.__context.telemetry.info(
            "Workflow execution resumed", type=FathomEvent.WORKFLOW_RESUMED
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

    async def __validate_state_sync(self, checkpoint: str) -> None:
        """
        Validate consistency between graph state and context objects.
        Logs warnings if state drift is detected.
        """

        try:
            snapshot = await self.__graph.aget_state(self.__config)
            graph_is_complete = snapshot.values.get(CommonStateKey.IS_COMPLETE, False)
            context_is_complete = self.__context.agent_state.is_complete

            if graph_is_complete != context_is_complete:
                logger.warning(
                    f"Executor [{checkpoint}]: State drift detected! "
                    f"Graph is_complete={graph_is_complete}, Context is_complete={context_is_complete}"
                )

            logger.debug(
                f"Executor [{checkpoint}]: State validation - "
                f"next_nodes={snapshot.next}, graph_keys={list(snapshot.values.keys())}"
            )
        except Exception as exception:
            logger.error(f"Executor [{checkpoint}]: State validation failed: {exception}")

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
            used = self.__context.agent_state.runtime.realignment.count
            budget = self.__context.agent_state.runtime.realignment.budget
            remaining = budget - used
            logger.info(
                f"Executor: Realignment triggered. Budget used: {used}/"
                f"{budget} (Remaining: {remaining})"
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

        # CRITICAL: Persist injected context to graph state for checkpoint recovery
        await self.__graph.aupdate_state(self.__config, update_dict)
        logger.info(
            f"Executor: Graph state updated with context injection. Keys updated: {list(update_dict.keys())}"
        )
