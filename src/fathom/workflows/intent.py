from __future__ import annotations

import asyncio
import contextlib
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from fathom.graph.nodes import NodeContext

from fathom.agent.planner import StepPlanner
from fathom.agent.strategies.intent import IntentStrategy
from fathom.interfaces import IMemoryProvider
from fathom.schemas.configuration import WorkflowConfig
from fathom.schemas.results import IntentResult
from fathom.schemas.screens import ScreenCapture
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool
from fathom.tools.vision import VisionTool
from fathom.workflows.base import BaseWorkflow

logger = getLogger(__name__)


class IntentWorkflow(BaseWorkflow[IntentResult]):
    """
    Workflow for executing a specific intent.

    Orchestrates the full execution of a goal-directed automation:
    1. Decompose complex intent into atomic sub-intents
    2. Execute each sub-intent sequentially using IntentStrategy
    3. Return combined result

    When ``use_langgraph=True``, the execution loop is replaced by a
    LangGraph StateGraph that drives the same underlying components.
    """

    def __init__(
        self,
        workflow_id: str,
        intent: str,
        vision: VisionTool,
        device: DeviceTool,
        capture: CaptureTool,
        memory: IMemoryProvider,
        *,
        configuration: Optional[WorkflowConfig] = None,
        use_langgraph: bool = False,
    ) -> None:
        """
        Initialize intent workflow.
        """

        super().__init__(workflow_id=workflow_id, configuration=configuration)

        self.__vision = vision
        self.__device = device
        self.__memory = memory
        self.__capture = capture
        self.__original_intent = intent
        self.__use_langgraph = use_langgraph

        self.__package_name = configuration.package_name if configuration else ""
        self.__planner = StepPlanner(vision_tool=vision)
        # Create strategy immediately since we no longer decompose
        self.__strategy = IntentStrategy(
            intent=intent,
            device=device,
            memory=memory,
            planner=self.__planner,
            capture=capture,
            workflow_id=workflow_id,
            step_timeout=configuration.step_timeout if configuration else 30.0,
            use_xml=configuration.use_xml_bounding_boxes if configuration else False,
            max_steps=configuration.max_steps if configuration else 10,
            package_name=self.__package_name,
        )

        self.__completion_reason = ""
        self.__final_screen: Optional[ScreenCapture] = None

    @property
    def name(self) -> str:
        """
        Returns the name of the workflow.
        """

        return "intent"

    @property
    def intent(self) -> str:
        """
        Returns the original intent.
        """

        return self.__original_intent

    async def execute(self) -> IntentResult:
        """
        Execute the intent workflow.

        Delegates to the LangGraph StateGraph when ``use_langgraph`` is
        ``True``; otherwise falls back to the original strategy loop.
        """

        logger.info(
            f"Executing intent: {self.__original_intent} (langgraph={self.__use_langgraph})"
        )

        if self.__use_langgraph:
            return await self.__execute_langgraph()

        return await self.__execute_classic()

    # ── Classic path (original strategy loop) ──────────────────────────

    async def __execute_classic(self) -> IntentResult:
        """Original while-loop strategy execution.

        Each step is wrapped in a race against the ``cancel_event`` so that
        an in-flight LLM call or device interaction is cancelled within
        milliseconds of the user pressing Ctrl+C, rather than waiting for
        the entire step to complete.
        """

        while await self.__strategy.should_continue():
            if self.is_cancelled():
                self.__completion_reason = "Workflow cancelled by user"
                break

            # Race step execution against cancellation
            step_task = asyncio.create_task(self.__strategy.execute_step())
            cancel_waiter = asyncio.create_task(self.cancel_event.wait())

            done, _pending = await asyncio.wait(
                {step_task, cancel_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if cancel_waiter in done and step_task not in done:
                logger.info("Classic execution: cancel received mid-step, aborting")
                step_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await step_task
                self.__completion_reason = "Workflow cancelled by user"
                break

            # Clean up cancel waiter
            cancel_waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cancel_waiter

            result = step_task.result()

            if result.step_result:
                self.record_step(result=result.step_result)

            if result.is_terminal:
                if result.status == result.status.ERROR:
                    logger.warning(f"Intent failed: {result.message}")
                break

        # Finalize
        success = self.__strategy.state.is_complete

        if not self.__completion_reason:
            if success:
                self.__completion_reason = "Goal successfully achieved"
            else:
                self.__completion_reason = "Execution failed or timed out"

        return IntentResult(
            metrics=self.__strategy.metrics,
            success=success,
            intent=self.__original_intent,
            steps_taken=self.steps_executed,
            final_screen=self.__final_screen,
            completion_reason=self.__completion_reason,
            step_results=self.recorded_steps,
        )

    # ── LangGraph path ─────────────────────────────────────────────────

    async def __execute_langgraph(self) -> IntentResult:
        """Execute using a LangGraph StateGraph.

        The graph invocation is wrapped in an asyncio task so that when
        ``cancel()`` is called (via SIGINT), a monitor coroutine can cancel
        the task immediately rather than waiting for the current node to
        finish.  Each graph node also checks ``ctx.is_cancelled`` at entry
        for belt-and-suspenders safety.
        """

        from fathom.graph.intent_graph import build_intent_graph

        config = self.configuration

        compiled_graph, node_ctx = build_intent_graph(
            intent=self.__original_intent,
            planner=self.__planner,
            device=self.__device,
            capture=self.__capture,
            memory=self.__memory,
            max_steps=config.max_steps if config else 100,
            use_xml=config.use_xml_bounding_boxes if config else False,
            step_timeout=config.step_timeout if config else 15.0,
            workflow_id=self.workflow_id,
            cancel_event=self.cancel_event,
            package_name=self.__package_name,
        )

        initial_state = {
            "intent": self.__original_intent,
            "max_steps": config.max_steps if config else 100,
            "use_xml": config.use_xml_bounding_boxes if config else False,
            "step_number": 0,
            "step_results": [],
            "is_complete": False,
            "should_retry": False,
        }

        # Run ainvoke in a task so we can cancel it from the monitor
        graph_task = asyncio.create_task(
            compiled_graph.ainvoke(
                initial_state,
                config={
                    "recursion_limit": 710,
                    "configurable": {"thread_id": self.workflow_id},
                },
            )
        )

        # Monitor: wait for either graph completion or cancellation
        cancel_waiter = asyncio.create_task(self.cancel_event.wait())

        done, pending = await asyncio.wait(
            {graph_task, cancel_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if cancel_waiter in done and graph_task not in done:
            # Cancellation was requested — cancel the in-flight graph task
            logger.info("LangGraph execution cancelled by user, cancelling task")
            graph_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await graph_task

            # Build a partial result from whatever node_ctx recorded
            self.__completion_reason = "Workflow cancelled by user"
            return self.__build_langgraph_result(
                node_ctx=node_ctx,
                final_state=None,
                cancelled=True,
            )

        # Clean up the cancel waiter if graph finished first
        cancel_waiter.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cancel_waiter

        final_state = graph_task.result()
        return self.__build_langgraph_result(
            node_ctx=node_ctx,
            final_state=final_state,
            cancelled=False,
        )

    def __build_langgraph_result(
        self,
        node_ctx: NodeContext,
        final_state: Optional[Dict[str, Any]],
        *,
        cancelled: bool,
    ) -> IntentResult:
        """Map graph terminal state to IntentResult."""

        if cancelled or final_state is None:
            # Use whatever the agent state recorded before cancellation
            step_results = (
                list(node_ctx.agent_state.results)
                if hasattr(node_ctx.agent_state, "results")
                else []
            )
            return IntentResult(
                metrics=node_ctx.metrics.to_report_dict(),
                success=False,
                intent=self.__original_intent,
                steps_taken=len(step_results),
                final_screen=self.__final_screen,
                completion_reason=self.__completion_reason or "Workflow cancelled by user",
                step_results=step_results,
            )

        success = final_state.get("is_complete", False)
        step_results = final_state.get("step_results", [])

        for sr in step_results:
            self.record_step(result=sr)

        completion_reason = final_state.get("completion_reason", "")
        if not completion_reason:
            completion_reason = (
                "Goal successfully achieved" if success else "Execution failed or timed out"
            )

        return IntentResult(
            metrics=node_ctx.metrics.to_report_dict(),
            success=success,
            intent=self.__original_intent,
            steps_taken=len(step_results),
            final_screen=self.__final_screen,
            completion_reason=completion_reason,
            step_results=step_results,
        )

    async def __should_continue(self) -> bool:
        """
        Required by base but unused in our overridden execute loop.
        """

        return not self.is_cancelled() and not (
            self.__strategy and self.__strategy.state.is_complete
        )

    def get_progress(self) -> Dict[str, Any]:
        """
        Get progress for the intent workflow.
        """

        progress = {
            "elapsed": self.elapsed,
            "intent": self.__original_intent,
            "steps_executed": self.steps_executed,
            "max_steps": self.configuration.max_steps,
        }
        if self.__strategy:
            progress["strategy"] = self.__strategy.get_progress()

        return progress
