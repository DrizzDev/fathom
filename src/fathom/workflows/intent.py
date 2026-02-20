from __future__ import annotations

import asyncio
import contextlib
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from fathom.graph.nodes import NodeContext

from fathom.agent.planner import StepPlanner
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

    Orchestrates the full execution of a goal-directed automation
    using a LangGraph StateGraph that drives planning, analysis,
    resolution, execution, and recording nodes.
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

        self.__package_name = configuration.package_name if configuration else ""
        self.__planner = StepPlanner(vision_tool=vision)

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
        Execute the intent workflow via the LangGraph StateGraph.
        """

        logger.info(f"Executing intent: {self.__original_intent}")

        from fathom.graph.checkpointer import build_checkpointer
        from fathom.graph.intent_graph import build_intent_graph

        config = self.configuration

        checkpointer = None
        if config and config.human_in_loop:
            checkpoint_path = Path("assets/checkpoints/intent") / f"{self.workflow_id}.sqlite"
            checkpointer = build_checkpointer(checkpoint_path)

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
            checkpointer=checkpointer,
            cancel_event=self.cancel_event,
            pause_event=self.pause_event,
            package_name=self.__package_name,
            human_in_loop=config.human_in_loop if config else False,
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

        graph_task = asyncio.create_task(
            compiled_graph.ainvoke(
                initial_state,
                config={
                    "recursion_limit": 710,
                    "configurable": {"thread_id": self.workflow_id},
                },
            )
        )

        cancel_waiter = asyncio.create_task(self.cancel_event.wait())

        done, pending = await asyncio.wait(
            {graph_task, cancel_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if cancel_waiter in done and graph_task not in done:
            logger.info("LangGraph execution cancelled by user, cancelling task")
            graph_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await graph_task

            self.__completion_reason = "Workflow cancelled by user"
            return self.__build_result(
                node_ctx=node_ctx,
                final_state=None,
                cancelled=True,
            )

        cancel_waiter.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cancel_waiter

        final_state = graph_task.result()
        return self.__build_result(
            node_ctx=node_ctx,
            final_state=final_state,
            cancelled=False,
        )

    def __build_result(
        self,
        node_ctx: NodeContext,
        final_state: Optional[Dict[str, Any]],
        *,
        cancelled: bool,
    ) -> IntentResult:
        """Map graph terminal state to IntentResult."""

        if cancelled or final_state is None:
            return IntentResult(
                metrics=node_ctx.metrics.to_report_dict(),
                success=False,
                intent=self.__original_intent,
                steps_taken=node_ctx.agent_state.step_count,
                final_screen=self.__final_screen,
                completion_reason=self.__completion_reason or "Workflow cancelled by user",
                step_results=[],
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

    def get_progress(self) -> Dict[str, Any]:
        """
        Get progress for the intent workflow.
        """

        return {
            "elapsed": self.elapsed,
            "intent": self.__original_intent,
            "steps_executed": self.steps_executed,
            "max_steps": self.configuration.max_steps,
        }
