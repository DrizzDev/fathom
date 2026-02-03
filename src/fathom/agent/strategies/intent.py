from __future__ import annotations

import asyncio
import time
from datetime import datetime
from logging import getLogger
from typing import Dict, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fathom.agent.planner import StepPlanner
from fathom.agent.reasoner import Reasoner
from fathom.agent.state import AgentState
from fathom.agent.strategies.base import ExecutionStrategy
from fathom.constants import ActionType, StrategyStatus
from fathom.interfaces import IMemoryProvider
from fathom.schemas.actions import Action, BoundingBox
from fathom.schemas.metrics import ExecutionMetrics
from fathom.schemas.results import ActionResult, StrategyResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step, StepResult
from fathom.services.hierarchy import HierarchyService
from fathom.services.history import HistoryService
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool
from fathom.tools.vision.processing.annotator import ImageAnnotator
from fathom.utils.coordinates import CoordinateConverter

logger = getLogger(__name__)
console = Console()


class IntentStrategy(ExecutionStrategy):
    """
    Strategy for executing a specific intent.
    """

    def __init__(
        self,
        intent: str,
        planner: StepPlanner,
        device: DeviceTool,
        capture: CaptureTool,
        memory: IMemoryProvider,
        *,
        max_steps: int = 20,
        use_xml: bool = False,
        step_timeout: float = 15.0,
        workflow_id: str = "default",
    ) -> None:
        self.__intent = intent
        self.__planner = planner
        self.__device = device
        self.__capture = capture
        self.__memory = memory
        self.__max_steps = max_steps
        self.__use_xml = use_xml
        self.__reasoner = Reasoner(intent)
        self.__state = AgentState(intent, max_steps=max_steps)
        self.__hierarchy = HierarchyService(device)
        self.__history = HistoryService(workflow_id)
        self.__start = time.time()
        self.__metrics = ExecutionMetrics()

    @property
    def metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Return reporting metrics.
        """
        return self.__metrics.to_report_dict()

    @property
    def name(self) -> str:
        """
        Strategy name.
        """
        return "intent"

    @property
    def state(self) -> AgentState:
        """
        Agent state.
        """
        return self.__state

    async def execute_step(self) -> StrategyResult:
        """
        Executes one step with accurate metric tracking.
        """
        start = time.time()

        # 1. GROUNDING PHASE
        grounding_start = time.time()
        if self.__use_xml:
            # Parallel capture and dump
            screen_task = self.__capture.capture_stable(timeout=2000)
            xml_task = self.__device.dump_hierarchy()
            screen, xml = await asyncio.gather(screen_task, xml_task)
            self.__metrics.record("hierarchy_dump", time.time() - grounding_start)
        else:
            screen = await self.__capture.capture_stable(timeout=2000)
            xml = None

        grounding_duration = time.time() - grounding_start
        self.__metrics.record("screenshot", grounding_duration)

        if not screen:
            return StrategyResult(status=StrategyStatus.ERROR, message="Capture failed")

        # Update screen capture with computed state
        state_obj = self.__capture.compute_state(screen)
        screen = screen.model_copy(update={"state": state_obj})
        is_new_screen = self.__state.update_screen(state_obj)

        planning_screen = screen
        hierarchy_proc_duration = 0.0
        if self.__use_xml and xml:
            proc_start = time.time()
            if len(xml) < 200:
                logger.info("Hierarchy too small, likely loading...")
                await asyncio.sleep(1.0)
                return StrategyResult(status=StrategyStatus.CONTINUE, message="Loading...")

            annotated, _ = await self.__hierarchy.process_xml_and_screen(screen, xml)
            hierarchy_proc_duration = time.time() - proc_start
            self.__metrics.record("hierarchy_processing", hierarchy_proc_duration)
            if annotated:
                planning_screen = annotated.model_copy(update={"state": state_obj})

        # 2. ANALYSIS PHASE
        analysis_start = time.time()
        plan = await self.__planner.plan_step(
            self.__state, self.__reasoner, planning_screen, use_xml=self.__use_xml
        )
        analysis_duration = time.time() - analysis_start
        self.__metrics.record("analysis", analysis_duration)

        if plan.is_complete:
            return StrategyResult(status=StrategyStatus.COMPLETE, message=plan.reason)

        if not plan.step:
            if plan.should_retry:
                return StrategyResult(status=StrategyStatus.CONTINUE, message=plan.reason)
            return StrategyResult(status=StrategyStatus.ERROR, message=plan.reason)

        step = plan.step
        if self.__use_xml and step.action.label_id:
            step = await self.__resolve(step)

        # 3. BRAIN & THINKING TRANSPARENCY
        knowledge = await self.__memory.retrieve_knowledge(state_obj.visual_hash)
        prev_actions = knowledge.get("previous_actions", [])

        memory_lines = []
        if prev_actions:
            memory_lines.append(
                f"[bold cyan]Retrieved {len(prev_actions)} experiences for this screen:[/bold cyan]"
            )
            for idx, exp in enumerate(prev_actions, 1):
                status = "✓" if exp.get("success") else "✗"
                memory_lines.append(f"  {idx}. {status} {exp.get('action')} on {exp.get('target')}")
        else:
            memory_lines.append("[dim]No prior experience for this screen hash.[/dim]")

        memory_info = "\n".join(memory_lines) + "\n"

        console.print(
            Panel(
                f"{memory_info}\n"
                f"[cyan]Reasoning:[/cyan] {step.action.rationale}\n"
                f"[yellow]Action:[/yellow] {step.action.to_description()}",
                title=f"Step {self.__state.step_count + 1} Thinking",
                border_style="blue",
            )
        )

        # 4. EXECUTION PHASE
        exec_start = time.time()

        # Capture raw coordinates for background verification
        action_coords = await self.__get_action_coordinates(step.action)

        # Start verification annotation in background (zero latency impact)
        asyncio.create_task(self.__trace_background(screen.image, step.action, action_coords))

        result = await self.__execute(step.action)
        exec_duration = time.time() - exec_start
        self.__metrics.record("action", exec_duration)

        step_result = StepResult(
            step=step,
            success=result.success,
            screen_changed=True,
            pre_hash=state_obj.visual_hash,
            post_hash="0",
            duration=int((time.time() - start) * 1000),
        )
        self.__state.record_step(step_result)

        # Store experience
        asyncio.create_task(
            self.__memory.store_experience(
                visual_hash=step_result.pre_hash,
                action=step.action,
                success=result.success,
            )
        )
        self.__history.save_step(step_result)

        # 5. AUDIT & TIMING
        audit = Table.grid(padding=(0, 2))
        audit.add_column(style="dim")
        audit.add_column(justify="right")

        status_icon = "🆕" if is_new_screen else "🔄"
        audit.add_row(
            "Screen Status:", f"{status_icon} {state_obj.visual_hash[:8]} ({state_obj.activity})"
        )

        if self.__state.is_stuck:
            audit.add_row("[bold red]Loop Detected:[/bold red]", "YES")

        audit.add_row("Grounding (Stable Capture):", f"{grounding_duration * 1000:.0f}ms")
        if hierarchy_proc_duration > 0:
            audit.add_row("Hierarchy Processing:", f"{hierarchy_proc_duration * 1000:.0f}ms")

        if plan.metrics:
            if "memory_retrieval" in plan.metrics:
                audit.add_row(
                    "Memory Retrieval:", f"{plan.metrics['memory_retrieval'] * 1000:.0f}ms"
                )
            if "llm_analysis" in plan.metrics:
                audit.add_row("LLM Core Analysis:", f"{plan.metrics['llm_analysis'] * 1000:.0f}ms")

        audit.add_row("Total Analysis:", f"{analysis_duration * 1000:.0f}ms")
        audit.add_row("ADB Execution:", f"{result.duration}ms")
        audit.add_row("Step Overhead:", f"{(exec_duration * 1000) - result.duration:.0f}ms")

        total_ms = (time.time() - start) * 1000
        audit.add_row(
            "[bold white]Total Step Time:[/bold white]", f"[bold cyan]{total_ms:.0f}ms[/bold cyan]"
        )

        console.print(
            Panel(
                audit,
                title=f"Step {self.__state.step_count} Audit",
                title_align="right",
                border_style="dim",
            )
        )

        return StrategyResult(step_result=step_result, status=StrategyStatus.CONTINUE, message="OK")

    async def should_continue(self) -> bool:
        """
        Check stop conditions.
        """
        if self.__state.is_complete:
            return False
        return self.__state.can_continue

    def get_progress(self) -> Dict[str, object]:
        """
        Return workflow progress.
        """
        return {
            "intent": self.__intent,
            "step_count": self.__state.step_count,
            "is_complete": self.__state.is_complete,
        }

    async def __resolve(self, step: Step) -> Step:
        """
        Resolves label to physical coordinates.
        """
        mapping = self.__hierarchy.label_map
        if step.action.label_id in mapping:
            element = mapping[step.action.label_id]
            size = await self.__device.get_screen_size()
            if size[0] > 0 and size[1] > 0:
                nx = int((element["center_x"] / size[0]) * 1000)
                ny = int((element["center_y"] / size[1]) * 1000)
                action = step.action.model_copy(
                    update={"bbox": BoundingBox(x=nx - 50, y=ny - 50, width=100, height=100)}
                )
                return step.model_copy(update={"action": action})
        return step

    async def __get_action_coordinates(self, action: Action) -> Tuple[int, ...]:
        """
        Converts action bbox to pixel coordinates for verification.
        """
        size = await self.__device.get_screen_size()
        converter = CoordinateConverter(size[0], size[1])

        if action.action_type in (ActionType.TAP, ActionType.TYPE, ActionType.LONG_PRESS):
            if action.bbox:
                return converter.center_to_pixels(action.bbox)
            return (size[0] // 2, size[1] // 2)

        if action.action_type in (ActionType.SWIPE, ActionType.SCROLL):
            # For simplicity, use a generic swipe if no bbox
            if not action.bbox:
                return (size[0] // 2, size[1] * 3 // 4, size[0] // 2, size[1] // 4)
            # Default to swipe up logic if no direction in simple Action schema
            return converter.swipe_coordinates(action.bbox, "up")

        return ()

    async def __trace_background(
        self, image_data: bytes, action: Action, coords: Tuple[int, ...]
    ) -> None:
        """
        Saves annotated action image to assets/traces.
        """
        if not coords:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"step_{self.__state.step_count + 1}_{action.action_type.value}_{timestamp}.png"
        output_path = f"assets/traces/{filename}"

        ImageAnnotator.trace(
            image_data=image_data,
            output_path=output_path,
            action_type=action.action_type.value,
            coords=coords,
            label=action.to_description(),
        )

    async def __execute(self, action: Action) -> ActionResult:
        """
        Dispatches action to device.
        """
        size = await self.__device.get_screen_size()
        converter = CoordinateConverter(size[0], size[1])
        if action.action_type == ActionType.TAP:
            coords = (
                converter.center_to_pixels(action.bbox)
                if action.bbox
                else (size[0] // 2, size[1] // 2)
            )
            return await self.__device.tap(coords[0], coords[1])
        if action.action_type == ActionType.TYPE:
            return await self.__device.type_text(action.text or "")
        if action.action_type == ActionType.WAIT:
            duration = action.wait_duration or 1000
            await asyncio.sleep(duration / 1000.0)
            return ActionResult(success=True, duration=duration)
        return await self.__device.execute(action.model_dump())

    async def __capture_with_timeout(self) -> Optional[ScreenCapture]:
        """
        Safely capture screen.
        """
        try:
            return await self.__capture.capture()
        except Exception:
            return None
