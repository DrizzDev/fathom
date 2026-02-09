from __future__ import annotations

import asyncio
import time
from datetime import datetime
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fathom.agent.planner import StepPlanner
from fathom.agent.reasoner import Reasoner
from fathom.agent.state import AgentState
from fathom.agent.strategies.base import ExecutionStrategy
from fathom.constants import ActionType, StrategyStatus
from fathom.infrastructure.memory.ledger import Ledger
from fathom.interfaces import ILedger, IMemoryProvider
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.metrics import ExecutionMetrics
from fathom.schemas.results import ActionResult, PlanResult, StrategyResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.services.hierarchy import HierarchyService
from fathom.services.history import HistoryService
from fathom.services.ux import UXService
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool
from fathom.tools.vision.processing.annotator import ImageAnnotator
from fathom.utils.coordinates import CoordinateConverter

console = Console()
logger = getLogger(name=__name__)


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
        self.__ledger: ILedger = Ledger()

        self.__use_xml = use_xml
        self.__max_steps = max_steps

        self.__reasoner = Reasoner(intent=intent)
        self.__state = AgentState(intent=intent, max_steps=max_steps)

        self.__ux_service = UXService()
        self.__hierarchy = HierarchyService(device=device)
        self.__history = HistoryService(workflow_id=workflow_id)

        self.__start_time = time.time()
        self.__metrics = ExecutionMetrics()
        self.__memory_audit_trail: List[Dict[str, Any]] = []

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

        step_start = time.time()

        # 1. GROUNDING
        screen, xml, grounding_duration = await self.__perform_grounding()
        self.__metrics.record(operation="screenshot", duration=grounding_duration)

        if not screen:
            return StrategyResult(status=StrategyStatus.ERROR, message="Capture failed")

        state_object, is_new_screen = self.__update_state(screen=screen)

        # 2. HIERARCHY PROCESSING
        planning_screen, label_mapping, hierarchy_duration = await self.__process_hierarchy(
            screen=screen, xml=xml
        )
        if hierarchy_duration > 0:
            self.__metrics.record(operation="hierarchy_processing", duration=hierarchy_duration)

        # If loading detected during hierarchy processing
        if planning_screen is None:
            return StrategyResult(status=StrategyStatus.CONTINUE, message="Loading...")

        # 3. ANALYSIS & PLANNING
        plan, knowledge, analysis_duration = await self.__perform_analysis(
            state_object=state_object, planning_screen=planning_screen, label_mapping=label_mapping
        )
        self.__metrics.record(operation="analysis", duration=analysis_duration)

        if plan.is_complete:
            self.__finalize_audit()
            return StrategyResult(status=StrategyStatus.COMPLETE, message=plan.reason)

        if not plan.step:
            if plan.should_retry:
                return StrategyResult(status=StrategyStatus.CONTINUE, message=plan.reason)

            self.__finalize_audit()
            return StrategyResult(status=StrategyStatus.ERROR, message=plan.reason)

        step = plan.step
        if self.__use_xml and step.action.label_id:
            step = await self.__resolve(step=step)

        # 4. EXECUTION
        self.__render_ux(plan=plan, knowledge=knowledge, step=step, duration=analysis_duration)

        result, execution_duration, coordinates = await self.__execute_action_step(
            step=step, screen=screen
        )
        self.__metrics.record(operation="action", duration=execution_duration)

        # 5. RECORDING & AUDIT
        step_result = self.__record_result(
            step=step,
            result=result,
            step_start=step_start,
            coordinates=coordinates,
            state_object=state_object,
        )

        self.__audit_step(
            plan=plan,
            result=result,
            start_time=step_start,
            state_object=state_object,
            analysis=analysis_duration,
            is_new_screen=is_new_screen,
            execution=execution_duration,
            grounding=grounding_duration,
            hierarchy=hierarchy_duration,
        )

        return StrategyResult(step_result=step_result, status=StrategyStatus.CONTINUE, message="OK")

    async def __perform_grounding(self) -> Tuple[Optional[ScreenCapture], Optional[str], float]:
        """
        Captures screen and hierarchy.
        """

        start = time.time()

        if self.__use_xml:
            screen_task = self.__capture.capture_stable(timeout=2000)
            xml_task = self.__device.dump_hierarchy()
            screen, xml = await asyncio.gather(screen_task, xml_task)
        else:
            xml = None
            screen = await self.__capture.capture_stable(timeout=2000)

        return screen, xml, time.time() - start

    def __update_state(self, screen: ScreenCapture) -> Tuple[ScreenState, bool]:
        """
        Updates agent state with new screen.
        """

        state_object = self.__capture.compute_state(capture=screen)

        screen = screen.model_copy(update={"state": state_object})
        is_new_screen = self.__state.update_screen(screen=state_object)

        return state_object, is_new_screen

    async def __process_hierarchy(
        self, screen: ScreenCapture, xml: Optional[str]
    ) -> Tuple[Optional[ScreenCapture], Dict[str, Any], float]:
        """
        Processes XML hierarchy if available.
        Returns (planning_screen, label_mapping, duration).
        """

        if not (self.__use_xml and xml):
            return screen, {}, 0.0

        start = time.time()

        if len(xml) < 200:
            logger.info(msg="Screen appears to be loading (small XML)...")
            await asyncio.sleep(delay=1.0)
            return None, {}, time.time() - start

        annotated, label_mapping = await self.__hierarchy.process_xml_and_screen(
            screen=screen, xml=xml
        )
        return (annotated if annotated else screen), label_mapping, time.time() - start

    async def __perform_analysis(
        self,
        state_object: ScreenState,
        planning_screen: ScreenCapture,
        label_mapping: Dict[str, Any],
    ) -> Tuple[PlanResult, Dict[str, Any], float]:
        """
        Retrieves knowledge and plans the next step.
        """

        # Fetch knowledge
        knowledge = await self.__memory.retrieve_knowledge(visual_hash=state_object.visual_hash)

        # Inject ledger memory into knowledge for context
        entries = await self.__ledger.get_all()
        knowledge["memory_store"] = entries

        start = time.time()
        plan = await self.__planner.plan_step(
            state=self.__state,
            use_xml=self.__use_xml,
            capture=planning_screen,
            reasoner=self.__reasoner,
            elements=label_mapping if label_mapping else None,
        )
        return plan, knowledge, time.time() - start

    def __render_ux(
        self, plan: Any, knowledge: Dict[str, Any], step: Step, duration: float
    ) -> None:
        """
        Renders the step UX.
        """

        memory_lines = []
        previous_actions = knowledge.get("previous_actions", [])

        if previous_actions:
            memory_lines.append(
                f"[bold cyan]Retrieved {len(previous_actions)} experiences for this screen:[/bold cyan]"
            )
            for index, experience in enumerate(iterable=previous_actions, start=1):
                status = "✓" if experience.get("success") else "✗"
                memory_lines.append(
                    f"  {index}. {status} {experience.get('action')} on {experience.get('target')}"
                )
        else:
            memory_lines.append("[dim]No prior experience for this screen hash.[/dim]")

        memory_information = "\n".join(memory_lines) + "\n"

        if plan.metadata.get("tool_name"):
            self.__ux_service.render_tool_call(
                duration=duration,
                args=plan.metadata["tool_args"],
                tool_name=plan.metadata["tool_name"],
            )
        else:
            console.print(
                Panel(
                    renderable=f"{memory_information}\n"
                    f"[dim]Vision Target: None[/dim]\n"
                    f"[cyan]Reasoning:[/cyan] {step.action.rationale}\n"
                    f"[yellow]Action:[/yellow] {step.action.to_description()}",
                    title=f"Step {self.__state.step_count + 1} Thinking",
                    border_style="blue",
                )
            )

    async def __execute_action_step(
        self, step: Step, screen: ScreenCapture
    ) -> Tuple[ActionResult, float, Optional[List[int]]]:
        """
        Executes the planned action.
        """

        start = time.time()

        # Handle memory-only actions
        if step.action.action_type == ActionType.SAVE_MEMORY:
            if step.action.memory_updates:
                for key, value in step.action.memory_updates.items():
                    await self.__ledger.set(key=key, value=value)

            return ActionResult(success=True, duration=0), time.time() - start, None

        if step.action.action_type == ActionType.RETRIEVE_MEMORY:
            return ActionResult(success=True, duration=0), time.time() - start, None

        # Physical Action
        action_coordinates = await self.__get_action_coordinates(action=step.action)
        asyncio.create_task(
            coro=self.__trace_background(
                action=step.action, image_data=screen.image, coordinates=action_coordinates
            )
        )

        result = await self.__execute(action=step.action)

        # Side-effect memory updates
        if step.action.memory_updates:
            for key, value in step.action.memory_updates.items():
                await self.__ledger.set(key=key, value=value)

        # Convert coordinates to list for storage
        coords_list = list(action_coordinates) if action_coordinates else None

        center = None
        if coords_list:
            if len(coords_list) == 2:
                center = coords_list
            elif len(coords_list) == 4:
                center = [
                    (coords_list[0] + coords_list[2]) // 2,
                    (coords_list[1] + coords_list[3]) // 2,
                ]

        return result, time.time() - start, center

    def __record_result(
        self,
        step: Step,
        step_start: float,
        result: ActionResult,
        state_object: ScreenState,
        coordinates: Optional[List[int]] = None,
    ) -> StepResult:
        """
        Records the step result to history and memory.
        """

        step_result = StepResult(
            step=step,
            post_hash="0",
            screen_changed=True,
            success=result.success,
            pre_hash=state_object.visual_hash,
            duration=int((time.time() - step_start) * 1000),
        )
        self.__state.record_step(result=step_result)

        asyncio.create_task(
            coro=self.__memory.store_experience(
                visual_hash=step_result.pre_hash,
                action=step.action,
                success=result.success,
            )
        )
        self.__history.save_step(result=step_result, absolute_center=coordinates)
        return step_result

    def __audit_step(
        self,
        plan: PlanResult,
        grounding: float,
        hierarchy: float,
        analysis: float,
        execution: float,
        start_time: float,
        is_new_screen: bool,
        result: ActionResult,
        state_object: ScreenState,
    ) -> None:
        """
        Prints the step audit table.
        """

        audit = Table.grid(padding=(0, 2))
        audit.add_column(style="dim")
        audit.add_column(justify="right")

        status_icon = "🆕" if is_new_screen else "🔄"
        audit.add_row(
            "Screen Status:",
            f"{status_icon} {state_object.visual_hash[:8]} ({state_object.activity})",
        )

        if self.__state.is_stuck:
            audit.add_row("[bold red]Loop Detected:[/bold red]", "YES")

        audit.add_row("Grounding:", self.__format_time(milliseconds=grounding * 1000))
        if hierarchy > 0:
            audit.add_row("Hierarchy:", self.__format_time(milliseconds=hierarchy * 1000))

        if plan.metrics:
            if "memory_retrieval" in plan.metrics:
                audit.add_row(
                    "Memory Retrieval:",
                    self.__format_time(milliseconds=plan.metrics["memory_retrieval"] * 1000),
                )
            if "llm_analysis" in plan.metrics:
                audit.add_row(
                    "LLM Core Analysis:",
                    self.__format_time(milliseconds=plan.metrics["llm_analysis"] * 1000),
                )

        audit.add_row("Total Analysis:", self.__format_time(milliseconds=analysis * 1000))
        audit.add_row("ADB Execution:", self.__format_time(milliseconds=result.duration))
        audit.add_row(
            "Step Overhead:",
            self.__format_time(milliseconds=(execution * 1000) - result.duration),
        )

        total = (time.time() - start_time) * 1000
        audit.add_row(
            "[bold white]Total Step Time:[/bold white]",
            f"[bold cyan]{self.__format_time(milliseconds=total)}[/bold cyan]",
        )

        console.print(
            Panel(
                renderable=audit,
                border_style="dim",
                title_align="right",
                title=f"Step {self.__state.step_count} Audit",
            )
        )

    def __format_time(self, milliseconds: float) -> str:
        """
        Formats milliseconds to 'Xs [Yms]' format.
        """

        seconds = milliseconds / 1000.0
        return f"{seconds:.2f}s [{milliseconds:.0f}ms]"

    def __finalize_audit(self) -> None:
        """
        Prints final memory and context audit.
        """

        if not self.__memory_audit_trail:
            return

        audit_table = Table(
            title="Execution Context & Memory Audit", show_lines=True, header_style="bold magenta"
        )
        audit_table.add_column(header="Step", justify="center")
        audit_table.add_column(header="Hash / Knowledge (READ)", style="cyan")
        audit_table.add_column(header="Session Context (SENT)", style="green")
        audit_table.add_column(header="Action Result (WRITE)", style="yellow")

        for item in self.__memory_audit_trail:
            # Format knowledge
            knowledge = item["knowledge_retrieved"]
            knowledge_string = f"Hash: [dim]{item['visual_hash'][:12]}[/dim]\n"
            knowledge_string += f"Desc: {knowledge.get('description', 'N/A')}\n"

            past_actions = knowledge.get("previous_actions", [])

            if past_actions:
                knowledge_string += f"Past: {len(past_actions)} actions retrieved"
            else:
                knowledge_string += "Past: No prior experience"

            # Format context
            context = item["context_sent"]
            failures = context.get("relevant_failures", [])
            context_string = f"History: {context.get('compact_history')}\n"

            if failures:
                context_string += f"Failures Sent: {', '.join(failures)}"
            else:
                context_string += "Failures Sent: None"

            # Format action
            success_tag = (
                "[bold green]OK[/bold green]" if item["success"] else "[bold red]FAIL[/bold red]"
            )
            action_string = f"{item['action_stored']}\nStatus: {success_tag}"

            audit_table.add_row(
                str(item["step"]),
                knowledge_string,
                context_string,
                action_string,
            )

        console.print(renderable="\n")
        console.print(renderable=audit_table)

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
        label_id = step.action.label_id

        if label_id and label_id in mapping:
            element = mapping[label_id]
            size = await self.__device.get_screen_size()

            if size[0] > 0 and size[1] > 0:
                # Calculate normalized center (0-1000)
                normalized_x = int((element["center_x"] / size[0]) * 1000)
                normalized_y = int((element["center_y"] / size[1]) * 1000)

                # Ensure we don't produce a bounding box that clamps to 0,0
                # We use a 100x100 box centered at the element
                bounding_box_x = max(0, normalized_x - 50)
                bounding_box_y = max(0, normalized_y - 50)

                action = step.action.model_copy(
                    update={
                        "bounds": Bounds(x=bounding_box_x, y=bounding_box_y, width=100, height=100)
                    }
                )
                return step.model_copy(update={"action": action})

        if label_id:
            logger.warning(msg=f"Failed to resolve coordinates for label ID: {label_id}")

        return step

    async def __get_action_coordinates(self, action: Action) -> Tuple[int, ...]:
        """
        Converts action bounding box to pixel coordinates for verification.
        """

        size = await self.__device.get_screen_size()
        converter = CoordinateConverter(screen_width=size[0], screen_height=size[1])

        if action.action_type in (ActionType.TAP, ActionType.TYPE, ActionType.LONG_PRESS):
            if action.bounds:
                return converter.center_to_pixels(bounds=action.bounds)

            return (size[0] // 2, size[1] // 2)

        if action.action_type in (ActionType.SWIPE, ActionType.SCROLL):
            # For simplicity, use a generic swipe if no bounding box
            if not action.bounds:
                return (size[0] // 2, size[1] * 3 // 4, size[0] // 2, size[1] // 4)

            # Default to swipe up logic if no direction in simple Action schema
            return converter.swipe_coordinates(bounds=action.bounds, direction="up")

        return ()

    async def __trace_background(
        self,
        action: Action,
        image_data: bytes,
        coordinates: Tuple[int, ...],
    ) -> None:
        """
        Saves annotated action image to assets/traces.
        """

        if not coordinates:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"step_{self.__state.step_count + 1}_{action.action_type.value}_{timestamp}.png"
        output_path = f"assets/traces/{filename}"

        ImageAnnotator.trace(
            coords=coordinates,
            image_data=image_data,
            output_path=output_path,
            label=action.to_description(),
            action_type=action.action_type.value,
        )

    async def __execute(self, action: Action) -> ActionResult:
        """
        Dispatches action to device.
        """

        size = await self.__device.get_screen_size()
        converter = CoordinateConverter(screen_width=size[0], screen_height=size[1])

        if action.action_type == ActionType.TAP:
            coordinates = (
                converter.center_to_pixels(bounds=action.bounds)
                if action.bounds
                else (size[0] // 2, size[1] // 2)
            )
            return await self.__device.tap(x=coordinates[0], y=coordinates[1])

        if action.action_type == ActionType.TYPE:
            return await self.__device.type_text(text=action.text or "")

        if action.action_type in (
            ActionType.SWIPE_LEFT,
            ActionType.SWIPE_RIGHT,
            ActionType.SWIPE_UP,
            ActionType.SWIPE_DOWN,
        ):
            size = await self.__device.get_screen_size()
            converter = CoordinateConverter(screen_width=size[0], screen_height=size[1])
            direction = action.action_type.value.split("_")[1]
            coords = converter.swipe_coordinates(
                bounds=action.bounds or Bounds(x=200, y=200, width=600, height=600),
                direction=direction,
            )
            return await self.__device.swipe(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])

        if action.action_type == ActionType.WAIT:
            duration = action.wait_duration or 1000
            await asyncio.sleep(delay=duration / 1000.0)
            return ActionResult(success=True, duration=duration)

        return await self.__device.execute(request=action.model_dump())

    async def __capture_with_timeout(self) -> Optional[ScreenCapture]:
        """
        Safely capture screen.
        """

        try:
            return await self.__capture.capture()
        except Exception:
            return None
