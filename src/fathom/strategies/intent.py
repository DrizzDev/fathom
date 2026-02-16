"""
Intent-based execution strategy using hexagonal architecture.

Migrated from agent/strategies/intent.py with ports instead of tools.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from rich.console import Console

from fathom.core.agent.planner import StepPlanner
from fathom.core.agent.reasoner import Reasoner
from fathom.core.agent.state import AgentState
from fathom.constants import ActionType, SignalType
from fathom.constants.execution import VISUAL_HASH_LENGTH
from fathom.core.context.manager import ContextManager
from fathom.core.exceptions import StrategyError
from fathom.core.execution.engine import ExecutionEngine
from fathom.core.services.audit import AuditService
from fathom.core.services.hierarchy import HierarchyService
from fathom.core.services.history import HistoryService
from fathom.core.services.resolution import ReferenceResolutionService
from fathom.core.services.ux import UXService
from fathom.core.services.vision import VisionService
from fathom.interfaces.device import DevicePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.core.prompts.preprocessor import PromptPreprocessor
from fathom.schemas.actions import Bounds
from fathom.schemas.metrics import ExecutionMetrics
from fathom.schemas.results import ActionResult, ExecutionResult, PlanResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult

if TYPE_CHECKING:
    from fathom.base.paths import SharedPathManager

logger = getLogger(name=__name__)
console = Console()


class IntentStrategy:
    """
    Strategy for executing a specific intent using hexagonal architecture.

    Consolidates actions and perception through the DevicePort.
    """

    def __init__(
        self,
        engine: ExecutionEngine,
        context: ContextManager,
        intent: str,
        *,
        device: DevicePort,
        llm: LLMPort,
        memory: MemoryPort,
        storage: StoragePort,
        telemetry: TelemetryPort,
        signal: SignalPort,
        path_manager: SharedPathManager,
        max_steps: int = 20,
        use_xml: bool = False,
        workflow_id: str = "default",
        package_name: str = "unknown_app",
    ) -> None:
        """Initialize intent strategy with ports."""
        self.__engine = engine
        self.__context = context
        self.__intent = intent
        self.__package_name = package_name

        # Ports
        self.__device = device
        self.__llm = llm
        self.__memory = memory
        self.__storage = storage
        self.__telemetry = telemetry
        self.__signal = signal
        self.__path_manager = path_manager

        # Configuration
        self.__use_xml = use_xml
        self.__max_steps = max_steps
        self.__workflow_id = workflow_id

        # Create vision service
        vision_tool = VisionService(
            llm=llm,
            memory=memory,
            storage=storage,
            version="pro_xml" if use_xml else "pro",
            session_id=workflow_id,
            package_name=package_name,
        )

        # Agent components
        self.__planner = StepPlanner(vision_tool=vision_tool)
        self.__reasoner = Reasoner(intent=intent)
        self.__state = AgentState(intent=intent, max_steps=max_steps)

        # Services
        self.__ux_service = UXService()
        self.__audit_service = AuditService()
        self.__hierarchy = HierarchyService(device=device)
        self.__history = HistoryService(
            workflow_id=workflow_id,
            package_name=package_name,
            path_manager=path_manager,
        )
        self.__resolution = ReferenceResolutionService(ledger=memory)

        # Metrics
        self.__start_time = time.time()
        self.__metrics = ExecutionMetrics()

        # Cancellation support
        self.__cancelled = False

    async def execute(self, max_steps: int) -> ExecutionResult:
        """Execute intent-based workflow."""
        start_time = time.time()
        self.__context.set_roadmap(intent=self.__intent)

        success = False
        error = None

        try:
            while (
                self.__state.can_continue and not self.__state.is_complete and not self.__cancelled
            ):
                if self.__state.step_count >= max_steps:
                    break

                step_result = await self.__execute_step()
                if not step_result:
                    break

                if self.__state.is_complete:
                    success = True
                    break

            if self.__cancelled:
                error = "Execution cancelled by user"
            elif not success and not error:
                error = "Max steps reached or execution stopped"

        except StrategyError as exception:
            logger.exception(f"Intent strategy execution failed: {exception}")
            error = str(exception)
            success = False
        except Exception as exception:
            logger.exception(f"Unexpected error in intent strategy: {exception}")
            raise StrategyError("Intent execution failed unexpectedly") from exception

        duration = int((time.time() - start_time) * 1000)
        return ExecutionResult(success=success, duration=duration, error=error)

    async def __execute_step(self) -> Optional[StepResult]:
        """Execute a single step."""
        step_start = time.time()

        # 1. GROUNDING - Capture screen via DevicePort
        screen, xml, grounding_duration = await self.__perform_grounding()
        self.__metrics.record(operation="screenshot", duration=grounding_duration)

        if not screen:
            self.__telemetry.error("Screen capture failed")
            return None

        # 2. UPDATE STATE
        state, is_new_screen = await self.__update_state(screen=screen)

        # 3. ANALYSIS
        elements = screen.metadata.get("elements") if screen.metadata else None
        plan, knowledge, analysis_duration = await self.__perform_analysis(
            state=state, screen=screen, elements=elements
        )
        self.__metrics.record(operation="analysis", duration=analysis_duration)

        # Token usage
        if plan.metrics:
            self.__metrics.record_tokens(
                prompt=int(plan.metrics.get("prompt_tokens", 0)),
                completion=int(plan.metrics.get("completion_tokens", 0)),
                cached=int(plan.metrics.get("cached_tokens", 0)),
            )

        # UX
        if plan.step:
            self.__ux_service.render_fallback(
                reasoning=plan.reason or "No reasoning provided.",
                action=plan.step.action.to_description(),
                step_number=self.__state.step_count + 1,
            )

        # HITL checks
        if plan.step and plan.step.action.confidence < 0.5:
            # Request guidance if confidence is low
            question = f"Agent is uncertain ({plan.step.action.confidence:.1%}). What should it do?"
            try:
                user_guidance = await self.__signal.ask(prompt=question)
                if user_guidance.strip():
                    await self.__context.inject_user_guidance(guidance=user_guidance)
                    plan, knowledge, analysis_duration = await self.__perform_analysis(
                        state=state,
                        screen=screen,
                        additional_context=f"USER GUIDANCE: {user_guidance}",
                        elements=elements,
                    )
            except Exception:
                with contextlib.suppress(Exception):
                    pass

        # HITL: Check for injected context from signal
        if self.__signal.has_injected_context():
            injected = self.__signal.get_injected_context()
            if injected:
                self.__telemetry.info("Using injected context", context=injected)
                # Inject into context for LLM reasoning
                await self.__context.inject_user_guidance(guidance=injected)

                # Re-analyze with injected context
                # Format to make it clear this can override/modify the goal
                priority_context = (
                    f"{'=' * 60}\n"
                    f"🎯 USER INSTRUCTION (PRIORITY):\n"
                    f"{injected}\n\n"
                    f"Note: This user instruction takes priority. If it conflicts with the original goal, "
                    f"follow this instruction instead. If it adds a sub-goal, complete it as part of the workflow.\n"
                    f"{'=' * 60}"
                )

                plan, knowledge, analysis_duration = await self.__perform_analysis(
                    state=state,
                    screen=screen,
                    additional_context=priority_context,
                    elements=elements,
                )

        step = plan.step
        if not step:
            if plan.is_complete:
                self.__state.mark_complete(reason="Goal achieved according to plan")
            return None

        # Resolve references and coordinates
        step = await self.__resolve_references(step=step)
        if self.__use_xml and step.action.label_id:
            step = await self.__resolve_coordinates(step=step)

        # 4. EXECUTION
        result = await self.__engine.execute_step(
            step=step,
            pre_capture=screen,
            package_name=self.__package_name,
            session_id=self.__workflow_id,
        )

        # 5. RECORDING
        self.__state.record_step(result=result)

        # Save history
        self.__history.save_step(result=result, intent=self.__intent)

        await self.__memory.store_experience(
            visual_hash=state.visual_hash,
            action=step.action,
            success=result.success,
        )

        # Update context
        await self.__context.commit(
            observation=f"Screen: {state.activity}",
            thought=step.action.rationale,
            action=step.action,
        )

        self.__audit_service.log_step(
            plan=plan,
            state=state,
            result=ActionResult(
                success=result.success, duration=result.duration, error=result.error
            ),
            is_new_screen=is_new_screen,
            is_stuck=self.__state.is_stuck,
            step_count=self.__state.step_count,
            analysis_duration=analysis_duration,
            grounding_duration=grounding_duration,
            hierarchy_duration=0,
            execution_duration=result.duration / 1000.0,
            total_duration=time.time() - step_start,
        )

        return result

    async def __perform_grounding(self) -> Tuple[Optional[ScreenCapture], Optional[str], float]:
        """Capture screen and hierarchy via DevicePort."""
        start = time.time()
        try:
            # Direct calls to device port
            screenshot_bytes = await self.__device.capture_screen()
            width, height = await self.__device.get_screen_size()

            try:
                activity = await self.__device.get_current_package()
            except Exception:
                activity = "unknown"

            # Store screenshot
            storage_id = await self.__storage.save(
                data=screenshot_bytes,
                metadata={
                    "type": "screenshots",
                    "package_name": activity,
                    "session_id": self.__workflow_id,
                    "timestamp": time.time(),
                },
            )

            screen = ScreenCapture(
                image=screenshot_bytes,
                width=width,
                height=height,
                activity=activity,
                timestamp=int(time.time() * 1000),
                metadata={"storage_id": storage_id},
            )

            xml = None
            if self.__use_xml:
                xml = await self.__device.dump_hierarchy()

            if self.__use_xml and xml:
                annotated_screen, elements = await self.__hierarchy.process_xml_and_screen(
                    screen=screen,
                    xml=xml,
                    path_manager=self.__path_manager,
                    package_name=self.__package_name,
                    session_id=self.__workflow_id,
                    action_type=ActionType.TAP,
                )
                if annotated_screen:
                    screen = annotated_screen
                new_metadata = screen.metadata.copy()
                new_metadata["elements"] = elements
                screen = screen.model_copy(update={"metadata": new_metadata})

            return screen, xml, time.time() - start
        except Exception as exception:
            logger.exception(f"Grounding failed: {exception}")
            return None, None, time.time() - start

    async def __update_state(self, screen: ScreenCapture) -> Tuple[ScreenState, bool]:
        """Update agent state with new screen."""
        visual_hash = hashlib.sha256(screen.image).hexdigest()[:VISUAL_HASH_LENGTH]

        state = ScreenState(
            visual_hash=visual_hash,
            activity=screen.activity,
            timestamp=screen.timestamp,
            activity_hash=hashlib.md5(screen.activity.encode(), usedforsecurity=False).hexdigest()[
                :VISUAL_HASH_LENGTH
            ],
            structural_hash="0" * VISUAL_HASH_LENGTH,  # Not computed in this simplified version
        )

        is_new = self.__state.update_screen(screen=state)

        return state, is_new

    async def __perform_analysis(
        self,
        state: ScreenState,
        screen: ScreenCapture,
        additional_context: Optional[str] = None,
        elements: Optional[Dict[str, Any]] = None,
    ) -> Tuple[PlanResult, Dict[str, Any], float]:
        """Retrieves knowledge and plans next step."""
        start = time.time()
        entries = await self.__memory.get_all()
        knowledge = await self.__memory.retrieve_knowledge(visual_hash=state.visual_hash)
        knowledge["memory_store"] = entries

        smart_context = self.__state.get_smart_context()

        # Add user guidance if available
        user_guidance = self.__context.get_user_guidance()
        if user_guidance:
            guidance_str = "\n\nUSER GUIDANCE:\n" + "\n".join(f"- {g}" for g in user_guidance)
            smart_context = smart_context + guidance_str
            # Clear guidance after using it
            self.__context.clear_user_guidance()

        # Prompt preprocessing
        hints = PromptPreprocessor.extract_hints(
            intent=self.__state.intent, current_activity=state.activity or ""
        )
        hint_str = PromptPreprocessor.build_context_prefix(hints=hints)

        full_context = smart_context
        if hint_str:
            full_context = f"{hint_str}\n{smart_context}"

        if additional_context:
            full_context = f"{full_context}\n\n{additional_context}"

        # Create LLM task that can be cancelled
        console.print("[dim]🤖 Analyzing screen and planning next action...[/dim]")

        llm_task = asyncio.create_task(
            self.__planner.plan_step(
                state=self.__state,
                use_xml=self.__use_xml,
                capture=screen,
                reasoner=self.__reasoner,
                elements=elements,
                additional_context=full_context,
            )
        )

        # Poll for pause requests while LLM is working
        while not llm_task.done():
            # Check if pause requested
            signal = await self.__signal.check_signal()
            if signal == SignalType.ASK or signal == SignalType.PAUSE:
                # Cancel the LLM task immediately
                console.print("[yellow]⏸️  Cancelling current LLM analysis...[/yellow]")
                llm_task.cancel()
                try:
                    await llm_task
                except asyncio.CancelledError:
                    console.print("[green]✓ LLM analysis cancelled[/green]")

                # Wait for user to resume
                await self.__signal.wait_for_resume()

                # Check for injected context
                if self.__signal.has_injected_context():
                    injected = self.__signal.get_injected_context()
                    if injected:
                        # Add injected context and retry
                        console.print(
                            "[bold cyan]📝 Adding your context to LLM prompt:[/bold cyan]"
                        )
                        console.print(f"[italic]{injected}[/italic]\n")

                        # Format context to make it clear it can override/modify the goal
                        full_context = f"{full_context}\n\n{'=' * 60}\n🎯 USER INSTRUCTION (PRIORITY):\n{injected}\n\nNote: This user instruction takes priority. If it conflicts with the original goal, follow this instruction instead. If it adds a sub-goal, complete it as part of the workflow.\n{'=' * 60}"

                # Restart the LLM call with updated context
                console.print("[dim]🤖 Restarting analysis with your context...[/dim]")
                llm_task = asyncio.create_task(
                    self.__planner.plan_step(
                        state=self.__state,
                        use_xml=self.__use_xml,
                        capture=screen,
                        reasoner=self.__reasoner,
                        elements=elements,
                        additional_context=full_context,
                    )
                )

            # Small sleep to avoid busy waiting
            await asyncio.sleep(delay=0.1)

        # Get the result
        plan = await llm_task
        console.print("[green]✓ Analysis complete[/green]\n")

        return plan, knowledge, time.time() - start

    async def __resolve_references(self, step: Step) -> Step:
        """Resolve dynamic references in the action."""
        resolved_action = await self.__resolution.resolve(action=step.action)
        return step.model_copy(update={"action": resolved_action})

    async def __resolve_coordinates(self, step: Step) -> Step:
        """Resolves label to physical coordinates."""
        label_id = step.action.label_id
        mapping = self.__hierarchy.label_map
        if label_id and label_id in mapping:
            element = mapping[label_id]
            size = await self.__device.get_screen_size()
            if size[0] > 0 and size[1] > 0:
                normalized_x = int((element["center_x"] / size[0]) * 1000)
                normalized_y = int((element["center_y"] / size[1]) * 1000)
                updates = {
                    "bounds": Bounds(
                        x=normalized_x - 50, y=normalized_y - 50, width=100, height=100
                    ),
                    "target": element.get("text")
                    or element.get("content-desc")
                    or step.action.target,
                }
                action = step.action.model_copy(update=updates)
                return step.model_copy(update={"action": action})
        return step

    def get_progress(self) -> Dict[str, Any]:
        """Get execution progress."""
        return {
            "intent": self.__intent,
            "step_count": self.__state.step_count,
            "is_complete": self.__state.is_complete,
            "context": self.__context.get_full_context(),
            "metrics": self.__metrics.to_dict(),
        }

    def get_metrics(self) -> ExecutionMetrics:
        """Get execution metrics."""
        return self.__metrics

    def cancel(self) -> None:
        """Cancel the execution."""
        self.__cancelled = True
        self.__telemetry.warning("Intent strategy cancellation requested")
