"""
LangGraph node functions for the Fathom intent-execution graph.

Each function is a **thin adapter** that delegates to the existing Fathom
components (``StepPlanner``, ``StepExecutor``, ``AgentState``, etc.).
Mutable objects that don't belong in the LangGraph ``TypedDict`` state
(deques, services, etc.) are captured via a shared :class:`NodeContext`
closure that is constructed once and injected into each node factory.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Tuple

from fathom.agent.planner import StepPlanner
from fathom.agent.reasoner import Reasoner
from fathom.agent.state import AgentState
from fathom.constants import ActionType
from fathom.graph.state import FathomGraphState
from fathom.infrastructure.memory.ledger import Ledger
from fathom.infrastructure.memory.sqlite import SQLiteMemoryProvider
from fathom.interfaces import ILedger, IMemoryProvider
from fathom.prompts.preprocessor import PromptPreprocessor
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.metrics import ExecutionMetrics
from fathom.schemas.results import ActionResult, PlanResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.services.audit import AuditService
from fathom.services.classifier import TargetClassifier
from fathom.services.hierarchy import HierarchyService
from fathom.services.history import HistoryService
from fathom.services.resolution import ReferenceResolutionService
from fathom.services.ux import UXService
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool
from fathom.tools.vision.processing.annotator import ImageAnnotator
from fathom.utils.coordinates import CoordinateConverter
from fathom.utils.execution import execute_device_action, get_action_coordinates

logger = getLogger(__name__)


class CancellationError(Exception):
    """Raised when a graph node detects that the workflow has been cancelled."""


def _extract_package(activity: str) -> str:
    """Extract the package name from an activity string.

    Activity strings are typically ``com.example.app/.MainActivity``.
    Returns the portion before ``/``, or the full string if no ``/``
    is present.
    """

    if not activity:
        return ""
    return activity.split("/")[0]


def _maybe_switch_knowledge_db(
    ctx: "NodeContext",
    activity: str,
) -> None:
    """Switch the knowledge DB when the foreground app changes.

    Compares the package extracted from *activity* against
    ``ctx.current_package``.  When they differ and the memory provider
    is a :class:`SQLiteMemoryProvider`, ``switch_database`` is called so
    that all subsequent queries target the correct per-app knowledge
    graph.  ``ctx.current_package`` is updated accordingly.
    """

    new_pkg = _extract_package(activity)
    if not new_pkg or new_pkg == ctx.current_package:
        return

    if isinstance(ctx.memory, SQLiteMemoryProvider):
        new_db = f"assets/memory/{new_pkg}/knowledge.db"
        ctx.memory.switch_database(new_db)
        logger.info(
            "Package changed: %s -> %s, knowledge DB switched",
            ctx.current_package or "(initial)",
            new_pkg,
        )

    ctx.current_package = new_pkg


class NodeContext:
    """
    Shared mutable context for all graph nodes.

    Holds the heavy objects (device connections, memory, services, agent state)
    that **cannot** live inside the immutable ``FathomGraphState`` dict.
    Created once per workflow run and closed over by each node function.
    """

    def __init__(
        self,
        intent: str,
        planner: StepPlanner,
        device: DeviceTool,
        capture: CaptureTool,
        memory: IMemoryProvider,
        *,
        max_steps: int = 100,
        use_xml: bool = False,
        step_timeout: float = 15.0,
        workflow_id: str = "default",
        cancel_event: Optional[asyncio.Event] = None,
        package_name: str = "",
    ) -> None:
        self.intent = intent
        self.planner = planner
        self.device = device
        self.capture_tool = capture
        self.memory = memory
        self.use_xml = use_xml
        self.max_steps = max_steps
        self._cancel_event = cancel_event or asyncio.Event()
        self.current_package = package_name

        self.ledger: ILedger = Ledger()
        self.reasoner = Reasoner(intent=intent)
        self.agent_state = AgentState(intent=intent, max_steps=max_steps)
        self.metrics = ExecutionMetrics()

        self.ux_service = UXService()
        self.audit_service = AuditService()
        self.hierarchy = HierarchyService(device=device)
        self.history = HistoryService(workflow_id=workflow_id, intent=intent)
        self.resolution = ReferenceResolutionService(ledger=self.ledger)
        self.classifier = TargetClassifier()

    @property
    def is_cancelled(self) -> bool:
        """Fast non-blocking check for cancellation."""
        return self._cancel_event.is_set()


# ── Node factory ────────────────────────────────────────────────────────


def build_nodes(ctx: NodeContext) -> Dict[str, Callable[..., Any]]:
    """
    Return a dict of ``{node_name: async_callable}`` that each accept and
    return :class:`FathomGraphState`.  All nodes close over *ctx*.
    """

    async def ground_node(state: FathomGraphState) -> FathomGraphState:
        """Capture the screen and optionally the XML hierarchy."""

        if ctx.is_cancelled:
            logger.info("ground_node: cancelled, skipping capture")
            return {
                **state,
                "capture": None,
                "is_complete": True,
                "completion_reason": "Workflow cancelled by user",
            }

        # Lazy goal-state inference (matches IntentStrategy)
        if not ctx.history.goal_state:
            try:
                ctx.history.goal_state = await ctx.classifier.infer_goal_state(intent=ctx.intent)
            except Exception as e:
                logger.warning(f"Failed to infer goal state: {e}")

        start = time.time()
        if ctx.use_xml:
            screen_task = ctx.capture_tool.capture_stable(timeout=2000)
            xml_task = ctx.device.dump_hierarchy()
            screen, _xml = await asyncio.gather(screen_task, xml_task)
        else:
            screen = await ctx.capture_tool.capture_stable(timeout=2000)

        grounding_duration = time.time() - start
        ctx.metrics.record(operation="screenshot", duration=grounding_duration)

        if not screen:
            return {
                **state,
                "capture": None,
                "is_complete": False,
                "grounding_duration": grounding_duration,
                "completion_reason": "Capture failed",
            }

        # Update agent state with the new screen
        screen_state = ctx.capture_tool.compute_state(capture=screen)
        screen = screen.model_copy(update={"state": screen_state})
        is_new = ctx.agent_state.update_screen(screen=screen_state)

        # Detect package change and switch knowledge DB if needed
        _maybe_switch_knowledge_db(ctx=ctx, activity=screen.activity)

        return {
            **state,
            "capture": screen,
            "screen_state": screen_state,
            "is_new_screen": is_new,
            "grounding_duration": grounding_duration,
            # Reset per-step fields
            "plan": None,
            "planned_step": None,
            "step_result": None,
        }

    async def hierarchy_node(state: FathomGraphState) -> FathomGraphState:
        """Process XML hierarchy and annotate the screen image."""

        screen: Optional[ScreenCapture] = state.get("capture")
        if not screen:
            return {**state, "planning_screen": None, "elements": {}, "hierarchy_duration": 0.0}

        xml = screen.xml_content if ctx.use_xml else None

        if not (ctx.use_xml and xml):
            return {**state, "planning_screen": screen, "elements": {}, "hierarchy_duration": 0.0}

        start = time.time()
        xml_size_kb = len(xml.encode("utf-8")) / 1024

        if xml_size_kb < 0.2:
            logger.warning("XML too small, waiting for UI stability…")
            await asyncio.sleep(1.0)
            return {
                **state,
                "planning_screen": screen,
                "elements": {},
                "hierarchy_duration": time.time() - start,
            }

        try:
            annotated, mapping = await ctx.hierarchy.process_xml_and_screen(
                screen=screen, xml=xml, action_type=ActionType.TAP
            )
            duration = time.time() - start
            ctx.metrics.record(operation="hierarchy_processing", duration=duration)

            if annotated and annotated.image != screen.image:
                return {
                    **state,
                    "planning_screen": annotated,
                    "elements": mapping,
                    "hierarchy_duration": duration,
                }
            return {
                **state,
                "planning_screen": screen,
                "elements": mapping,
                "hierarchy_duration": duration,
            }
        except Exception as exc:
            logger.exception(f"Hierarchy processing error: {exc}")
            return {
                **state,
                "planning_screen": screen,
                "elements": {},
                "hierarchy_duration": time.time() - start,
            }

    async def analyze_node(state: FathomGraphState) -> FathomGraphState:
        """Run the LLM planner to decide the next action."""

        if ctx.is_cancelled:
            logger.info("analyze_node: cancelled, skipping LLM call")
            return {
                **state,
                "plan": None,
                "planned_step": None,
                "is_complete": True,
                "should_retry": False,
                "analysis_duration": 0.0,
                "completion_reason": "Workflow cancelled by user",
            }

        planning_screen: Optional[ScreenCapture] = state.get("planning_screen")
        screen_state: Optional[ScreenState] = state.get("screen_state")
        elements: Dict[str, Any] = state.get("elements", {})

        if planning_screen is None:
            return {
                **state,
                "plan": None,
                "planned_step": None,
                "analysis_duration": 0.0,
                "should_retry": False,
            }

        if screen_state is None:
            return {
                **state,
                "plan": None,
                "planned_step": None,
                "analysis_duration": 0.0,
                "should_retry": False,
            }

        # Retrieve knowledge
        entries = await ctx.ledger.get_all()
        knowledge = await ctx.memory.retrieve_knowledge(visual_hash=screen_state.visual_hash)
        knowledge["memory_store"] = entries

        start = time.time()
        smart_context = ctx.agent_state.get_smart_context()
        hints = PromptPreprocessor.extract_hints(
            intent=ctx.agent_state.intent, current_activity=screen_state.activity or ""
        )
        hint_str = PromptPreprocessor.build_context_prefix(hints)
        full_context = f"{hint_str}\n{smart_context}" if hint_str else smart_context

        plan = await ctx.planner.plan_step(
            state=ctx.agent_state,
            use_xml=ctx.use_xml,
            capture=planning_screen,
            reasoner=ctx.reasoner,
            elements=elements if elements else None,
            additional_context=full_context,
        )

        analysis_duration = time.time() - start
        ctx.metrics.record(operation="analysis", duration=analysis_duration)

        if plan.metrics:
            ctx.metrics.record_tokens(
                prompt=int(plan.metrics.get("prompt_tokens", 0)),
                completion=int(plan.metrics.get("completion_tokens", 0)),
                cached=int(plan.metrics.get("cached_tokens", 0)),
            )

        # Check validity
        if getattr(plan, "is_valid_action", True) is False:
            reason = getattr(plan, "validation_reasoning", "Invalid action")
            logger.warning(f"Invalid action: {reason}")
            ctx.agent_state.set_last_error(reason)
            return {
                **state,
                "plan": plan,
                "planned_step": None,
                "knowledge": knowledge,
                "analysis_duration": analysis_duration,
                "should_retry": True,
                "is_complete": False,
            }

        step = plan.step
        if not step:
            return {
                **state,
                "plan": plan,
                "planned_step": None,
                "knowledge": knowledge,
                "analysis_duration": analysis_duration,
                "should_retry": plan.should_retry,
                "is_complete": plan.is_complete,
                "completion_reason": plan.reason
                if plan.is_complete
                else state.get("completion_reason"),
            }

        # HARDWARE BACK override
        if step.action.action_type == ActionType.BACK:
            action = step.action.model_copy(update={"bounds": None})
            step = step.model_copy(update={"action": action})

        return {
            **state,
            "plan": plan,
            "planned_step": step,
            "knowledge": knowledge,
            "analysis_duration": analysis_duration,
            "should_retry": False,
            "is_complete": plan.is_complete,
            "completion_reason": plan.reason
            if plan.is_complete
            else state.get("completion_reason"),
        }

    async def resolve_node(state: FathomGraphState) -> FathomGraphState:
        """Resolve dynamic references ($memory, $env) and XML coordinates."""

        step: Optional[Step] = state.get("planned_step")
        if not step:
            return state

        # Resolve references
        resolved_action = await ctx.resolution.resolve(action=step.action)
        step = step.model_copy(update={"action": resolved_action})

        # Resolve XML label → coordinates
        if ctx.use_xml and step.action.label_id:
            label_id = step.action.label_id
            mapping = ctx.hierarchy.label_map

            if label_id and label_id in mapping:
                element = mapping[label_id]
                size = await ctx.device.get_screen_size()

                if size[0] > 0 and size[1] > 0:
                    normalized_x = int((element["center_x"] / size[0]) * 1000)
                    normalized_y = int((element["center_y"] / size[1]) * 1000)
                    element_name = element.get("text") or element.get("content-desc")

                    updates: Dict[str, Any] = {
                        "bounds": Bounds(
                            width=100,
                            height=100,
                            x=normalized_x - 50,
                            y=normalized_y - 50,
                        )
                    }
                    if element_name and str(element_name).strip():
                        updates["target"] = str(element_name).strip()
                        updates["natural_language_target"] = str(element_name).strip()

                    action = step.action.model_copy(update=updates)
                    step = step.model_copy(update={"action": action})

        return {**state, "planned_step": step}

    async def execute_node(state: FathomGraphState) -> FathomGraphState:
        """Execute the planned action on the device."""

        if ctx.is_cancelled:
            logger.info("execute_node: cancelled, skipping device action")
            return {
                **state,
                "step_result": None,
                "execution_duration": 0.0,
            }

        step: Optional[Step] = state.get("planned_step")
        plan: Optional[PlanResult] = state.get("plan")
        capture: Optional[ScreenCapture] = state.get("capture")
        analysis_duration: float = state.get("analysis_duration", 0.0)

        if not step or not capture:
            return {**state, "step_result": None, "execution_duration": 0.0}

        # UX rendering
        if plan and plan.metadata.get("tool_name"):
            ctx.ux_service.render_tool_call(
                duration=analysis_duration,
                args=plan.metadata["tool_args"],
                tool_name=plan.metadata["tool_name"],
            )
        elif plan:
            ctx.ux_service.render_fallback(
                reasoning=step.action.rationale,
                action=step.action.to_description(),
                step_number=ctx.agent_state.step_count + 1,
            )

        step_start = time.time()

        # Handle memory-only actions
        if step.action.action_type == ActionType.SAVE_MEMORY:
            if step.action.memory_updates:
                for key, value in step.action.memory_updates.items():
                    await ctx.ledger.set(key=key, value=value)
            action_result = ActionResult(success=True, duration=0)
            execution_duration = time.time() - step_start
            ctx.metrics.record(operation="action", duration=execution_duration)
            return {
                **state,
                "step_result": _build_step_result(
                    step, state.get("screen_state"), action_result, step_start
                ),
                "execution_duration": execution_duration,
            }

        if step.action.action_type == ActionType.RETRIEVE_MEMORY:
            action_result = ActionResult(success=True, duration=0)
            execution_duration = time.time() - step_start
            ctx.metrics.record(operation="action", duration=execution_duration)
            return {
                **state,
                "step_result": _build_step_result(
                    step, state.get("screen_state"), action_result, step_start
                ),
                "execution_duration": execution_duration,
            }

        # Physical action
        coordinates = await get_action_coordinates(ctx.device, step.action)
        await _trace_background(ctx, step.action, capture.image, coordinates)
        action_result = await execute_device_action(ctx.device, step.action)

        if step.action.memory_updates:
            for key, value in step.action.memory_updates.items():
                await ctx.ledger.set(key=key, value=value)

        execution_duration = time.time() - step_start
        ctx.metrics.record(operation="action", duration=execution_duration)

        return {
            **state,
            "step_result": _build_step_result(
                step, state.get("screen_state"), action_result, step_start
            ),
            "execution_duration": execution_duration,
        }

    async def record_node(state: FathomGraphState) -> FathomGraphState:
        """Record the step result into agent state, memory, and audit logs."""

        if ctx.is_cancelled:
            logger.info("record_node: cancelled, skipping recording")
            return {**state, "is_complete": True, "completion_reason": "Workflow cancelled by user"}

        step_result: Optional[StepResult] = state.get("step_result")
        plan: Optional[PlanResult] = state.get("plan")
        screen_state: Optional[ScreenState] = state.get("screen_state")
        knowledge: Dict[str, Any] = state.get("knowledge", {})

        if not step_result:
            return state

        step = step_result.step

        # Extract screen_description from plan metadata for classifier context
        screen_description = ""
        if plan and plan.metadata:
            tool_args = plan.metadata.get("tool_args", {})
            screen_description = str(tool_args.get("screen_description", ""))

        # Classification: prefer VLM-provided target_type/script_target; fallback to classifier
        generalized_target = None
        is_positional = False
        target_type = getattr(step.action, "target_type", None)
        script_target = getattr(step.action, "script_target", None)

        if target_type in ("positional", "dynamic") and script_target:
            generalized_target = script_target.strip()
            is_positional = target_type == "positional"
        else:
            target_text = step.action.natural_language_target or step.action.target
            if target_text:
                try:
                    classification = await ctx.classifier.classify_and_generalize(
                        target=target_text,
                        intent=ctx.intent,
                        rationale=step.action.rationale,
                        screen_description=screen_description,
                    )
                    if classification.description != target_text:
                        generalized_target = classification.description
                        is_positional = classification.is_positional
                except Exception as e:
                    logger.warning(f"Target classification failed: {e}")

        if generalized_target:
            step_result = step_result.model_copy(
                update={"generalized_target": generalized_target, "is_positional": is_positional}
            )

        # Record in agent state
        ctx.agent_state.record_step(result=step_result)

        # Persist experience asynchronously
        asyncio.create_task(
            ctx.memory.store_experience(
                action=step.action,
                success=step_result.success,
                visual_hash=step_result.pre_hash,
            )
        )

        # Compute absolute center for history export
        center: Optional[List[int]] = None
        if step.action.bounds:
            size = await ctx.device.get_screen_size()
            converter = CoordinateConverter(screen_width=size[0], screen_height=size[1])
            cx, cy = converter.center_to_pixels(bounds=step.action.bounds)
            center = [cx, cy]

        ctx.history.save_step(result=step_result, absolute_center=center)

        # Audit
        if screen_state:
            ctx.audit_service.record_context(
                knowledge=knowledge,
                success=step_result.success,
                visual_hash=screen_state.visual_hash,
                step_number=ctx.agent_state.step_count,
                context=ctx.agent_state.build_context(),
                action_description=step.action.to_description(),
            )

            if plan:
                ctx.audit_service.log_step(
                    plan=plan,
                    state=screen_state,
                    result=ActionResult(success=step_result.success, duration=step_result.duration),
                    is_new_screen=state.get("is_new_screen", False),
                    is_stuck=ctx.agent_state.is_stuck,
                    step_count=ctx.agent_state.step_count,
                    analysis_duration=state.get("analysis_duration", 0.0),
                    grounding_duration=state.get("grounding_duration", 0.0),
                    hierarchy_duration=state.get("hierarchy_duration", 0.0),
                    execution_duration=state.get("execution_duration", 0.0),
                    total_duration=state.get("grounding_duration", 0.0)
                    + state.get("analysis_duration", 0.0)
                    + state.get("execution_duration", 0.0),
                )

        # Append to accumulated results list
        results: List[StepResult] = list(state.get("step_results", []))
        results.append(step_result)

        return {
            **state,
            "step_result": step_result,
            "step_results": results,
            "step_number": ctx.agent_state.step_count,
        }

    return {
        "ground": ground_node,
        "hierarchy": hierarchy_node,
        "analyze": analyze_node,
        "resolve": resolve_node,
        "execute": execute_node,
        "record": record_node,
    }


# ── Routing functions ───────────────────────────────────────────────────


def make_route_after_ground(ctx: NodeContext) -> Callable[[FathomGraphState], str]:
    """Return a router that checks whether grounding succeeded."""

    def route(state: FathomGraphState) -> str:
        if ctx.is_cancelled:
            return "done"
        if state.get("capture") is None:
            return "done"
        return "hierarchy"

    return route


def make_route_after_analyze(ctx: NodeContext) -> Callable[[FathomGraphState], str]:
    """Return a router that decides what happens after analysis."""

    def route(state: FathomGraphState) -> str:
        if ctx.is_cancelled:
            return "done"

        # Complete — no step to execute
        if state.get("is_complete") and state.get("planned_step") is None:
            ctx.audit_service.print_session_summary()
            return "done"

        # Complete but has a final physical action to run
        if state.get("is_complete") and state.get("planned_step") is not None:
            return "resolve"

        # Retry — go back to grounding for a fresh screen
        if state.get("should_retry"):
            return "ground"

        # No step at all (edge case — planner returned nothing)
        if state.get("planned_step") is None:
            ctx.audit_service.print_session_summary()
            return "done"

        return "resolve"

    return route


def make_route_after_record(ctx: NodeContext) -> Callable[[FathomGraphState], str]:
    """Return a router that decides whether to loop or terminate."""

    def route(state: FathomGraphState) -> str:
        if ctx.is_cancelled:
            return "done"

        if state.get("is_complete"):
            return "done"

        if not ctx.agent_state.can_continue:
            return "done"

        return "ground"

    return route


# ── Private helpers ─────────────────────────────────────────────────────


def _build_step_result(
    step: Step,
    screen_state: Optional[ScreenState],
    result: ActionResult,
    step_start: float,
) -> StepResult:
    return StepResult(
        step=step,
        post_hash="0",
        screen_changed=True,
        success=result.success,
        pre_hash=screen_state.visual_hash if screen_state else "0",
        duration=int((time.time() - step_start) * 1000),
        generalized_target=None,
    )


# _get_action_coordinates and _execute_device_action have been moved to
# fathom.utils.execution and are imported at the top of this module as
# ``get_action_coordinates`` and ``execute_device_action``.


def _coords_to_center(coordinates: Tuple[int, ...]) -> Optional[List[int]]:
    if not coordinates:
        return None
    coords_list = list(coordinates)
    if len(coords_list) == 2:
        return coords_list
    if len(coords_list) == 4:
        return [(coords_list[0] + coords_list[2]) // 2, (coords_list[1] + coords_list[3]) // 2]
    return None


async def _trace_background(
    ctx: NodeContext, action: Action, image_data: bytes, coordinates: Tuple[int, ...]
) -> None:
    if not coordinates:
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = (
        f"step__{ctx.agent_state.step_count + 1}__{action.action_type.value}__{timestamp}.png"
    )
    path = f"assets/traces/{filename}"
    ImageAnnotator.trace(
        output_path=path,
        coords=coordinates,
        image_data=image_data,
        label=action.to_description(),
        action_type=action.action_type.value,
    )
