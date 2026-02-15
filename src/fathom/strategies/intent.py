"""
Intent-based execution strategy using hexagonal architecture.

Migrated from agent/strategies/intent.py with ports instead of tools.
"""

from __future__ import annotations

import time
from logging import getLogger
from typing import Any, Dict, Optional, Tuple

from fathom.agent.planner import StepPlanner
from fathom.agent.reasoner import Reasoner
from fathom.agent.state import AgentState
from fathom.constants import ActionType
from fathom.core.context.manager import ContextManager
from fathom.core.execution.engine import ExecutionEngine
from fathom.infrastructure.memory.ledger import Ledger
from fathom.interfaces.device import DevicePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.prompts.preprocessor import PromptPreprocessor
from fathom.schemas.metrics import ExecutionMetrics
from fathom.schemas.results import ActionResult, ExecutionResult, PlanResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.services.audit import AuditService
from fathom.services.history import HistoryService
from fathom.services.resolution import ReferenceResolutionService
from fathom.services.ux import UXService

logger = getLogger(name=__name__)


class IntentStrategy:
    """
    Strategy for executing a specific intent using hexagonal architecture.
    
    This is the REAL implementation migrated from agent/strategies/intent.py
    but adapted to use ports (DevicePort, LLMPort, etc.) instead of tools.
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
        max_steps: int = 20,
        use_xml: bool = False,
        workflow_id: str = "default",
    ) -> None:
        """Initialize intent strategy with ports."""
        self.__engine = engine
        self.__context = context
        self.__intent = intent
        
        # Ports
        self.__device = device
        self.__llm = llm
        self.__memory = memory
        self.__storage = storage
        self.__telemetry = telemetry
        
        # Configuration
        self.__use_xml = use_xml
        self.__max_steps = max_steps
        
        # Agent components (reuse existing logic)
        self.__planner = StepPlanner(vision_tool=None)  # Will need LLM port
        self.__reasoner = Reasoner(intent=intent)
        self.__state = AgentState(intent=intent, max_steps=max_steps)
        self.__ledger = Ledger()
        
        # Services (reuse existing)
        self.__ux_service = UXService()
        self.__audit_service = AuditService()
        self.__history = HistoryService(workflow_id=workflow_id)
        self.__resolution = ReferenceResolutionService(ledger=self.__ledger)
        
        # Metrics
        self.__start_time = time.time()
        self.__metrics = ExecutionMetrics()

    async def execute(self, max_steps: int) -> ExecutionResult:
        """
        Execute intent-based workflow.
        
        This is the REAL execution loop from agent/strategies/intent.py
        """
        start_time = time.time()
        self.__context.set_roadmap(intent=self.__intent)
        
        success = False
        error = None
        
        try:
            # Main execution loop
            while self.__state.can_continue and not self.__state.is_complete:
                if self.__state.step_count >= max_steps:
                    break
                
                # Execute one step
                step_result = await self.__execute_step()
                
                if not step_result:
                    break
                
                # Check if complete
                if self.__state.is_complete:
                    success = True
                    break
            
            if not success and not error:
                error = "Max steps reached or execution stopped"
                
        except Exception as e:
            logger.exception(f"Intent strategy execution failed: {e}")
            error = str(e)
            success = False
        
        duration = int((time.time() - start_time) * 1000)
        
        return ExecutionResult(
            success=success,
            duration=duration,
            error=error,
        )

    async def __execute_step(self) -> Optional[StepResult]:
        """
        Execute a single step (migrated from agent/strategies/intent.py).
        """
        step_start = time.time()
        
        # 1. GROUNDING - Capture screen
        screen, xml, grounding_duration = await self.__perform_grounding()
        self.__metrics.record(operation="screenshot", duration=grounding_duration)
        
        if not screen:
            self.__telemetry.error("Screen capture failed")
            return None
        
        # 2. UPDATE STATE
        state, is_new_screen = await self.__update_state(screen=screen)
        
        # 3. ANALYSIS - Get knowledge and plan next step
        plan, knowledge, analysis_duration = await self.__perform_analysis(state=state, screen=screen)
        self.__metrics.record(operation="analysis", duration=analysis_duration)
        
        # Track token usage
        if plan.metrics:
            self.__metrics.record_tokens(
                prompt=int(plan.metrics.get("prompt_tokens", 0)),
                completion=int(plan.metrics.get("completion_tokens", 0)),
                cached=int(plan.metrics.get("cached_tokens", 0)),
            )
        
        # Check if we have a step to execute
        step = plan.step
        if not step:
            if plan.is_complete:
                self.__state.mark_complete()
                return None
            return None
        
        # HARDWARE BACK: Always use keycode for back actions
        if step.action.action_type == ActionType.BACK:
            action = step.action.model_copy(update={"bounds": None})
            step = step.model_copy(update={"action": action})
        
        # RESOLUTION: Resolve dynamic references ($memory, $env)
        step = await self.__resolve_references(step=step)
        
        # 4. EXECUTION - Execute through engine
        result = await self.__engine.execute_step(step=step)
        
        # 5. RECORDING
        self.__state.record_step(result=result)
        
        # Store experience in memory
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
        
        # Audit logging
        self.__audit_service.log_step(
            plan=plan,
            state=state,
            result=ActionResult(success=result.success, duration=result.duration, error=result.error),
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
        """Capture screen and hierarchy."""
        start = time.time()
        
        try:
            # Capture screenshot through device port
            screenshot_bytes = await self.__device.capture_screen()
            
            # Store screenshot
            storage_id = await self.__storage.save(
                data=screenshot_bytes,
                metadata={"type": "screenshot", "timestamp": time.time()},
            )
            
            screen = ScreenCapture(
                image_data=screenshot_bytes,
                storage_id=storage_id,
                timestamp=time.time(),
                image=screenshot_bytes,  # For compatibility
            )
            
            # Get XML hierarchy if enabled
            xml = None
            if self.__use_xml:
                # XML hierarchy would come from device port if supported
                pass
            
            return screen, xml, time.time() - start
            
        except Exception as e:
            logger.exception(f"Grounding failed: {e}")
            return None, None, time.time() - start

    async def __update_state(self, screen: ScreenCapture) -> Tuple[ScreenState, bool]:
        """Update agent state with new screen."""
        # Compute screen state (hash, activity, etc.)
        import hashlib
        visual_hash = hashlib.sha256(screen.image_data).hexdigest()[:16]
        
        # Get current package/activity
        try:
            package = await self.__device.get_current_package()
        except:
            package = "unknown"
        
        state = ScreenState(
            visual_hash=visual_hash,
            activity=package,
            timestamp=screen.timestamp,
        )
        
        is_new = self.__state.update_screen(screen=state)
        
        return state, is_new

    async def __perform_analysis(
        self,
        state: ScreenState,
        screen: ScreenCapture,
    ) -> Tuple[PlanResult, Dict[str, Any], float]:
        """Retrieve knowledge and plan next step using LLM."""
        start = time.time()
        
        # Get memory entries
        entries = await self.__ledger.get_all()
        knowledge = await self.__memory.retrieve_knowledge(visual_hash=state.visual_hash)
        knowledge["memory_store"] = entries
        
        # Build context
        smart_context = self.__state.get_smart_context()
        
        # Prompt preprocessing
        hints = PromptPreprocessor.extract_hints(
            intent=self.__state.intent,
            current_activity=state.activity or ""
        )
        hint_str = PromptPreprocessor.build_context_prefix(hints)
        
        full_context = smart_context
        if hint_str:
            full_context = f"{hint_str}\n{smart_context}"
        
        # Plan next step using planner (which uses LLM)
        plan = await self.__planner.plan_step(
            state=self.__state,
            use_xml=self.__use_xml,
            capture=screen,
            reasoner=self.__reasoner,
            elements=None,
            additional_context=full_context,
        )
        
        return plan, knowledge, time.time() - start

    async def __resolve_references(self, step: Step) -> Step:
        """Resolve dynamic references in the action."""
        resolved_action = await self.__resolution.resolve(action=step.action)
        return step.model_copy(update={"action": resolved_action})

    def get_progress(self) -> Dict[str, Any]:
        """Get execution progress."""
        return {
            "intent": self.__intent,
            "step_count": self.__state.step_count,
            "is_complete": self.__state.is_complete,
            "context": self.__context.get_full_context(),
        }
