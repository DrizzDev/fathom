"""
Intent-based execution strategy using hexagonal architecture.

Migrated from agent/strategies/intent.py with ports instead of tools.
"""


from __future__ import annotations

import hashlib
import time
from logging import getLogger
from typing import Any, Dict, Optional, Tuple

from fathom.adapters.vision import (
    ImageStorageAdapter,
    LLMVisionProvider,
    MemoryProviderAdapter,
)
from fathom.agent.planner import StepPlanner
from fathom.agent.reasoner import Reasoner
from fathom.agent.state import AgentState
from fathom.constants import ActionType
from fathom.constants.execution import VISUAL_HASH_LENGTH, SignalType
from fathom.core.context.manager import ContextManager
from fathom.core.exceptions import StrategyError
from fathom.core.execution.engine import ExecutionEngine
from fathom.infrastructure.memory.ledger import Ledger
from fathom.interfaces.device import DevicePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.signal import SignalPort
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
from fathom.tools.vision.gemini import GeminiVisionTool

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
        signal: SignalPort,
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
        self.__signal = signal
        
        # Configuration
        self.__use_xml = use_xml
        self.__max_steps = max_steps
        
        # Create adapters to bridge new ports to old interfaces
        vision_provider = LLMVisionProvider(llm=llm)
        memory_provider = MemoryProviderAdapter(memory=memory)
        image_storage = ImageStorageAdapter(storage=storage)
        
        # Ledger for session state
        self.__ledger = Ledger()
        
        # Create vision tool with adapters
        vision_tool = GeminiVisionTool(
            model=vision_provider,
            memory=memory_provider,
            ledger=self.__ledger,
            local_storage=image_storage,
            version="pro_xml" if use_xml else "pro",
            session_id=workflow_id,
        )
        
        # Agent components (reuse existing logic with proper wiring)
        self.__planner = StepPlanner(vision_tool=vision_tool)
        self.__reasoner = Reasoner(intent=intent)
        self.__state = AgentState(intent=intent, max_steps=max_steps)
        
        # Services (reuse existing)
        self.__ux_service = UXService()
        self.__audit_service = AuditService()
        self.__history = HistoryService(workflow_id=workflow_id)
        self.__resolution = ReferenceResolutionService(ledger=self.__ledger)
        
        # Metrics
        self.__start_time = time.time()
        self.__metrics = ExecutionMetrics()
        
        # Cancellation support
        self.__cancelled = False

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
            while self.__state.can_continue and not self.__state.is_complete and not self.__cancelled:
                if self.__state.step_count >= max_steps:
                    break
                
                # Check for manual pause signal
                signal = await self.__signal.check_signal()
                if signal == SignalType.ASK.value:
                    # User requested pause - wait for resume
                    await self.__signal.wait_for_resume()
                
                # Execute one step
                step_result = await self.__execute_step()
                
                if not step_result:
                    break
                
                # Check if complete
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
        
        # HITL: Check if agent is stuck or uncertain and needs human help
        if plan.step and plan.step.action.confidence < 0.5:
            # Agent is uncertain - ask for help
            self.__telemetry.warning(
                "Agent uncertain about next action",
                confidence=plan.step.action.confidence,
                step=self.__state.step_count
            )
            
            # Request human input through signal port
            question = (
                f"The agent is uncertain (confidence: {plan.step.action.confidence:.1%}) about what to do next.\n"
                f"Current screen: {state.activity}\n"
                f"Intent: {self.__intent}\n"
                f"Suggested action: {plan.step.action.to_description() if plan.step else 'None'}\n\n"
                f"What should the agent do? (Provide guidance or press Enter to continue)"
            )
            
            try:
                user_guidance = await self.__signal.request_input(prompt=question)
                
                if user_guidance.strip():
                    # Inject user guidance into context for next analysis
                    await self.__context.inject_user_guidance(guidance=user_guidance)
                    self.__telemetry.info("User guidance received", guidance=user_guidance)
                    
                    # Re-analyze with user guidance
                    plan, knowledge, analysis_duration = await self.__perform_analysis(
                        state=state,
                        screen=screen,
                        additional_context=f"USER GUIDANCE: {user_guidance}"
                    )
            except Exception as exception:
                self.__telemetry.warning(f"Failed to get user input: {exception}")
        
        # HITL: Check for injected context from signal
        if hasattr(self.__signal, 'has_injected_context') and self.__signal.has_injected_context():
            injected = self.__signal.get_injected_context()
            if injected:
                self.__telemetry.info("Using injected context", context=injected)
                # Inject into context for LLM reasoning
                await self.__context.inject_user_guidance(guidance=injected)
                
                # Re-analyze with injected context
                plan, knowledge, analysis_duration = await self.__perform_analysis(
                    state=state,
                    screen=screen,
                    additional_context=f"USER CONTEXT: {injected}"
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
            
            # Get screen dimensions
            width, height = await self.__device.get_screen_size()
            
            # Get current activity
            try:
                activity = await self.__device.get_current_package()
            except:
                activity = "unknown"
            
            # Store screenshot
            storage_id = await self.__storage.save(
                data=screenshot_bytes,
                metadata={"type": "screenshot", "timestamp": time.time()},
            )
            
            screen = ScreenCapture(
                width=width,
                height=height,
                activity=activity,
                image=screenshot_bytes,
                timestamp=int(time.time() * 1000),  # milliseconds as int
                metadata={"storage_id": storage_id},
            )
            
            # Get XML hierarchy if enabled
            xml = None
            if self.__use_xml:
                # XML hierarchy would come from device port if supported
                pass
            
            return screen, xml, time.time() - start
            
        except StrategyError as exception:
            logger.exception(f"Grounding failed: {exception}")
            return None, None, time.time() - start
        except Exception as exception:
            logger.exception(f"Unexpected error in grounding: {exception}")
            raise StrategyError("Grounding failed unexpectedly") from exception

    async def __update_state(self, screen: ScreenCapture) -> Tuple[ScreenState, bool]:
        """Update agent state with new screen."""
        visual_hash = hashlib.sha256(screen.image).hexdigest()[:VISUAL_HASH_LENGTH]
        
        state = ScreenState(
            visual_hash=visual_hash,
            activity=screen.activity,
            timestamp=screen.timestamp,
            activity_hash=hashlib.md5(screen.activity.encode()).hexdigest()[:VISUAL_HASH_LENGTH],
            structural_hash="0" * VISUAL_HASH_LENGTH,  # Not computed in this simplified version
        )
        
        is_new = self.__state.update_screen(screen=state)
        
        return state, is_new

    async def __perform_analysis(
        self,
        state: ScreenState,
        screen: ScreenCapture,
        additional_context: Optional[str] = None,
    ) -> Tuple[PlanResult, Dict[str, Any], float]:
        """
        Retrieve knowledge and plan next step using LLM.
        
        Args:
            state: Current screen state
            screen: Screen capture
            additional_context: Optional additional context (e.g., user guidance)
        
        Returns:
            Tuple of (plan, knowledge, duration)
        """
        start = time.time()
        
        # Get memory entries
        entries = await self.__ledger.get_all()
        knowledge = await self.__memory.retrieve_knowledge(visual_hash=state.visual_hash)
        knowledge["memory_store"] = entries
        
        # Build context
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
            intent=self.__state.intent,
            current_activity=state.activity or ""
        )
        hint_str = PromptPreprocessor.build_context_prefix(hints)
        
        full_context = smart_context
        if hint_str:
            full_context = f"{hint_str}\n{smart_context}"
        
        # Add additional context if provided (e.g., from HITL)
        if additional_context:
            full_context = f"{full_context}\n\n{additional_context}"
        
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
            "metrics": self.__metrics.to_dict(),
        }

    def get_metrics(self) -> ExecutionMetrics:
        """Get execution metrics."""
        return self.__metrics
    
    def cancel(self) -> None:
        """Cancel the execution."""
        self.__cancelled = True
        self.__telemetry.warning("Intent strategy cancellation requested")
