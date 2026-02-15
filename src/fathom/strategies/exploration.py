"""
Exploration-based execution strategy using hexagonal architecture.

Migrated from agent/strategies/exploration.py with ports instead of tools.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from collections import defaultdict
from logging import getLogger
from typing import Any, Dict, List, Optional, Set, Tuple

from fathom.constants import ActionType
from fathom.core.context.manager import ContextManager
from fathom.core.execution.engine import ExecutionEngine
from fathom.interfaces.device import DevicePort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.results import ExecutionResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step

logger = getLogger(name=__name__)


class ScreenNode:
    """
    Node in the screen graph representing a unique screen state.
    
    Migrated from agent/strategies/exploration.py
    """

    def __init__(self, fingerprint: str, activity: str) -> None:
        self.__activity = activity
        self.__fingerprint = fingerprint

        self.__visits = 0
        self.__last = 0.0
        self.__actions: Set[str] = set()
        self.__transitions: Dict[str, str] = {}

    @property
    def fingerprint(self) -> str:
        """Unique identifier for this screen state."""
        return self.__fingerprint

    @property
    def activity(self) -> str:
        """Activity name for this screen."""
        return self.__activity

    @property
    def visits(self) -> int:
        """Number of times this screen has been visited."""
        return self.__visits

    @property
    def actions(self) -> Set[str]:
        """Actions that can be performed from this screen."""
        return self.__actions

    @property
    def transitions(self) -> Dict[str, str]:
        """Transitions from this screen to other screens."""
        return self.__transitions

    def record_visit(self) -> None:
        """Records a visit to this screen."""
        self.__visits += 1
        self.__last = time.time()

    def record_action(self, description: str, destination: str) -> None:
        """Records an action and its result."""
        self.__actions.add(description)
        self.__transitions[description] = destination

    def should_explore(self, limit: int = 5) -> bool:
        """Checks if exploration limit reached."""
        return self.__visits < limit


class ExplorationGraph:
    """
    Graph of discovered screens and transitions.
    
    Migrated from agent/strategies/exploration.py
    """

    def __init__(self) -> None:
        self.__nodes: Dict[str, ScreenNode] = {}
        self.__edges: List[Tuple[str, str, str]] = []

    @property
    def nodes(self) -> Dict[str, ScreenNode]:
        """All discovered screen nodes."""
        return self.__nodes

    @property
    def edges(self) -> List[Tuple[str, str, str]]:
        """All transitions between screens."""
        return self.__edges

    def add_screen(self, state: ScreenState) -> ScreenNode:
        """Adds or updates a screen."""
        key = state.visual_hash
        if key not in self.__nodes:
            self.__nodes[key] = ScreenNode(fingerprint=key, activity=state.activity)

        node = self.__nodes[key]
        node.record_visit()

        return node

    def record_transition(self, origin: str, destination: str, action: str) -> None:
        """Records a transition."""
        if origin in self.__nodes:
            self.__nodes[origin].record_action(description=action, destination=destination)

        self.__edges.append((origin, action, destination))

    def get_stats(self) -> Dict[str, Any]:
        """Calculates coverage stats."""
        total = sum(len(node.actions) for node in self.__nodes.values())
        unexplored = sum(1 for node in self.__nodes.values() if node.should_explore())
        activities = len({node.activity for node in self.__nodes.values()})

        return {
            "total_actions": total,
            "unexplored": unexplored,
            "activities": activities,
            "unique_screens": len(self.__nodes),
            "total_transitions": len(self.__edges),
        }


class ActionGenerator:
    """
    Generates exploratory actions for unknown UI states.
    
    Migrated from agent/strategies/exploration.py
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.__rng = random.Random(seed)  # nosec
        self.__failures: Dict[str, int] = defaultdict(int)

    def generate(self, node: ScreenNode, width: int, height: int) -> Action:
        """Selects the best exploratory action."""
        if node.visits <= 2:
            return self.__tap(width=width, height=height)

        if node.visits <= 4:
            return self.__scroll()

        return self.__back()

    def __tap(self, width: int, height: int) -> Action:
        """Random tap."""
        _ = width
        _ = height

        x = self.__rng.randint(50, 950)
        y = self.__rng.randint(100, 900)

        return Action(
            confidence=0.3,
            rationale="Exploratory tap",
            action_type=ActionType.TAP,
            target=f"random tap at ({x}, {y})",
            bounds=Bounds(x=x, y=y, width=50, height=50),
        )

    def __scroll(self) -> Action:
        """Random scroll."""
        direction = self.__rng.choice(["up", "down"])

        return Action(
            confidence=0.4,
            action_type=ActionType.SCROLL,
            rationale=f"Scrolling {direction}",
            target=f"exploration scroll {direction}",
        )

    def __back(self) -> Action:
        """Back navigation."""
        return Action(
            confidence=0.5,
            target="back navigation",
            action_type=ActionType.BACK,
            rationale="Exploring parent path",
        )


class ExplorationStrategy:
    """
    Strategy for autonomous application mapping using hexagonal architecture.
    
    This is the REAL implementation migrated from agent/strategies/exploration.py
    but adapted to use ports (DevicePort, StoragePort, etc.) instead of tools.
    """
    
    def __init__(
        self,
        engine: ExecutionEngine,
        context: ContextManager,
        *,
        device: DevicePort,
        storage: StoragePort,
        telemetry: TelemetryPort,
        max_steps: int = 100,
        timeout: float = 3600.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize exploration strategy with ports.
        
        Args:
            engine: Execution engine for running steps
            context: Context manager for tracking execution state
            device: Device port for screen capture and device info
            storage: Storage port for saving screenshots
            telemetry: Telemetry port for logging
            max_steps: Maximum number of exploration steps
            timeout: Maximum exploration time in seconds
            seed: Random seed for reproducible exploration
        """
        self.__engine = engine
        self.__context = context
        
        # Ports
        self.__device = device
        self.__storage = storage
        self.__telemetry = telemetry
        
        # Configuration
        self.__max_steps = max_steps
        self.__timeout = timeout
        
        # Exploration state
        self.__steps = 0
        self.__start = time.time()
        self.__graph = ExplorationGraph()
        self.__generator = ActionGenerator(seed=seed)
        
        # Tracking
        self.__last: Optional[Action] = None
        self.__current: Optional[ScreenState] = None
    
    @property
    def graph(self) -> ExplorationGraph:
        """Exploration graph."""
        return self.__graph
    
    async def execute(self, max_steps: int) -> ExecutionResult:
        """
        Execute exploration workflow.
        
        This is the REAL execution loop from agent/strategies/exploration.py
        """
        start_time = time.time()
        self.__context.set_roadmap(intent="Explore application structure")
        
        success = False
        error = None
        
        try:
            # Main exploration loop
            while await self.__should_continue():
                if self.__steps >= max_steps:
                    break
                
                # Execute one exploration step
                step_success = await self.__execute_step()
                
                if not step_success:
                    # Continue even if individual step fails
                    self.__telemetry.warning("Exploration step failed, continuing")
            
            success = True
            
        except Exception as e:
            logger.exception(f"Exploration strategy execution failed: {e}")
            error = str(e)
            success = False
        
        duration = int((time.time() - start_time) * 1000)
        
        return ExecutionResult(
            success=success,
            duration=duration,
            error=error,
        )
    
    async def __execute_step(self) -> bool:
        """
        Execute one discovery step.
        
        Migrated from agent/strategies/exploration.py
        """
        try:
            # 1. Capture screen
            capture = await self.__capture_screen()
            if not capture:
                return False
            
            # 2. Compute screen state
            state = await self.__compute_state(capture=capture)
            fingerprint = state.visual_hash
            
            # 3. Update graph
            node = self.__graph.add_screen(state=state)
            
            # 4. Record transition from previous screen
            if self.__last and self.__current:
                self.__graph.record_transition(
                    destination=fingerprint,
                    origin=self.__current.visual_hash,
                    action=self.__last.to_description(),
                )
            
            self.__current = state
            
            # 5. Generate exploratory action
            size = await self.__device.get_screen_size()
            action = self.__generator.generate(node=node, width=size[0], height=size[1])
            
            # 6. Create step
            step = Step(
                action=action,
                screen_hash=fingerprint,
                step_number=self.__steps,
            )
            
            # 7. Execute through engine
            result = await self.__engine.execute_step(step=step)
            
            self.__steps += 1
            self.__last = action
            
            # 8. Stability wait
            await asyncio.sleep(delay=0.5)
            
            # 9. Capture post-action screen
            post_capture = await self.__capture_screen()
            if post_capture:
                post_state = await self.__compute_state(capture=post_capture)
                screen_changed = fingerprint != post_state.visual_hash
                
                # Update context
                await self.__context.commit(
                    observation=f"Screen: {state.activity} -> {post_state.activity if screen_changed else 'same'}",
                    thought=action.rationale,
                    action=action,
                )
            
            return result.success
            
        except Exception as e:
            logger.exception(f"Exploration step failed: {e}")
            return False
    
    async def __capture_screen(self) -> Optional[ScreenCapture]:
        """Capture screen through device port."""
        try:
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
            
            return ScreenCapture(
                width=width,
                height=height,
                activity=activity,
                image=screenshot_bytes,
                timestamp=int(time.time() * 1000),  # milliseconds as int
                metadata={"storage_id": storage_id},
            )
            
        except Exception as e:
            logger.exception(f"Screen capture failed: {e}")
            return None
    
    async def __compute_state(self, capture: ScreenCapture) -> ScreenState:
        """Compute screen state from capture."""
        # Compute visual hash
        visual_hash = hashlib.sha256(capture.image).hexdigest()[:16]
        
        return ScreenState(
            visual_hash=visual_hash,
            activity=capture.activity,
            timestamp=capture.timestamp,
            activity_hash=hashlib.md5(capture.activity.encode()).hexdigest()[:16],
            structural_hash="0" * 16,  # Not computed in this simplified version
        )
    
    async def __should_continue(self) -> bool:
        """
        Stop conditions.
        
        Migrated from agent/strategies/exploration.py
        """
        if self.__steps >= self.__max_steps:
            return False

        return (time.time() - self.__start) < self.__timeout
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Discovery metrics.
        
        Migrated from agent/strategies/exploration.py
        """
        return {
            "steps": self.__steps,
            "stats": self.__graph.get_stats(),
            "elapsed": time.time() - self.__start,
            "context": self.__context.get_full_context(),
        }
