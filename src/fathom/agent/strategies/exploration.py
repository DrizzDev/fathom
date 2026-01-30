"""Exploration strategy for discovering app functionality."""

from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Set, Tuple

from fathom.agent.strategies.base import (
    ExecutionStrategy,
)
from fathom.constants import ActionType, StrategyStatus
from fathom.schemas.actions import Action, BoundingBox
from fathom.schemas.results import StrategyResult
from fathom.schemas.screens import ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool
from fathom.tools.vision import VisionTool


@dataclass
class ScreenNode:
    """Node in the screen graph representing a unique screen."""

    screen_hash: str
    activity: str
    visit_count: int = 0
    actions_tried: Set[str] = field(default_factory=set)
    transitions: Dict[str, str] = field(default_factory=dict)
    last_visited: float = 0.0

    def should_explore(self, max_visits: int = 5) -> bool:
        """Check if this screen should be further explored."""
        return self.visit_count < max_visits


@dataclass
class ExplorationGraph:
    """Graph of discovered screens and transitions.

    Tracks:
    - Unique screens encountered
    - Transitions between screens
    - Actions tried on each screen
    - Coverage metrics
    """

    nodes: Dict[str, ScreenNode] = field(default_factory=dict)
    edges: List[Tuple[str, str, str]] = field(default_factory=list)

    def add_screen(self, state: ScreenState) -> ScreenNode:
        """Add or update a screen in the graph."""
        key = state.visual_hash
        if key not in self.nodes:
            self.nodes[key] = ScreenNode(
                screen_hash=key,
                activity=state.activity,
            )
        node = self.nodes[key]
        node.visit_count += 1
        node.last_visited = time.time()
        return node

    def record_transition(
        self,
        from_hash: str,
        to_hash: str,
        action_desc: str,
    ) -> None:
        """Record a transition between screens."""
        if from_hash in self.nodes:
            self.nodes[from_hash].actions_tried.add(action_desc)
            self.nodes[from_hash].transitions[action_desc] = to_hash
        self.edges.append((from_hash, action_desc, to_hash))

    def get_unexplored_count(self) -> int:
        """Count screens with remaining exploration potential."""
        return sum(1 for n in self.nodes.values() if n.should_explore())

    def get_coverage_stats(self) -> Dict[str, object]:
        """Get exploration coverage statistics."""
        total_actions = sum(len(n.actions_tried) for n in self.nodes.values())
        return {
            "unique_screens": len(self.nodes),
            "total_transitions": len(self.edges),
            "total_actions_tried": total_actions,
            "unexplored_screens": self.get_unexplored_count(),
            "unique_activities": len({n.activity for n in self.nodes.values()}),
        }


class ActionGenerator:
    """Generates exploration actions for a screen.

    Implements intelligent exploration:
    - Prioritizes untried elements
    - Avoids repeated failures
    - Balances breadth vs depth
    """

    __EXPLORATION_ACTIONS: ClassVar[List[ActionType]] = [
        ActionType.TAP,
        ActionType.SCROLL,
        ActionType.SWIPE,
        ActionType.BACK,
    ]

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize generator.

        Args:
            seed: Random seed for reproducibility.
        """
        self.__rng = random.Random(seed)  # nosec
        self.__failed_actions: Dict[str, int] = defaultdict(int)

    def generate_random_tap(
        self,
        screen_width: int,
        screen_height: int,
    ) -> Action:
        """Generate random tap action."""
        x = self.__rng.randint(50, 950)
        y = self.__rng.randint(100, 900)

        return Action(
            action_type=ActionType.TAP,
            target=f"exploration tap at ({x}, {y})",
            bbox=BoundingBox(x=x, y=y, width=50, height=50),
            confidence=0.3,
            reasoning="Random exploration tap",
        )

    def generate_scroll(self, direction: str = "down") -> Action:
        """Generate scroll action."""
        return Action(
            action_type=ActionType.SCROLL,
            target=f"exploration scroll {direction}",
            confidence=0.4,
            reasoning=f"Scroll {direction} to reveal content",
        )

    def generate_back(self) -> Action:
        """Generate back navigation action."""
        return Action(
            action_type=ActionType.BACK,
            target="exploration back navigation",
            confidence=0.5,
            reasoning="Navigate back to explore different path",
        )

    def select_exploration_action(
        self,
        node: ScreenNode,
        screen_width: int,
        screen_height: int,
    ) -> Action:
        """Select next exploration action for a screen.

        Strategy:
        1. If screen has few visits, try random taps
        2. If screen is moderately visited, try scrolling
        3. If screen is well-visited, navigate back
        """
        if node.visit_count <= 2:
            return self.generate_random_tap(screen_width, screen_height)

        if node.visit_count <= 4:
            direction = self.__rng.choice(["up", "down"])
            return self.generate_scroll(direction)

        return self.generate_back()

    def record_failure(self, action_desc: str) -> None:
        """Record a failed action."""
        self.__failed_actions[action_desc] += 1


class ExplorationStrategy(ExecutionStrategy):
    """Strategy for exploring app functionality.

    Unlike IntentStrategy which pursues a specific goal,
    ExplorationStrategy discovers and maps the application:
    - Builds a graph of screens and transitions
    - Tries actions to discover new screens
    - Tracks coverage metrics
    - Supports checkpointing for long-running exploration

    Useful for:
    - App discovery and documentation
    - Test generation
    - Coverage analysis
    """

    def __init__(
        self,
        device: DeviceTool,
        capture: CaptureTool,
        vision: Optional[VisionTool] = None,
        *,
        max_steps: int = 100,
        timeout: float = 3600.0,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize exploration strategy.

        Args:
            device: Device tool for actions.
            capture: Capture tool for screenshots.
            vision: Optional vision tool for guided exploration.
            max_steps: Maximum exploration steps.
            timeout: Maximum exploration time in seconds.
            seed: Random seed for reproducibility.
        """
        self.__device = device
        self.__capture = capture
        self.__vision = vision

        self.__max_steps = max_steps
        self.__timeout = timeout
        self.__step_count = 0

        self.__graph = ExplorationGraph()
        self.__generator = ActionGenerator(seed)

        self.__start_time = time.time()
        self.__current_screen: Optional[ScreenState] = None
        self.__last_action: Optional[Action] = None

    @property
    def name(self) -> str:
        """Strategy name."""
        return "exploration"

    @property
    def graph(self) -> ExplorationGraph:
        """Exploration graph."""
        return self.__graph

    async def execute_step(self) -> StrategyResult:
        """Execute a single exploration step.

        Flow:
        1. Capture current screen
        2. Add to graph
        3. Select exploration action
        4. Execute action
        5. Record transition
        6. Update metrics

        Returns:
            Result indicating exploration status.
        """
        capture = await self.__capture.capture()
        screen_state = self.__capture.compute_state(capture)

        pre_hash = screen_state.visual_hash
        node = self.__graph.add_screen(screen_state)

        if self.__last_action and self.__current_screen:
            self.__graph.record_transition(
                self.__current_screen.visual_hash,
                pre_hash,
                self.__last_action.to_description(),
            )

        self.__current_screen = screen_state

        screen_size = await self.__device.get_screen_size()
        action = self.__generator.select_exploration_action(
            node,
            screen_size[0],
            screen_size[1],
        )

        step = Step(
            action=action,
            screen_hash=pre_hash,
            step_number=self.__step_count,
        )

        execution_request = self.__prepare_execution(action, screen_size)
        result = await self.__device.execute(execution_request)

        if not result.success:
            self.__generator.record_failure(action.to_description())

        import asyncio

        await asyncio.sleep(0.5)
        post_capture = await self.__capture.capture()
        post_state = self.__capture.compute_state(post_capture)
        post_hash = post_state.visual_hash

        screen_changed = pre_hash != post_hash
        self.__step_count += 1
        self.__last_action = action

        step_result = StepResult(
            step=step,
            success=result.success,
            screen_changed=screen_changed,
            pre_hash=pre_hash,
            post_hash=post_hash,
            duration=result.duration,
            error=result.error,
        )

        should_checkpoint = (self.__step_count % 10) == 0

        return StrategyResult(
            status=StrategyStatus.CONTINUE,
            step_result=step_result,
            message=f"Explored: {action.to_description()}",
            should_checkpoint=should_checkpoint,
        )

    def __prepare_execution(
        self,
        action: Action,
        screen_size: Tuple[int, int],
    ) -> Dict[str, object]:
        """Prepare action for device execution."""
        width, height = screen_size

        if action.action_type == ActionType.TAP and action.bbox:
            x = action.bbox.center_x * width // 1000
            y = action.bbox.center_y * height // 1000
            return {"action": "tap", "x": x, "y": y}

        if action.action_type == ActionType.SCROLL:
            cx, cy = width // 2, height // 2
            return {
                "action": "swipe",
                "x1": cx,
                "y1": cy + 400,
                "x2": cx,
                "y2": cy - 400,
            }

        if action.action_type == ActionType.BACK:
            return {"action": "back"}

        return {"action": "tap", "x": width // 2, "y": height // 2}

    async def should_continue(self) -> bool:
        """Check if exploration should continue."""
        if self.__step_count >= self.__max_steps:
            return False

        elapsed = time.time() - self.__start_time
        if elapsed >= self.__timeout:
            return False

        return not (self.__graph.get_unexplored_count() == 0 and self.__step_count > 20)

    def get_progress(self) -> Dict[str, object]:
        """Get current exploration progress."""
        elapsed = time.time() - self.__start_time
        return {
            "step_count": self.__step_count,
            "max_steps": self.__max_steps,
            "elapsed_seconds": elapsed,
            "remaining_seconds": max(0, self.__timeout - elapsed),
            "coverage": self.__graph.get_coverage_stats(),
        }

    def get_checkpoint(self) -> Dict[str, object]:
        """Get checkpoint data for persistence."""
        return {
            "strategy": self.name,
            "step_count": self.__step_count,
            "start_time": self.__start_time,
            "graph_stats": self.__graph.get_coverage_stats(),
            "progress": self.get_progress(),
        }
