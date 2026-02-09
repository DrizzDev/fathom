from __future__ import annotations

import asyncio
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from fathom.agent.strategies.base import ExecutionStrategy
from fathom.constants import ActionType, StrategyStatus
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.results import StrategyResult
from fathom.schemas.screens import ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool
from fathom.tools.vision import VisionTool


class ScreenNode:
    """
    Node in the screen graph representing a unique screen state.
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
        """
        Unique identifier for this screen state.
        """

        return self.__fingerprint

    @property
    def activity(self) -> str:
        """
        Activity name for this screen.
        """

        return self.__activity

    @property
    def visits(self) -> int:
        """
        Number of times this screen has been visited.
        """

        return self.__visits

    @property
    def actions(self) -> Set[str]:
        """
        Actions that can be performed from this screen.
        """

        return self.__actions

    @property
    def transitions(self) -> Dict[str, str]:
        """
        Transitions from this screen to other screens.
        """

        return self.__transitions

    def record_visit(self) -> None:
        """
        Records a visit to this screen.
        """

        self.__visits += 1
        self.__last = time.time()

    def record_action(self, description: str, destination: str) -> None:
        """
        Records an action and its result.
        """

        self.__actions.add(description)
        self.__transitions[description] = destination

    def should_explore(self, limit: int = 5) -> bool:
        """
        Checks if exploration limit reached.
        """

        return self.__visits < limit


class ExplorationGraph:
    """
    Graph of discovered screens and transitions.
    """

    def __init__(self) -> None:
        self.__nodes: Dict[str, ScreenNode] = {}
        self.__edges: List[Tuple[str, str, str]] = []

    @property
    def nodes(self) -> Dict[str, ScreenNode]:
        """
        All discovered screen nodes.
        """

        return self.__nodes

    @property
    def edges(self) -> List[Tuple[str, str, str]]:
        """
        All transitions between screens.
        """

        return self.__edges

    def add_screen(self, state: ScreenState) -> ScreenNode:
        """
        Adds or updates a screen.
        """

        key = state.visual_hash
        if key not in self.__nodes:
            self.__nodes[key] = ScreenNode(fingerprint=key, activity=state.activity)

        node = self.__nodes[key]
        node.record_visit()

        return node

    def record_transition(self, origin: str, destination: str, action: str) -> None:
        """
        Records a transition.
        """

        if origin in self.__nodes:
            self.__nodes[origin].record_action(description=action, destination=destination)

        self.__edges.append((origin, action, destination))

    def get_stats(self) -> Dict[str, Any]:
        """
        Calculates coverage stats.
        """

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
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.__rng = random.Random(seed)  # nosec
        self.__failures: Dict[str, int] = defaultdict(int)

    def generate(self, node: ScreenNode, width: int, height: int) -> Action:
        """
        Selects the best exploratory action.
        """

        if node.visits <= 2:
            return self.__tap(width=width, height=height)

        if node.visits <= 4:
            return self.__scroll()

        return self.__back()

    def __tap(self, width: int, height: int) -> Action:
        """
        Random tap.
        """

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
        """
        Random scroll.
        """

        direction = self.__rng.choice(["up", "down"])

        return Action(
            confidence=0.4,
            action_type=ActionType.SCROLL,
            rationale=f"Scrolling {direction}",
            target=f"exploration scroll {direction}",
        )

    def __back(self) -> Action:
        """
        Back navigation.
        """

        return Action(
            confidence=0.5,
            target="back navigation",
            action_type=ActionType.BACK,
            rationale="Exploring parent path",
        )


class ExplorationStrategy(ExecutionStrategy):
    """
    Strategy for autonomous application mapping.
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
        self.__device = device
        self.__capture = capture

        self.__vision = vision
        self.__max_steps = max_steps

        self.__steps = 0
        self.__timeout = timeout

        self.__graph = ExplorationGraph()
        self.__generator = ActionGenerator(seed=seed)

        self.__start = time.time()
        self.__last: Optional[Action] = None
        self.__current: Optional[ScreenState] = None

    @property
    def name(self) -> str:
        """
        Strategy name.
        """

        return "exploration"

    @property
    def graph(self) -> ExplorationGraph:
        """
        Exploration graph.
        """

        return self.__graph

    async def execute_step(self) -> StrategyResult:
        """
        Executes one discovery step.
        """

        capture = await self.__capture.capture()
        state = self.__capture.compute_state(capture=capture)

        fingerprint = state.visual_hash

        node = self.__graph.add_screen(state=state)

        if self.__last and self.__current:
            self.__graph.record_transition(
                destination=fingerprint,
                origin=self.__current.visual_hash,
                action=self.__last.to_description(),
            )

        self.__current = state
        size = await self.__device.get_screen_size()
        action = self.__generator.generate(node=node, width=size[0], height=size[1])

        step = Step(
            action=action,
            screen_hash=fingerprint,
            step_number=self.__steps,
        )

        # Implementation of simple execution for exploration
        result = await self.__device.execute(request=action.model_dump())

        self.__steps += 1
        self.__last = action

        # Stability wait
        await asyncio.sleep(delay=0.5)

        post_capture = await self.__capture.capture()
        post_state = self.__capture.compute_state(capture=post_capture)

        step_result = StepResult(
            step=step,
            error=result.error,
            pre_hash=fingerprint,
            success=result.success,
            duration=result.duration,
            post_hash=post_state.visual_hash,
            screen_changed=fingerprint != post_state.visual_hash,
        )

        return StrategyResult(
            step_result=step_result,
            status=StrategyStatus.CONTINUE,
            message=f"Explored: {action.to_description()}",
        )

    async def should_continue(self) -> bool:
        """
        Stop conditions.
        """

        if self.__steps >= self.__max_steps:
            return False

        return (time.time() - self.__start) < self.__timeout

    def get_progress(self) -> Dict[str, Any]:
        """
        Discovery metrics.
        """

        return {
            "steps": self.__steps,
            "stats": self.__graph.get_stats(),
            "elapsed": time.time() - self.__start,
        }
