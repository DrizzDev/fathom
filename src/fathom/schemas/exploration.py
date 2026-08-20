from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel

from fathom.constants import ActionType
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.screens import ScreenState


class BFSQueueEntry(BaseModel):
    """
    A screen enqueued for breadth-first exploration, with the action and path that reached it.
    """

    screen_hash: str
    parent_hash: str

    depth: int
    action_from_parent: Action
    path_from_root: List[Tuple[str, Action]]


class ScreenNode:
    """
    A discovered screen in the exploration graph, tracking its visits, tried actions, and transitions.
    """

    def __init__(self, fingerprint: str, activity: str) -> None:
        """
        Initialize screen node with fingerprint and activity.
        """

        self.__activity = activity
        self.__fingerprint = fingerprint

        self.__visits = 0
        self.__last = 0.0
        self.__actions: Set[str] = set()
        self.__transitions: Dict[str, str] = {}

    @property
    def fingerprint(self) -> str:
        """
        Stable fingerprint identifying this screen state.
        """

        return self.__fingerprint

    @property
    def activity(self) -> str:
        """
        Android activity name for this screen.
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
        Action descriptions already recorded from this screen.
        """

        return self.__actions

    @property
    def transitions(self) -> Dict[str, str]:
        """
        Maps each recorded action description to its destination screen hash.
        """

        return self.__transitions

    def record_visit(self) -> None:
        """
        Updates visit count and timestamp.
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
    Graph of discovered screens and the transitions between them.
    """

    def __init__(self) -> None:
        """
        Initialize an empty exploration graph.
        """

        self.__nodes: Dict[str, ScreenNode] = {}
        self.__edges: List[Tuple[str, str, str]] = []

    @property
    def nodes(self) -> Dict[str, ScreenNode]:
        """
        All discovered screen nodes, keyed by visual hash.
        """

        return self.__nodes

    @property
    def edges(self) -> List[Tuple[str, str, str]]:
        """
        All recorded transitions as (origin, action, destination) triples.
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
    Generates exploratory actions, escalating tap to scroll to back as a screen's visit count grows.
    """

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        tap_margin_x: int = 50,
        tap_margin_y: int = 100,
        tap_max_x: int = 950,
        tap_max_y: int = 900,
        tap_target_size: int = 50,
    ) -> None:
        """
        Initialize action generator with optional random seed and tap configuration.

        Args:
            seed: Random seed for deterministic exploration
            tap_margin_x: Horizontal margin from screen edges
            tap_margin_y: Vertical margin from screen edges
            tap_max_x: Maximum X coordinate for taps
            tap_max_y: Maximum Y coordinate for taps
            tap_target_size: Size of tap target bounds
        """

        self.__rng = random.Random(seed)  # nosec
        self.__failures: Dict[str, int] = defaultdict(int)
        self.__tap_margin_x = tap_margin_x
        self.__tap_margin_y = tap_margin_y
        self.__tap_max_x = tap_max_x
        self.__tap_max_y = tap_max_y
        self.__tap_target_size = tap_target_size

    def generate(self, node: ScreenNode, width: int, height: int) -> Action:
        """
        Selects the best exploratory action.

        Args:
            node: Screen node with visit history
            width: Screen width in pixels
            height: Screen height in pixels

        Returns:
            Action to execute for exploration
        """

        if node.visits <= 2:
            return self.__tap(width=width, height=height)

        if node.visits <= 4:
            return self.__scroll()

        return self.__back()

    def __tap(self, width: int, height: int) -> Action:
        """
        Random tap within screen bounds.

        Args:
            width: Screen width in pixels
            height: Screen height in pixels

        Returns:
            Tap action with random coordinates
        """

        x = self.__rng.randint(
            self.__tap_margin_x, min(self.__tap_max_x, width - self.__tap_margin_x)
        )
        y = self.__rng.randint(
            self.__tap_margin_y, min(self.__tap_max_y, height - self.__tap_margin_y)
        )

        return Action(
            confidence=0.3,
            rationale="Exploratory tap",
            action_type=ActionType.TAP,
            target=f"random tap at ({x}, {y})",
            bounds=Bounds(x=x, y=y, width=self.__tap_target_size, height=self.__tap_target_size),
        )

    def __scroll(self) -> Action:
        """
        Random scroll.

        Returns:
            Scroll action with random direction
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

        Returns:
            Back navigation action
        """

        return Action(
            confidence=0.5,
            target="back navigation",
            action_type=ActionType.BACK,
            rationale="Exploring parent path",
        )
