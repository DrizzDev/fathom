"""
Domain entities for exploration workflows.

These entities represent the core domain concepts for application exploration
and screen graph discovery.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from fathom.constants import ActionType
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.screens import ScreenState


class ScreenNode:
    """
    Node in the screen graph representing a unique screen state.

    Tracks visits, actions, and transitions for a discovered screen.
    """

    def __init__(self, fingerprint: str, activity: str) -> None:
        """Initialize screen node with fingerprint and activity."""
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

    Maintains the complete exploration state including all discovered screens
    and the transitions between them.
    """

    def __init__(self) -> None:
        """Initialize empty exploration graph."""
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

    Uses heuristics to select appropriate exploratory actions based on
    screen visit history and exploration state.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize action generator with optional random seed."""
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
