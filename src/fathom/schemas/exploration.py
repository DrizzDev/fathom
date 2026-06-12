from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from fathom.constants import ActionType
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.screens import ScreenState


class BFSQueueEntry(BaseModel):
    """
    Entry in the BFS exploration queue.
    Represents a screen to explore with its parent and path information.
    """

    screen_hash: str
    parent_hash: str

    depth: int
    action_from_parent: Action
    path_from_root: List[Tuple[str, Action]]


class ScreenNode:
    """
    Node in the screen graph representing a unique screen state.
    Tracks visits, actions, and transitions for a discovered screen.
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
        Returns Unique identifier for this screen state.
        """

        return self.__fingerprint

    @property
    def activity(self) -> str:
        """
        Returns Activity name for this screen.
        """

        return self.__activity

    @property
    def visits(self) -> int:
        """
        Returns Number of times this screen has been visited.
        """

        return self.__visits

    @property
    def actions(self) -> Set[str]:
        """
        Returns Actions that can be performed from this screen.
        """

        return self.__actions

    @property
    def transitions(self) -> Dict[str, str]:
        """
        Returns Dictionary mapping action descriptions to destination hashes
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
    Graph of discovered screens and transitions.
    Maintains the complete exploration state including all discovered screens and the transitions between them.
    """

    def __init__(self) -> None:
        """
        Initialize empty exploration graph.
        Creates empty nodes and edges collections.
        """

        self.__nodes: Dict[str, ScreenNode] = {}
        self.__edges: List[Tuple[str, str, str]] = []

    @property
    def nodes(self) -> Dict[str, ScreenNode]:
        """
        Returns All discovered screen nodes.
        """

        return self.__nodes

    @property
    def edges(self) -> List[Tuple[str, str, str]]:
        """
        Returns All transitions between screens.
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
    Uses heuristics to select appropriate exploratory actions based on screen visit history and exploration state.
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


class ExploredScreen(BaseModel):
    """
    A single screen discovered during exploration.
    """

    hash: str = Field(description="Canonical visual hash identifying the screen")
    activity: str = Field(description="Android activity the screen belongs to")
    description: Optional[str] = Field(
        default=None, description="Human-readable summary of the screen"
    )
    visits: int = Field(default=0, ge=0, description="Number of times the screen was visited")


class ScreenTransition(BaseModel):
    """
    A recorded transition from one screen to another.
    """

    source: str = Field(description="Visual hash of the originating screen")
    destination: str = Field(description="Visual hash of the resulting screen")
    action: str = Field(description="Action type that drove the transition")
    target: Optional[str] = Field(default=None, description="Element the action addressed")
    count: int = Field(default=1, ge=1, description="Times this transition was observed")


class ExplorationStats(BaseModel):
    """
    Aggregate coverage counts for an exploration run.
    """

    screens: int = Field(default=0, ge=0, description="Unique screens discovered")
    transitions: int = Field(default=0, ge=0, description="Total recorded transitions")
    visits: int = Field(default=0, ge=0, description="Total screen visits across the run")
    activities: List[str] = Field(
        default_factory=list, description="Distinct activities discovered"
    )
    unexplored: int = Field(default=0, ge=0, description="Screens not yet fully explored")


class ExplorationSnapshot(BaseModel):
    """
    Serialisable view of the explored screen graph.
    """

    screens: List[ExploredScreen] = Field(
        default_factory=list, description="All discovered screens"
    )
    transitions: List[ScreenTransition] = Field(
        default_factory=list, description="All recorded transitions"
    )
    stats: ExplorationStats = Field(
        default_factory=ExplorationStats, description="Aggregate coverage statistics"
    )
