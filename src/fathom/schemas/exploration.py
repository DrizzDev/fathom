from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from fathom.schemas.actions import Action


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
