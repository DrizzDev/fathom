from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants import ActionType
from fathom.constants.exploration import ExpectedOutcome
from fathom.schemas.actions import Action
from fathom.schemas.steps import StepResult


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


class ActionOutcome(BaseModel):
    """
    Outcome of a recently executed action, fed back into the scan as feedback.
    """

    model_config = ConfigDict(frozen=True)

    kind: ActionType = Field(description="Action type that was executed")
    target: str = Field(default="", description="Element the action addressed")
    success: bool = Field(description="Whether the device reported the action succeeded")
    screen_changed: bool = Field(description="Whether the screen changed after the action")
    expected: Optional[ExpectedOutcome] = Field(
        default=None, description="Screen effect the action predicted, for verification"
    )

    @classmethod
    def from_step_result(cls, *, result: StepResult) -> "ActionOutcome":
        """
        Projects an executed step result into a compact feedback outcome.
        """

        action = result.step.action
        return cls(
            kind=action.action_type,
            target=action.natural_language_target or action.target or "",
            success=result.success,
            screen_changed=result.screen_changed,
            expected=action.expected_outcome,
        )


class TriedAction(BaseModel):
    """
    An action already exercised on a screen and the screen it led to.
    """

    model_config = ConfigDict(frozen=True)

    action_type: str = Field(description="Action type that was tried")
    target: str = Field(default="", description="Element the action addressed")
    coord_bucket: Optional[str] = Field(
        default=None, description="Coordinate bucket of the tap, when grounded"
    )
    destination_hash: str = Field(description="Canonical hash of the resulting screen")
    destination_description: Optional[str] = Field(
        default=None, description="Description of the resulting screen, when known"
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
