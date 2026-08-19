from __future__ import annotations

from typing import Optional, assert_never

from fathom.constants import ActionType
from fathom.core.capability.gesture import GestureNormalizer
from fathom.schemas.actions import Action
from fathom.schemas.requirement import (
    CommandRequirement,
    PressRequirement,
    SwipeRequirement,
    TypeRequirement,
    WaitRequirement,
)


class CommandMatcher:
    """
    Matches a canonical command requirement against the executed action by operation and canonical payload only.

    Free-text target, surface, and condition descriptions are authored independently by the decomposer
    and the planner, so they are never compared for equality; the target correlation is the runtime
    binding, checked by the advancement policy, not a string match here.
    """

    def __init__(self, *, gestures: Optional[GestureNormalizer] = None) -> None:
        """
        Bind the gesture normalizer used to canonicalize legacy directional swipe aliases.
        """

        self.__gestures = gestures if gestures is not None else GestureNormalizer()

    def matches(self, *, requirement: CommandRequirement, action: Action) -> bool:
        """
        Return whether the executed action's operation and canonical payload satisfy the requirement.
        """

        if isinstance(requirement, PressRequirement):
            return action.action_type is requirement.operation

        if isinstance(requirement, TypeRequirement):
            return action.action_type is ActionType.TYPE and action.text == requirement.text

        if isinstance(requirement, SwipeRequirement):
            canonical = self.__swipe(action=action)
            return canonical is not None and canonical.direction == requirement.direction

        if isinstance(requirement, WaitRequirement):
            return (
                action.action_type is ActionType.WAIT and action.wait_duration == requirement.bound
            )

        assert_never(requirement)

    def __swipe(self, *, action: Action) -> Optional[SwipeRequirement]:
        """
        Canonicalize a directional swipe action to its finger direction; the free-text surface never gates.
        """

        try:
            return self.__gestures.canonical(operation=action.action_type)
        except ValueError:
            return None
