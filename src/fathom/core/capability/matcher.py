from __future__ import annotations

from typing import Optional

from fathom.constants import GESTURE_SCROLL_DIRECTION, ActionType
from fathom.core.capability.gesture import GestureNormalizer
from fathom.schemas.actions import Action
from fathom.schemas.requirement import (
    CommandRequirement,
    NavigationRequirement,
    PressRequirement,
    ScrollRequirement,
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

        if isinstance(requirement, ScrollRequirement):
            return (
                action.action_type is ActionType.SCROLL
                and GESTURE_SCROLL_DIRECTION.get(action.action_type) == requirement.direction
            )

        if isinstance(requirement, SwipeRequirement):
            return self.__swipe(action=action) == requirement

        if isinstance(requirement, WaitRequirement):
            return (
                action.action_type is ActionType.WAIT
                and action.wait_duration == requirement.bound
            )

        return self.__navigation(action=action, requirement=requirement)

    def __swipe(self, *, action: Action) -> Optional[SwipeRequirement]:
        """
        Canonicalize a legacy directional swipe action into a swipe requirement for comparison.
        """

        try:
            canonical = self.__gestures.canonical(
                operation=action.action_type, target=action.surface
            )
        except ValueError:
            return None

        return canonical

    @staticmethod
    def __navigation(*, action: Action, requirement: NavigationRequirement) -> bool:
        """
        Return whether a navigation action exactly matches the requested operation.
        """

        return action.action_type is requirement.operation
