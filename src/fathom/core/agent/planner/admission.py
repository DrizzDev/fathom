from __future__ import annotations

from typing import Optional

from fathom.core.capability.matcher import CommandMatcher
from fathom.schemas.actions import Action
from fathom.schemas.requirement import CommandRequirement
from fathom.schemas.subgoal import GoalState
from fathom.schemas.success import CommandSuccess


class RequirementAdmitter:
    """
    Admits the active command requirement onto a step only when the executed action matches it.
    """

    def __init__(self, *, matcher: Optional[CommandMatcher] = None) -> None:
        """
        Bind the command matcher that proves an executed action against the active requirement.
        """

        self.__matcher = matcher if matcher is not None else CommandMatcher()

    def admit(
        self, *, current_sub_goal: Optional[GoalState], action: Action
    ) -> Optional[CommandRequirement]:
        """
        Derive the requirement solely from the active CommandSuccess, admitted only after the matcher proves it.

        A model-supplied requirement is never trusted; preparatory or recovery actions carry no
        requirement and so cannot complete a command goal.
        """

        if current_sub_goal is None:
            return None

        success = current_sub_goal.success
        if not isinstance(success, CommandSuccess):
            return None

        if self.__matcher.matches(requirement=success.requirement, action=action):
            return success.requirement

        return None
