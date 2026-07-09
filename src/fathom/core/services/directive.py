from __future__ import annotations

from typing import Optional

from fathom.constants import ActionType
from fathom.constants.capability import CompletionMode
from fathom.core.capability.catalog import CommandCatalog
from fathom.schemas.subgoal import SubGoalKind


class DirectivePolicy:
    """
    Projects a sub-goal's action directive into the completion-gate sub-goal kind it selects.
    """

    def __init__(self, *, catalog: CommandCatalog) -> None:
        """
        Bind the projection to the command catalog whose completion modes it reads.
        """

        self.__catalog = catalog

    def kind(self, *, directive: Optional[ActionType]) -> SubGoalKind:
        """
        Return the sub-goal kind for a directive: claim-verified directives validate, the rest act.
        """

        if directive is None:
            return SubGoalKind.ACTION

        completion = self.__catalog.profile(action_type=directive).completion
        if completion is CompletionMode.CLAIM_VERIFIED:
            return SubGoalKind.VALIDATION

        return SubGoalKind.ACTION
