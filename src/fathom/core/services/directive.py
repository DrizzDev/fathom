from __future__ import annotations

from typing import Optional

from fathom.constants import ActionType
from fathom.constants.capability import CompletionMode
from fathom.constants.subgoal import TaskProof
from fathom.core.capability.catalog import CommandCatalog
from fathom.schemas.subgoal import SubGoal, SubGoalKind
from fathom.schemas.tasks import Task


class DirectivePolicy:
    """
    Projects a sub-goal's action directive into its completion-gate kind and typed proof requirement.
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

        if self.__completion(directive=directive) is CompletionMode.CLAIM_VERIFIED:
            return SubGoalKind.VALIDATION

        return SubGoalKind.ACTION

    def project(self, *, sub_goal: SubGoal) -> Task:
        """
        Project a sub-goal into the consolidated task, resolving its proof requirement once.
        """

        completion = self.__proof(sub_goal=sub_goal)

        return Task(
            index=sub_goal.index,
            kind=sub_goal.kind,
            completion=completion,
            criterion=sub_goal.criterion,
            directive=sub_goal.directive,
            description=sub_goal.description,
        )

    def __proof(self, *, sub_goal: SubGoal) -> CompletionMode:
        """
        Resolve the proof requirement: kind and command family first, then the decomposer's durable bit.
        """

        if sub_goal.kind is SubGoalKind.VALIDATION:
            return CompletionMode.CLAIM_VERIFIED

        base = self.__completion(directive=sub_goal.directive)
        if base in {CompletionMode.CAPTURE_VERIFIED, CompletionMode.TERMINAL}:
            return base

        if sub_goal.proof is TaskProof.DURABLE:
            return CompletionMode.OUTCOME_VERIFIED

        return base

    def __completion(self, *, directive: Optional[ActionType]) -> CompletionMode:
        """
        Return the catalog's completion mode, defaulting to the strict screen-verified path.
        """

        if directive is None:
            return CompletionMode.SCREEN_VERIFIED

        return self.__catalog.profile(action_type=directive).completion
