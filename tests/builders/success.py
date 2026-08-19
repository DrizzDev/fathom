from __future__ import annotations

from typing import Optional

from fathom.constants import ActionType
from fathom.constants.success import CaptureNameProvenance
from fathom.schemas.capture import CaptureIdentity
from fathom.schemas.requirement import CommandRequirement, PressRequirement
from fathom.schemas.subgoal import GoalState, Progress, SubGoal, SubGoalStatus
from fathom.schemas.success import (
    CaptureSuccess,
    CommandSuccess,
    ObservationRequirement,
    ObservedSuccess,
    SourceLocation,
    SourceSpan,
)


class SuccessFixtures:
    """
    Canonical builders for the three success variants and their supporting value objects.
    """

    @staticmethod
    def observation(assertion: str = "target screen displayed") -> ObservationRequirement:
        """
        Build an observation requirement.
        """

        return ObservationRequirement(assertion=assertion)

    @classmethod
    def observed(cls, *, assertion: str = "target screen displayed") -> ObservedSuccess:
        """
        Build an observed success.
        """

        return ObservedSuccess(observation=cls.observation(assertion))

    @staticmethod
    def source(quote: str, intent: str) -> SourceSpan:
        """
        Build a source span whose location exactly spans the quote in the intent.
        """

        start = intent.find(quote)
        return SourceSpan(quote=quote, location=SourceLocation(start=start, end=start + len(quote)))

    @classmethod
    def command(
        cls,
        *,
        requirement: Optional[CommandRequirement] = None,
        postcondition: Optional[ObservationRequirement] = None,
        quote: str = "tap",
        intent: str = "tap the target",
    ) -> CommandSuccess:
        """
        Build a command success bound to a source span in the intent.
        """

        return CommandSuccess(
            requirement=requirement
            if requirement is not None
            else PressRequirement(operation=ActionType.TAP, target="Login"),
            source=cls.source(quote=quote, intent=intent),
            postcondition=postcondition,
        )

    @staticmethod
    def capture(
        *,
        name: str = "price",
        subject: str = "item price",
        provenance: CaptureNameProvenance = CaptureNameProvenance.USER,
    ) -> CaptureSuccess:
        """
        Build a capture success with an exact capture identity and descriptive subject.
        """

        return CaptureSuccess(
            target=CaptureIdentity(name=name, provenance=provenance), subject=subject
        )


class ProgressFixtures:
    """
    Canonical builders for mutable sub-goal progress and goal state.
    """

    @staticmethod
    def progress(
        *,
        status: SubGoalStatus = SubGoalStatus.IN_PROGRESS,
        attempts: int = 0,
        recovery: int = 0,
        limit: int = 8,
    ) -> Progress:
        """
        Build a mutable Progress with explicit counters.
        """

        return Progress(status=status, attempts=attempts, recovery=recovery, limit=limit)

    @classmethod
    def goal_state(
        cls,
        *,
        goal: SubGoal,
        status: SubGoalStatus = SubGoalStatus.IN_PROGRESS,
        attempts: int = 0,
        recovery: int = 0,
        limit: int = 8,
    ) -> GoalState:
        """
        Build a GoalState pairing an immutable goal with explicit progress.
        """

        return GoalState(
            goal=goal,
            progress=cls.progress(status=status, attempts=attempts, recovery=recovery, limit=limit),
        )
