from __future__ import annotations

from typing import List, Optional

from fathom.constants.completion import CompletionEvidence
from fathom.schemas.completion import CompletionVerdict
from fathom.schemas.outcomes import ActionOutcome, OutcomeStatus
from fathom.schemas.tasks import ExecutionTask, ExecutionTaskState, TaskKind, TaskStatus


class CompletionService:
    """
    Fuses model task status with observed outcome to decide task completion.
    """

    def evaluate(
        self,
        *,
        task: ExecutionTask,
        status: Optional[TaskStatus],
        outcome: Optional[ActionOutcome],
    ) -> CompletionVerdict:
        """
        Return the completion verdict for the active task.
        """

        if task.over_budget:
            return self.__exhausted(task=task)

        if status == TaskStatus.BLOCKED:
            return self.__blocked(task=task)

        if status == TaskStatus.MET and task.kind == TaskKind.VALIDATION:
            return self.__succeeded(task=task)

        if status == TaskStatus.MET and self.__effective(outcome=outcome):
            return self.__succeeded(task=task)

        return self.__active(task=task, status=status, outcome=outcome)

    @staticmethod
    def __effective(*, outcome: Optional[ActionOutcome]) -> bool:
        """
        Return whether the observed outcome confirms an effective action.
        """

        return outcome is not None and outcome.status == OutcomeStatus.EFFECTIVE

    @staticmethod
    def __exhausted(*, task: ExecutionTask) -> CompletionVerdict:
        """
        Build the verdict for a task that exhausted its attempt budget.
        """

        return CompletionVerdict(
            complete=False,
            next_state=ExecutionTaskState.FAILED,
            missing=[CompletionEvidence.BUDGET_EXHAUSTED],
            reason=(
                f"Task {task.identifier!r} exhausted attempt budget "
                f"({task.attempts.count}/{task.attempts.limit})."
            ),
        )

    @staticmethod
    def __blocked(*, task: ExecutionTask) -> CompletionVerdict:
        """
        Build the verdict for a task reported BLOCKED by the model.
        """

        return CompletionVerdict(
            complete=False,
            next_state=ExecutionTaskState.BLOCKED,
            missing=[CompletionEvidence.OUTCOME_BLOCKED],
            reason=f"Task {task.identifier!r} reported blocked by the model.",
        )

    @staticmethod
    def __succeeded(*, task: ExecutionTask) -> CompletionVerdict:
        """
        Build the verdict for a task confirmed by model verdict and effective outcome.
        """

        return CompletionVerdict(
            complete=True,
            next_state=ExecutionTaskState.SUCCEEDED,
            reason=f"Task {task.identifier!r} criterion met with effective outcome.",
        )

    @staticmethod
    def __active(
        *,
        task: ExecutionTask,
        status: Optional[TaskStatus],
        outcome: Optional[ActionOutcome],
    ) -> CompletionVerdict:
        """
        Build the verdict for a task that should keep running.
        """

        missing: List[CompletionEvidence] = []

        if status != TaskStatus.MET:
            missing.append(CompletionEvidence.TASK_STATUS_MET)

        requires_effective_outcome = task.kind != TaskKind.VALIDATION
        if requires_effective_outcome and (
            outcome is None or outcome.status != OutcomeStatus.EFFECTIVE
        ):
            missing.append(CompletionEvidence.OUTCOME_EFFECTIVE)

        return CompletionVerdict(
            complete=False,
            missing=missing,
            next_state=ExecutionTaskState.ACTIVE,
            reason=f"Task {task.identifier!r} not yet complete.",
        )
