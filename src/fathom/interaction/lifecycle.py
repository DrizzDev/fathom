from __future__ import annotations

from typing import Dict, Final, FrozenSet, Optional

from fathom.constants.collaboration import (
    POLICY_SCOPES_REQUIRING_WORKSPACE,
    TERMINAL_IDEMPOTENCY_STATES,
    TERMINAL_JOB_STATES,
    TERMINAL_TASK_STATES,
    IdempotencyState,
    JobState,
    PolicyScope,
    TaskState,
)
from fathom.core.exceptions import InteractionError

TASK_TRANSITIONS: Final[Dict[TaskState, FrozenSet[TaskState]]] = {
    TaskState.QUEUED: frozenset({TaskState.RUNNING, TaskState.CANCELLED, TaskState.DELETED}),
    TaskState.RUNNING: frozenset(
        {
            TaskState.FAILED,
            TaskState.EXPIRED,
            TaskState.BLOCKED,
            TaskState.WAITING,
            TaskState.SUCCEEDED,
            TaskState.CANCELLED,
        }
    ),
    TaskState.BLOCKED: frozenset({TaskState.RUNNING, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.WAITING: frozenset({TaskState.RUNNING, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.SUCCEEDED: frozenset(),
    TaskState.FAILED: frozenset(),
    TaskState.CANCELLED: frozenset(),
    TaskState.EXPIRED: frozenset(),
    TaskState.DELETED: frozenset(),
}


class Lifecycle:
    """
    Pure interaction lifecycle policy for tasks and messages.
    """

    def validate_task_transition(self, *, current: TaskState, target: TaskState) -> None:
        """
        Validate that a task can move from the current state to the target state.
        """

        if current == target:
            return

        allowed = TASK_TRANSITIONS[current]
        if target not in allowed:
            raise InteractionError(
                f"Task cannot transition from '{current.value}' to '{target.value}'."
            )

    def validate_message_recording(self, *, task_state: Optional[TaskState]) -> None:
        """
        Validate that a message can be recorded for the optional task state.
        """

        if task_state in TERMINAL_TASK_STATES:
            raise InteractionError(
                f"Message cannot be recorded for terminal task state '{task_state.value}'."
            )

    def validate_task_lineage(
        self,
        *,
        task: str,
        parent: Optional[str],
        root: Optional[str],
    ) -> None:
        """
        Validate that task lineage references are internally consistent.
        """

        if parent is None and root not in {None, task}:
            raise InteractionError("Root task must reference itself when no parent exists.")

        if parent is not None and root is None:
            raise InteractionError("Child task must include a root task identifier.")

    def validate_policy_scope(self, *, scope: PolicyScope, workspace: Optional[str]) -> None:
        """
        Validate that policy scope matches the workspace boundary.
        """

        if scope in POLICY_SCOPES_REQUIRING_WORKSPACE and workspace is None:
            raise InteractionError("Workspace policy must include a workspace.")

        if scope == PolicyScope.TENANT and workspace is not None:
            raise InteractionError("Tenant policy must not include a workspace.")

    def validate_job_claim(self, *, state: JobState) -> None:
        """
        Validate that a job can be claimed for processing.
        """

        if state != JobState.PENDING:
            raise InteractionError(f"Job state '{state.value}' cannot be claimed.")

    def validate_job_finish(self, *, state: JobState, target: JobState) -> None:
        """
        Validate that a job can move to the requested terminal state.
        """

        if target not in TERMINAL_JOB_STATES:
            raise InteractionError(f"Job target state '{target.value}' is not terminal.")

        if state == target:
            return

        if state != JobState.CLAIMED:
            raise InteractionError(f"Job state '{state.value}' cannot finish as '{target.value}'.")

    def validate_request_finish(
        self,
        *,
        state: IdempotencyState,
        target: IdempotencyState,
    ) -> None:
        """
        Validate that an idempotency record can move to the requested terminal state.
        """

        if target not in TERMINAL_IDEMPOTENCY_STATES:
            raise InteractionError(f"Idempotency target state '{target.value}' is not terminal.")

        if state == target:
            return

        if state != IdempotencyState.STARTED:
            raise InteractionError(
                f"Idempotency state '{state.value}' cannot finish as '{target.value}'."
            )
