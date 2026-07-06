from __future__ import annotations

from logging import getLogger

from fathom.authoring.agent import AuthoringAgent
from fathom.constants.authoring import AuthoringKind, AuthoringMode, AuthoringStatus
from fathom.core.exceptions import InvariantViolation
from fathom.interfaces.authoring import AuthoringPort
from fathom.schemas.authoring import AuthoringConfiguration, AuthoringResponse, AuthoringTask

logger = getLogger(__name__)


class AuthoringRunner:
    """
    Orchestrates configurable script-authoring tasks without owning execution or persistence.
    """

    def __init__(self, *, agent: AuthoringAgent, configuration: AuthoringConfiguration) -> None:
        """
        Bind the single authoring agent and authoring configuration.
        """

        self.__agent = agent
        self.__configuration = configuration

    async def author(self, *, task: AuthoringTask, author: AuthoringPort) -> AuthoringResponse:
        """
        Author a script task when enabled; otherwise return an explicit skip.
        """

        if not self.enabled(kind=task.kind):
            logger.info(
                "authoring skipped by configuration",
                extra={
                    "event": "authoring.skipped",
                    "execution.id": task.execution_id,
                    "authoring.task.kind": task.kind.value,
                    "authoring.reason": "authoring task disabled",
                },
            )
            return AuthoringResponse(
                status=AuthoringStatus.SKIPPED,
                reason=f"{task.kind.value} authoring is disabled by configuration.",
            )

        logger.info(
            "authoring started",
            extra={
                "event": "authoring.started",
                "execution.id": task.execution_id,
                "authoring.step": task.step_number,
                "authoring.task.kind": task.kind.value,
            },
        )
        response = await self.__agent.author(task=task, authoring=author)
        logger.info(
            "authoring completed",
            extra={
                "event": "authoring.completed",
                "execution.id": task.execution_id,
                "authoring.task.kind": task.kind.value,
                "authoring.status": response.status.value,
                "authoring.has_script": response.has_script,
            },
        )
        return response

    def enabled(self, *, kind: AuthoringKind) -> bool:
        """
        Return whether the task kind is enabled by the current configuration.
        """

        if kind is AuthoringKind.RUN:
            return self.__configuration.run.enabled

        if kind is AuthoringKind.STEP:
            return self.__configuration.step.mode is not AuthoringMode.DISABLED

        if kind is AuthoringKind.REPAIR:
            return True

        raise InvariantViolation(f"Unsupported authoring task kind '{kind.value}'.")
