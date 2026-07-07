from __future__ import annotations

import asyncio
from logging import getLogger
from typing import Set

from fathom.authoring.application.runner import AuthoringRunner
from fathom.authoring.evidence import AuthoringEvidenceBuilder
from fathom.constants.authoring import AuthoringKind, AuthoringStatus
from fathom.interfaces.authoring import AuthoringDraftStore, AuthoringPort, AuthoringScheduler
from fathom.interfaces.evidence import EvidenceSource
from fathom.schemas.authoring import AuthoringTask
from fathom.schemas.authoring.draft import AuthoringDraft
from fathom.schemas.flow import Evidence, RunObjective

logger = getLogger(__name__)


class StepAuthoringScheduler(AuthoringScheduler):
    """
    Schedules optional per-step authoring drafts without blocking execution.
    """

    def __init__(
        self,
        *,
        author: AuthoringPort,
        source: EvidenceSource,
        runner: AuthoringRunner,
        drafts: AuthoringDraftStore,
        builder: AuthoringEvidenceBuilder,
    ) -> None:
        """
        Bind evidence, runner, authoring source, and draft persistence.
        """

        self.__source = source
        self.__author = author
        self.__runner = runner
        self.__drafts = drafts
        self.__builder = builder
        self.__tasks: Set[asyncio.Task[None]] = set()

    def schedule_step(self, *, execution_id: str, objective: RunObjective, step_index: int) -> None:
        """
        Schedule optional authoring for one persisted execution step.
        """

        if not self.__runner.enabled(kind=AuthoringKind.STEP):
            logger.info(
                "step authoring skipped by configuration",
                extra={
                    "event": "authoring.step.schedule_skipped",
                    "execution.id": execution_id,
                    "authoring.step": step_index,
                },
            )
            return

        try:
            task = asyncio.create_task(
                self.__author_step(
                    objective=objective,
                    step_index=step_index,
                    execution_id=execution_id,
                )
            )
        except RuntimeError as exception:
            logger.warning(
                "step authoring could not be scheduled",
                extra={
                    "event": "authoring.step.schedule_failed",
                    "execution.id": execution_id,
                    "authoring.step": step_index,
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                },
            )
            return

        self.__tasks.add(task)
        task.add_done_callback(self.__tasks.discard)

    async def drain(self) -> None:
        """
        Await scheduled step authoring work.
        """

        if not self.__tasks:
            return

        results = await asyncio.gather(*tuple(self.__tasks), return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.warning(
                    "step authoring task failed during drain",
                    extra={
                        "event": "authoring.step.drain_failed",
                        "exception.type": type(result).__name__,
                        "exception.message": str(result),
                    },
                )

    async def __author_step(
        self,
        *,
        step_index: int,
        execution_id: str,
        objective: RunObjective,
    ) -> None:
        """
        Build, author, and persist one step draft.
        """

        try:
            evidence = await self.__source.read(execution_id=execution_id, objective=objective)
            if not self.__authorable_step(evidence=evidence, step_index=step_index):
                logger.info(
                    "step authoring skipped for discarded evidence step",
                    extra={
                        "event": "authoring.step.discarded",
                        "execution.id": execution_id,
                        "authoring.step": step_index,
                    },
                )
                return

            response = await self.__runner.author(
                author=self.__author,
                task=AuthoringTask(
                    step_number=step_index,
                    kind=AuthoringKind.STEP,
                    intent=objective.intent,
                    execution_id=execution_id,
                    evidence=self.__builder.build_step(evidence=evidence, step_index=step_index),
                ),
            )
            await self.__drafts.save(
                draft=AuthoringDraft(
                    step_index=step_index,
                    status=response.status,
                    reason=response.reason,
                    kind=AuthoringKind.STEP,
                    execution_id=execution_id,
                    artifact=response.artifact,
                )
            )
        except Exception as exception:  # noqa: BLE001 - background authoring must not break execution.
            logger.exception(
                "step authoring failed",
                extra={
                    "event": "authoring.step.failed",
                    "execution.id": execution_id,
                    "authoring.step": step_index,
                    "authoring.status": AuthoringStatus.FAILED.value,
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                },
            )

    @staticmethod
    def __authorable_step(*, evidence: Evidence, step_index: int) -> bool:
        """
        Return whether the distilled evidence still contains the requested step.
        """

        if step_index in evidence.discarded:
            return False

        return any(step.index == step_index for step in evidence.steps)
