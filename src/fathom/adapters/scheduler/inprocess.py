from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from logging import getLogger
from typing import Dict, Optional

from fathom.constants.collaboration import JobCode, JobState
from fathom.core.exceptions import InteractionError
from fathom.interfaces.interaction import InteractionPort
from fathom.interfaces.scheduler import JobHandlerPort, JobSchedulerPort
from fathom.schemas.configuration import InProcessJobSchedulerConfiguration
from fathom.schemas.interaction import (
    ClaimJob,
    FinishJob,
    Job,
    Outcome,
    RecoverJob,
    RescheduleJob,
)

logger = getLogger(__name__)


class InProcessJobScheduler(JobSchedulerPort):
    """
    Asyncio-loop scheduler for hosts that want local durable-job dispatch.
    """

    def __init__(
        self,
        *,
        interaction: InteractionPort,
        configuration: InProcessJobSchedulerConfiguration,
    ) -> None:
        """
        Initialize the scheduler with a durable interaction store.
        """

        self.__logger = getLogger(".".join((__name__, self.__class__.__name__)))

        self.__interaction = interaction
        self.__configuration = configuration

        self.__stopping = asyncio.Event()
        self.__last_recovery: Optional[datetime] = None
        self.__task: Optional[asyncio.Task[None]] = None

    async def start(self, *, handler: JobHandlerPort) -> None:
        """
        Start the scheduler loop once.
        """

        if self.__task is not None and not self.__task.done():
            raise InteractionError("Job scheduler is already running.")

        self.__logger.info(
            "Starting in-process job scheduler",
            extra=self.__log_extra(event="scheduler_start"),
        )

        self.__stopping.clear()
        self.__task = asyncio.create_task(self.__loop(handler=handler))

    async def stop(self) -> None:
        """
        Stop the scheduler loop and wait for it to exit.
        """

        self.__stopping.set()

        if self.__task is not None:
            await self.__task
            self.__task = None

        self.__logger.info(
            "Stopped in-process job scheduler",
            extra=self.__log_extra(event="scheduler_stop"),
        )

    async def __loop(self, *, handler: JobHandlerPort) -> None:
        """
        Recover stale work, claim available jobs, and dispatch them.
        """

        while not self.__stopping.is_set():
            try:
                if self.__should_recover():
                    self.__logger.info(
                        "Recovering stale scheduler jobs",
                        extra=self.__log_extra(event="scheduler_recover_stale"),
                    )
                    await self.__recover_stale()
                    self.__last_recovery = self.__now()

                claimed = await self.__dispatch_batch(handler=handler)
            except Exception:
                self.__logger.exception(
                    "Job scheduler iteration failed; backing off",
                    extra=self.__log_extra(event="scheduler_iteration_failed"),
                )
                await self.__sleep(duration=self.__configuration.failure_backoff)
                continue

            if claimed == 0:
                await self.__sleep_or_stop()

    def __should_recover(self) -> bool:
        """
        Decide whether the recovery cadence has elapsed since the last sweep.
        """

        if self.__last_recovery is None:
            return True

        elapsed = (self.__now() - self.__last_recovery).total_seconds() * 1000

        return elapsed >= self.__configuration.recovery_interval

    async def __dispatch_batch(self, *, handler: JobHandlerPort) -> int:
        """
        Claim and dispatch up to one configured batch of jobs.
        """

        claimed = 0

        for _ in range(self.__configuration.batch_size):
            job = await self.__claim()

            if job is None:
                self.__logger.debug(
                    "No job available for scheduler dispatch",
                    extra=self.__log_extra(
                        event="scheduler_claim_empty",
                        claimed=claimed,
                    ),
                )
                return claimed

            claimed += 1
            await self.__dispatch(handler=handler, job=job)

        return claimed

    async def __claim(self) -> Job | None:
        """
        Claim the next available job across configured kinds.
        """

        claimed = self.__now()
        if not self.__configuration.kinds:
            return await self.__interaction.claim_job(
                request=ClaimJob(
                    claimed=claimed,
                    owner=self.__configuration.owner,
                    tenant=self.__configuration.tenant,
                )
            )

        for kind in self.__configuration.kinds:
            job = await self.__interaction.claim_job(
                request=ClaimJob(
                    kind=kind,
                    claimed=claimed,
                    owner=self.__configuration.owner,
                    tenant=self.__configuration.tenant,
                )
            )
            if job is not None:
                return job

        return None

    async def __dispatch(self, *, handler: JobHandlerPort, job: Job) -> None:
        """
        Execute one claimed job and persist its terminal result.
        """

        if job.attempts > self.__configuration.max_attempts:
            self.__logger.warning(
                "Scheduler claimed job beyond attempt budget",
                extra=self.__job_log_extra(
                    event="scheduler_job_attempt_budget_exceeded",
                    job=job,
                ),
            )
            await self.__finish_max_attempts(job=job)
            return

        try:
            result = await handler.handle(job=job)
        except Exception as exception:
            self.__logger.exception(
                "Scheduler handler failed for claimed job",
                extra=self.__job_log_extra(
                    event="scheduler_job_handler_failed",
                    job=job,
                    attempt=job.attempts,
                ),
            )
            if job.attempts >= self.__configuration.max_attempts:
                await self.__finish_max_attempts(job=job)
                return

            await self.__reschedule(job=job, detail=str(exception))
            return

        self.__logger.info(
            "Scheduler handler completed claimed job",
            extra=self.__job_log_extra(
                event="scheduler_job_completed",
                job=job,
                result_state=result.state.value,
            ),
        )
        await self.__interaction.finish_job(
            request=FinishJob(
                state=result.state,
                job=job.identity.id,
                finished=self.__now(),
                outcome=result.outcome,
                tenant=job.identity.tenant,
                owner=self.__configuration.owner,
            )
        )

    async def __finish_max_attempts(self, *, job: Job) -> None:
        """
        Mark a job failed once its configured attempt budget is exhausted.
        """

        self.__logger.warning(
            "Finishing scheduler job as failed after maximum attempts",
            extra=self.__job_log_extra(
                event="scheduler_job_max_attempts",
                job=job,
                max_attempts=self.__configuration.max_attempts,
            ),
        )
        await self.__interaction.finish_job(
            request=FinishJob(
                job=job.identity.id,
                tenant=job.identity.tenant,
                owner=self.__configuration.owner,
                state=JobState.FAILED,
                outcome=Outcome(
                    code=JobCode.PERMANENT_ERROR,
                    detail="Job exceeded maximum scheduler attempts.",
                ),
                finished=self.__now(),
            )
        )

    async def __recover_stale(self) -> None:
        """
        Release stale claimed jobs so they can be retried after backoff.
        """

        before = self.__now() - timedelta(milliseconds=self.__configuration.lease)
        available = self.__retry_available()

        if not self.__configuration.kinds:
            await self.__interaction.recover_jobs(
                request=RecoverJob(
                    before=before,
                    available_at=available,
                    tenant=self.__configuration.tenant,
                    limit=self.__configuration.batch_size,
                )
            )
            return

        for kind in self.__configuration.kinds:
            await self.__interaction.recover_jobs(
                request=RecoverJob(
                    kind=kind,
                    before=before,
                    available_at=available,
                    tenant=self.__configuration.tenant,
                    limit=self.__configuration.batch_size,
                )
            )

    async def __reschedule(self, *, job: Job, detail: str) -> None:
        """
        Release a failed handler attempt for retry after configured backoff.
        """

        self.__logger.info(
            "Rescheduling scheduler job after handler failure",
            extra=self.__job_log_extra(
                event="scheduler_job_reschedule",
                job=job,
                attempt=job.attempts,
            ),
        )
        await self.__interaction.reschedule_job(
            request=RescheduleJob(
                detail=detail,
                job=job.identity.id,
                attempts=job.attempts,
                rescheduled=self.__now(),
                tenant=job.identity.tenant,
                owner=self.__configuration.owner,
                available_at=self.__retry_available(),
            )
        )

    async def __sleep_or_stop(self) -> None:
        """
        Sleep until the next poll interval or until stop is requested.
        """

        await self.__sleep(duration=self.__configuration.poll_interval)

    async def __sleep(self, *, duration: int) -> None:
        """
        Sleep up to the configured duration unless the loop is asked to stop.
        """

        if duration <= 0:
            return

        try:
            await asyncio.wait_for(
                self.__stopping.wait(),
                timeout=duration / 1000,
            )
        except asyncio.TimeoutError:
            return

    def __now(self) -> datetime:
        """
        Return the scheduler clock value.
        """

        return datetime.now(timezone.utc)

    def __log_extra(self, *, event: str, **values: object) -> Dict[str, object]:
        """
        Build structured log context for scheduler lifecycle events.
        """

        return {
            "event": event,
            "owner": self.__configuration.owner,
            "tenant": self.__configuration.tenant,
            "component": "fathom_in_process_scheduler",
            **values,
        }

    def __job_log_extra(self, *, event: str, job: Job, **values: object) -> Dict[str, object]:
        """
        Build structured log context for one scheduler job decision.
        """

        return self.__log_extra(
            event=event,
            job=job.identity.id,
            kind=job.kind.value,
            state=job.state.value,
            attempts=job.attempts,
            **values,
        )

    def __retry_available(self) -> datetime:
        """
        Return the next retry availability timestamp.
        """

        return self.__now() + timedelta(milliseconds=self.__configuration.retry_backoff)
