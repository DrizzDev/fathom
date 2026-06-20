from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fathom.adapters.interaction.pypika.sqlite import SQLiteInteraction
from fathom.adapters.scheduler.inprocess import InProcessJobScheduler
from fathom.constants.collaboration import (
    ActorKind,
    JobCode,
    JobKind,
    JobState,
    MembershipRole,
    TaskKind,
    TaskState,
)
from fathom.interfaces.scheduler import JobHandlerPort
from fathom.schemas.configuration import InProcessJobSchedulerConfiguration
from fathom.schemas.interaction import (
    Assignment,
    CreateActor,
    CreateThread,
    Identity,
    Job,
    JobQuery,
    JoinThread,
    Lineage,
    OpenTask,
    Outcome,
    Plan,
    ScheduleJob,
)
from fathom.schemas.scheduler import JobHandlerResult


class _RecordingHandler(JobHandlerPort):
    """
    Capture calls and return a configurable terminal result.
    """

    def __init__(
        self,
        *,
        result: JobHandlerResult | None = None,
        exception: Exception | None = None,
    ) -> None:
        """
        Store the configured handler result or exception.
        """

        self.__result = result or JobHandlerResult(
            state=JobState.COMPLETED,
            outcome=Outcome(code=JobCode.COMPLETED),
        )
        self.__exception = exception
        self.calls = 0

    async def handle(self, *, job: Job) -> JobHandlerResult:
        """
        Capture one handler call and return or raise the configured outcome.
        """

        self.calls += 1
        if self.__exception is not None:
            raise self.__exception
        return self.__result


class _FlakyInteraction:
    """
    Decorate an interaction to inject a single transient failure.
    """

    def __init__(self, *, delegate: SQLiteInteraction) -> None:
        """
        Wrap a real interaction and fail the first claim call.
        """

        self.__delegate = delegate
        self.__failed = False

    def __getattr__(self, name: str) -> object:
        """
        Forward all methods except claim_job to the wrapped interaction.
        """

        return getattr(self.__delegate, name)

    async def claim_job(self, *, request) -> object:
        """
        Fail once, then delegate job claiming to the wrapped interaction.
        """

        if not self.__failed:
            self.__failed = True
            raise RuntimeError("transient claim failure")
        return await self.__delegate.claim_job(request=request)


class TestInProcessJobSchedulerResilience(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the in-process scheduler loop resilience.
    """

    def setUp(self) -> None:
        """
        Create an isolated SQLite interaction store for each scheduler test.
        """

        self.__temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.__temporary_directory.cleanup)
        path = Path(self.__temporary_directory.name) / "scheduler.db"
        self.__interaction = SQLiteInteraction(path=path)
        self.__now = datetime.now(tz=timezone.utc) - timedelta(seconds=10)

    async def __seed_one_job(self) -> None:
        """
        Seed one tenant, thread, task, and pending job.
        """

        await self.__interaction.create_actor(
            request=CreateActor(
                identity=Identity(id="actor-1", tenant="tenant-1"),
                kind=ActorKind.HUMAN,
                name="Aman",
                created=self.__now,
            )
        )
        await self.__interaction.create_thread(
            request=CreateThread(
                identity=Identity(id="thread-1", tenant="tenant-1"),
                title="t",
                creator="actor-1",
                created=self.__now,
            )
        )
        await self.__interaction.join_thread(
            request=JoinThread(
                identity=Identity(id="m-1", tenant="tenant-1"),
                thread="thread-1",
                actor="actor-1",
                role=MembershipRole.OWNER,
                joined=self.__now,
            )
        )
        await self.__interaction.open_task(
            request=OpenTask(
                identity=Identity(id="task-1", tenant="tenant-1"),
                thread="thread-1",
                assignment=Assignment(creator="actor-1", assignee="actor-1"),
                lineage=Lineage(),
                kind=TaskKind.AGENT,
                state=TaskState.RUNNING,
                plan=Plan(objective="test"),
                created=self.__now,
            )
        )
        await self.__interaction.schedule_job(
            request=ScheduleJob(
                identity=Identity(id="job-1", tenant="tenant-1"),
                thread="thread-1",
                task="task-1",
                kind=JobKind.MEMORY,
                available=self.__now,
                created=self.__now,
            )
        )

    async def test_loop_survives_transient_claim_failure(self) -> None:
        """
        Continue dispatching after a transient claim error and the configured backoff.
        """

        await self.__seed_one_job()
        flaky = _FlakyInteraction(delegate=self.__interaction)
        handler = _RecordingHandler()
        scheduler = InProcessJobScheduler(
            interaction=flaky,  # type: ignore[arg-type]
            configuration=InProcessJobSchedulerConfiguration(
                tenant="tenant-1",
                owner="worker-1",
                poll_interval=10,
                failure_backoff=10,
                recovery_interval=10_000,
                batch_size=1,
            ),
        )

        await scheduler.start(handler=handler)
        for _ in range(50):
            await asyncio.sleep(0.05)
            if handler.calls > 0:
                break
        await scheduler.stop()

        self.assertEqual(1, handler.calls)

    async def test_handler_failure_does_not_kill_loop(self) -> None:
        """
        Reschedule after a handler exception and continue dispatching the next job.
        """

        await self.__seed_one_job()
        handler = _RecordingHandler(exception=RuntimeError("handler failed"))
        scheduler = InProcessJobScheduler(
            interaction=self.__interaction,
            configuration=InProcessJobSchedulerConfiguration(
                tenant="tenant-1",
                owner="worker-1",
                poll_interval=10,
                retry_backoff=0,
                recovery_interval=10_000,
                max_attempts=2,
                batch_size=1,
            ),
        )

        await scheduler.start(handler=handler)
        for _ in range(50):
            await asyncio.sleep(0.05)
            if handler.calls >= 2:
                break
        await scheduler.stop()

        self.assertGreaterEqual(handler.calls, 2)

    async def test_finish_uses_owner_for_lease_protection(self) -> None:
        """
        Successful dispatch finalizes the job under the scheduler's owner identity.
        """

        await self.__seed_one_job()
        handler = _RecordingHandler()
        scheduler = InProcessJobScheduler(
            interaction=self.__interaction,
            configuration=InProcessJobSchedulerConfiguration(
                tenant="tenant-1",
                owner="worker-1",
                poll_interval=10,
                recovery_interval=10_000,
                batch_size=1,
            ),
        )

        await scheduler.start(handler=handler)
        for _ in range(100):
            await asyncio.sleep(0.05)
            if handler.calls > 0:
                break
        await scheduler.stop()

        self.assertGreaterEqual(handler.calls, 1)
        jobs = await self.__interaction.get_jobs(
            query=JobQuery(tenant="tenant-1", thread="thread-1")
        )
        self.assertEqual(1, len(jobs))
        self.assertEqual(JobState.COMPLETED, jobs[0].state)
