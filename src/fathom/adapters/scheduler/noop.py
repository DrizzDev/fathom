from __future__ import annotations

from fathom.interfaces.scheduler import JobHandlerPort, JobSchedulerPort


class NoopJobScheduler(JobSchedulerPort):
    """
    Scheduler adapter that explicitly disables background job dispatch.
    """

    async def start(self, *, handler: JobHandlerPort) -> None:
        """
        Accept the handler but do not dispatch any jobs.
        """

    async def stop(self) -> None:
        """
        No scheduler resources are held.
        """
