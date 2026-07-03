from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.interaction import Job
from fathom.schemas.scheduler import JobHandlerResult


class JobHandlerPort(ABC):
    """
    Application-level handler that executes one claimed durable job.
    """

    @abstractmethod
    async def handle(self, *, job: Job) -> JobHandlerResult:
        """
        Execute one claimed job and return its terminal outcome.
        """

        raise NotImplementedError


class JobSchedulerPort(ABC):
    """
    Host-neutral dispatcher that drives durable interaction jobs.
    """

    @abstractmethod
    async def start(self, *, handler: JobHandlerPort) -> None:
        """
        Start dispatching available jobs to a handler.
        """

        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop dispatching jobs and wait for active scheduler work to drain.
        """

        raise NotImplementedError
