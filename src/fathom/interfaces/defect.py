"""
Backend-neutral ports for exploration defect detection and storage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from fathom.schemas.defect import Defect, ScreenSnapshot, StepSignals


class InlineDefectDetectorPort(ABC):
    """
    Derives defects from a single step's runtime signals during the crawl.
    """

    @abstractmethod
    def inspect_step(self, *, signals: StepSignals) -> List[Defect]:
        """
        Returns the defects evidenced by one completed step.
        """

        raise NotImplementedError


class ScreenDefectDetectorPort(ABC):
    """
    Detects defects on a single captured screen after the crawl.
    """

    @abstractmethod
    async def inspect_screen(self, *, snapshot: ScreenSnapshot) -> List[Defect]:
        """
        Returns the defects found on one screen.
        """

        raise NotImplementedError


class DefectRepositoryPort(ABC):
    """
    Persists and retrieves defects for an exploration run.
    """

    @abstractmethod
    async def record(self, *, session: str, defect: Defect) -> None:
        """
        Persists one defect, deduplicating by its natural signature.
        """

        raise NotImplementedError

    @abstractmethod
    async def for_screen(self, *, session: str, screen: str) -> List[Defect]:
        """
        Returns the defects recorded for one screen.
        """

        raise NotImplementedError

    @abstractmethod
    async def for_run(self, *, session: str) -> List[Defect]:
        """
        Returns every defect recorded for the run.
        """

        raise NotImplementedError
