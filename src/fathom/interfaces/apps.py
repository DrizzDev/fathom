from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.app_graph.state import Completion, Description, Observation, Start


class ApplicationsPort(ABC):
    """Persists evidenced exploration knowledge under an application boundary."""

    @abstractmethod
    async def start(self, *, start: Start) -> None:
        """Register an idempotent Block before exploration begins."""

    @abstractmethod
    async def describe(self, *, description: Description) -> None:
        """Attach model interpretation to a captured page."""

    @abstractmethod
    async def observe(self, *, observation: Observation) -> None:
        """Atomically merge one idempotent discovery event."""

    @abstractmethod
    async def complete(self, *, completion: Completion) -> None:
        """Persist the terminal block outcome."""
