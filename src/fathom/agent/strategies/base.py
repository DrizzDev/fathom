from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.results import StrategyResult


class ExecutionStrategy(ABC):
    """
    Abstract base for execution strategies.

    Strategies determine how to execute steps and when to stop.
    Different strategies have different termination conditions and recovery behaviors.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Strategy name.
        """

        raise NotImplementedError

    @abstractmethod
    async def execute_step(self) -> StrategyResult:
        """
        Execute a single step.

        Returns:
            Result indicating whether to continue.
        """

        raise NotImplementedError

    @abstractmethod
    async def should_continue(self) -> bool:
        """
        Check if execution should continue.

        Returns:
            True if more steps should be executed.
        """

        raise NotImplementedError

    @abstractmethod
    def get_progress(self) -> dict[str, object]:
        """
        Get current progress information.

        Returns:
            Progress data for reporting.
        """

        raise NotImplementedError
