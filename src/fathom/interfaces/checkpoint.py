"""
Backend-neutral ports for LangGraph checkpoint storage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from fathom.schemas.checkpoint import ExplorationCheckpoint


@runtime_checkable
class LangGraphCheckpointer(Protocol):
    """
    Subset of langgraph.checkpoint.base.BaseCheckpointSaver that the intent strategy depends on.
    """

    async def aget(self, config: Any) -> Any:
        """
        Return the latest checkpoint for the given configurable thread, or None.
        """

    async def aget_tuple(self, config: Any) -> Any:
        """
        Return the latest checkpoint tuple including parent metadata, or None.
        """

    async def aput(
        self,
        config: Any,
        checkpoint: Any,
        metadata: Any,
        new_versions: Any,
    ) -> Any:
        """
        Persist a new checkpoint snapshot for the configurable thread.
        """

    async def aput_writes(
        self,
        config: Any,
        writes: Any,
        task_id: str,
        task_path: str = "",
    ) -> Any:
        """
        Persist pending per-task write rows associated with the active checkpoint.
        """

    async def alist(
        self,
        config: Any,
        *,
        filter: Any = None,
        before: Any = None,
        limit: Any = None,
    ) -> Any:
        """
        List historical checkpoints for the configurable thread.
        """


@runtime_checkable
class CheckpointStore(Protocol):
    """
    Backend-neutral lifecycle for LangGraph checkpoint persistence per workflow.
    """

    def open(self, *, workflow_id: str) -> AbstractAsyncContextManager[LangGraphCheckpointer]:
        """
        Acquire a workflow-scoped LangGraph checkpointer as an async context.
        """

    async def discard(self, *, workflow_id: str) -> None:
        """
        Remove all persisted checkpoint state for a completed workflow.
        """

    async def sweep_stale(self) -> list[str]:
        """
        Remove orphaned checkpoint state older than the configured retention; return removed workflow identifiers.
        """


class ExplorationCheckpointPort(ABC):
    """
    Persistence contract for the exploration DFS checkpoint, enabling cross-run resume.
    """

    @abstractmethod
    async def save(self, *, workflow_id: str, checkpoint: ExplorationCheckpoint) -> None:
        """
        Persist the latest DFS checkpoint for a workflow, replacing any prior one.
        """

        raise NotImplementedError

    @abstractmethod
    async def load(self, *, workflow_id: str) -> Optional[ExplorationCheckpoint]:
        """
        Load the saved DFS checkpoint for a workflow, or None when absent or incompatible.
        """

        raise NotImplementedError
