from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path


@runtime_checkable
class HistoryPaths(Protocol):
    """
    Resolves a run's on-disk history directory.
    """

    def get_history_directory(self, *, session_id: str) -> Path:
        """
        Return the directory holding the given run's recorded history.
        """
        ...
