from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class HierarchyPort(ABC):
    """
    Abstract interface for hierarchy extraction strategies.
    """

    @abstractmethod
    async def dump_hierarchy(self) -> Optional[str]:
        """
        Extract hierarchy for the current execution target.
        """

        raise NotImplementedError
