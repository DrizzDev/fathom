from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from fathom.schemas.actions import Action
from fathom.schemas.experience import Experience
from fathom.schemas.screens import ScreenState


class MemoryPort(ABC):
    """
    Session key-value store plus cross-run recall of screen observations and action outcomes.
    """

    @abstractmethod
    async def set(self, *, key: str, value: str) -> None:
        """
        Store key-value pair in session.
        """

        raise NotImplementedError

    @abstractmethod
    async def get(self, *, key: str) -> Optional[str]:
        """
        Retrieve value by key.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_all(self) -> Dict[str, str]:
        """
        Get all session data.
        """

        raise NotImplementedError

    @abstractmethod
    async def store_observation(self, *, screen: ScreenState, description: Optional[str]) -> None:
        """
        Store screen observation for future recall.
        """

        raise NotImplementedError

    @abstractmethod
    async def store_experience(self, *, visual_hash: str, action: Action, success: bool) -> None:
        """
        Store action outcome for learning.
        """

        raise NotImplementedError

    async def store_outcome(self, *, experience: Experience) -> None:
        """
        Store the typed outcome of one executed action; default is a no-op for adapters without persistence.
        """

        return None

    @abstractmethod
    async def retrieve_knowledge(self, *, visual_hash: str) -> Dict[str, Any]:
        """
        Retrieve everything known about a screen.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_all_knowledge(self) -> Dict[str, Any]:
        """
        Retrieve a summary of all stored knowledge.
        """

        raise NotImplementedError
