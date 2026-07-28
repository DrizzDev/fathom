from __future__ import annotations

from typing import Any, Dict, Optional

from fathom.interfaces import ILedger, IMemoryProvider
from fathom.interfaces.memory import MemoryPort
from fathom.schemas.actions import Action
from fathom.schemas.experience import Experience
from fathom.schemas.screens import ScreenState


class SQLiteMemory(MemoryPort):
    """
    SQLite adapter for memory persistence.
    """

    def __init__(self, ledger: ILedger, provider: IMemoryProvider) -> None:
        """
        Initialize SQLite memory adapter with injected providers.
        """

        self.__ledger = ledger
        self.__provider = provider

    async def set(self, *, key: str, value: str) -> None:
        """
        Store key-value pair in session.
        """

        await self.__ledger.set(key=key, value=value)

    async def get(self, *, key: str) -> Optional[str]:
        """
        Retrieve value by key.
        """

        return await self.__ledger.get(key=key)

    async def get_all(self) -> Dict[str, str]:
        """
        Get all session data.
        """

        return await self.__ledger.get_all()

    async def store_observation(self, *, screen: ScreenState, description: Optional[str]) -> None:
        """
        Store screen observation for future recall.
        """

        await self.__provider.store_observation(screen=screen, description=description)

    async def store_outcome(self, *, experience: Experience) -> None:
        """
        Store the typed outcome of one executed action.
        """

        await self.__provider.store_outcome(experience=experience)

    async def store_experience(self, *, visual_hash: str, action: Action, success: bool) -> None:
        """
        Store action outcome for learning.
        """

        await self.__provider.store_experience(
            visual_hash=visual_hash, action=action, success=success
        )

    async def retrieve_knowledge(self, *, visual_hash: str) -> Dict[str, Any]:
        """
        Retrieve everything known about a screen.
        """

        return await self.__provider.retrieve_knowledge(visual_hash=visual_hash)

    async def get_all_knowledge(self) -> Dict[str, Any]:
        """
        Retrieve a summary of all stored knowledge.
        """

        return await self.__provider.get_all_knowledge()
