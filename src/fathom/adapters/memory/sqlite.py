"""SQLite memory adapter - wraps existing memory logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from fathom.infrastructure.memory.ledger import Ledger
from fathom.infrastructure.memory.sqlite import SQLiteMemoryProvider
from fathom.interfaces.memory import MemoryPort
from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenState

if TYPE_CHECKING:
    from fathom.base.paths import SharedPathManager


class SQLiteMemory(MemoryPort):
    """
    SQLite adapter for memory persistence.

    This adapter wraps the existing SQLiteMemoryProvider and Ledger logic without modifications.
    All code delegates to existing implementations to preserve exact behavior.
    """

    def __init__(
        self,
        path_manager: SharedPathManager,
        *,
        knowledge_path: str = "assets/memory/knowledge.db",
        ledger_path: str = "assets/memory/ledger.db",
    ) -> None:
        """Initialize SQLite memory adapter."""
        k_path = str(path_manager.get_knowledge_db_path())
        l_path = str(path_manager.get_ledger_db_path())

        # Use existing implementations as-is
        self.__provider = SQLiteMemoryProvider(database_path=k_path)
        self.__ledger = Ledger(database_path=l_path)

    async def set(self, *, key: str, value: str) -> None:
        """Store key-value pair in session."""
        await self.__ledger.set(key=key, value=value)

    async def get(self, *, key: str) -> Optional[str]:
        """Retrieve value by key."""
        return await self.__ledger.get(key=key)

    async def get_all(self) -> Dict[str, str]:
        """Get all session data."""
        return await self.__ledger.get_all()

    async def store_observation(self, *, screen: ScreenState, description: Optional[str]) -> None:
        """Store screen observation for future recall."""
        await self.__provider.store_observation(screen=screen, description=description)

    async def store_experience(self, *, visual_hash: str, action: Action, success: bool) -> None:
        """Store action outcome for learning."""
        await self.__provider.store_experience(
            visual_hash=visual_hash, action=action, success=success
        )

    async def retrieve_knowledge(self, *, visual_hash: str) -> Dict[str, Any]:
        """Retrieve everything known about a screen."""
        return await self.__provider.retrieve_knowledge(visual_hash=visual_hash)

    async def get_all_knowledge(self) -> Dict[str, Any]:
        """Retrieve a summary of all stored knowledge."""
        return await self.__provider.get_all_knowledge()
