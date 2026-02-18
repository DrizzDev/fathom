from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import aiosqlite

from fathom.interfaces import ILedger


class Ledger(ILedger):
    """
    Persistent Key-Value storage using SQLite.
    Separated from the core Knowledge Graph logic.
    """

    def __init__(self, database_path: str = "assets/memory/ledger.db") -> None:
        self.__initialized = False
        self.__path = Path(database_path)
        self.__path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize_key(key: str) -> str:
        """
        Normalizes a memory key: lowercase, strip whitespace, replace spaces
        with underscores. Acts as a safety net for consistent key matching.
        """

        return key.strip().lower().replace(" ", "_")

    async def __initialize(self) -> None:
        """
        Initializes the ledger table.
        """

        if self.__initialized:
            return

        async with aiosqlite.connect(self.__path) as db:
            await db.execute(
                "CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, value TEXT, updated_at INTEGER)"
            )
            await db.commit()

        self.__initialized = True

    async def set(self, key: str, value: str) -> None:
        """
        Stores a ledger entry. Key is normalized before writing.
        """

        await self.__initialize()
        normalized = self.normalize_key(key)

        async with aiosqlite.connect(self.__path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO entries (key, value, updated_at) VALUES (?, ?, ?)",
                (normalized, value, int(time.time())),
            )
            await db.commit()

    async def get(self, key: str) -> Optional[str]:
        """
        Retrieves a value by key. Key is normalized before lookup.
        """

        await self.__initialize()
        normalized = self.normalize_key(key)

        async with (
            aiosqlite.connect(self.__path) as db,
            db.execute("SELECT value FROM entries WHERE key = ?", (normalized,)) as cursor,
        ):
            row = await cursor.fetchone()
            return row[0] if row else None

    async def get_all(self) -> Dict[str, str]:
        """
        Retrieves all ledger entries.
        """

        await self.__initialize()
        result = {}

        async with (
            aiosqlite.connect(self.__path) as db,
            db.execute("SELECT key, value FROM entries") as cursor,
        ):
            async for row in cursor:
                result[row[0]] = row[1]

        return result
