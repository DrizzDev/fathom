from __future__ import annotations

import time
from logging import getLogger
from pathlib import Path  # noqa: TC003
from typing import Any, Dict, Optional

import aiosqlite

from fathom.interfaces import ILedger

logger = getLogger(__name__)


class Ledger(ILedger):
    """
    Persistent Key-Value storage using SQLite.
    Separated from the core Knowledge Graph logic.
    """

    def __init__(self, database_path: Path) -> None:
        """
        Initialize ledger with explicit database path.
        """

        self.__initialized = False
        self.__path = database_path
        self.__path.parent.mkdir(parents=True, exist_ok=True)

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
        Stores a ledger entry.

        NOTE: System state keys are REJECTED to prevent memory pollution.

        Ledger is reserved for user-actionable memory only.
        System state (GCC context, internal state) should use separate storage.
        """

        if not key or not isinstance(key, str):
            logger.error(f"[LEDGER] Invalid key: {key}")
            raise ValueError(f"Key must be a non-empty string, got: {key}")

        if not isinstance(value, str):
            logger.error(f"[LEDGER] Invalid value type for key={key}: {type(value)}")
            raise ValueError(f"Value must be a string, got: {type(value)}")

        # DEFENSIVE: Reject system state keys to prevent pollution
        SYSTEM_KEY_PREFIXES = ("context:", "ctx_v3:", "ctx_")
        if key.startswith(SYSTEM_KEY_PREFIXES):
            logger.warning(
                f"[LEDGER] REJECTED system key | "
                f"key={key} | "
                f"reason=Ledger is for user-actionable memory only"
            )
            raise ValueError(
                f"System keys are not allowed in Ledger. "
                f"Key '{key}' starts with forbidden prefix: {SYSTEM_KEY_PREFIXES}. "
                f"Use separate storage for system state."
            )

        await self.__initialize()

        try:
            async with aiosqlite.connect(self.__path) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO entries (key, value, updated_at) VALUES (?, ?, ?)",
                    (key, value, int(time.time())),
                )
                await db.commit()

            logger.info(f"[LEDGER] SET | key={key} | value_length={len(value)}")
        except Exception as exception:
            logger.error(f"[LEDGER] SET FAILED | key={key} | error={exception}")
            raise

    async def get(self, key: str) -> Optional[str]:
        """
        Retrieves a value by key.
        """

        await self.__initialize()

        async with (
            aiosqlite.connect(self.__path) as db,
            db.execute("SELECT value FROM entries WHERE key = ?", (key,)) as cursor,
        ):
            row = await cursor.fetchone()
            result = row[0] if row else None

        logger.info(f"[LEDGER] GET | key={key} | found={result is not None}")
        return result

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

        logger.info(f"[LEDGER] GET_ALL | total_entries={len(result)} | keys={list(result.keys())}")
        return result

    async def health_check(self) -> Dict[str, Any]:
        """
        Performs a health check on the ledger. Returns diagnostic information.
        """

        await self.__initialize()

        try:
            async with aiosqlite.connect(self.__path) as db:
                # Check table exists
                async with db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='entries'"
                ) as cursor:
                    table_exists = await cursor.fetchone() is not None

                # Count entries
                async with db.execute("SELECT COUNT(*) FROM entries") as cursor:
                    row = await cursor.fetchone()
                    entry_count = row[0] if row else 0

                # Get oldest and newest timestamps
                async with db.execute(
                    "SELECT MIN(updated_at), MAX(updated_at) FROM entries"
                ) as cursor:
                    row = await cursor.fetchone()
                    oldest_ts = row[0] if row and row[0] else None
                    newest_ts = row[1] if row and row[1] else None

                return {
                    "healthy": True,
                    "oldest_entry": oldest_ts,
                    "newest_entry": newest_ts,
                    "entry_count": entry_count,
                    "table_exists": table_exists,
                    "database_path": str(self.__path),
                }
        except Exception as exception:
            logger.error(f"[LEDGER] Health check failed: {exception}")
            return {
                "healthy": False,
                "error": str(exception),
                "database_path": str(self.__path),
            }
