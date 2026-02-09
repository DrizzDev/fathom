from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import aiosqlite

from fathom.interfaces import IMemoryProvider
from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenState


class SQLiteMemoryProvider(IMemoryProvider):
    """
    SQLite implementation of the persistent memory layer.
    Handles raw database operations for the Knowledge Graph.
    """

    def __init__(self, database_path: str = "assets/memory/knowledge.db") -> None:
        self.__initialized = False
        self.__path = Path(database_path)
        self.__path.parent.mkdir(parents=True, exist_ok=True)

    async def __initialize(self) -> None:
        """
        Initializes the database schema if it doesn't exist.
        """

        if self.__initialized:
            return

        async with aiosqlite.connect(self.__path) as db:
            await db.execute(
                "CREATE TABLE IF NOT EXISTS screens (visual_hash TEXT PRIMARY KEY, activity TEXT, description TEXT, last_seen INTEGER)"
            )
            await db.execute(
                "CREATE TABLE IF NOT EXISTS experience (id INTEGER PRIMARY KEY AUTOINCREMENT, visual_hash TEXT, action_json TEXT, success BOOLEAN, rationale TEXT, timestamp INTEGER)"
            )
            await db.commit()
        self.__initialized = True

    async def store_observation(self, screen: ScreenState, description: Optional[str]) -> None:
        """
        Stores a screen observation in the database.
        """

        await self.__initialize()

        async with aiosqlite.connect(self.__path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO screens (visual_hash, activity, description, last_seen) VALUES (?, ?, ?, ?)",
                (screen.visual_hash, screen.activity, description, int(time.time())),
            )
            await db.commit()

    async def store_experience(self, visual_hash: str, action: Action, success: bool) -> None:
        """
        Stores an action experience in the database.
        """

        await self.__initialize()

        async with aiosqlite.connect(self.__path) as db:
            await db.execute(
                "INSERT INTO experience (visual_hash, action_json, success, rationale, timestamp) VALUES (?, ?, ?, ?, ?)",
                (
                    visual_hash,
                    action.model_dump_json(),
                    success,
                    action.rationale,
                    int(time.time()),
                ),
            )
            await db.commit()

    async def retrieve_knowledge(self, visual_hash: str) -> Dict[str, Any]:
        """
        Retrieves everything known about a specific screen by its visual hash.
        """

        await self.__initialize()
        knowledge: Dict[str, Any] = {"description": None, "previous_actions": []}

        async with aiosqlite.connect(self.__path) as db:
            # 1. Get screen description
            async with db.execute(
                "SELECT description FROM screens WHERE visual_hash = ?", (visual_hash,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    knowledge["description"] = row[0]

            # 2. Get recent experiences (last 5)
            async with db.execute(
                "SELECT action_json, success FROM experience WHERE visual_hash = ? ORDER BY timestamp DESC LIMIT 5",
                (visual_hash,),
            ) as cursor:
                async for row in cursor:
                    try:
                        data = json.loads(row[0])
                        knowledge["previous_actions"].append(
                            {
                                "action": data.get("action_type"),
                                "target": data.get("target") or "element",
                                "success": bool(row[1]),
                            }
                        )
                    except (json.JSONDecodeError, AttributeError):
                        continue

        return knowledge

    async def get_all_knowledge(self) -> Dict[str, Any]:
        """
        Retrieves a summary of all stored knowledge for reporting.
        """

        await self.__initialize()
        summary: Dict[str, Any] = {"screens": [], "experience_count": 0}

        async with aiosqlite.connect(self.__path) as db:
            # Get screens
            async with db.execute(
                "SELECT visual_hash, activity, description FROM screens ORDER BY last_seen DESC"
            ) as cursor:
                async for row in cursor:
                    summary["screens"].append(
                        {"hash": row[0], "activity": row[1], "description": row[2]}
                    )

            # Get total experience count
            async with db.execute("SELECT COUNT(*) FROM experience") as cursor:
                row = await cursor.fetchone()
                summary["experience_count"] = row[0] if row else 0

        return summary
