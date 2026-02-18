from __future__ import annotations

import json
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional

import aiosqlite

from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenState

logger = getLogger(__name__)


class MemoryService:
    """
    Persisted memory layer for UI Knowledge Graph.
    Responsibility: Store and retrieve screen states, transitions, and action outcomes.
    """

    def __init__(self, database_path: str = "assets/memory/knowledge.db") -> None:
        self.__initialized = False
        self.__database_path = Path(database_path)
        self.__database_path.parent.mkdir(parents=True, exist_ok=True)

    async def __ensure_initialized(self) -> None:
        """
        Initializes the database schema if it doesn't exist.
        """

        if self.__initialized:
            return

        async with aiosqlite.connect(self.__database_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS screens (
                    visual_hash TEXT PRIMARY KEY,
                    activity TEXT,
                    description TEXT,
                    last_seen INTEGER
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS transitions (
                    source_hash TEXT,
                    action_json TEXT,
                    destination_hash TEXT,
                    PRIMARY KEY (source_hash, action_json)
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS experience (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visual_hash TEXT,
                    action_json TEXT,
                    success BOOLEAN,
                    rationale TEXT,
                    timestamp INTEGER
                )
            """)

            await db.commit()

        self.__initialized = True

    async def record_observation(
        self, screen: ScreenState, description: Optional[str] = None
    ) -> None:
        """
        Stores or updates knowledge about a specific screen.
        """

        await self.__ensure_initialized()
        async with aiosqlite.connect(self.__database_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO screens (visual_hash, activity, description, last_seen) VALUES (?, ?, ?, ?)",
                (screen.visual_hash, screen.activity, description, int(screen.timestamp)),
            )
            await db.commit()

    async def record_transition(
        self, source_hash: str, action: Action, destination_hash: str
    ) -> None:
        """
        Stores a directional link between two screens via an action.
        """

        await self.__ensure_initialized()

        async with aiosqlite.connect(self.__database_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO transitions (source_hash, action_json, destination_hash) VALUES (?, ?, ?)",
                (source_hash, action.model_dump_json(), destination_hash),
            )
            await db.commit()

    async def record_experience(self, visual_hash: str, action: Action, success: bool) -> None:
        """
        Stores the outcome of an action on a specific screen.
        """

        await self.__ensure_initialized()

        async with aiosqlite.connect(self.__database_path) as db:
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

    async def get_screen_knowledge(self, visual_hash: str) -> Dict[str, Any]:
        """
        Retrieves everything known about a specific screen.
        """

        await self.__ensure_initialized()

        knowledge: Dict[str, Any] = {
            "description": None,
            "previous_actions": [],
            "successful_transitions": [],
        }

        async with aiosqlite.connect(self.__database_path) as db:
            # 1. Basic Info
            async with db.execute(
                "SELECT description FROM screens WHERE visual_hash = ?", (visual_hash,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    knowledge["description"] = row[0]

            # 2. Historical Experience
            async with db.execute(
                "SELECT action_json, success FROM experience WHERE visual_hash = ? ORDER BY timestamp DESC LIMIT 5",
                (visual_hash,),
            ) as cursor:
                async for row in cursor:
                    try:
                        action_data = json.loads(row[0])
                    except (json.JSONDecodeError, TypeError):
                        continue
                    previous_actions = knowledge.get("previous_actions")
                    if isinstance(previous_actions, list):
                        previous_actions.append(
                            {
                                "success": bool(row[1]),
                                "target": action_data.get("target"),
                                "action": action_data.get("action_type"),
                            }
                        )

        return knowledge
