from __future__ import annotations

import json
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

from fathom.interfaces import IMemoryProvider
from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenState

logger = getLogger(__name__)


class SQLiteMemoryProvider(IMemoryProvider):
    """
    SQLite implementation of the persistent memory layer.
    Handles raw database operations for the Knowledge Graph.
    """

    def __init__(
        self,
        database_path: str = "assets/memory/knowledge.db",
        *,
        readonly: bool = False,
    ) -> None:
        self.__initialized = False
        self.__readonly = readonly
        self.__path = Path(database_path)
        self.__path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """Returns the database file path."""
        return self.__path

    @property
    def readonly(self) -> bool:
        """Whether this provider silently drops all write operations."""
        return self.__readonly

    def switch_database(self, new_path: str) -> None:
        """Switch to a different database file.

        Safe to call between async operations because every query opens a
        fresh ``aiosqlite.connect()`` — there is no persistent connection to
        invalidate.  No-op when *new_path* resolves to the current path.
        """

        candidate = Path(new_path)
        if candidate == self.__path:
            return

        self.__path = candidate
        self.__path.parent.mkdir(parents=True, exist_ok=True)
        self.__initialized = False
        logger.info("Memory provider switched to %s", self.__path)

    async def __initialize(self) -> None:
        """
        Initializes the database schema if it doesn't exist.
        """

        if self.__initialized:
            return

        async with aiosqlite.connect(self.__path) as db:
            await db.execute(
                "CREATE TABLE IF NOT EXISTS screens ("
                "visual_hash TEXT PRIMARY KEY, "
                "activity TEXT, "
                "description TEXT, "
                "first_seen INTEGER, "
                "last_seen INTEGER, "
                "visit_count INTEGER DEFAULT 0"
                ")"
            )
            await db.execute(
                "CREATE TABLE IF NOT EXISTS experience ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "visual_hash TEXT, "
                "action_json TEXT, "
                "success BOOLEAN, "
                "rationale TEXT, "
                "timestamp INTEGER"
                ")"
            )
            await db.execute(
                "CREATE TABLE IF NOT EXISTS transitions ("
                "source_hash TEXT NOT NULL, "
                "destination_hash TEXT NOT NULL, "
                "action_type TEXT NOT NULL, "
                "action_target TEXT, "
                "action_json TEXT NOT NULL, "
                "count INTEGER DEFAULT 1, "
                "first_seen INTEGER, "
                "last_seen INTEGER, "
                "PRIMARY KEY (source_hash, action_type, action_target)"
                ")"
            )
            await db.commit()

            # Migrate existing databases that lack new columns
            await self.__migrate(db)

        self.__initialized = True

    async def __migrate(self, db: aiosqlite.Connection) -> None:
        """
        Backward-compatible migration for databases created before the
        knowledge graph extension. Adds missing columns silently.
        """

        async with (
            db.execute("PRAGMA table_info(screens)") as cursor,
        ):
            columns = {row[1] async for row in cursor}

        migrations_applied = False

        if "first_seen" not in columns:
            await db.execute("ALTER TABLE screens ADD COLUMN first_seen INTEGER")
            # Backfill first_seen from last_seen for existing rows
            await db.execute("UPDATE screens SET first_seen = last_seen WHERE first_seen IS NULL")
            migrations_applied = True

        if "visit_count" not in columns:
            await db.execute("ALTER TABLE screens ADD COLUMN visit_count INTEGER DEFAULT 0")
            # Backfill: existing rows get visit_count = 1 (they were seen at least once)
            await db.execute("UPDATE screens SET visit_count = 1 WHERE visit_count = 0")
            migrations_applied = True

        if migrations_applied:
            await db.commit()
            logger.info("Applied schema migrations to knowledge database")

    async def store_observation(self, screen: ScreenState, description: Optional[str]) -> None:
        """
        Stores a screen observation in the database.
        Increments visit_count on subsequent observations of the same screen.
        """

        if self.__readonly:
            return

        await self.__initialize()
        now = int(time.time())

        async with aiosqlite.connect(self.__path) as db:
            await db.execute(
                "INSERT INTO screens (visual_hash, activity, description, first_seen, last_seen, visit_count) "
                "VALUES (?, ?, ?, ?, ?, 1) "
                "ON CONFLICT(visual_hash) DO UPDATE SET "
                "activity = excluded.activity, "
                "description = COALESCE(excluded.description, screens.description), "
                "last_seen = excluded.last_seen, "
                "visit_count = screens.visit_count + 1",
                (screen.visual_hash, screen.activity, description, now, now),
            )
            await db.commit()

    async def store_experience(self, visual_hash: str, action: Action, success: bool) -> None:
        """
        Stores an action experience in the database.
        """

        if self.__readonly:
            return

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
        knowledge: Dict[str, Any] = {
            "description": None,
            "previous_actions": [],
            "transitions": [],
            "visit_count": 0,
        }

        async with aiosqlite.connect(self.__path) as db:
            # 1. Get screen description and visit count
            async with db.execute(
                "SELECT description, visit_count FROM screens WHERE visual_hash = ?", (visual_hash,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    knowledge["description"] = row[0]
                    knowledge["visit_count"] = row[1]

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

            # 3. Get known transitions from this screen (with destination descriptions)
            async with db.execute(
                "SELECT t.destination_hash, t.action_type, t.action_target, t.count, s.description "
                "FROM transitions t "
                "LEFT JOIN screens s ON t.destination_hash = s.visual_hash "
                "WHERE t.source_hash = ? ORDER BY t.count DESC LIMIT 10",
                (visual_hash,),
            ) as cursor:
                async for row in cursor:
                    knowledge["transitions"].append(
                        {
                            "destination": row[0],
                            "action_type": row[1],
                            "action_target": row[2],
                            "count": row[3],
                            "destination_description": row[4],
                        }
                    )

        return knowledge

    async def store_transition(
        self,
        source_hash: str,
        action: Action,
        destination_hash: str,
    ) -> None:
        """
        Stores or updates a screen transition edge in the knowledge graph.
        Increments count on repeated observations of the same transition.
        """

        if self.__readonly:
            return

        await self.__initialize()
        now = int(time.time())

        action_type = (
            action.action_type.value
            if hasattr(action.action_type, "value")
            else str(action.action_type)
        )
        action_target = action.natural_language_target or action.target or ""

        async with aiosqlite.connect(self.__path) as db:
            await db.execute(
                "INSERT INTO transitions "
                "(source_hash, destination_hash, action_type, action_target, action_json, count, first_seen, last_seen) "
                "VALUES (?, ?, ?, ?, ?, 1, ?, ?) "
                "ON CONFLICT(source_hash, action_type, action_target) DO UPDATE SET "
                "destination_hash = excluded.destination_hash, "
                "action_json = excluded.action_json, "
                "count = transitions.count + 1, "
                "last_seen = excluded.last_seen",
                (
                    source_hash,
                    destination_hash,
                    action_type,
                    action_target,
                    action.model_dump_json(),
                    now,
                    now,
                ),
            )
            await db.commit()

    async def retrieve_transitions(self, visual_hash: str) -> List[Dict[str, Any]]:
        """
        Retrieves all outgoing transitions from a screen.
        """

        await self.__initialize()
        transitions: List[Dict[str, Any]] = []

        async with (
            aiosqlite.connect(self.__path) as db,
            db.execute(
                "SELECT destination_hash, action_type, action_target, count, last_seen "
                "FROM transitions WHERE source_hash = ? ORDER BY count DESC",
                (visual_hash,),
            ) as cursor,
        ):
            async for row in cursor:
                transitions.append(
                    {
                        "destination_hash": row[0],
                        "action_type": row[1],
                        "action_target": row[2],
                        "count": row[3],
                        "last_seen": row[4],
                    }
                )

        return transitions

    async def get_all_screens(self) -> List[Dict[str, Any]]:
        """
        Retrieves all screens with full metadata.
        """

        await self.__initialize()
        screens: List[Dict[str, Any]] = []

        async with (
            aiosqlite.connect(self.__path) as db,
            db.execute(
                "SELECT visual_hash, activity, description, first_seen, last_seen, visit_count "
                "FROM screens ORDER BY last_seen DESC"
            ) as cursor,
        ):
            async for row in cursor:
                screens.append(
                    {
                        "visual_hash": row[0],
                        "activity": row[1],
                        "description": row[2],
                        "first_seen": row[3],
                        "last_seen": row[4],
                        "visit_count": row[5],
                    }
                )

        return screens

    async def get_all_transitions(self) -> List[Dict[str, Any]]:
        """
        Retrieves all transitions in the knowledge graph.
        """

        await self.__initialize()
        transitions: List[Dict[str, Any]] = []

        async with (
            aiosqlite.connect(self.__path) as db,
            db.execute(
                "SELECT source_hash, destination_hash, action_type, action_target, count, first_seen, last_seen "
                "FROM transitions ORDER BY count DESC"
            ) as cursor,
        ):
            async for row in cursor:
                transitions.append(
                    {
                        "source_hash": row[0],
                        "destination_hash": row[1],
                        "action_type": row[2],
                        "action_target": row[3],
                        "count": row[4],
                        "first_seen": row[5],
                        "last_seen": row[6],
                    }
                )

        return transitions

    async def get_all_knowledge(self) -> Dict[str, Any]:
        """
        Retrieves a summary of all stored knowledge for reporting.
        """

        await self.__initialize()
        summary: Dict[str, Any] = {"screens": [], "experience_count": 0, "transition_count": 0}

        async with aiosqlite.connect(self.__path) as db:
            # Get screens
            async with db.execute(
                "SELECT visual_hash, activity, description, visit_count FROM screens ORDER BY last_seen DESC"
            ) as cursor:
                async for row in cursor:
                    summary["screens"].append(
                        {
                            "hash": row[0],
                            "activity": row[1],
                            "description": row[2],
                            "visit_count": row[3],
                        }
                    )

            # Get total experience count
            async with db.execute("SELECT COUNT(*) FROM experience") as cursor:
                exp_row = await cursor.fetchone()
                summary["experience_count"] = exp_row[0] if exp_row else 0

            # Get total transition count
            async with db.execute("SELECT COUNT(*) FROM transitions") as cursor:
                trans_row = await cursor.fetchone()
                summary["transition_count"] = trans_row[0] if trans_row else 0

        return summary
