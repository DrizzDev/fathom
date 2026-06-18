from __future__ import annotations

import json
import time
from logging import getLogger
from pathlib import Path  # noqa: TC003
from typing import Any, Dict, List, Optional

import aiosqlite

from fathom.constants.execution import LAUNCHER_PACKAGES
from fathom.interfaces import IMemoryProvider
from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenState

logger = getLogger(__name__)


class SQLiteMemoryProvider(IMemoryProvider):
    """
    SQLite implementation of the persistent knowledge-graph memory layer.

    Handles raw database operations for screens, experiences, and transitions.
    The database is the source of truth; the in-memory graph is a read-through
    cache loaded from get_all_screens / get_all_transitions on startup.
    """

    def __init__(self, database_path: Path, *, readonly: bool = False) -> None:
        """
        Initialize the provider with an explicit database path.
        """

        self.__initialized = False
        self.__readonly = readonly
        self.__path = database_path
        self.__path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """
        Returns the database file path.
        """

        return self.__path

    @property
    def readonly(self) -> bool:
        """
        Whether this provider silently drops all write operations.
        """

        return self.__readonly

    def switch_database(self, new_path: Path) -> None:
        """
        Switch to a different database file, re-initialising lazily.
        """

        if new_path == self.__path:
            return

        self.__path = new_path
        self.__path.parent.mkdir(parents=True, exist_ok=True)
        self.__initialized = False
        logger.info("Memory provider switched to %s", self.__path)

    async def __initialize(self) -> None:
        """
        Creates the schema if absent and migrates older databases.
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
                "visit_count INTEGER DEFAULT 0, "
                "rich_description TEXT"
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
            await self.__migrate(db=db)

        self.__initialized = True

    async def __migrate(self, *, db: aiosqlite.Connection) -> None:
        """
        Adds knowledge-graph columns to databases created before this schema.
        """

        async with db.execute("PRAGMA table_info(screens)") as cursor:
            screen_columns = {row[1] async for row in cursor}

        migrated = False

        if "first_seen" not in screen_columns:
            await db.execute("ALTER TABLE screens ADD COLUMN first_seen INTEGER")
            await db.execute("UPDATE screens SET first_seen = last_seen WHERE first_seen IS NULL")
            migrated = True

        if "visit_count" not in screen_columns:
            await db.execute("ALTER TABLE screens ADD COLUMN visit_count INTEGER DEFAULT 0")
            await db.execute("UPDATE screens SET visit_count = 1 WHERE visit_count = 0")
            migrated = True

        if "rich_description" not in screen_columns:
            await db.execute("ALTER TABLE screens ADD COLUMN rich_description TEXT")
            migrated = True

        if "exhausted" not in screen_columns:
            await db.execute("ALTER TABLE screens ADD COLUMN exhausted INTEGER DEFAULT 0")
            migrated = True

        if "relevance" not in screen_columns:
            await db.execute("ALTER TABLE screens ADD COLUMN relevance TEXT DEFAULT 'unscoped'")
            migrated = True

        for column in ("activity_hash", "xml_hash", "interaction_hash"):
            if column not in screen_columns:
                await db.execute(f"ALTER TABLE screens ADD COLUMN {column} TEXT")
                migrated = True

        async with db.execute("PRAGMA table_info(transitions)") as cursor:
            transition_columns = {row[1] async for row in cursor}

        for column in ("coord_bucket", "coord_region", "element_category"):
            if column not in transition_columns:
                await db.execute(f"ALTER TABLE transitions ADD COLUMN {column} TEXT")
                migrated = True

        if migrated:
            await db.commit()
            logger.info("Applied schema migrations to knowledge database")

    async def store_observation(self, screen: ScreenState, description: Optional[str]) -> None:
        """
        Upserts a screen observation, incrementing visit_count on revisits.
        """

        if self.__readonly:
            return

        await self.__initialize()
        now = int(time.time())

        async with aiosqlite.connect(self.__path) as db:
            await db.execute(
                "INSERT INTO screens "
                "(visual_hash, activity, description, first_seen, last_seen, visit_count, "
                " activity_hash, xml_hash, interaction_hash) "
                "VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?) "
                "ON CONFLICT(visual_hash) DO UPDATE SET "
                "activity = excluded.activity, "
                "description = CASE "
                "    WHEN screens.description IS NOT NULL "
                "         AND TRIM(screens.description) != '' "
                "         AND LOWER(TRIM(screens.description)) "
                "             NOT IN ('unknown', 'tool-based analysis') "
                "        THEN screens.description "
                "    ELSE COALESCE(excluded.description, screens.description) "
                "END, "
                "last_seen = excluded.last_seen, "
                "visit_count = screens.visit_count + 1, "
                "activity_hash = COALESCE(screens.activity_hash, excluded.activity_hash), "
                "xml_hash = COALESCE(screens.xml_hash, excluded.xml_hash), "
                "interaction_hash = COALESCE(screens.interaction_hash, excluded.interaction_hash)",
                (
                    screen.visual_hash,
                    screen.activity,
                    description,
                    now,
                    now,
                    screen.activity_hash,
                    screen.xml_hash,
                    screen.interaction_hash,
                ),
            )
            await db.commit()

    async def update_rich_description(self, visual_hash: str, rich_description: str) -> None:
        """
        Updates the rich_description column for an existing screen.
        """

        if self.__readonly:
            return

        await self.__initialize()

        async with aiosqlite.connect(self.__path) as db:
            await db.execute(
                "UPDATE screens SET rich_description = ? WHERE visual_hash = ?",
                (rich_description, visual_hash),
            )
            await db.commit()

    async def mark_exhausted(self, visual_hash: str) -> None:
        """
        Flags a screen as fully explored so a later run can skip re-scanning it.
        """

        if self.__readonly:
            return

        await self.__initialize()

        async with aiosqlite.connect(self.__path) as db:
            await db.execute(
                "UPDATE screens SET exhausted = 1 WHERE visual_hash = ?", (visual_hash,)
            )
            await db.commit()

    async def set_relevance(self, visual_hash: str, relevance: str) -> None:
        """
        Persists how a screen relates to the focus so a later run keeps focus-awareness.
        """

        if self.__readonly:
            return

        await self.__initialize()

        async with aiosqlite.connect(self.__path) as db:
            await db.execute(
                "UPDATE screens SET relevance = ? WHERE visual_hash = ?",
                (relevance, visual_hash),
            )
            await db.commit()

    async def store_experience(self, visual_hash: str, action: Action, success: bool) -> None:
        """
        Stores an action outcome for a screen.
        """

        if self.__readonly:
            return

        await self.__initialize()

        async with aiosqlite.connect(self.__path) as db:
            await db.execute(
                "INSERT INTO experience "
                "(visual_hash, action_json, success, rationale, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
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
        Retrieves the description, recent experiences, and transitions for a screen.
        """

        await self.__initialize()
        knowledge: Dict[str, Any] = {
            "description": None,
            "previous_actions": [],
            "transitions": [],
            "visit_count": 0,
        }

        async with aiosqlite.connect(self.__path) as db:
            async with db.execute(
                "SELECT activity, description, visit_count FROM screens WHERE visual_hash = ?",
                (visual_hash,),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    activity = str(row[0] or "")
                    if activity.split("/")[0] in LAUNCHER_PACKAGES:
                        return knowledge
                    knowledge["description"] = row[1]
                    knowledge["visit_count"] = row[2]

            async with db.execute(
                "SELECT action_json, success FROM experience "
                "WHERE visual_hash = ? ORDER BY timestamp DESC LIMIT 5",
                (visual_hash,),
            ) as cursor:
                async for row in cursor:
                    try:
                        data = json.loads(row[0])
                    except (json.JSONDecodeError, AttributeError):
                        continue
                    knowledge["previous_actions"].append(
                        {
                            "action": data.get("action_type"),
                            "target": data.get("target") or "element",
                            "success": bool(row[1]),
                        }
                    )

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
        self, source_hash: str, action: Action, destination_hash: str
    ) -> None:
        """
        Upserts a transition edge, incrementing count on repeats.
        """

        if self.__readonly:
            return

        await self.__initialize()
        now = int(time.time())

        action_type = action.action_type.value
        action_target = action.natural_language_target or action.target or ""
        coord_bucket = action.bounds.coord_bucket() if action.bounds else None
        coord_region = action.region
        element_category = action.element_category

        async with aiosqlite.connect(self.__path) as db:
            await db.execute(
                "INSERT INTO transitions "
                "(source_hash, destination_hash, action_type, action_target, coord_bucket, "
                " coord_region, element_category, action_json, count, first_seen, last_seen) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?) "
                "ON CONFLICT(source_hash, action_type, action_target) DO UPDATE SET "
                "destination_hash = excluded.destination_hash, "
                "coord_bucket = COALESCE(excluded.coord_bucket, transitions.coord_bucket), "
                "coord_region = COALESCE(excluded.coord_region, transitions.coord_region), "
                "element_category = COALESCE(excluded.element_category, transitions.element_category), "
                "action_json = excluded.action_json, "
                "count = transitions.count + 1, "
                "last_seen = excluded.last_seen",
                (
                    source_hash,
                    destination_hash,
                    action_type,
                    action_target,
                    coord_bucket,
                    coord_region,
                    element_category,
                    action.model_dump_json(),
                    now,
                    now,
                ),
            )
            await db.commit()

    async def retrieve_transitions(self, visual_hash: str) -> List[Dict[str, Any]]:
        """
        Retrieves all outgoing transitions from a screen, most frequent first.
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
        Retrieves all screens with full metadata for graph hydration.
        """

        await self.__initialize()
        screens: List[Dict[str, Any]] = []

        async with (
            aiosqlite.connect(self.__path) as db,
            db.execute(
                "SELECT visual_hash, activity, description, first_seen, last_seen, "
                "visit_count, rich_description, activity_hash, xml_hash, interaction_hash, "
                "exhausted, relevance "
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
                        "rich_description": row[6],
                        "activity_hash": row[7],
                        "xml_hash": row[8],
                        "interaction_hash": row[9],
                        "exhausted": bool(row[10]),
                        "relevance": row[11],
                    }
                )

        return screens

    async def get_all_transitions(self) -> List[Dict[str, Any]]:
        """
        Retrieves all transition edges for graph hydration.
        """

        await self.__initialize()
        transitions: List[Dict[str, Any]] = []

        async with (
            aiosqlite.connect(self.__path) as db,
            db.execute(
                "SELECT source_hash, destination_hash, action_type, action_target, coord_bucket, "
                "coord_region, element_category, count, first_seen, last_seen "
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
                        "coord_bucket": row[4],
                        "coord_region": row[5],
                        "element_category": row[6],
                        "count": row[7],
                        "first_seen": row[8],
                        "last_seen": row[9],
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
            async with db.execute(
                "SELECT visual_hash, activity, description, visit_count "
                "FROM screens ORDER BY last_seen DESC"
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

            async with db.execute("SELECT COUNT(*) FROM experience") as cursor:
                experience_row = await cursor.fetchone()
                summary["experience_count"] = experience_row[0] if experience_row else 0

            async with db.execute("SELECT COUNT(*) FROM transitions") as cursor:
                transition_row = await cursor.fetchone()
                summary["transition_count"] = transition_row[0] if transition_row else 0

        return summary
