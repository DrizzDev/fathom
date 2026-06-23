"""
SQLite-backed defect repository sharing the knowledge database.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Iterable, List

import aiosqlite

from fathom.interfaces.defect import DefectRepositoryPort
from fathom.schemas.defect import Defect


class SqliteDefectRepository(DefectRepositoryPort):
    """
    Persists defects in the knowledge database, deduplicating by signature.
    """

    def __init__(self, *, database_path: Path) -> None:
        self.__path = database_path
        self.__path.parent.mkdir(parents=True, exist_ok=True)
        self.__initialized = False

    async def __initialize(self) -> None:
        """
        Creates the defects table when absent.
        """

        if self.__initialized:
            return

        async with aiosqlite.connect(self.__path) as db:
            await db.execute(
                "CREATE TABLE IF NOT EXISTS defects ("
                "session TEXT NOT NULL, "
                "signature TEXT NOT NULL, "
                "screen TEXT NOT NULL, "
                "severity TEXT NOT NULL, "
                "occurrence INTEGER NOT NULL DEFAULT 1, "
                "payload TEXT NOT NULL, "
                "PRIMARY KEY (session, signature)"
                ")"
            )
            await db.commit()

        self.__initialized = True

    async def record(self, *, session: str, defect: Defect) -> None:
        """
        Persists one defect, incrementing its occurrence on a repeat signature.
        """

        await self.__initialize()

        async with aiosqlite.connect(self.__path) as db:
            await db.execute(
                "INSERT INTO defects "
                "(session, signature, screen, severity, occurrence, payload) "
                "VALUES (?, ?, ?, ?, 1, ?) "
                "ON CONFLICT(session, signature) DO UPDATE SET occurrence = occurrence + 1",
                (
                    session,
                    defect.signature,
                    defect.evidence.screen,
                    defect.severity.value,
                    defect.model_dump_json(),
                ),
            )
            await db.commit()

    async def for_screen(self, *, session: str, screen: str) -> List[Defect]:
        """
        Returns the defects recorded for one screen.
        """

        await self.__initialize()

        async with (
            aiosqlite.connect(self.__path) as db,
            db.execute(
                "SELECT payload, occurrence FROM defects WHERE session = ? AND screen = ?",
                (session, screen),
            ) as cursor,
        ):
            rows = await cursor.fetchall()

        return self.__to_defects(rows=rows)

    async def for_run(self, *, session: str) -> List[Defect]:
        """
        Returns every defect recorded for the run.
        """

        await self.__initialize()

        async with (
            aiosqlite.connect(self.__path) as db,
            db.execute(
                "SELECT payload, occurrence FROM defects WHERE session = ?",
                (session,),
            ) as cursor,
        ):
            rows = await cursor.fetchall()

        return self.__to_defects(rows=rows)

    @staticmethod
    def __to_defects(*, rows: Iterable[aiosqlite.Row]) -> List[Defect]:
        """
        Rehydrates persisted rows into defects with their stored occurrence count.
        """

        return [
            Defect.model_validate_json(row[0]).model_copy(update={"occurrence": row[1]})
            for row in rows
        ]
