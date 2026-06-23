from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fathom.constants.defect import DefectSignal, DefectSource
from fathom.infrastructure.defect.sqlite import SqliteDefectRepository
from fathom.schemas.defect import Defect, DefectEvidence


class SqliteDefectRepositoryTest(unittest.IsolatedAsyncioTestCase):
    """
    Verifies persistence, signature dedup, and per-screen/per-run queries.
    """

    @staticmethod
    def __defect(
        *,
        signal: DefectSignal = DefectSignal.DEAD_TAP,
        screen: str = "home",
        excerpt: str = "Buy",
    ) -> Defect:
        """
        Builds an inline defect anchored to a screen and control.
        """

        return Defect.from_signal(
            signal=signal,
            source=DefectSource.INLINE,
            summary="x",
            evidence=DefectEvidence(screen=screen, excerpt=excerpt),
        )

    async def test_record_and_for_run(self) -> None:
        """
        A recorded defect is returned for the run with occurrence one.
        """

        with tempfile.TemporaryDirectory() as tmp:
            repository = SqliteDefectRepository(database_path=Path(tmp) / "knowledge.db")

            await repository.record(session="wf", defect=self.__defect())

            run = await repository.for_run(session="wf")
            self.assertEqual(len(run), 1)
            self.assertEqual(run[0].signal, DefectSignal.DEAD_TAP)
            self.assertEqual(run[0].occurrence, 1)

    async def test_duplicate_signature_increments_occurrence(self) -> None:
        """
        Re-recording the same signature collapses to one row with a higher count.
        """

        with tempfile.TemporaryDirectory() as tmp:
            repository = SqliteDefectRepository(database_path=Path(tmp) / "knowledge.db")

            await repository.record(session="wf", defect=self.__defect())
            await repository.record(session="wf", defect=self.__defect())

            run = await repository.for_run(session="wf")
            self.assertEqual(len(run), 1)
            self.assertEqual(run[0].occurrence, 2)

    async def test_distinct_signatures_are_separate_rows(self) -> None:
        """
        The same signal on different controls is two distinct defects.
        """

        with tempfile.TemporaryDirectory() as tmp:
            repository = SqliteDefectRepository(database_path=Path(tmp) / "knowledge.db")

            await repository.record(session="wf", defect=self.__defect(excerpt="Buy"))
            await repository.record(session="wf", defect=self.__defect(excerpt="Cancel"))

            self.assertEqual(len(await repository.for_run(session="wf")), 2)

    async def test_for_screen_filters_by_screen(self) -> None:
        """
        for_screen returns only the defects on that screen.
        """

        with tempfile.TemporaryDirectory() as tmp:
            repository = SqliteDefectRepository(database_path=Path(tmp) / "knowledge.db")

            await repository.record(session="wf", defect=self.__defect(screen="home"))
            await repository.record(session="wf", defect=self.__defect(screen="cart"))

            home = await repository.for_screen(session="wf", screen="home")
            self.assertEqual([defect.evidence.screen for defect in home], ["home"])

    async def test_sessions_are_isolated(self) -> None:
        """
        Defects from one run do not leak into another.
        """

        with tempfile.TemporaryDirectory() as tmp:
            repository = SqliteDefectRepository(database_path=Path(tmp) / "knowledge.db")

            await repository.record(session="wf-1", defect=self.__defect())
            await repository.record(session="wf-2", defect=self.__defect())

            self.assertEqual(len(await repository.for_run(session="wf-1")), 1)
            self.assertEqual(len(await repository.for_run(session="wf-2")), 1)


if __name__ == "__main__":
    unittest.main()
