"""
Pins for the per-workflow SQLite checkpoint store path layout and sanitization rules.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from fathom.adapters.checkpoint.sqlite import SqliteCheckpointSweeper
from fathom.schemas.checkpoint import SqliteCheckpointPolicy


class SqliteCheckpointSweeperTest(unittest.IsolatedAsyncioTestCase):
    """
    The sweeper must remove only orphaned per-workflow files and preserve the legacy shared file.
    """

    def setUp(self) -> None:
        """
        Reset the throttle clock so each test sees a fresh sweeper window.
        """

        setattr(SqliteCheckpointSweeper, "_SqliteCheckpointSweeper__last_swept_at", 0.0)

    async def test_sweep_removes_aged_per_workflow_files_and_sidecars(self) -> None:
        """
        Files older than sweep_age are removed along with their -wal and -shm sidecars.
        """

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            stale_db = directory / "checkpoints__workflow-stale.db"
            stale_wal = directory / "checkpoints__workflow-stale.db-wal"
            stale_shm = directory / "checkpoints__workflow-stale.db-shm"
            for path in (stale_db, stale_wal, stale_shm):
                path.write_bytes(b"")
                old = time.time() - 7200
                __import__("os").utime(path, (old, old))

            fresh_db = directory / "checkpoints__workflow-fresh.db"
            fresh_db.write_bytes(b"")

            sweeper = SqliteCheckpointSweeper(
                directory=directory,
                sweep_age=3600,
                sweep_min_interval=0,
            )

            removed = sweeper.sweep()

            self.assertIn("workflow-stale", removed)
            self.assertFalse(stale_db.exists())
            self.assertFalse(stale_wal.exists())
            self.assertFalse(stale_shm.exists())
            self.assertTrue(fresh_db.exists())

    async def test_sweep_preserves_legacy_shared_file(self) -> None:
        """
        The legacy shared checkpoints.db must never be auto-deleted by the sweeper.
        """

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            legacy = directory / "checkpoints.db"
            legacy.write_bytes(b"")
            old = time.time() - 7200
            __import__("os").utime(legacy, (old, old))

            sweeper = SqliteCheckpointSweeper(
                directory=directory,
                sweep_age=3600,
                sweep_min_interval=0,
            )

            removed = sweeper.sweep()

            self.assertEqual(removed, [])
            self.assertTrue(legacy.exists())

    async def test_throttle_skips_consecutive_invocations(self) -> None:
        """
        A sweep called within the throttle interval must skip the filesystem scan.
        """

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            stale = directory / "checkpoints__workflow-x.db"
            stale.write_bytes(b"")
            old = time.time() - 7200
            __import__("os").utime(stale, (old, old))

            sweeper = SqliteCheckpointSweeper(
                directory=directory,
                sweep_age=3600,
                sweep_min_interval=600,
            )

            first = sweeper.sweep()
            self.assertIn("workflow-x", first)

            stale.write_bytes(b"")
            __import__("os").utime(stale, (old, old))

            second = sweeper.sweep()
            self.assertEqual(second, [])
            self.assertTrue(stale.exists())


class SqliteCheckpointPolicyDefaultsTest(unittest.TestCase):
    """
    Defaults on SqliteCheckpointPolicy must remain conservative for the production fix.
    """

    def test_busy_timeout_default_is_fail_fast(self) -> None:
        """
        Default busy_timeout must remain low to prevent runaway lock waits.
        """

        policy = SqliteCheckpointPolicy()

        self.assertLessEqual(policy.busy_timeout, 10_000)
