from __future__ import annotations

import re
import time
from contextlib import asynccontextmanager
from logging import getLogger
from typing import TYPE_CHECKING, Any, AsyncIterator, ClassVar, Final, List, Optional

import aiosqlite

from fathom.core.exceptions import CheckpointStoreError
from fathom.interfaces.checkpoint import LangGraphCheckpointer
from fathom.schemas.checkpoint import SqliteCheckpointPolicy

if TYPE_CHECKING:
    from pathlib import Path

logger = getLogger(__name__)


class _WorkflowIdentifierSanitizer:
    """
    Convert raw workflow identifiers into filesystem-safe stems.
    """

    __DEFAULT: Final[str] = "default"
    __PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_-]")

    @classmethod
    def sanitize(cls, *, workflow_id: Optional[str]) -> str:
        """
        Return a filesystem-safe identifier; empty or None resolves to the default sentinel.
        """

        if workflow_id is None:
            return cls.__DEFAULT

        stripped = workflow_id.strip()

        if not stripped:
            return cls.__DEFAULT

        return cls.__PATTERN.sub("_", stripped)


class SqliteCheckpointSweeper:
    """
    Throttled defensive backstop that deletes orphaned per-workflow checkpoint files.
    """

    __FILENAME_SUFFIX: ClassVar[str] = ".db"
    __FILENAME_PREFIX: ClassVar[str] = "checkpoints__"
    __LEGACY_SHARED_FILENAME: ClassVar[str] = "checkpoints.db"
    __SIDECAR_SUFFIXES: ClassVar[tuple[str, ...]] = (".db", ".db-wal", ".db-shm")

    __last_swept_at: ClassVar[float] = 0.0

    def __init__(
        self,
        *,
        directory: Path,
        sweep_age: int,
        sweep_min_interval: int,
    ) -> None:
        """
        Bind the directory to scan, file age threshold, and minimum invocation interval.
        """

        self.__directory = directory
        self.__sweep_age = sweep_age
        self.__sweep_min_interval = sweep_min_interval

    def sweep(self) -> list[str]:
        """
        Sweep eligible orphaned files when throttle interval has elapsed; return removed workflow identifiers.
        """

        now = time.time()
        elapsed_since_last = now - SqliteCheckpointSweeper.__last_swept_at

        if elapsed_since_last < self.__sweep_min_interval:
            logger.info(
                "checkpoint sweep skipped",
                extra={
                    "reason": "throttled",
                    "elapsed": elapsed_since_last,
                    "min_interval": self.__sweep_min_interval,
                    "event": "fathom.checkpoint.sweep.skipped",
                },
            )
            return []

        SqliteCheckpointSweeper.__last_swept_at = now

        if not self.__directory.exists():
            logger.info(
                "checkpoint sweep completed; directory absent",
                extra={
                    "removed.count": 0,
                    "directory": str(self.__directory),
                    "event": "fathom.checkpoint.sweep.completed",
                },
            )
            return []

        removed: List[str] = []
        threshold = now - self.__sweep_age

        for path in self.__directory.glob(
            f"{SqliteCheckpointSweeper.__FILENAME_PREFIX}*{SqliteCheckpointSweeper.__FILENAME_SUFFIX}"
        ):
            if path.name == SqliteCheckpointSweeper.__LEGACY_SHARED_FILENAME:
                continue

            try:
                if path.stat().st_mtime < threshold:
                    identifier = path.stem.removeprefix(SqliteCheckpointSweeper.__FILENAME_PREFIX)
                    self.__remove_sidecars(stem=path.stem)
                    removed.append(identifier)
            except OSError as exception:
                logger.warning(
                    "checkpoint sweep failed for %s: %s",
                    path,
                    exception,
                    extra={
                        "checkpoint.path": str(path),
                        "checkpoint.identifier": path.stem,
                        "exception.message": str(exception),
                        "event": "fathom.checkpoint.sweep.failed",
                        "exception.type": type(exception).__name__,
                    },
                )

        logger.info(
            "checkpoint sweep completed",
            extra={
                "removed.count": len(removed),
                "directory": str(self.__directory),
                "event": "fathom.checkpoint.sweep.completed",
            },
        )

        return removed

    def __remove_sidecars(self, *, stem: str) -> None:
        """
        Unlink the primary .db plus any -wal and -shm sidecar files for one workflow.
        """

        for suffix in SqliteCheckpointSweeper.__SIDECAR_SUFFIXES:
            (self.__directory / f"{stem}{suffix}").unlink(missing_ok=True)


class SqliteCheckpointStore:
    """
    Per-workflow SQLite-backed LangGraph checkpoint store with fail-fast pragmas.
    """

    __FILENAME_SUFFIX: ClassVar[str] = ".db"
    __FILENAME_PREFIX: ClassVar[str] = "checkpoints__"
    __SIDECAR_SUFFIXES: ClassVar[tuple[str, ...]] = (".db", ".db-wal", ".db-shm")

    def __init__(
        self,
        *,
        directory: Path,
        policy: SqliteCheckpointPolicy,
        serde: Optional[Any] = None,
    ) -> None:
        """
        Bind the checkpoint directory, operational policy, and optional LangGraph serializer.
        """

        self.__directory = directory
        self.__policy = policy
        self.__serde = serde
        self.__directory.mkdir(parents=True, exist_ok=True)
        self.__sweeper = SqliteCheckpointSweeper(
            directory=self.__directory,
            sweep_age=self.__policy.sweep_age,
            sweep_min_interval=self.__policy.sweep_min_interval,
        )

    @asynccontextmanager
    async def open(self, *, workflow_id: str) -> AsyncIterator[LangGraphCheckpointer]:
        """
        Open a per-workflow aiosqlite connection, apply pragmas, and yield an AsyncSqliteSaver bound to it.
        """

        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        identifier = _WorkflowIdentifierSanitizer.sanitize(workflow_id=workflow_id)
        path = self.__path_for(identifier=identifier)

        try:
            async with aiosqlite.connect(str(path)) as connection:
                await self.__apply_pragmas(connection=connection)
                logger.info(
                    "checkpoint store opened",
                    extra={
                        "workflow.id": workflow_id,
                        "checkpoint.path": str(path),
                        "checkpoint.backend": "sqlite",
                        "checkpoint.identifier": identifier,
                        "event": "fathom.checkpoint.opened",
                        "checkpoint.busy_timeout": self.__policy.busy_timeout,
                    },
                )
                yield self.__build_saver(saver_class=AsyncSqliteSaver, connection=connection)

        except Exception as exception:
            logger.error(
                "checkpoint store open failed: %s",
                exception,
                exc_info=True,
                extra={
                    "workflow.id": workflow_id,
                    "checkpoint.path": str(path),
                    "checkpoint.backend": "sqlite",
                    "exception.message": str(exception),
                    "event": "fathom.checkpoint.open.failed",
                    "exception.type": type(exception).__name__,
                },
            )
            raise CheckpointStoreError(
                f"could not open SQLite checkpoint store for workflow '{workflow_id}': {exception}"
            ) from exception

    async def discard(self, *, workflow_id: str) -> None:
        """
        Unlink the per-workflow checkpoint database file and its sidecars for a completed workflow.
        """

        identifier = _WorkflowIdentifierSanitizer.sanitize(workflow_id=workflow_id)

        logger.info(
            "checkpoint discard started",
            extra={
                "workflow.id": workflow_id,
                "checkpoint.backend": "sqlite",
                "checkpoint.identifier": identifier,
                "event": "fathom.checkpoint.discard.started",
            },
        )

        removed_paths: List[str] = []
        stem = f"{SqliteCheckpointStore.__FILENAME_PREFIX}{identifier}"

        try:
            for suffix in SqliteCheckpointStore.__SIDECAR_SUFFIXES:
                path = self.__directory / f"{stem}{suffix}"
                if path.exists():
                    path.unlink(missing_ok=True)
                    removed_paths.append(str(path))

        except OSError as exception:
            logger.warning(
                "checkpoint discard failed: %s",
                exception,
                extra={
                    "workflow.id": workflow_id,
                    "checkpoint.backend": "sqlite",
                    "checkpoint.identifier": identifier,
                    "exception.message": str(exception),
                    "exception.type": type(exception).__name__,
                    "event": "fathom.checkpoint.discard.failed",
                },
            )
            return

        logger.info(
            "checkpoint discard completed",
            extra={
                "workflow.id": workflow_id,
                "checkpoint.backend": "sqlite",
                "removed.paths": removed_paths,
                "checkpoint.identifier": identifier,
                "event": "fathom.checkpoint.discard.completed",
            },
        )

    async def sweep_stale(self) -> list[str]:
        """
        Sweep orphaned checkpoint files older than the configured retention; return removed workflow identifiers.
        """

        return self.__sweeper.sweep()

    async def __apply_pragmas(self, *, connection: Any) -> None:
        """
        Apply defensive SQLite pragmas on a freshly opened checkpoint connection.
        """

        await connection.execute(f"PRAGMA busy_timeout = {self.__policy.busy_timeout}")
        await connection.commit()

    def __build_saver(self, *, saver_class: Any, connection: Any) -> Any:
        """
        Instantiate the LangGraph saver with the injected serde when provided.
        """

        if self.__serde is None:
            return saver_class(conn=connection)

        return saver_class(conn=connection, serde=self.__serde)

    def __path_for(self, *, identifier: str) -> Path:
        """
        Resolve the on-disk path for a workflow's primary checkpoint database file.
        """

        return (
            self.__directory
            / f"{SqliteCheckpointStore.__FILENAME_PREFIX}{identifier}{SqliteCheckpointStore.__FILENAME_SUFFIX}"
        )
