from __future__ import annotations

import asyncio
import time
from logging import getLogger
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from fathom.constants.flow import IssueCode
from fathom.constants.generation import (
    BASELINE_METADATA_FILENAME,
    BASELINE_SCRIPT_FILENAME,
    ScriptSource,
    ScriptStatus,
)
from fathom.core.services.generation.baseline import BaselineScriptService
from fathom.interfaces.evidence import EvidenceSource
from fathom.interfaces.paths import HistoryPaths
from fathom.interfaces.script import ScriptRefresher
from fathom.schemas.flow import Issue, RunObjective
from fathom.schemas.generation import BaselineArtifact, ScriptFileMetadata

if TYPE_CHECKING:
    from pathlib import Path

logger = getLogger(__name__)


class BaselineRefresher(ScriptRefresher):
    """
    Coalesced, exception-safe background refresher that persists the deterministic baseline artifact atomically.
    """

    def __init__(
        self,
        *,
        source: EvidenceSource,
        path_manager: HistoryPaths,
        baseline: BaselineScriptService,
    ) -> None:
        """
        Bind the evidence source, the baseline builder, and the history path resolver.
        """

        self.__source = source
        self.__baseline = baseline
        self.__path_manager = path_manager

        self.__task: Optional[asyncio.Task[None]] = None
        self.__pending: Optional[Tuple[str, RunObjective]] = None

    def schedule(self, *, execution_id: str, objective: RunObjective) -> None:
        """
        Record the latest refresh request and start a single worker if none is running; never blocks.
        """

        self.__pending = (execution_id, objective)

        if self.__task is not None and not self.__task.done():
            logger.info(
                "baseline refresh coalesced into in-flight worker",
                extra={
                    "event": "script.baseline.refresh.coalesced",
                    "execution.id": execution_id,
                    "script.package": objective.package,
                },
            )
            return

        try:
            self.__task = asyncio.create_task(self.__worker())
        except RuntimeError as exception:
            logger.warning(
                "baseline refresh could not be scheduled; left pending for the next drain",
                extra={
                    "event": "script.baseline.refresh.schedule_failed",
                    "execution.id": execution_id,
                    "script.package": objective.package,
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                },
            )

    async def drain(self) -> None:
        """
        Run any pending refresh that never started, then await the in-flight one, so the latest artifact lands.
        """

        if self.__pending is not None and (self.__task is None or self.__task.done()):
            self.__task = asyncio.create_task(self.__worker())

        task = self.__task
        awaited = task is not None and not task.done()
        if task is not None and not task.done():
            await task

        logger.info(
            "baseline refresh drained",
            extra={"event": "script.baseline.refresh.drained", "script.awaited": awaited},
        )

    async def __worker(self) -> None:
        """
        Process pending requests latest-wins until none remain, so bursts collapse to one trailing refresh.
        """

        while self.__pending is not None:
            execution_id, objective = self.__pending
            self.__pending = None
            await self.__refresh(execution_id=execution_id, objective=objective)

    async def __refresh(self, *, execution_id: str, objective: RunObjective) -> None:
        """
        Build and persist the baseline; any unexpected failure is captured as a failed artifact, never raised.
        """

        started = time.perf_counter()
        logger.info(
            "baseline refresh started",
            extra={
                "event": "script.baseline.refresh.started",
                "execution.id": execution_id,
                "script.package": objective.package,
                "script.source": ScriptSource.BASELINE.value,
            },
        )

        try:
            evidence = await self.__source.read(execution_id=execution_id, objective=objective)
            artifact = self.__baseline.build(evidence=evidence)
        except Exception as exception:  # noqa: BLE001 — background refresh must never raise into the caller
            logger.warning(
                "baseline refresh failed before an artifact was built",
                extra={
                    "event": "script.baseline.refresh.failed",
                    "execution.id": execution_id,
                    "script.source": ScriptSource.BASELINE.value,
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                    "duration.ms": round((time.perf_counter() - started) * 1000, 3),
                },
            )
            artifact = self.__failure(exception=exception)
            self.__persist(
                execution_id=execution_id, package_name=objective.package, artifact=artifact
            )
            return

        self.__log_built(execution_id=execution_id, artifact=artifact, started=started)
        self.__persist(execution_id=execution_id, package_name=objective.package, artifact=artifact)

    def __log_built(self, *, execution_id: str, artifact: BaselineArtifact, started: float) -> None:
        """
        Record the baseline build outcome with issue codes, skipped reasons, and line count.
        """

        metadata = artifact.metadata
        skipped: Dict[str, int] = {}

        for step in metadata.skipped:
            skipped[step.reason.value] = skipped.get(step.reason.value, 0) + 1

        context = {
            "execution.id": execution_id,
            "script.skipped_reasons": skipped,
            "script.status": metadata.status.value,
            "script.source": ScriptSource.BASELINE.value,
            "script.issue_count": len(metadata.issues),
            "script.skipped_count": len(metadata.skipped),
            "script.line_count": len((artifact.text or "").splitlines()),
            "script.issue_codes": [issue.code.value for issue in metadata.issues],
            "duration.ms": round((time.perf_counter() - started) * 1000, 3),
        }

        if metadata.status is ScriptStatus.GENERATED:
            logger.info(
                "baseline refresh generated a script",
                extra={"event": "script.baseline.refresh.generated", **context},
            )
        else:
            logger.warning(
                "baseline refresh build failed the fidelity or syntax gate",
                extra={"event": "script.baseline.refresh.failed", **context},
            )

    @staticmethod
    def __failure(*, exception: Exception) -> BaselineArtifact:
        """
        Build a failed baseline artifact describing an unexpected refresh error.
        """

        issue = Issue(
            code=IssueCode.UNRENDERABLE_VALUE,
            message=f"Baseline refresh raised before a script could be produced: {exception}",
        )
        return BaselineArtifact(
            text=None,
            metadata=ScriptFileMetadata(
                source=ScriptSource.BASELINE, status=ScriptStatus.FAILED, issues=(issue,)
            ),
        )

    def __persist(
        self, *, execution_id: str, package_name: str, artifact: BaselineArtifact
    ) -> None:
        """
        Atomically write the metadata sidecar and, when generated, the baseline script; never raises.
        """

        try:
            directory = self.__path_manager.get_history_directory(session_id=execution_id)
            script_path = self.__scoped(
                directory=directory,
                package_name=package_name,
                filename=BASELINE_SCRIPT_FILENAME,
            )
            metadata_path = self.__scoped(
                directory=directory,
                package_name=package_name,
                filename=BASELINE_METADATA_FILENAME,
            )

            if artifact.text is not None:
                self.__write(path=script_path, content=artifact.text)

            elif script_path.exists():
                script_path.unlink()

            self.__write(
                path=metadata_path,
                content=artifact.metadata.model_dump_json(),
            )
        except Exception as exception:  # noqa: BLE001 — best-effort; must not crash the loop task
            logger.warning(
                "failed to persist baseline artifact",
                extra={
                    "event": "script.baseline.refresh.persist_failed",
                    "execution.id": execution_id,
                    "script.package": package_name,
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                },
            )
            return

        logger.info(
            "baseline artifact persisted",
            extra={
                "event": "script.baseline.refresh.persisted",
                "execution.id": execution_id,
                "script.package": package_name,
                "script.metadata_file": metadata_path.name,
                "script.status": artifact.metadata.status.value,
                "script.script_file": script_path.name if artifact.text is not None else None,
            },
        )

    @staticmethod
    def __scoped(*, directory: Path, filename: str, package_name: str) -> Path:
        """
        Scope an artifact filename by package so multi-package sessions never collide.
        """

        stem, _, ext = filename.rpartition(".")
        scoped = f"{stem}__{package_name}.{ext}" if stem and ext else f"{filename}__{package_name}"

        return directory / scoped

    @staticmethod
    def __write(*, path: Path, content: str) -> None:
        """
        Write the content to a temp file and rename it into place so readers never see a partial artifact.
        """

        temporary = path.with_suffix(f"{path.suffix}.tmp")
        with temporary.open(mode="w") as handle:
            handle.write(content)

        temporary.replace(path)
