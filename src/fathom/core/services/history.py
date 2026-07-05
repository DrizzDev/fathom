from __future__ import annotations

import asyncio
import json
import time
from logging import getLogger
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

import yaml

from fathom.base.timing import time_it
from fathom.constants.flow import IssueCode
from fathom.constants.generation import (
    BASELINE_METADATA_FILENAME,
    BASELINE_SCRIPT_FILENAME,
    COMPLETION_ASSERTIONS_FILENAME,
    ScriptArtifactMode,
    ScriptArtifactScope,
    ScriptSource,
    ScriptStatus,
)
from fathom.core.artifact.pipeline import ArtifactPipeline
from fathom.interfaces.storage import StoragePort
from fathom.schemas.artifact import ArtifactRecord, ScriptPayload
from fathom.schemas.flow import CompletionAssertion, Issue, RunObjective
from fathom.schemas.generation import (
    BaselineArtifact,
    GenerationResult,
    ScriptFileMetadata,
)
from fathom.schemas.steps import StepGoal, StepResult

if TYPE_CHECKING:
    from pathlib import Path

    from fathom.interfaces.authoring import AuthoringScheduler
    from fathom.interfaces.paths import HistoryPaths
    from fathom.interfaces.script import ScriptRefresher


logger = getLogger(__name__)


class HistoryService:
    """
    Service responsible for persisting execution history and generating scripts.
    All outputs are saved to assets/history/{date}/{package}/{session}/ directory.
    """

    __EXECUTION = ScriptArtifactScope.EXECUTION.value

    def __init__(
        self,
        execution_id: str,
        package_name: str,
        path_manager: HistoryPaths,
        *,
        storage: Optional[StoragePort] = None,
        pipeline: Optional[ArtifactPipeline] = None,
        refresher: Optional["ScriptRefresher"] = None,
        authoring: Optional["AuthoringScheduler"] = None,
        artifact_mode: ScriptArtifactMode = ScriptArtifactMode.NORMAL,
    ) -> None:
        self.__execution_id = execution_id
        self.__package_name = package_name

        self.__storage = storage
        self.__path_manager = path_manager

        self.__pipeline = pipeline
        self.__refresher = refresher
        self.__authoring = authoring
        self.__artifact_mode = artifact_mode

        self.__background_tasks: Set[asyncio.Task[Any]] = set()
        self.__persistence_tasks: Set[asyncio.Task[Any]] = set()
        self.__persistence_chain: Optional[asyncio.Task[None]] = None

    def __log_failure(
        self,
        *,
        event: str,
        message: str,
        exception: BaseException,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Emit a structured failure log carrying the run id and the exception type and message.
        """

        logger.warning(
            message,
            extra={
                "event": event,
                "execution.id": self.__execution_id,
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
                **(context or {}),
            },
        )

    def __fire_and_forget(self, coroutine: Any) -> None:
        """
        Schedules a coroutine as a background task.
        """

        try:
            task = asyncio.create_task(coroutine)
            self.__background_tasks.add(task)
            task.add_done_callback(self.__background_tasks.discard)
        except Exception as exception:
            self.__log_failure(
                exception=exception,
                event="script.history.background_task_failed",
                message="fire-and-forget storage task could not be scheduled",
            )

    def enqueue_save_step(
        self,
        *,
        result: StepResult,
        intent: str = "",
        goal: Optional[StepGoal] = None,
        package_name: Optional[str] = None,
        execution_activity: Optional[str] = None,
        absolute_center: Optional[List[int]] = None,
        on_complete: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> None:
        """
        Queue ordered history persistence without blocking the step lifecycle.
        """

        async def __run_persistence() -> None:
            script_data = await self.save_step(
                goal=goal,
                result=result,
                intent=intent,
                package_name=package_name,
                absolute_center=absolute_center,
                execution_activity=execution_activity,
            )

            if on_complete and script_data:
                await on_complete(script_data)

        previous_task = self.__persistence_chain

        async def __run_serialized() -> None:
            if previous_task and previous_task is not asyncio.current_task():
                try:
                    await previous_task
                except Exception as exception:
                    self.__log_failure(
                        event="script.history.persistence_chain_failed",
                        message="previous history persistence task failed",
                        exception=exception,
                    )

            await __run_persistence()

        try:
            task = asyncio.create_task(__run_serialized())
            self.__persistence_chain = task
            self.__persistence_tasks.add(task)
            task.add_done_callback(self.__persistence_tasks.discard)
        except Exception as exception:
            self.__log_failure(
                event="script.history.persistence_enqueue_failed",
                message="history persistence task could not be queued",
                exception=exception,
            )

    @time_it(operation="history.flush_pending_operations")
    async def flush_pending_operations(self) -> None:
        """
        Wait for queued history persistence and artifact uploads to complete.
        """

        if self.__persistence_chain:
            try:
                await self.__persistence_chain
            finally:
                self.__persistence_chain = None

        if self.__refresher is not None:
            try:
                await self.__refresher.drain()
            except Exception as exception:
                self.__log_failure(
                    event="script.baseline.refresh.drain_failed",
                    message="baseline refresh drain failed during finalization",
                    exception=exception,
                )

        if self.__authoring is not None:
            try:
                await self.__authoring.drain()
            except Exception as exception:
                self.__log_failure(
                    exception=exception,
                    event="authoring.step.drain_failed",
                    message="step authoring drain failed during finalization",
                )

        if self.__background_tasks:
            background_results = await asyncio.gather(
                *tuple(self.__background_tasks),
                return_exceptions=True,
            )
            self.__log_background_failures(
                failures=background_results,
                category="history.background_storage",
            )

        if self.__persistence_tasks:
            persistence_results = await asyncio.gather(
                *tuple(self.__persistence_tasks),
                return_exceptions=True,
            )
            self.__log_background_failures(
                failures=persistence_results,
                category="history.persistence_queue",
            )

    async def drain_background_tasks(self) -> None:
        """
        Await all pending background tasks. Delegates to flush_pending_operations.
        """

        await self.flush_pending_operations()

    @time_it(operation="history.save_step")
    async def save_step(
        self,
        result: StepResult,
        *,
        intent: str = "",
        goal: Optional[StepGoal] = None,
        package_name: Optional[str] = None,
        execution_activity: Optional[str] = None,
        absolute_center: Optional[List[int]] = None,
    ) -> str:
        """
        Saves a single step result and updates associated artifact files.
        Returns the current script if already generated.
        """

        resolved_package_name = self.__resolve_package_name(package_name=package_name)
        history = self.__load_history(package_name=resolved_package_name)

        record = result.to_record(
            goal=goal,
            activity=resolved_package_name,
            absolute_center=absolute_center,
        ).model_dump()

        record["timestamp"] = int(time.time() * 1000)
        record["screen_changed"] = result.screen_changed

        # Tag with pre-action activity so script generation can ground launcher transitions.
        if execution_activity:
            record["execution_activity"] = execution_activity

        history["history"].append(record)

        if self.__artifact_mode is ScriptArtifactMode.DEBUG:
            await self.__save_json(data=history, package_name=resolved_package_name)
            await self.__save_yaml(history=history["history"], package_name=resolved_package_name)

        self.__append_execution_trace(record=record)

        logger.info(
            "script step persisted to history",
            extra={
                "event": "script.history.step_saved",
                "execution.id": self.__execution_id,
                "script.success": record.get("success"),
                "script.package": resolved_package_name,
                "script.step": record.get("step_number"),
                "script.execution_activity": execution_activity,
                "script.event_type": record.get("event_type"),
                "script.action_type": record.get("action_type"),
            },
        )

        self.__schedule_refresh(intent=intent, history=history["history"])
        self.__schedule_authoring(intent=intent, history=history["history"], record=record)

        return self.__read_existing_script(package_name=resolved_package_name)

    def __schedule_authoring(
        self,
        *,
        intent: str,
        record: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> None:
        """
        Schedule optional per-step authoring against the persisted execution trace.
        """

        if self.__authoring is None:
            return

        if not intent.strip() or not history:
            return

        package_name = self.__resolve_export_package_name(history=history)
        if not package_name.strip():
            return

        step_index = int(record.get("step_number", 0))

        self.__authoring.schedule_step(
            step_index=step_index,
            execution_id=self.__execution_id,
            objective=RunObjective(intent=intent, goal=intent, package=package_name),
        )

    def __schedule_refresh(self, *, intent: str, history: List[Dict[str, Any]]) -> None:
        """
        Schedule a non-blocking baseline refresh against the freshly persisted history.
        """

        if self.__refresher is None:
            self.__log_schedule_skipped(reason="no_refresher", package_name=None)
            return

        if not intent.strip():
            self.__log_schedule_skipped(reason="blank_intent", package_name=None)
            return

        if not history:
            self.__log_schedule_skipped(reason="no_execution_trace", package_name=None)
            return

        package_name = self.__resolve_export_package_name(history=history)
        if not package_name.strip():
            self.__log_schedule_skipped(reason="unresolved_package", package_name=None)
            return

        self.__refresher.schedule(
            execution_id=self.__execution_id,
            objective=RunObjective(intent=intent, goal=intent, package=package_name),
        )
        logger.info(
            "baseline refresh scheduled",
            extra={
                "event": "script.baseline.refresh.scheduled",
                "script.package": package_name,
                "execution.id": self.__execution_id,
            },
        )

    def __log_schedule_skipped(self, *, reason: str, package_name: Optional[str]) -> None:
        """
        Record why a baseline refresh was not scheduled for this step.
        """

        logger.info(
            "baseline refresh schedule skipped",
            extra={
                "event": "script.baseline.refresh.schedule_skipped",
                "execution.id": self.__execution_id,
                "script.skip_reason": reason,
                "script.package": package_name,
            },
        )

    def __append_execution_trace(self, *, record: Dict[str, Any]) -> None:
        """
        Append one record to the single ordered execution-level trace (the script-gen source).
        """

        path = self.__execution_trace_path()
        data: Dict[str, Any] = {
            "execution_id": self.__execution_id,
            "history": [],
        }

        if path.exists():
            try:
                with path.open(mode="r") as handle:
                    data = json.load(fp=handle)
            except Exception as exception:  # nosec
                backup_path = path.with_suffix(f".corrupt.{int(time.time())}.json")
                self.__log_failure(
                    event="script.history.execution_trace_corrupt",
                    message="execution trace corrupted; preserving a backup",
                    exception=exception,
                    context={"script.backup": backup_path.name},
                )
                try:
                    path.replace(backup_path)
                except Exception as backup_exception:  # nosec
                    self.__log_failure(
                        event="script.history.execution_trace_backup_failed",
                        message="failed to preserve the corrupt execution trace",
                        exception=backup_exception,
                    )
                data = {
                    "execution_id": self.__execution_id,
                    "history": [],
                }

        data["history"].append(record)

        temporary_path = path.with_suffix(".tmp")

        try:
            with temporary_path.open(mode="w") as handle:
                handle.write(json.dumps(obj=data, indent=2))

            temporary_path.replace(path)
        except Exception as exception:
            logger.warning(
                "execution trace append failed",
                extra={
                    "event": "script.history.execution_trace_failed",
                    "execution.id": self.__execution_id,
                    "script.step": record.get("step_number"),
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                },
            )
            raise

        logger.info(
            "execution trace appended",
            extra={
                "event": "script.history.execution_trace_appended",
                "execution.id": self.__execution_id,
                "script.step": record.get("step_number"),
                "script.record_count": len(data["history"]),
            },
        )

    def __execution_trace_path(self) -> "Path":
        """
        Resolve the path of the single ordered execution trace, distinct from per-package files.
        """

        directory = self.__path_manager.get_history_directory(session_id=self.__execution_id)
        return directory / "history__execution.json"

    def __load_execution_history(self) -> List[Dict[str, Any]]:
        """
        Load the ordered execution trace records, or an empty list when absent or unreadable.
        """

        path = self.__execution_trace_path()

        if not path.exists():
            return []

        try:
            with path.open(mode="r") as handle:
                data: Dict[str, Any] = json.load(fp=handle)
        except Exception as exception:  # nosec
            self.__log_failure(
                event="script.history.execution_trace_unreadable",
                message="execution trace unreadable at finalization; treating as empty",
                exception=exception,
            )
            return []

        records: List[Dict[str, Any]] = data.get("history", [])
        return records

    @time_it(operation="history.load_history")
    def __load_history(self, *, package_name: str) -> Dict[str, Any]:
        """
        Loads existing history from the JSON artifact.
        """

        path = self.__get_history_file_path(package_name=package_name, filename="history.json")
        data: Dict[str, Any] = {
            "execution_id": self.__execution_id,
            "history": [],
        }

        if not path.exists() and package_name != self.__package_name:
            previous_path = self.__get_history_file_path(
                filename="history.json",
                package_name=self.__package_name,
            )
            if previous_path.exists():
                path = previous_path

        if path.exists():
            try:
                with path.open(mode="r") as handle:
                    data = json.load(fp=handle)
            except Exception as exception:  # nosec
                backup_path = path.with_suffix(f".corrupt.{int(time.time())}.json")
                self.__log_failure(
                    event="script.history.file_corrupt",
                    message="history file corrupted; preserving a backup",
                    exception=exception,
                    context={"script.package": package_name, "script.backup": backup_path.name},
                )
                try:
                    path.replace(backup_path)
                except Exception as backup_exception:  # nosec
                    self.__log_failure(
                        event="script.history.file_backup_failed",
                        message="failed to preserve the corrupt history file",
                        exception=backup_exception,
                        context={"script.package": package_name},
                    )

        return data

    @time_it(operation="history.save_json")
    async def __save_json(self, data: Dict[str, Any], *, package_name: str) -> None:
        """
        Writes history to structured JSON format.
        """

        path = self.__get_history_file_path(package_name=package_name, filename="history.json")

        json_data = json.dumps(obj=data, indent=2)
        with path.open(mode="w") as handle:
            handle.write(json_data)

        if self.__storage:
            self.__fire_and_forget(
                self.__storage.save(
                    data=json_data.encode("utf-8"),
                    metadata={
                        "category": "history",
                        "filename": "history.json",
                        "package_name": package_name,
                        "session_id": self.__execution_id,
                    },
                )
            )

    @time_it(operation="history.save_yaml")
    async def __save_yaml(self, history: List[Dict[str, Any]], *, package_name: str) -> None:
        """
        Generates a YAML representation of the execution.
        """

        path = self.__get_history_file_path(package_name=package_name, filename="history.yaml")
        steps = [
            self.__build_yaml_item(index=index, record=item)
            for index, item in enumerate(iterable=history, start=1)
        ]

        yaml_data = yaml.dump(indent=2, data=steps, sort_keys=False, default_flow_style=False)

        with path.open(mode="w") as handle:
            handle.write(yaml_data)

        if self.__storage:
            self.__fire_and_forget(
                self.__storage.save(
                    data=yaml_data.encode("utf-8"),
                    metadata={
                        "category": "history",
                        "filename": "history.yaml",
                        "package_name": package_name,
                        "session_id": self.__execution_id,
                    },
                )
            )

    def save_completion_assertions(self, *, assertions: Tuple[CompletionAssertion, ...]) -> None:
        """
        Persist terminal verifier assertions for script authoring evidence.
        """

        if not assertions:
            return

        path = self.__get_history_file_path(
            package_name=self.__EXECUTION,
            filename=COMPLETION_ASSERTIONS_FILENAME,
        )
        payload = [assertion.model_dump(mode="json") for assertion in assertions]

        with path.open(mode="w") as handle:
            handle.write(json.dumps(obj=payload, indent=2))

        logger.info(
            "completion assertions persisted",
            extra={
                "event": "script.completion.assertions.persisted",
                "execution.id": self.__execution_id,
                "assertion.count": len(assertions),
            },
        )

    @time_it(operation="history.read_baseline_outcome")
    async def read_baseline_outcome(self, *, step_number: int) -> BaselineArtifact:
        """
        Read the latest persisted baseline; promote a generated one to the canonical script, else report failure.
        """

        artifact_scope = self.__EXECUTION
        logger.info(
            "baseline read started",
            extra={
                "event": "script.baseline.read.started",
                "script.scope": artifact_scope,
                "execution.id": self.__execution_id,
            },
        )

        artifact = self.__read_baseline(artifact_scope=artifact_scope)

        if artifact is None:
            return self.__baseline_unavailable(
                message="No baseline script artifact was available at finalization."
            )

        logger.info(
            "baseline read completed",
            extra={
                "event": "script.baseline.read.completed",
                "script.scope": artifact_scope,
                "execution.id": self.__execution_id,
                "script.status": artifact.metadata.status.value,
                "script.issue_codes": [issue.code.value for issue in artifact.metadata.issues],
                "script.line_count": len((artifact.text or "").splitlines()),
            },
        )

        if artifact.metadata.status is ScriptStatus.GENERATED and (artifact.text or "").strip():
            promoted = await self.__promote_baseline(
                artifact=artifact, artifact_scope=artifact_scope, step_number=step_number
            )
            if promoted and self.__artifact_mode is ScriptArtifactMode.NORMAL:
                self.__cleanup_baseline(artifact_scope=artifact_scope)

        return artifact

    async def __promote_baseline(
        self, *, artifact: BaselineArtifact, artifact_scope: str, step_number: int
    ) -> bool:
        """
        Promote a generated baseline to the canonical script; promotion is best-effort and never fatal.
        """

        try:
            await self.__persist_script(
                step_number=step_number,
                package_name=artifact_scope,
                source=ScriptSource.BASELINE,
                result=GenerationResult(
                    text=artifact.text or "", attempts=1, review=artifact.metadata.review
                ),
            )
        except Exception as exception:  # noqa: BLE001 — promotion is best-effort; the event still ships
            logger.warning(
                "baseline promotion to canonical script failed",
                extra={
                    "event": "script.baseline.promote_failed",
                    "script.scope": artifact_scope,
                    "execution.id": self.__execution_id,
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                },
            )
            return False

        logger.info(
            "baseline promoted to canonical script",
            extra={
                "event": "script.baseline.promoted",
                "script.scope": artifact_scope,
                "execution.id": self.__execution_id,
                "script.line_count": len((artifact.text or "").splitlines()),
            },
        )
        return True

    def __cleanup_baseline(self, *, artifact_scope: str) -> None:
        """
        Remove baseline handoff artifacts after successful promotion in normal artifact mode.
        """

        for filename in (BASELINE_SCRIPT_FILENAME, BASELINE_METADATA_FILENAME):
            path = self.__get_history_file_path(package_name=artifact_scope, filename=filename)
            try:
                path.unlink(missing_ok=True)
            except Exception as exception:  # noqa: BLE001 — cleanup must not invalidate promotion
                self.__log_failure(
                    event="script.baseline.cleanup_failed",
                    message="baseline handoff artifact cleanup failed",
                    exception=exception,
                    context={"script.scope": artifact_scope, "script.path": path.name},
                )
                continue

        logger.info(
            "baseline handoff artifacts cleaned up",
            extra={
                "event": "script.baseline.cleaned",
                "script.scope": artifact_scope,
                "execution.id": self.__execution_id,
            },
        )

    def __read_baseline(self, *, artifact_scope: str) -> Optional[BaselineArtifact]:
        """
        Load the persisted baseline script and its metadata sidecar, or None when no metadata exists.
        """

        metadata_path = self.__get_history_file_path(
            package_name=artifact_scope, filename=BASELINE_METADATA_FILENAME
        )
        if not metadata_path.exists():
            logger.info(
                "baseline artifact missing at finalization",
                extra={
                    "event": "script.baseline.read.missing",
                    "script.scope": artifact_scope,
                    "execution.id": self.__execution_id,
                },
            )
            return None

        try:
            with metadata_path.open(mode="r") as handle:
                metadata = ScriptFileMetadata.model_validate_json(handle.read())
        except Exception as exception:  # nosec
            logger.warning(
                "baseline metadata unreadable at finalization",
                extra={
                    "event": "script.baseline.read.failed_metadata",
                    "execution.id": self.__execution_id,
                    "script.scope": artifact_scope,
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                },
            )
            return None

        text: Optional[str] = None
        text_path = self.__get_history_file_path(
            package_name=artifact_scope, filename=BASELINE_SCRIPT_FILENAME
        )

        if text_path.exists():
            try:
                with text_path.open(mode="r") as handle:
                    text = handle.read()
            except Exception as exception:  # nosec
                logger.warning(
                    "baseline script text unreadable at finalization",
                    extra={
                        "event": "script.baseline.read.failed_text",
                        "execution.id": self.__execution_id,
                        "script.scope": artifact_scope,
                        "exception.type": type(exception).__name__,
                        "exception.message": str(exception),
                    },
                )
                return self.__baseline_unavailable(
                    message="Baseline script text was unreadable at finalization."
                )

        if metadata.status is ScriptStatus.GENERATED and not (text or "").strip():
            logger.warning(
                "baseline metadata marked generated but script text was unavailable",
                extra={
                    "event": "script.baseline.read.missing_text",
                    "script.scope": artifact_scope,
                    "execution.id": self.__execution_id,
                },
            )
            return self.__baseline_unavailable(
                message="Baseline metadata was generated but script text was missing or empty."
            )

        return BaselineArtifact(text=text, metadata=metadata)

    @staticmethod
    def __baseline_unavailable(*, message: str) -> BaselineArtifact:
        """
        Build a failed baseline artifact carrying an actionable unavailable diagnostic.
        """

        return BaselineArtifact(
            metadata=ScriptFileMetadata(
                status=ScriptStatus.FAILED,
                source=ScriptSource.BASELINE,
                issues=(Issue(code=IssueCode.BASELINE_UNAVAILABLE, message=message),),
            )
        )

    async def __persist_script(
        self,
        *,
        step_number: int,
        package_name: str,
        result: Optional[GenerationResult],
        source: ScriptSource = ScriptSource.QUALITY,
    ) -> str:
        """
        Persist a produced script as script.txt plus its sidecar and artifact, or keep the existing one.
        """

        if result is None or not result.text.strip():
            return self.__read_existing_script(package_name=package_name)

        path = self.__get_history_file_path(package_name=package_name, filename="script.txt")

        logger.info(
            "script artifact persist started",
            extra={
                "event": "script.artifact.persist.started",
                "script.step": step_number,
                "script.package": package_name,
                "execution.id": self.__execution_id,
                "script.line_count": len(result.text.splitlines()),
            },
        )

        try:
            with path.open(mode="w") as handle:
                handle.write(result.text)
            self.__write_script_metadata(package_name=package_name, result=result, source=source)
        except Exception as exception:
            logger.warning(
                "script artifact persist failed",
                extra={
                    "event": "script.artifact.persist.failed",
                    "script.package": package_name,
                    "execution.id": self.__execution_id,
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                },
            )
            raise

        logger.info(
            "script artifact persisted",
            extra={
                "event": "script.artifact.persisted",
                "script.path": path.name,
                "script.step": step_number,
                "script.package": package_name,
                "execution.id": self.__execution_id,
            },
        )

        await self.__emit_script_artifact(
            content=result.text,
            step_number=step_number,
            package_name=package_name,
            partial=result.review.partial,
            review_reason=result.review.reason,
        )

        return result.text

    def __write_script_metadata(
        self, *, package_name: str, result: GenerationResult, source: ScriptSource
    ) -> None:
        """
        Persist a typed sidecar describing the script's review state next to script.txt.
        """

        path = self.__get_history_file_path(package_name=package_name, filename="script.meta.json")
        metadata = ScriptFileMetadata(source=source, review=result.review)

        with path.open(mode="w") as handle:
            handle.write(metadata.model_dump_json())

    async def __emit_script_artifact(
        self,
        *,
        content: str,
        step_number: int,
        package_name: str,
        partial: bool = False,
        review_reason: Optional[str] = None,
    ) -> None:
        """
        Hand the generated automation script to the artifact pipeline.

        Replaces the legacy ``__storage.save(category="history", ...)``
        direct upload — the pipeline owns durable EFS staging, async
        cloud upload, and replay-on-crash for every artifact kind.
        """

        if self.__pipeline is None:
            return

        try:
            await self.__pipeline.emit(
                record=ArtifactRecord(
                    step_number=step_number,
                    package_name=package_name,
                    session_id=self.__execution_id,
                    created=int(time.time() * 1000),
                    payload=ScriptPayload(
                        content=content, partial=partial, review_reason=review_reason
                    ),
                ),
            )
        except Exception as exception:
            logger.warning(
                "script artifact pipeline emit failed",
                extra={
                    "event": "script.artifact.pipeline.failed",
                    "script.step": step_number,
                    "script.package": package_name,
                    "execution.id": self.__execution_id,
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                },
            )
            raise

        logger.info(
            "script artifact emitted to pipeline",
            extra={
                "event": "script.artifact.pipeline.emitted",
                "script.step": step_number,
                "script.package": package_name,
                "execution.id": self.__execution_id,
            },
        )

    def __read_existing_script(self, *, package_name: str) -> str:
        """
        Return script.txt content if it already exists.
        """

        path = self.__get_history_file_path(package_name=package_name, filename="script.txt")
        if not path.exists():
            return ""

        with path.open(mode="r") as handle:
            return handle.read()

    def __resolve_export_package_name(self, history: List[Dict[str, Any]]) -> str:
        """
        Resolve best package name for OPEN_APP from recorded runtime activity.
        """

        for item in reversed(history):
            activity_raw = str(item.get("activity") or "").strip()

            if not activity_raw or activity_raw.lower() == "unknown":
                continue

            if "/" in activity_raw:
                activity_raw = activity_raw.split("/", 1)[0].strip()

            if activity_raw:
                return activity_raw

        return self.__package_name

    def __resolve_package_name(self, *, package_name: Optional[str]) -> str:
        """
        Resolve the active package name for history artifact persistence.
        """

        if package_name and package_name.strip():
            self.__package_name = package_name

        return self.__package_name

    def __log_background_failures(self, *, failures: List[Any], category: str) -> None:
        """
        Log any exceptions surfaced from gathered background tasks.
        """

        for failure in failures:
            if isinstance(failure, Exception):
                self.__log_failure(
                    event="script.history.background_task_failed",
                    message="background task finished with an error",
                    exception=failure,
                    context={"script.category": category},
                )

    def __get_history_file_path(self, *, package_name: str, filename: str) -> Path:
        """
        Resolve a history artifact path. Package is embedded in the filename so a
        single session that touches multiple packages does not collide its histories.
        """

        directory = self.__path_manager.get_history_directory(session_id=self.__execution_id)
        stem, _, ext = filename.rpartition(".")
        scoped = f"{stem}__{package_name}.{ext}" if stem and ext else f"{filename}__{package_name}"

        return directory / scoped

    def __build_yaml_item(self, index: int, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Constructs a structured dictionary for a YAML step.
        """

        target = record.get("natural_language_target") or record.get("target") or "UI Element"

        return {
            "step": index,
            "target": target,
            "center": record.get("center"),
            "bounding_box": record.get("bounds"),
            "event_type": record.get("event_type", "action"),
            "action_type": record.get("action_type", "wait"),
            "metadata": {
                "success": record.get("success"),
                "duration": record.get("duration"),
                "timestamp": record.get("timestamp"),
                "rationale": record.get("rationale"),
            },
        }
