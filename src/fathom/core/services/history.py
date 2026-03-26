from __future__ import annotations

import asyncio
import importlib
import json
import time
from logging import getLogger
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Set

from fathom.base.timing import time_it
from fathom.core.services.exporter import ScriptExporter
from fathom.interfaces.storage import StoragePort
from fathom.schemas.steps import StepResult

if TYPE_CHECKING:
    from pathlib import Path

    from fathom.base.paths import SharedPathManager

yaml: Any
try:
    yaml = importlib.import_module("yaml")
except ImportError:
    yaml = None


logger = getLogger(__name__)


class HistoryService:
    """
    Service responsible for persisting execution history and generating scripts.
    All outputs are saved to assets/history/{date}/{package}/{session}/ directory.
    """

    def __init__(
        self,
        workflow_id: str,
        package_name: str,
        exporter: ScriptExporter,
        path_manager: SharedPathManager,
        storage: Optional[StoragePort] = None,
    ) -> None:
        self.__workflow_id = workflow_id
        self.__package_name = package_name

        self.__storage = storage
        self.__exporter = exporter
        self.__path_manager = path_manager

        self.__background_tasks: Set[asyncio.Task[Any]] = set()
        self.__persistence_tasks: Set[asyncio.Task[Any]] = set()
        self.__persistence_chain: Optional[asyncio.Task[None]] = None

    def __fire_and_forget(self, coroutine: Any) -> None:
        """
        Schedules a coroutine as a background task.
        """

        try:
            task = asyncio.create_task(coroutine)
            self.__background_tasks.add(task)
            task.add_done_callback(self.__background_tasks.discard)
        except Exception as exception:
            logger.exception(f"Failed to execute FAF task. Got exception {exception}")

    def enqueue_save_step(
        self,
        *,
        result: StepResult,
        intent: str = "",
        package_name: Optional[str] = None,
        absolute_center: Optional[List[int]] = None,
        execution_activity: Optional[str] = None,
        on_complete: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> None:
        """
        Queue ordered history persistence without blocking the step lifecycle.
        """

        async def __run_persistence() -> None:
            script_data = await self.save_step(
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
                    logger.exception("Previous history persistence task failed: %s", exception)

            await __run_persistence()

        try:
            task = asyncio.create_task(__run_serialized())
            self.__persistence_chain = task
            self.__persistence_tasks.add(task)
            task.add_done_callback(self.__persistence_tasks.discard)
        except Exception as exception:
            logger.exception("Failed to queue history persistence task: %s", exception)

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
        package_name: Optional[str] = None,
        absolute_center: Optional[List[int]] = None,
        execution_activity: Optional[str] = None,
    ) -> str:
        """
        Saves a single step result and updates associated artifact files.
        Returns the current script if already generated.
        """

        resolved_package_name = self.__resolve_package_name(package_name=package_name)
        history = self.__load_history(package_name=resolved_package_name)

        record = result.to_record(
            absolute_center=absolute_center,
            activity=resolved_package_name,
        ).model_dump()

        record["timestamp"] = int(time.time() * 1000)
        record["screen_changed"] = result.screen_changed

        # Tag with pre-action activity so the exporter can filter launcher steps.
        if execution_activity:
            record["execution_activity"] = execution_activity

        history["history"].append(record)

        await self.__save_json(data=history, package_name=resolved_package_name)
        await self.__save_yaml(history=history["history"], package_name=resolved_package_name)

        return self.__read_existing_script(package_name=resolved_package_name)

    @time_it(operation="history.get_current_script")
    async def get_current_script(self, intent: str) -> str:
        """
        Retrieves (or generates) the latest script based on saved history.
        """

        await self.flush_pending_operations()
        history = self.__load_history(package_name=self.__package_name)

        return await self.__update_script(
            intent=intent,
            package_name=self.__package_name,
            history=history.get("history", []),
        )

    @time_it(operation="history.load_history")
    def __load_history(self, *, package_name: str) -> Dict[str, Any]:
        """
        Loads existing history from the JSON artifact.
        """

        path = self.__get_history_file_path(package_name=package_name, filename="history.json")
        data: Dict[str, Any] = {"workflow_id": self.__workflow_id, "history": []}

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
                logger.error(
                    "History file is corrupted; preserving backup at %s. Original error: %s",
                    backup_path,
                    exception,
                )
                try:
                    path.replace(backup_path)
                except Exception as backup_exception:  # nosec
                    logger.warning(
                        "Failed to preserve corrupt history backup: %s",
                        backup_exception,
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
                        "session_id": self.__workflow_id,
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

        if yaml:
            yaml_data = yaml.dump(
                indent=2,
                data=steps,
                sort_keys=False,
                default_flow_style=False,
            )
        else:
            yaml_data = self.__build_manual_yaml_string(steps=steps)

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
                        "session_id": self.__workflow_id,
                    },
                )
            )

    @time_it(operation="history.update_script")
    async def __update_script(
        self,
        history: List[Dict[str, Any]],
        intent: str,
        *,
        package_name: str,
    ) -> str:
        """
        Generates and persists a final natural language script.
        """

        path = self.__get_history_file_path(package_name=package_name, filename="script.txt")

        export_package_name = self.__resolve_export_package_name(history=history)
        script_data = await self.__exporter.export_with_llm(
            step_results=history,
            goal_state=intent,
            package_name=export_package_name,
            intent=intent,
        )

        if script_data is None or not script_data.strip():
            return self.__read_existing_script(package_name=package_name)

        with path.open(mode="w") as handle:
            handle.write(script_data)

        if self.__storage:
            self.__fire_and_forget(
                self.__storage.save(
                    data=script_data.encode("utf-8"),
                    metadata={
                        "category": "history",
                        "filename": "script.txt",
                        "package_name": package_name,
                        "session_id": self.__workflow_id,
                    },
                )
            )

        return script_data

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

        if package_name and str(package_name).strip():
            self.__package_name = str(package_name)

        return self.__package_name

    def __log_background_failures(self, *, failures: List[Any], category: str) -> None:
        """
        Log any exceptions surfaced from gathered background tasks.
        """

        for failure in failures:
            if isinstance(failure, Exception):
                logger.error(
                    "Background task failure in %s: %s",
                    category,
                    failure,
                    exc_info=(type(failure), failure, failure.__traceback__),
                )

    def __get_history_file_path(self, *, package_name: str, filename: str) -> Path:
        """
        Resolve a history artifact path for the current package context.
        """

        directory = self.__path_manager.get_history_directory(
            package_name=package_name,
            session_id=self.__workflow_id,
        )
        return directory / filename

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

    def __build_manual_yaml_string(self, steps: List[Dict[str, Any]]) -> str:
        """
        Fallback YAML writer if PyYAML is unavailable. Returns YAML string.
        """

        lines = []

        for step in steps:
            lines.append(f"- step: {step['step']}")
            lines.append(f'  action_type: "{step["action_type"]}"')
            lines.append(f'  event_type: "{step.get("event_type", "action")}"')
            lines.append(f'  target: "{step["target"]}"')
            lines.append(f"  bounding_box: {step.get('bounding_box')}")
            lines.append(f"  center: {step.get('center')}")

            meta = step["metadata"]
            rationale = str(object=meta.get("rationale", "")).replace('"', '\\"')
            lines.append("  metadata:")
            lines.append(f"    success: {str(object=meta.get('success')).lower()}")
            lines.append(f"    duration: {meta.get('duration')}")
            lines.append(f"    timestamp: {meta.get('timestamp')}")
            lines.append(f'    rationale: "{rationale}"')
            lines.append("")

        return "\n".join(lines)
