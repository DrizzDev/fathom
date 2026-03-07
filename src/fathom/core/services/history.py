from __future__ import annotations

import asyncio
import json
import time
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None

from fathom.core.services.exporter import ScriptExporter
from fathom.interfaces.storage import StoragePort
from fathom.schemas.steps import StepResult

if TYPE_CHECKING:
    from fathom.base.paths import SharedPathManager


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
        path_manager: SharedPathManager,
        exporter: ScriptExporter,
        storage: Optional[StoragePort] = None,
    ) -> None:
        self.__workflow_id = workflow_id
        self.__package_name = package_name

        self.__storage = storage
        self.__directory = path_manager.get_history_directory(
            package_name=package_name, session_id=workflow_id
        )
        self.__exporter = exporter
        self.__background_tasks: set[asyncio.Task[Any]] = set()

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

    async def save_step(
        self,
        result: StepResult,
        *,
        intent: str = "",
        absolute_center: Optional[List[int]] = None,
        activity: Optional[str] = None,
    ) -> str:
        """
        Saves a single step result and updates associated artifact files.
        Returns the current script if already generated.
        """

        history = self.__load_history()

        record = result.to_record(absolute_center=absolute_center, activity=activity).model_dump()
        record["timestamp"] = int(time.time() * 1000)
        record["screen_changed"] = result.screen_changed

        history["history"].append(record)

        await self.__save_json(data=history)
        await self.__save_yaml(history=history["history"])
        return self.__read_existing_script()

    async def get_current_script(self, intent: str) -> str:
        """
        Retrieves (or generates) the latest script based on saved history.
        """

        history = self.__load_history()
        return await self.__update_script(history=history.get("history", []), intent=intent)

    def __load_history(self) -> Dict[str, Any]:
        """
        Loads existing history from the JSON artifact.
        """

        path = self.__directory / "history.json"
        data: Dict[str, Any] = {"workflow_id": self.__workflow_id, "history": []}

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

    async def __save_json(self, data: Dict[str, Any]) -> None:
        """
        Writes history to structured JSON format.
        """

        path = self.__directory / "history.json"

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
                        "session_id": self.__workflow_id,
                        "package_name": self.__package_name,
                    },
                )
            )

    async def __save_yaml(self, history: List[Dict[str, Any]]) -> None:
        """
        Generates a YAML representation of the execution.
        """

        path = self.__directory / "history.yaml"
        steps = [
            self.__build_yaml_item(index=index, record=item)
            for index, item in enumerate(iterable=history, start=1)
        ]

        if yaml:
            yaml_data = yaml.dump(
                data=steps,
                indent=2,
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
                        "session_id": self.__workflow_id,
                        "package_name": self.__package_name,
                    },
                )
            )

    async def __update_script(self, history: List[Dict[str, Any]], intent: str) -> str:
        """
        Generates and persists a final natural language script.
        """

        path = self.__directory / "script.txt"

        export_package_name = self.__resolve_export_package_name(history=history)
        script_data = await self.__exporter.export_with_llm(
            step_results=history,
            goal_state=intent,
            package_name=export_package_name,
            intent=intent,
        )

        if script_data is None or not script_data.strip():
            return self.__read_existing_script()

        with path.open(mode="w") as handle:
            handle.write(script_data)

        if self.__storage:
            self.__fire_and_forget(
                self.__storage.save(
                    data=script_data.encode("utf-8"),
                    metadata={
                        "category": "history",
                        "filename": "script.txt",
                        "session_id": self.__workflow_id,
                        "package_name": self.__package_name,
                    },
                )
            )

        return script_data

    def __read_existing_script(self) -> str:
        """
        Return script.txt content if it already exists.
        """

        path = self.__directory / "script.txt"
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
