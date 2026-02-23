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
        storage: Optional[StoragePort] = None,
    ) -> None:
        self.__workflow_id = workflow_id
        self.__package_name = package_name

        self.__storage = storage
        self.__directory = path_manager.get_history_directory(
            package_name=package_name, session_id=workflow_id
        )
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
    ) -> str:
        """
        Saves a single step result and updates associated artifact files.
        Returns the generated natural language script.
        """

        history = self.__load_history()

        record = result.to_record(absolute_center=absolute_center).model_dump()
        record["timestamp"] = int(time.time() * 1000)
        record["screen_changed"] = result.screen_changed

        history["history"].append(record)

        await self.__save_json(data=history)
        await self.__save_yaml(history=history["history"])
        return await self.__update_script(history=history["history"], intent=intent)

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
                _ = exception
                pass

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
        Generates a natural language test script with smart validation.
        """

        step_number = 1
        lines = []
        path = self.__directory / "script.txt"

        for index, record in enumerate(iterable=history):
            action_type = record.get("action_type", "unknown")
            raw_target = record.get("natural_language_target") or record.get("target") or "element"

            # 1. Resolve Target (Generalize if not in intent)
            target = self.__resolve_script_target(
                target=raw_target, intent=intent, action=action_type
            )

            # 2. Smart Validation (if previous screen changed)
            if index > 0:
                previous = history[index - 1]
                if previous.get("screen_changed") and target.lower() not in (
                    "none",
                    "element",
                    "ui element",
                    "a visible item",
                ):
                    lines.append(f"{step_number}. Validate {target} is visible")
                    step_number += 1

            # 3. Action Description
            description = self.__build_description(
                action=action_type, target=target, text=record.get("text")
            )
            lines.append(f"{step_number}. {description}")
            step_number += 1

        script_data = "\n".join(lines) + "\n"
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

    def __resolve_script_target(self, target: str, intent: str, action: str) -> str:
        """
        Checks if target is meaningful to intent, otherwise generalizes.
        """

        if not target or not intent:
            return "element"

        target_lower = target.lower()
        intent_lower = intent.lower()

        if target_lower in intent_lower:
            return target

        # Fuzzy overlap check
        target_words = set(target_lower.replace("_", " ").split())
        filler = {
            "the",
            "a",
            "an",
            "on",
            "in",
            "to",
            "of",
            "is",
            "and",
            "or",
            "item",
            "button",
            "icon",
            "area",
            "field",
        }
        meaningful = target_words - filler
        if not meaningful:
            return target

        intent_words = set(intent_lower.replace("_", " ").split())
        overlap = meaningful & intent_words

        if len(overlap) >= len(meaningful) * 0.5:
            return target

        return "a visible item" if action in ("tap", "long_press") else "the current view"

    def __build_description(self, action: str, target: str, text: Optional[str]) -> str:
        """
        Constructs a human-readable action description.
        """

        if action == "tap":
            return f"Tap on {target}"

        if action == "type":
            return f"Type '{text}' into {target}"

        if "swipe" in action:
            direction = action.split(sep="_")[-1] if "_" in action else "content"
            return f"Swipe {direction} on {target}"

        if action in ("back", "press_back"):
            return "Press back button"

        if action in ("home", "press_home"):
            return "Press home button"

        if action == "enter":
            return "Press enter"

        if action == "wait":
            return f"Wait for {target}"

        if action == "scroll":
            return f"Scroll until you see {target}"

        if action == "long_press":
            return f"Long press on {target}"

        if action == "complete":
            return f"Validate {target} (Goal complete)"

        return f"{action.replace('_', ' ').capitalize()} on {target}"

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
