from __future__ import annotations

import json
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

try:
    import yaml
except ImportError:
    yaml = cast("Any", None)

from fathom.schemas.steps import StepResult
from fathom.services.text_normalization import describe_action, describe_validation

logger = getLogger(__name__)


class HistoryService:
    """
    Service responsible for persisting execution history.
    Generates structured JSON logs and YAML test scripts.
    """

    def __init__(self, workflow_id: str, intent: str = "", package_name: str = "") -> None:
        self.__workflow_id = workflow_id
        self.__intent = intent
        self.__package_name = package_name
        self.__base_directory = Path("assets/history")
        self.__base_directory.mkdir(parents=True, exist_ok=True)
        self.goal_state: str = ""

    def set_package_name(self, package_name: str) -> None:
        """Update the package name used for script export (e.g. after the app launches)."""
        if package_name:
            self.__package_name = package_name

    def save_step(
        self,
        result: StepResult,
        absolute_center: Optional[List[int]] = None,
        activity: Optional[str] = None,
    ) -> None:
        """
        Saves a single step result to the workflow history files.
        """

        history_data = self.__load_history()

        record = result.to_record(absolute_center=absolute_center, activity=activity).model_dump()
        record["timestamp"] = int(time.time() * 1000)

        history_data["history"].append(record)

        self.__save_json(data=history_data)
        self.__save_yaml(history=history_data["history"])

    def __load_history(self) -> Dict[str, Any]:
        """
        Loads existing history from disk.
        Returns empty history if file doesn't exist or is corrupted.
        """

        path = self.__base_directory / f"{self.__workflow_id}.json"
        data: Dict[str, Any] = {"workflow_id": self.__workflow_id, "history": []}

        if path.exists():
            try:
                with path.open(mode="r") as handle:
                    data = json.load(fp=handle)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse history JSON from {path}: {e}")
            except (IOError, OSError) as e:
                logger.warning(f"Failed to read history file {path}: {e}")
            except Exception as e:  # nosec
                logger.warning(f"Unexpected error loading history from {path}: {e}", exc_info=True)

        return data

    def __save_json(self, data: Dict[str, Any]) -> None:
        """
        Writes the history data to JSON.
        """

        path = self.__base_directory / f"{self.__workflow_id}.json"
        with path.open(mode="w") as handle:
            json.dump(obj=data, fp=handle, indent=2)

    def __save_yaml(self, history: List[Dict[str, Any]]) -> None:
        """
        Orchestrates the YAML script generation.
        """

        path = self.__base_directory / f"{self.__workflow_id}.yaml"
        steps = [
            self.__build_yaml_item(index=index, record=item)
            for index, item in enumerate(iterable=history, start=1)
        ]

        if yaml:
            with path.open(mode="w") as handle:
                yaml.dump(
                    indent=2,
                    data=steps,
                    stream=handle,
                    sort_keys=False,
                    default_flow_style=None,
                )
        else:
            self.__write_manual_yaml(path=path, steps=steps)

    def __build_yaml_item(self, index: int, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Constructs a structured dictionary for a YAML step.
        """

        # Improved target resolution for YAML
        target = self.__resolve_target_name(record=record)

        return {
            "step": index,
            "target": target,
            "center": record.get("center"),
            "bounding_box": record.get("bounds"),
            "event_type": record.get("event_type", "action"),
            "action_type": record.get("action_type", "wait"),
            "command": self.__describe_command(record=record),
            "metadata": {
                "success": record.get("success"),
                "duration": record.get("duration"),
                "timestamp": record.get("timestamp"),
                "rationale": record.get("rationale"),
            },
        }

    def __resolve_target_name(self, record: Dict[str, Any]) -> str:
        """
        Resolves the best human-readable target name.
        """

        target = record.get("target")
        natural_language_target = record.get("natural_language_target")

        # Trust natural language target if present
        if natural_language_target and str(natural_language_target).strip():
            return str(natural_language_target).strip()

        # Fallback to technical target
        if target and str(target).strip():
            return str(target).strip()

        return "UI Element"

    def __describe_command(self, record: Dict[str, Any]) -> str:
        """
        Generates a readable command description.
        """

        action_type = str(object=record.get("action_type", "wait")).lower()
        event_type = str(object=record.get("event_type", "action")).lower()
        target = self.__resolve_target_name(record=record)

        if event_type == "validation":
            return describe_validation(
                target=target,
                explicit=False,
                complete=(action_type == "complete"),
            )

        if action_type == "complete":
            return "Goal completed"
        return describe_action(action_type=action_type, target=target, text=record.get("text"))

    def __write_manual_yaml(self, path: Path, steps: List[Dict[str, Any]]) -> None:
        """
        Fallback YAML writer if PyYAML is unavailable.
        """

        lines = []

        for step in steps:
            lines.append(f"- step: {step['step']}")
            lines.append(f'  command: "{step["command"]}"')
            lines.append(f'  action_type: "{step["action_type"]}"')
            lines.append(f'  event_type: "{step.get("event_type", "action")}"')
            lines.append(f'  target: "{step["target"]}"')
            lines.append(f"  bounding_box: {step.get('bounding_box')}")
            lines.append(f"  center: {step.get('center')}")

            metadata = step["metadata"]
            rationale = str(object=metadata.get("rationale", "")).replace('"', '\\"')
            lines.append("  metadata:")
            lines.append(f"    success: {str(object=metadata.get('success')).lower()}")
            lines.append(f"    duration: {metadata.get('duration')}")
            lines.append(f"    timestamp: {metadata.get('timestamp')}")
            lines.append(f'    rationale: "{rationale}"')
            lines.append("")

        with path.open(mode="w") as handle:
            handle.write("\n".join(lines))
