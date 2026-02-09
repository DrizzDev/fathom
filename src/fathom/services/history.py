from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None

from fathom.schemas.steps import StepResult


class HistoryService:
    """
    Service responsible for persisting execution history.
    Generates structured JSON logs and YAML test scripts.
    """

    def __init__(self, workflow_id: str) -> None:
        self.__workflow_id = workflow_id
        self.__base_directory = Path("assets/history")
        self.__base_directory.mkdir(parents=True, exist_ok=True)

    def save_step(self, result: StepResult, absolute_center: Optional[List[int]] = None) -> None:
        """
        Saves a single step result to the workflow history files.
        """

        history_data = self.__load_history()

        record = result.to_record(absolute_center=absolute_center).model_dump()
        record["timestamp"] = int(time.time() * 1000)

        history_data["history"].append(record)

        self.__save_json(data=history_data)
        self.__save_yaml(history=history_data["history"])
        self.__append_text_audit(record=record)

    def __load_history(self) -> Dict[str, Any]:
        """
        Loads existing history from disk.
        """

        path = self.__base_directory / f"{self.__workflow_id}.json"
        data: Dict[str, Any] = {"workflow_id": self.__workflow_id, "history": []}

        if path.exists():
            try:
                with path.open(mode="r") as handle:
                    data = json.load(fp=handle)
            except Exception:  # nosec
                pass

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
            "command": self.__describe_command(record=record),
            "action_type": record.get("action_type", "wait"),
            "target": target,
            "bounding_box": record.get("bounds"),
            "center": record.get("center"),
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

        # Prioritize natural language target if it's not generic
        if natural_language_target and str(object=natural_language_target).lower() not in (
            "ui element",
            "element",
            "none",
            "label",
        ):
            return str(object=natural_language_target)

        # Fallback to technical target if it's not generic
        if target and str(object=target).lower() not in (
            "ui element",
            "element",
            "none",
            "label",
        ):
            return str(object=target)

        return "UI Element"

    def __describe_command(self, record: Dict[str, Any]) -> str:
        """
        Generates a readable command description.
        """

        action_type = record.get("action_type", "wait")
        target = self.__resolve_target_name(record=record)

        if action_type == "type":
            return f"Type '{record.get('text', '')}' in {target}"

        if action_type == "tap":
            return f"Tap on {target}"

        if action_type == "swipe":
            return f"Swipe on {target}"

        if action_type == "scroll":
            return f"Scroll {target}"

        if action_type == "back":
            return "Press back button"

        if action_type == "home":
            return "Press home button"

        if action_type == "complete":
            return "Goal completed"

        return f"{str(object=action_type).capitalize()} on {target}"

    def __append_text_audit(self, record: Dict[str, Any]) -> None:
        """
        Appends a readable audit line to output.txt.
        Format: Tap on [Target]
        """
        path = Path("output.txt")
        line = self.__describe_command(record=record)

        # Append to file
        with path.open(mode="a") as handle:
            handle.write(line + "\n")

    def __write_manual_yaml(self, path: Path, steps: List[Dict[str, Any]]) -> None:
        """
        Fallback YAML writer if PyYAML is unavailable.
        """

        lines = []

        for step in steps:
            lines.append(f"- step: {step['step']}")
            lines.append(f'  command: "{step["command"]}"')
            lines.append(f'  action_type: "{step["action_type"]}"')
            lines.append(f'  target: "{step["target"]}"')
            lines.append(f"  bounding_box: {step.get('bounding_box')}")
            lines.append(f"  center: {step.get('center')}")

            metadata = step["metadata"]
            rationale = str(object=metadata.get("rationale", "")).replace('"', '\\"')
            lines.append(
                f'  metadata:\n    timestamp: {metadata.get("timestamp")}\n    success: {str(object=metadata.get("success")).lower()}\n    rationale: "{rationale}"'
            )
            lines.append("")

        with path.open(mode="w") as handle:
            handle.write("\n".join(lines))
