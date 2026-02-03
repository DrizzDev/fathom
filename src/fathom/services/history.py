from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
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
        self.__base_dir = Path("assets/history")
        self.__base_dir.mkdir(parents=True, exist_ok=True)

    def save_step(self, result: StepResult, absolute_center: Optional[List[int]] = None) -> None:
        """
        Saves a single step result to the workflow history files.
        """
        history = self.__load_history()

        record = result.to_record(absolute_center=absolute_center).model_dump()
        record["timestamp"] = int(time.time() * 1000)

        history["history"].append(record)

        self.__save_json(history)
        self.__save_yaml(history["history"])

    def __load_history(self) -> Dict[str, Any]:
        """
        Loads existing history from disk.
        """
        path = self.__base_dir / f"{self.__workflow_id}.json"
        data: Dict[str, Any] = {"workflow_id": self.__workflow_id, "history": []}

        if path.exists():
            try:
                with path.open("r") as handle:
                    data = json.load(handle)
            except Exception:  # nosec
                pass
        return data

    def __save_json(self, data: Dict[str, Any]) -> None:
        """
        Writes the history data to JSON.
        """
        path = self.__base_dir / f"{self.__workflow_id}.json"
        with path.open("w") as handle:
            json.dump(data, handle, indent=2)

    def __save_yaml(self, history: List[Dict[str, Any]]) -> None:
        """
        Orchestrates the YAML script generation.
        """
        path = self.__base_dir / f"{self.__workflow_id}.yaml"
        steps = [self.__build_yaml_item(index, item) for index, item in enumerate(history, 1)]

        if yaml:
            with path.open("w") as handle:
                yaml.dump(steps, handle, sort_keys=False, indent=2, default_flow_style=None)
        else:
            self.__write_manual_yaml(path, steps)

    def __build_yaml_item(self, index: int, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Constructs a structured dictionary for a YAML step.
        """
        return {
            "step": index,
            "command": self.__describe_command(record),
            "action_type": record.get("action_type", "wait"),
            "target": record.get("target", "element"),
            "coordinates": {"bbox": record.get("bbox"), "center": record.get("center")},
            "metadata": {
                "timestamp": record.get("timestamp"),
                "success": record.get("success"),
                "duration": record.get("duration"),
                "rationale": record.get("rationale"),
            },
        }

    def __describe_command(self, record: Dict[str, Any]) -> str:
        """
        Generates a readable command description.
        """
        action = record.get("action_type", "wait")
        target = record.get("target", "element")

        if action == "type":
            return f"Type '{record.get('text', '')}' in {target}"
        if action == "tap":
            return f"Tap on {target}"
        if action == "swipe":
            return f"Swipe on {target}"
        if action == "scroll":
            return f"Scroll {target}"
        if action == "back":
            return "Press back button"
        if action == "home":
            return "Press home button"
        if action == "complete":
            return "Goal completed"

        return f"{action.capitalize()} on {target}"

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

            coords = step["coordinates"]
            lines.append(
                f"  coordinates:\n    bbox: {coords.get('bbox')}\n    center: {coords.get('center')}"
            )

            meta = step["metadata"]
            rationale = str(meta.get("rationale", "")).replace('"', '\\"')
            lines.append(
                f'  metadata:\n    timestamp: {meta.get("timestamp")}\n    success: {str(meta.get("success")).lower()}\n    rationale: "{rationale}"'
            )
            lines.append("")

        with path.open("w") as handle:
            handle.write("\n".join(lines))
