"""History service for persisting execution traces."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None

from fathom.schemas.steps import StepResult

if TYPE_CHECKING:
    from pathlib import Path

    from fathom.base.paths import SharedPathManager


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
    ) -> None:
        self.__workflow_id = workflow_id
        self.__directory = path_manager.get_history_directory(
            package_name=package_name, session_id=workflow_id
        )

    def save_step(
        self,
        result: StepResult,
        *,
        intent: str = "",
        absolute_center: Optional[List[int]] = None,
    ) -> None:
        """
        Saves a single step result and updates associated artifact files.
        """

        history = self.__load_history()

        record = result.to_record(absolute_center=absolute_center).model_dump()
        record["timestamp"] = int(time.time() * 1000)
        record["screen_changed"] = result.screen_changed

        history["history"].append(record)

        self.__save_json(data=history)
        self.__save_yaml(history=history["history"])
        self.__update_script(history=history["history"], intent=intent)

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

    def __save_json(self, data: Dict[str, Any]) -> None:
        """
        Writes history to structured JSON format.
        """

        path = self.__directory / "history.json"
        with path.open(mode="w") as handle:
            json.dump(obj=data, fp=handle, indent=2)

    def __save_yaml(self, history: List[Dict[str, Any]]) -> None:
        """
        Generates a YAML representation of the execution.
        """

        path = self.__directory / "history.yaml"
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

    def __update_script(self, history: List[Dict[str, Any]], intent: str) -> None:
        """
        Generates a natural language test script with smart validation.
        """

        lines = []
        step_number = 1
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

        with path.open(mode="w") as handle:
            handle.write("\n".join(lines) + "\n")

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

    def __write_manual_yaml(self, path: Path, steps: List[Dict[str, Any]]) -> None:
        """
        Fallback YAML writer if PyYAML is unavailable.
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

        with path.open(mode="w") as handle:
            handle.write("\n".join(lines))
