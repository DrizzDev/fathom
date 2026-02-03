from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fathom.schemas.steps import StepResult


class HistoryService:
    """
    Service responsible for persisting execution history.
    Generates structured JSON logs and a readable command sequence.
    """

    def __init__(self, workflow_id: str) -> None:
        self.__workflow_id = workflow_id
        self.__base_dir = Path("assets/history")
        self.__base_dir.mkdir(parents=True, exist_ok=True)

    def save_step(
        self,
        result: StepResult,
        absolute_center: Optional[List[int]] = None,
    ) -> None:
        """
        Saves a single step result to the workflow history file.
        """
        history_file = self.__base_dir / f"{self.__workflow_id}.json"

        data: Dict[str, Any] = {"workflow_id": self.__workflow_id, "history": []}

        if history_file.exists():
            try:
                with history_file.open("r") as f:
                    data = json.load(f)
            except Exception:
                pass

        step_record = result.to_record(absolute_center=absolute_center).model_dump()
        step_record["timestamp_ms"] = int(time.time() * 1000)

        history_list = data.get("history")
        if isinstance(history_list, list):
            history_list.append(step_record)

        with history_file.open("w") as f:
            json.dump(data, f, indent=2)

        # Also generate the structured command sequence (no extension)
        self.__save_command_sequence(data["history"])

    def __save_command_sequence(self, history: List[Dict[str, Any]]) -> None:
        """
        Generates a structured command sequence file without an extension.
        Format:
        Step [Index]:
          Command: [Description]
          Action: [Type]
          Target: [Element]
          BBox: [x1, y1, x2, y2] (normalized)
          Center: [x, y] (absolute)
          Metadata: { ... }
        """
        # File name with no extension
        sequence_file = self.__base_dir / f"{self.__workflow_id}"
        lines = []

        for index, step in enumerate(history, start=1):
            action_type = step.get("action_type", "wait")
            target = step.get("target", "element")
            
            # 1. Header
            lines.append(f"Step {index}:")
            
            # 2. Command Description
            if action_type == "type":
                command = f"Type '{step.get('text', '')}' in {target}"
            elif action_type == "tap":
                command = f"Tap on {target}"
            elif action_type == "swipe":
                command = f"Swipe on {target}"
            elif action_type == "scroll":
                command = f"Scroll {target}"
            elif action_type == "long_press":
                command = f"Long press on {target}"
            elif action_type == "back":
                command = "Press back button"
            elif action_type == "home":
                command = "Press home button"
            elif action_type == "wait":
                command = f"Wait for {target}"
            elif action_type == "complete":
                command = "Goal completed"
            else:
                command = f"{action_type.capitalize()} on {target}"
            
            lines.append(f"  Command: {command}")
            lines.append(f"  Action: {action_type}")
            lines.append(f"  Target: {target}")

            # 3. Metadata (BBox and Center)
            bbox = step.get("bbox")
            center = step.get("center")
            
            if bbox:
                lines.append(f"  BBox: {bbox} (normalized)")
            
            if center:
                lines.append(f"  Center: {center} (absolute)")
            
            # 4. Outcome
            lines.append(f"  Success: {step.get('success')}")
            lines.append(f"  Duration: {step.get('duration')}ms")
            
            lines.append("") # Empty line between steps

        with sequence_file.open("w") as f:
            f.write("\n".join(lines))