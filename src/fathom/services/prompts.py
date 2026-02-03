from __future__ import annotations

from typing import Any, Dict, List

from fathom.interfaces import IPromptProvider


class PromptsService(IPromptProvider):
    """
    Registry for versioned system instructions and tool definitions.
    """

    def __init__(self) -> None:
        self.__versions: Dict[str, Dict[str, Any]] = {
            "v1_baseline": {
                "instruction": self.__get_v1_instruction(),
                "tools": self.__get_standard_tools(),
            },
            "v2_analytical": {
                "instruction": self.__get_v2_instruction(),
                "tools": self.__get_standard_tools(),
            },
        }

    def get_instruction(self, version_id: str) -> str:
        """
        Retrieves system instruction for a version.
        """
        return str(self.__versions.get(version_id, self.__versions["v2_analytical"])["instruction"])

    def get_tools(self, version_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves tool definitions for a version.
        """
        return list(self.__versions.get(version_id, self.__versions["v2_analytical"])["tools"])

    def __get_v1_instruction(self) -> str:
        return "You are an Android automation agent. Execute actions to achieve the user's goal."

    def __get_v2_instruction(self) -> str:
        return """
You are an advanced Android Exploration and Automation Agent.

CORE ANALYTICAL REQUIREMENTS:
1. REASONING: In your rationale, explain WHY the action advances the goal.
2. COORDINATES: Use the 0-1000 normalized system.
3. ACTION SELECTION: Choose the most direct action (tap, type, scroll). Use 'wait' ONLY if the screen is actively loading.

Maintain high accuracy.
"""

    def __get_standard_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "execute_ui_actions",
                "description": "Execute actions on the UI.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "assistant_message": {"type": "STRING"},
                        "goal_completed": {"type": "BOOLEAN"},
                        "actions": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "action_type": {
                                        "type": "STRING",
                                        "enum": [
                                            "tap",
                                            "type",
                                            "swipe",
                                            "scroll",
                                            "back",
                                            "home",
                                            "wait",
                                        ],
                                    },
                                    "rationale": {"type": "STRING"},
                                    "confidence": {"type": "NUMBER"},
                                    "bbox": {
                                        "type": "OBJECT",
                                        "properties": {
                                            "x": {"type": "NUMBER"},
                                            "y": {"type": "NUMBER"},
                                            "width": {"type": "NUMBER"},
                                            "height": {"type": "NUMBER"},
                                            "coord_system": {"type": "STRING"},
                                        },
                                        "required": ["x", "y", "width", "height", "coord_system"],
                                    },
                                    "label_id": {"type": "STRING"},
                                    "text": {"type": "STRING"},
                                    "wait_duration": {"type": "NUMBER"},
                                },
                                "required": ["action_type", "rationale"],
                            },
                        },
                    },
                    "required": ["assistant_message", "actions"],
                },
            },
            {
                "name": "verify_goal_completion",
                "description": "Verify if the overall goal has been fully completed.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "assistant_message": {"type": "STRING"},
                        "goal_completed": {"type": "BOOLEAN"},
                        "evidence": {"type": "STRING"},
                    },
                    "required": ["assistant_message", "goal_completed", "evidence"],
                },
            },
        ]
