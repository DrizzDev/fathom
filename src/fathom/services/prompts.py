from __future__ import annotations

from typing import Any, Dict, List

from fathom.interfaces import IPromptProvider
from fathom.prompts.analysis import (
    FLASH_VISION_PROMPT,
    FLASH_XML_PROMPT,
    PRO_VISION_PROMPT,
    PRO_XML_PROMPT,
)


class PromptsService(IPromptProvider):
    """
    Registry for versioned system instructions and tool definitions.
    Dynamically selects prompts based on model tier and grounding strategy.
    """

    def __init__(self) -> None:
        self.__instructions: Dict[str, str] = {
            "pro_xml": PRO_XML_PROMPT,
            "flash_xml": FLASH_XML_PROMPT,
            "pro_vision": PRO_VISION_PROMPT,
            "flash_vision": FLASH_VISION_PROMPT,
        }

    def get_instruction(self, version_id: str) -> str:
        """
        Retrieves system instruction template for a version.
        """

        return self.__instructions.get(version_id, PRO_XML_PROMPT)

    def select_version(self, model_name: str, use_xml: bool) -> str:
        """
        Dynamically determines the optimal prompt version.
        """

        is_flash_model = "flash" in model_name.lower()

        tier = "flash" if is_flash_model else "pro"
        strategy = "xml" if use_xml else "vision"

        return f"{tier}_{strategy}"

    def get_tools(self, version_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves tool definitions.
        """

        return self.__get_standard_tools(is_flash_model="flash" in version_id)

    def __get_standard_tools(self, is_flash_model: bool) -> List[Dict[str, Any]]:
        """
        Returns tool definitions.
        """

        # For Flash models, we add 'screen_observation' to the schema to enforce CoT
        properties: Dict[str, Any] = {
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
                        "natural_language_target": {
                            "type": "STRING",
                            "description": "Descriptive name (e.g. 'Search Bar').",
                        },
                        "label_id": {"type": "STRING"},
                        "text": {"type": "STRING"},
                        "wait_duration": {"type": "NUMBER"},
                    },
                    "required": ["action_type", "rationale", "natural_language_target"],
                },
            },
        }

        required_fields = ["assistant_message", "actions"]

        if is_flash_model:
            properties["screen_observation"] = {
                "type": "STRING",
                "description": "Chain-of-Thought: Describe the screen contents before acting.",
            }
            required_fields.insert(0, "screen_observation")

        return [
            {
                "name": "execute_ui_actions",
                "description": "Execute actions on the UI.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": properties,
                    "required": required_fields,
                },
            }
        ]
