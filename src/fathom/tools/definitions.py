from __future__ import annotations

from typing import Any, Dict, List


class ToolRegistry:
    """
    Registry for tool definitions used by the Vision Language Model.

    Exploration-only registry — provides explore_ui and describe_screen tools.
    """

    @classmethod
    def get_exploration_tools(cls) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns tool definitions for the exploration VLM call.
        """

        return {
            "function_declarations": [
                cls.__explore_ui(),
            ]
        }

    @staticmethod
    def __explore_ui() -> Dict[str, Any]:
        """
        Definition for explore_ui tool.

        Dedicated exploration tool — identifies and taps the next untried
        interactive element on the current screen.  Stripped of all
        validation, delta, memory, and script-export fields that are
        irrelevant to the exploration workflow.
        """

        return {
            "name": "explore_ui",
            "description": (
                "Identify and tap the next untried interactive element on the current screen "
                "to discover new app screens. "
                "Use when there are untried interactive elements visible. "
                "Do NOT use when all visible interactive elements appear in the ALREADY TRIED list — "
                "set content_exhausted=true instead. "
                "SIDE EFFECTS: Taps a UI element on the device, which may navigate to a new screen."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "assistant_message": {
                        "type": "STRING",
                        "description": (
                            "Brief reasoning for choosing this element. "
                            "State what it is, why it has not been tried, and what you expect will happen. "
                            "Example: 'Tapping Settings icon — untried P4 secondary action, "
                            "likely leads to a new settings screen.'"
                        ),
                    },
                    "action": {
                        "type": "OBJECT",
                        "description": "The exploration action to execute on the device.",
                        "properties": {
                            "action_type": {
                                "type": "STRING",
                                "description": "Physical action to perform on the element.",
                                "enum": [
                                    "tap",
                                    "scroll",
                                    "swipe_up",
                                    "swipe_down",
                                    "back",
                                    "long_press",
                                ],
                            },
                            "rationale": {
                                "type": "STRING",
                                "description": (
                                    "Why this element was chosen over other untried elements. "
                                    "Reference priority level (P1-P5). "
                                    "Example: 'P1 navigation tab — higher priority than "
                                    "P3 list items below.'"
                                ),
                            },
                            "target_name": {
                                "type": "STRING",
                                "description": (
                                    "Human-readable label exactly as it appears on screen. "
                                    "Use the visible text or icon description. "
                                    "Examples: 'Home tab', 'Search icon', 'Add to Cart button', "
                                    "'3rd restaurant card', 'Hamburger menu icon'."
                                ),
                            },
                            "bbox": {
                                "type": "OBJECT",
                                "description": (
                                    "Bounding box of the element in normalized 0-1000 coordinates. "
                                    "x,y = top-left corner. width and height extend rightward and downward."
                                ),
                                "properties": {
                                    "x": {
                                        "type": "INTEGER",
                                        "description": "Top-left X (0-1000).",
                                    },
                                    "y": {
                                        "type": "INTEGER",
                                        "description": "Top-left Y (0-1000).",
                                    },
                                    "width": {
                                        "type": "INTEGER",
                                        "description": "Width from x (0-1000).",
                                    },
                                    "height": {
                                        "type": "INTEGER",
                                        "description": "Height from y (0-1000).",
                                    },
                                },
                            },
                            "element_category": {
                                "type": "STRING",
                                "description": (
                                    "What kind of UI element this is. "
                                    "Matches the priority system: navigation=P1, primary_action=P2, "
                                    "list_item=P3, secondary_action=P4, in_page_control=P5, "
                                    "overlay_dismiss=special."
                                ),
                                "enum": [
                                    "navigation",
                                    "primary_action",
                                    "list_item",
                                    "secondary_action",
                                    "in_page_control",
                                    "overlay_dismiss",
                                ],
                            },
                            "expected_outcome": {
                                "type": "STRING",
                                "description": (
                                    "What you expect will happen after tapping this element. "
                                    "Example: tapping a tab expects 'new_screen', "
                                    "tapping a toggle expects 'in_screen_change'."
                                ),
                                "enum": [
                                    "new_screen",
                                    "in_screen_change",
                                    "dialog_or_popup",
                                    "scroll_content",
                                    "dismiss_overlay",
                                    "go_back",
                                ],
                            },
                            "overlay_detected": {
                                "type": "BOOLEAN",
                                "description": "Set true ONLY when this action dismisses an overlay, popup, or modal.",
                            },
                            "confidence": {
                                "type": "NUMBER",
                                "description": (
                                    "How confident you are that this element is interactive and untried (0.0-1.0). "
                                    "0.9+ = clearly visible interactive element. "
                                    "Below 0.7 = uncertain whether the element is tappable."
                                ),
                            },
                        },
                        "required": ["action_type", "rationale", "target_name"],
                    },
                    "screen_description": {
                        "type": "STRING",
                        "description": (
                            "1-2 sentence summary identifying the screen's purpose and app section. "
                            "Example: 'Home feed showing recommended restaurants with search bar "
                            "and bottom navigation for Home, Search, Orders, and Profile.'"
                        ),
                    },
                    "content_exhausted": {
                        "type": "BOOLEAN",
                        "description": (
                            "Set true ONLY when every visible interactive element appears in the "
                            "ALREADY TRIED list AND scrolling has been attempted or is not applicable. "
                            "Defaults to false. NEVER set true while untried elements are visible."
                        ),
                    },
                },
                "required": ["assistant_message", "action", "screen_description"],
            },
        }

    @classmethod
    def get_screen_translation_tools(cls) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns tool definitions for the screen translation VLM call.
        """

        return {
            "function_declarations": [
                cls.__describe_screen(),
            ]
        }

    @staticmethod
    def __describe_screen() -> Dict[str, Any]:
        """
        Definition for describe_screen tool.

        Provides a thorough rich-text translation of all visible UI
        designs, features, and content on a mobile app screen.
        """

        return {
            "name": "describe_screen",
            "description": "Provide a thorough translation of all visible UI designs and features on the screen.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "layout_and_structure": {
                        "type": "STRING",
                        "description": "Detailed description of page layout, regions, spacing, visual hierarchy, header/body/footer arrangement.",
                    },
                    "navigation": {
                        "type": "STRING",
                        "description": "Navigation elements: top/bottom bars, tabs, menus, breadcrumbs, back buttons, sidebars and their labels.",
                    },
                    "content": {
                        "type": "STRING",
                        "description": "All visible text content, headings, images, media, cards, lists, badges, tags, and data displayed.",
                    },
                    "interactive_elements": {
                        "type": "STRING",
                        "description": "All interactive controls: buttons with labels, inputs, toggles, switches, checkboxes, dropdowns, sliders, links, and their states.",
                    },
                    "visual_design": {
                        "type": "STRING",
                        "description": "Colors, typography, iconography, branding, shadows, borders, rounded corners, and overall design language.",
                    },
                    "summary": {
                        "type": "STRING",
                        "description": "2-4 sentence prose summary of the screen's purpose, primary user task, and overall user experience.",
                    },
                },
                "required": [
                    "layout_and_structure",
                    "navigation",
                    "content",
                    "interactive_elements",
                    "visual_design",
                    "summary",
                ],
            },
        }
