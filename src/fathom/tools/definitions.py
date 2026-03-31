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
                cls.__describe_screen(),
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
                                    "Stable element identifier in this EXACT format:\n"
                                    "  {element_type}_{region}_{index}\n\n"
                                    "- element_type: button, tab, card, icon, input, chip, toggle, link, image\n"
                                    "- region: top_bar, content, bottom_nav, modal, fab, footer\n"
                                    "- index: 1-based position within that region (left-to-right, top-to-bottom)\n\n"
                                    "Examples: tab_bottom_nav_1, card_content_3, icon_top_bar_2, "
                                    "input_content_1, chip_content_4, button_modal_1\n\n"
                                    "NEVER use runtime text, product names, prices, or placeholder content. "
                                    "The same element must get the SAME target_name every time this screen is seen."
                                ),
                            },
                            "tap_target": {
                                "type": "OBJECT",
                                "description": (
                                    "CENTER point of the element to tap, in normalized 0-1000 coordinates. "
                                    "x = horizontal center of the element (0 = left edge, 1000 = right edge). "
                                    "y = vertical center of the element (0 = top edge, 1000 = bottom edge). "
                                    "Place the point at the visual CENTER of the element, not a corner."
                                ),
                                "properties": {
                                    "x": {
                                        "type": "INTEGER",
                                        "description": "Horizontal center of the element (0-1000).",
                                    },
                                    "y": {
                                        "type": "INTEGER",
                                        "description": "Vertical center of the element (0-1000).",
                                    },
                                },
                            },
                            "element_category": {
                                "type": "STRING",
                                "description": (
                                    "What kind of UI element this is. "
                                    "Must match the priority system: "
                                    "global_navigation=P1 (bottom nav, sidebar, top-level tab bar), "
                                    "primary_action=P2 (Add/Create/Search/Buy buttons, FABs, search bars), "
                                    "content_item=P3 (cards, list items, product tiles with detail arrows), "
                                    "filter_or_category=P4 (category chips, filter pills, sort, horizontal carousels), "
                                    "secondary_control=P5 (overflow menus, toggles, share, settings, profile icons), "
                                    "overlay_dismiss=special (popups, modals, banners)."
                                ),
                                "enum": [
                                    "global_navigation",
                                    "primary_action",
                                    "content_item",
                                    "filter_or_category",
                                    "secondary_control",
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
                        "required": ["action_type", "rationale", "target_name", "tap_target"],
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

        Design-blueprint description of a unique activity screen.  The output
        must be detailed enough for another LLM to recreate the screen
        image purely from the text — exact colors, sizes, positions, and
        element inventory.
        """

        return {
            "name": "describe_screen",
            "description": (
                "Produce a design-blueprint description of the current activity screen. "
                "The description must be detailed enough for an LLM to recreate the "
                "screen image purely from this text. Focus on DESIGN, not data."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "activity_name": {
                        "type": "STRING",
                        "description": (
                            "The Android activity name this screen belongs to, "
                            "exactly as shown in the context (e.g. "
                            "'in.swiggy.android/in.swiggy.android.imPdp.views.IMPdpActivity'). "
                            "One description per unique activity."
                        ),
                    },
                    "screen_purpose": {
                        "type": "STRING",
                        "description": (
                            "1-2 sentences: what this activity screen is for, "
                            "which app section it belongs to, and the primary user task."
                        ),
                    },
                    "layout_blueprint": {
                        "type": "STRING",
                        "description": (
                            "Top-to-bottom spatial blueprint of the screen. For each region describe: "
                            "position (top/middle/bottom, left/right/center/full-width), "
                            "approximate height as percentage of screen, "
                            "background color (hex or name), "
                            "and what it contains. "
                            "Example: 'Top 8%: status bar (dark, system icons). "
                            'Next 6%: white app bar with back arrow (left), title "Menu" (center bold 18sp), '
                            "cart icon with red badge (right). "
                            "Next 30%: hero image carousel (full-width, 16:9 aspect). "
                            "Remaining: scrollable content on #F5F5F5 background.'"
                        ),
                    },
                    "component_inventory": {
                        "type": "STRING",
                        "description": (
                            "One component per line using this format:\n"
                            "  [Region] type | generic-label | position | size | colors | shape | state\n\n"
                            "RULES:\n"
                            "- generic-label: Use the GENERIC element name, NEVER runtime data.\n"
                            "  GOOD: Search bar, Product card, Category chip, Add to cart button, "
                            "Restaurant card, Price label, Rating badge\n"
                            "  BAD: Search for Cake, 99 Slice by Olio Pizza, 5 items | ₹717, Tim Hortons\n"
                            "- If an element shows dynamic text (placeholder, price, name), "
                            "describe the element TYPE only: 'Search bar with placeholder' not 'Search for Sweets'.\n"
                            "- Group lines by region (Top bar, Content area, Bottom nav, etc).\n"
                            "- One line per component. No prose sentences.\n"
                            "- Be exhaustive — every icon, divider, badge, and label."
                        ),
                    },
                    "design_tokens": {
                        "type": "STRING",
                        "description": (
                            "Visual design system tokens observed: "
                            "primary color, accent color, background colors, "
                            "text colors (heading/body/caption/link), "
                            "font sizes (heading/subheading/body/caption approximate sp), "
                            "corner radii, elevation/shadow patterns, "
                            "spacing rhythm (padding/margin patterns), "
                            "icon style (outlined/filled/rounded, approximate size)."
                        ),
                    },
                },
                "required": [
                    "activity_name",
                    "screen_purpose",
                    "layout_blueprint",
                    "component_inventory",
                    "design_tokens",
                ],
            },
        }
