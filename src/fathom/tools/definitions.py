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
                                "description": (
                                    "Physical action to perform on the element. "
                                    "TAP discrete elements. SCROLL / SWIPE_UP / SWIPE_DOWN "
                                    "a scrollable area to reveal content below the fold. "
                                    "SWIPE_LEFT / SWIPE_RIGHT horizontal carousels or "
                                    "swipeable tab strips to expose more items. "
                                    "TYPE into a search bar or input (also set `text`). "
                                    "LONG_PRESS to reveal context menus. BACK to escape."
                                ),
                                "enum": [
                                    "tap",
                                    "type",
                                    "scroll",
                                    "swipe_up",
                                    "swipe_down",
                                    "swipe_left",
                                    "swipe_right",
                                    "back",
                                    "long_press",
                                ],
                            },
                            "rationale": {
                                "type": "STRING",
                                "description": (
                                    "Why this element was chosen over the other untried elements "
                                    "visible on screen. Focus on what's novel about it — do not "
                                    "restate the priority bucket (that's in element_category)."
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
                            "text": {
                                "type": "STRING",
                                "description": (
                                    "Text to type. REQUIRED when action_type is 'type'; ignored "
                                    "otherwise. Use a short, generic query that exercises the "
                                    "input's flow without overfitting (e.g. 'pizza' in a food app, "
                                    "'news' in a reader, 'a' as a cheap wildcard). Omit for any "
                                    "non-type action."
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
                            "region": {
                                "type": "STRING",
                                "description": (
                                    "Which region of the screen the element lives in. "
                                    "top_bar = status/app bar at the top (back, title, bell, cart, hamburger). "
                                    "bottom_nav = persistent tab bar at the bottom. "
                                    "content = main scrollable area between top_bar and bottom_nav. "
                                    "modal = elements inside a modal, dialog, or bottom sheet. "
                                    "overlay = permission prompt, cookie banner, tooltip, tutorial overlay. "
                                    "fab = floating action button (usually bottom-right circle). "
                                    "footer = persistent non-nav bar at the bottom (e.g. 'Apply filters')."
                                ),
                                "enum": [
                                    "top_bar",
                                    "bottom_nav",
                                    "content",
                                    "modal",
                                    "overlay",
                                    "fab",
                                    "footer",
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

        Rich functional description of a unique activity screen — what is on
        it, what each element does, and what a user can achieve here.  Uses
        stable, meaningful labels and excludes volatile runtime data so the
        per-activity description stays stable across revisits.
        """

        return {
            "name": "describe_screen",
            "description": (
                "Describe what is on the current activity screen: each element and what it "
                "does, and what a user can achieve here. Use stable, meaningful labels "
                "(button/tab/section names, what a card represents); do NOT include volatile "
                "content (specific prices, individual item names)."
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
                            "1-2 sentences: what this screen is for, which app section it "
                            "belongs to, and the primary tasks a user can do here."
                        ),
                    },
                    "elements": {
                        "type": "STRING",
                        "description": (
                            "Every interactive or informative element on the screen, one per "
                            "line, grouped by region (Top bar, Content, Bottom nav, etc). For "
                            "each: what it is + its stable label + what it does or where it leads.\n"
                            "Format:  [Region] element — label — what it does\n"
                            "GOOD: 'Top bar: Cart icon with item-count badge — opens the cart'; "
                            "'Content: Restaurant card — opens that restaurant's menu'; "
                            "'Bottom nav: Orders tab — switches to order history'.\n"
                            "Use stable labels (button/tab/section names, what a card represents). "
                            "Do NOT include volatile content (specific prices, individual item "
                            "names like '99 Slice Pizza' or '₹717') — describe the element TYPE "
                            "and its function instead.\n"
                            "Be exhaustive — every icon, tab, field, card type, and button."
                        ),
                    },
                    "achievable_actions": {
                        "type": "STRING",
                        "description": (
                            "The concrete things a user can accomplish on this screen, one per "
                            "line. Focus on outcomes and tasks, not individual taps.\n"
                            "Example: 'Search for restaurants'; 'Filter results by cuisine'; "
                            "'Open a restaurant to view its menu'; 'Switch to Orders or Profile "
                            "via the bottom navigation'."
                        ),
                    },
                },
                "required": [
                    "activity_name",
                    "screen_purpose",
                    "elements",
                    "achievable_actions",
                ],
            },
        }
