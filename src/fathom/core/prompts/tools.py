from __future__ import annotations

from typing import Any, Callable, Dict, FrozenSet, List, Optional

from fathom.constants.defect import DETECT_DEFECTS_TOOL, VISION_DEFECT_SIGNALS, DefectSeverity
from fathom.constants.screen import ScreenCategory
from fathom.constants.tools import ToolName


class ToolRegistry:
    """
    Registry for tool definitions used by the Vision Language Model.
    """

    @classmethod
    def get_all_definitions(cls) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns all tool definitions in a format compatible with Gemini API.
        """

        return {"function_declarations": [factory() for factory in cls.__factories().values()]}

    @classmethod
    def definitions(
        cls,
        *,
        names: FrozenSet[ToolName],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return the function-declarations payload for the requested tools.
        """

        factories = cls.__factories()
        ordered = [factories[name]() for name in factories if name in names]

        return {"function_declarations": ordered}

    @classmethod
    def __factories(cls) -> Dict[ToolName, Callable[[], Dict[str, Any]]]:
        """
        Return the name → declaration-factory map in catalog order.
        """

        return {
            ToolName.ASK_USER: cls.__ask_user,
            ToolName.EXECUTE_UI: cls.__execute_ui,
            ToolName.VERIFY_GOAL: cls.__verify_goal,
            ToolName.STORE_MEMORY: cls.__store_memory,
            ToolName.RECALL_MEMORY: cls.__recall_memory,
            ToolName.VALIDATE_STATE: cls.__validate_state,
        }

    @classmethod
    def get_export_definitions(
        cls, *, action_ids: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns tool definitions for script export composition.

        Args:
            action_ids: When provided, constrains action ID fields to only
                        these values via enum in the tool schema.
        """

        return {"function_declarations": [cls.__emit_script(action_ids=action_ids)]}

    @classmethod
    def get_exploration_definitions(cls) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns the explore_ui and describe_screen tools for exploration scans.
        """

        return {"function_declarations": [cls.__explore_ui(), cls.__describe_screen()]}

    @classmethod
    def get_translation_definitions(cls) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns the describe_screen tool for a standalone screen translation.
        """

        return {"function_declarations": [cls.__describe_screen()]}

    @classmethod
    def get_defect_definitions(cls) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns the detect_defects tool for a screenshot defect-inspection call.
        """

        return {"function_declarations": [cls.__detect_defects()]}

    @staticmethod
    def __explore_ui() -> Dict[str, Any]:
        """
        Definition for the explore_ui tool: pick the next untried element.
        """

        return {
            "name": "explore_ui",
            "description": (
                "Identify and tap the next untried interactive element on the current screen "
                "to discover new app screens. "
                "Use when there are untried interactive elements visible. "
                "Do NOT use when all visible interactive elements appear in the ALREADY TRIED "
                "list - set content_exhausted=true instead. "
                "SIDE EFFECTS: Taps a UI element on the device, which may navigate to a new screen."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "assistant_message": {
                        "type": "STRING",
                        "description": (
                            "Brief reasoning for choosing this element. State what it is, why it "
                            "has not been tried, and what you expect will happen."
                        ),
                    },
                    "action": {
                        "type": "OBJECT",
                        "description": "The exploration action to execute on the device.",
                        "properties": {
                            "action_type": {
                                "type": "STRING",
                                "description": (
                                    "Physical action to perform. TAP discrete elements. "
                                    "SCROLL / SWIPE_UP / SWIPE_DOWN a scrollable area to reveal "
                                    "content below the fold. SWIPE_LEFT / SWIPE_RIGHT horizontal "
                                    "carousels. TYPE into a search bar (also set `text`). "
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
                                    "visible on screen. Focus on what is novel about it."
                                ),
                            },
                            "target_name": {
                                "type": "STRING",
                                "description": (
                                    "Human-readable label exactly as it appears on screen. "
                                    "Examples: 'Home tab', 'Search icon', 'Add to Cart button'."
                                ),
                            },
                            "text": {
                                "type": "STRING",
                                "description": (
                                    "Text to type. REQUIRED when action_type is 'type'; ignored "
                                    "otherwise. Use a short, generic query (e.g. 'pizza', 'news', "
                                    "'a' as a cheap wildcard). Omit for any non-type action."
                                ),
                            },
                            "tap_target": {
                                "type": "OBJECT",
                                "description": (
                                    "CENTER point of the element to tap, in normalized 0-1000 "
                                    "coordinates. Place the point at the visual CENTER, not a corner."
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
                                    "What kind of UI element this is, matching the priority system: "
                                    "global_navigation=P1, primary_action=P2, content_item=P3, "
                                    "filter_or_category=P4, secondary_control=P5, "
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
                                    "top_bar, bottom_nav, content, modal, overlay, fab, or footer."
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
                                    "What you expect will happen after interacting with this element."
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
                                "description": (
                                    "Set true ONLY when this action dismisses an overlay, popup, "
                                    "or modal."
                                ),
                            },
                            "confidence": {
                                "type": "NUMBER",
                                "description": (
                                    "How confident you are that this element is interactive and "
                                    "untried (0.0-1.0)."
                                ),
                            },
                        },
                        "required": ["action_type", "rationale", "target_name", "tap_target"],
                    },
                    "screen_description": {
                        "type": "STRING",
                        "description": (
                            "1-2 sentence summary identifying the screen's purpose and app section."
                        ),
                    },
                    "content_exhausted": {
                        "type": "BOOLEAN",
                        "description": (
                            "Set true ONLY when every visible interactive element appears in the "
                            "ALREADY TRIED list AND scrolling has been attempted or is not "
                            "applicable. Defaults to false. NEVER set true while untried elements "
                            "are visible."
                        ),
                    },
                    "focus_relevance": {
                        "type": "STRING",
                        "description": (
                            "When GOAL names a focus, classify how THIS screen relates to it: "
                            "'on_focus' if part of the focused section/flow, 'leads_toward' if it "
                            "routes toward it, 'off_focus' if unrelated. Use 'unscoped' when GOAL is "
                            "generic or absent."
                        ),
                        "enum": [
                            "on_focus",
                            "leads_toward",
                            "off_focus",
                            "unscoped",
                        ],
                    },
                },
                "required": ["assistant_message", "action", "screen_description"],
            },
        }

    @staticmethod
    def __describe_screen() -> Dict[str, Any]:
        """
        Definition for the describe_screen tool: functional screen description.
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
                            "The Android activity name this screen belongs to, exactly as shown "
                            "in the context. One description per unique activity."
                        ),
                    },
                    "screen_category": {
                        "type": "STRING",
                        "description": (
                            "The functional kind of screen this is, from the user's point of "
                            "view. Choose the single best fit from the allowed values."
                        ),
                        "enum": [category.value for category in ScreenCategory],
                    },
                    "screen_purpose": {
                        "type": "STRING",
                        "description": (
                            "1-2 sentences: what this screen is for, which app section it belongs "
                            "to, and the primary tasks a user can do here."
                        ),
                    },
                    "elements": {
                        "type": "STRING",
                        "description": (
                            "Every interactive or informative element, one per line, grouped by "
                            "region (Top bar, Content, Bottom nav). For each: what it is + its "
                            "stable label + what it does or where it leads. Use stable labels; do "
                            "NOT include volatile content - describe the element TYPE and its "
                            "function. Be exhaustive."
                        ),
                    },
                    "achievable_actions": {
                        "type": "STRING",
                        "description": (
                            "The concrete things a user can accomplish on this screen, one per "
                            "line. Focus on outcomes and tasks, not individual taps."
                        ),
                    },
                },
                "required": [
                    "activity_name",
                    "screen_category",
                    "screen_purpose",
                    "elements",
                    "achievable_actions",
                ],
            },
        }

    @staticmethod
    def __detect_defects() -> Dict[str, Any]:
        """
        Definition for the detect_defects tool: report user-visible screen defects.
        """

        return {
            "name": DETECT_DEFECTS_TOOL,
            "description": (
                "Report every user-visible UI or content defect on the current screenshot. "
                "Return an empty list when the screen looks correct."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "defects": {
                        "type": "ARRAY",
                        "description": "User-visible defects on this screen; empty when none.",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "signal": {
                                    "type": "STRING",
                                    "description": "Which kind of defect this is.",
                                    "enum": [signal.value for signal in VISION_DEFECT_SIGNALS],
                                },
                                "severity": {
                                    "type": "STRING",
                                    "description": (
                                        "How badly it degrades the screen; omit for a default."
                                    ),
                                    "enum": [severity.value for severity in DefectSeverity],
                                },
                                "summary": {
                                    "type": "STRING",
                                    "description": "One-line description of the defect.",
                                },
                                "bounds": {
                                    "type": "OBJECT",
                                    "description": (
                                        "Normalized 0-1000 box around the defect, when localizable."
                                    ),
                                    "properties": {
                                        "x": {
                                            "type": "INTEGER",
                                            "description": "Left edge (0-1000).",
                                        },
                                        "y": {
                                            "type": "INTEGER",
                                            "description": "Top edge (0-1000).",
                                        },
                                        "width": {
                                            "type": "INTEGER",
                                            "description": "Width (0-1000).",
                                        },
                                        "height": {
                                            "type": "INTEGER",
                                            "description": "Height (0-1000).",
                                        },
                                    },
                                },
                            },
                            "required": ["signal", "summary"],
                        },
                    },
                },
                "required": ["defects"],
            },
        }

    @staticmethod
    def __execute_ui() -> Dict[str, Any]:
        """
        Definition for execute_ui tool.
        """

        return {
            "name": "execute_ui",
            "description": (
                "Execute a sequence of UI actions on the device to achieve a specific sub-goal "
                "or the final goal. Use this to interact with the app, including explicit "
                "validation checks via action_type='validate'. "
                "IMPORTANT: When launching a target app (when a package_name is known), prefer "
                "signaling app completion via 'goal_completed: true' or 'sub_goal_completed: true' "
                "rather than emitting an explicit 'tap' action on the app icon. The system will "
                "normalize app launch intents automatically. "
                "CRITICAL: For every UI action you MUST provide a concrete, user-facing target "
                "phrase via 'target_name' or 'script_target' (e.g., 'Search box', "
                "'Add to cart button', 'the first search result'). NEVER use placeholders like "
                "'UI Element', 'element', 'button', 'label', 'icon', 'field', or 'text' as the "
                "only target description."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "assistant_message": {
                        "type": "STRING",
                        "description": "A message to the user explaining the reasoning behind these actions.",
                    },
                    "action": {
                        "type": "OBJECT",
                        "description": "The single UI action to execute for this turn.",
                        "properties": {
                            # --- Critical execution fields (prioritized) ---
                            "action_type": {
                                "type": "STRING",
                                "description": "The type of action to perform. Use swipe_* for all scrolling gestures.",
                                "enum": [
                                    "tap",
                                    "type",
                                    "swipe_left",
                                    "swipe_right",
                                    "swipe_up",
                                    "swipe_down",
                                    "wait",
                                    "validate",
                                    "home",
                                    "back",
                                ],
                            },
                            "label_id": {
                                "type": "STRING",
                                "description": "The ID of the element from the manifest (e.g. '4'). REQUIRED when the target or scroll container exists in the manifest.",
                            },
                            "bbox": {
                                "type": "OBJECT",
                                "description": (
                                    "Bounding box for the action target. x,y are TOP-LEFT and "
                                    "width,height extend right/down. The rectangle MUST hug the "
                                    "visible glyph or icon pixels of the specific interactive "
                                    "control only — exclude surrounding card, shadow, halo, and "
                                    "empty padding. The runtime taps the GEOMETRIC CENTER of this "
                                    "rectangle. Use normalized values (0-1000) for visually "
                                    "estimated regions; set coordinate_system='pixel' when "
                                    "copying manifest or screenshot-space bounds."
                                ),
                                "properties": {
                                    "x": {"type": "INTEGER"},
                                    "y": {"type": "INTEGER"},
                                    "width": {"type": "INTEGER"},
                                    "height": {"type": "INTEGER"},
                                    "coordinate_system": {
                                        "type": "STRING",
                                        "enum": ["normalized", "pixel"],
                                        "description": "Coordinate system for bbox values. Default is 'normalized' (0-1000). Set 'pixel' only when using raw pixel coordinates.",
                                    },
                                },
                            },
                            "target_name": {
                                "type": "STRING",
                                "description": (
                                    "EXACT visible text or glyph label of the target control as "
                                    "rendered on screen, verbatim. Do NOT append interaction-kind "
                                    "suffixes such as 'button', 'icon', 'tab', 'link', 'chip', "
                                    "'cell', or 'row' — action_type already carries the "
                                    "interaction. For unlabelled icons, describe the visible "
                                    "symbol concisely (e.g., 'Magnifying glass'). MUST NOT be a "
                                    "generic placeholder like 'element', 'UI Element', 'button', "
                                    "'label', or 'icon' alone."
                                ),
                            },
                            "text_to_type": {
                                "type": "STRING",
                                "description": "Text to type (only for 'type' action).",
                            },
                            "wait_duration": {
                                "type": "NUMBER",
                                "description": "Duration to wait in seconds (e.g. 2.0, 5.0). Use this for 'wait' actions to specify how long to pause.",
                            },
                            # --- Execution signals ---
                            "confidence": {
                                "type": "NUMBER",
                                "description": "Required confidence level (0.0-1.0) for this action.",
                            },
                            "is_valid": {
                                "type": "BOOLEAN",
                                "description": "Self-correction: Is this action valid given the current screen state?",
                            },
                            # --- Non-critical metadata ---
                            "rationale": {
                                "type": "STRING",
                                "description": "Why this specific action is being taken.",
                            },
                            "validation_reason": {
                                "type": "STRING",
                                "description": "Reasoning for the validity judgment.",
                            },
                            "condition": {
                                "type": "STRING",
                                "description": (
                                    "Present-tense sentence describing the visible guard "
                                    "this action depends on. MANDATORY whenever "
                                    "is_conditional=true. Examples: 'Popup is visible', "
                                    "'Section is collapsed', 'Error banner is displayed', "
                                    "'Main menu is visible'. For a conditional wait, "
                                    "describe the awaited state (e.g., 'Search results are visible'), "
                                    "not the act of waiting."
                                ),
                            },
                            "is_conditional": {
                                "type": "BOOLEAN",
                                "description": (
                                    "Set true when this action should be executed only "
                                    "under a visible guard. When true, the 'condition' "
                                    "field is REQUIRED — describe the guard in the present tense."
                                ),
                            },
                            "conditional_type": {
                                "type": "STRING",
                                "enum": ["blocker", "transient", "error", "optional"],
                                "description": (
                                    "Condition class when is_conditional=true: "
                                    "blocker (overlay/popup/permission dialog blocking the UI), "
                                    "transient (spinner, skeleton shimmer, or splash screen that will auto-resolve), "
                                    "error (red/orange error banner, toast, or validation message), "
                                    "optional (non-blocking informational element)."
                                ),
                            },
                            "overlay_detected": {
                                "type": "BOOLEAN",
                                "description": (
                                    "Set true when the screenshot shows an overlay blocking the main UI "
                                    "(dimmed scrim, modal dialog, bottom sheet, permission prompt, or banner). "
                                    "This action must dismiss it."
                                ),
                            },
                            "export_target": {
                                "type": "STRING",
                                "description": (
                                    "The canonical phrase for this action in exported test scripts. "
                                    "Must be specific and human-readable (e.g., 'Search box', "
                                    "'the first search result', 'Add to cart button'). "
                                    "REQUIRED for tap, type, long_press, scroll, swipe, and wait actions. "
                                    "NEVER use generic placeholders like 'element', 'UI Element', "
                                    "'button', 'label', 'icon', 'field', or 'text' alone."
                                ),
                            },
                            "target_type": {
                                "type": "STRING",
                                "enum": ["stable", "positional", "dynamic"],
                                "description": "How the target should be referenced in exported scripts: stable (fixed label), positional (ordinal in list), or dynamic (content that may change). Leave unset if unsure.",
                            },
                            "script_target": {
                                "type": "STRING",
                                "description": (
                                    "When target_type is 'positional' or 'dynamic', the "
                                    "exact natural-language phrase that should appear in "
                                    "exported scripts (e.g. 'the first search result', "
                                    "'the promotional banner', 'the selected cart item'). "
                                    "Treat this field as REQUIRED whenever target_type is "
                                    "'positional' or 'dynamic'. The phrase MUST be specific "
                                    "and user-facing, not a generic placeholder."
                                ),
                            },
                            "scroll_target": {
                                "type": "STRING",
                                "description": (
                                    "REQUIRED for all scroll/swipe actions. The element or section being "
                                    "scrolled to find (e.g., 'Vitamins and supplements', 'Lab tests and "
                                    "packages'). Use the exact phrase from the UI when possible. "
                                    "When the manifest exposes the scrollable container, ground the action with that "
                                    "container's label_id and use scroll_target only for the intended content. "
                                    "Must not be empty for swipe_up, swipe_down, swipe_left, swipe_right, or scroll."
                                ),
                            },
                            "wait_subject": {
                                "type": "STRING",
                                "description": (
                                    "REQUIRED for all wait actions. What we're waiting for (e.g., 'app to "
                                    "load', 'search results to appear', 'Home page content'). Describe the "
                                    "expected state or element. Must not be empty for wait actions."
                                ),
                            },
                            "validation_subject": {
                                "type": "STRING",
                                "description": "For validate actions: what specifically is being validated (e.g., 'login status', 'banner visibility', 'item alignment'). Be specific about the validation target.",
                            },
                            "target_is_generic": {
                                "type": "BOOLEAN",
                                "description": "Set to true when this action taps/selects a non-specific target (e.g., 'any item', 'random category', 'first result'). Signals that target should be generalized in export.",
                            },
                            "target_element_type": {
                                "type": "STRING",
                                "enum": [
                                    "button",
                                    "icon",
                                    "option",
                                    "link",
                                    "field",
                                    "text",
                                    "checkbox",
                                ],
                                "description": "For tap/interact actions: the element type/role (button, icon, option, etc.). Helps refine target descriptions when product-specific elements are tapped.",
                            },
                            "validation_pattern": {
                                "type": "STRING",
                                "enum": ["blocker", "transient", "error", "generic"],
                                "description": (
                                    "For validate actions: blocker (modal/scrim/permission dialog blocking content), "
                                    "transient (spinner, progress bar, or shimmer placeholder still visible), "
                                    "error (red/orange text, error icon, or 'try again' message on screen), "
                                    "or generic (general state check like toggle position or text presence)."
                                ),
                            },
                            "wait_pattern": {
                                "type": "STRING",
                                "enum": ["ad", "splash", "load", "search", "generic"],
                                "description": (
                                    "For wait actions: ad (full-screen interstitial ad with countdown/skip button), "
                                    "splash (branded launch screen with app logo, no interactive elements), "
                                    "load (spinner, progress bar, skeleton/shimmer placeholders, or 'Loading...' text), "
                                    "search (waiting for search results list to populate), or generic."
                                ),
                            },
                        },
                        "required": ["action_type", "rationale", "is_valid", "confidence"],
                    },
                    "goal_completed": {
                        "type": "BOOLEAN",
                        "description": "True if the user's high-level goal is fully achieved after these actions.",
                    },
                    "goal_completion_reason": {
                        "type": "STRING",
                        "description": "Explicit reason why the goal is complete (e.g., 'Order placed successfully', 'Feature verified on screen'). Required when goal_completed=true.",
                    },
                    "sub_goal_completed": {
                        "type": "BOOLEAN",
                        "description": "True if the current sub-goal is completed after these actions. CONSTRAINT: You CANNOT skip sub-goals. All sub-goals must be executed in order. If you cannot complete the current sub-goal, ask the user for help or explain the blockage. Do NOT emit any signal that means 'skip this sub-goal'.",
                    },
                    "subgoal_completion_reason": {
                        "type": "STRING",
                        "description": "Explicit reason why the sub-goal is complete (e.g., 'Item added to cart', 'User authenticated'). Required when sub_goal_completed=true.",
                    },
                    "completion_criteria_met": {
                        "type": "ARRAY",
                        "description": "List of criteria/conditions that triggered completion (e.g., ['payment_processed', 'order_confirmed']). Use for multi-condition completions.",
                        "items": {"type": "STRING"},
                    },
                    "content_exhausted": {
                        "type": "BOOLEAN",
                        "description": "True when scrolling/swiping reveals no more new content and the list/feed appears exhausted.",
                    },
                    "previous_screen_summary": {
                        "type": "STRING",
                        "description": "Optional short summary of the previous screen for semantic delta analysis.",
                    },
                    "current_screen_summary": {
                        "type": "STRING",
                        "description": "Optional short summary of the current screen for semantic delta analysis.",
                    },
                    "delta_observed": {
                        "type": "BOOLEAN",
                        "description": "Whether a meaningful screen change was observed since the previous screenshot. Strongly RECOMMENDED when the model can assess semantic deltas; omit only when unsure or when no comparison is possible.",
                    },
                    "delta_reasoning": {
                        "type": "STRING",
                        "description": "Optional rationale behind the delta_observed hint.",
                    },
                    "delta_confidence": {
                        "type": "NUMBER",
                        "description": "Confidence score (0.0-1.0) for delta_observed. Strongly RECOMMENDED whenever delta_observed is provided; omit only when the model cannot estimate confidence.",
                    },
                    "visible_anchors": {
                        "type": "ARRAY",
                        "description": "Optional key UI anchors currently visible (e.g., top card title, footer label).",
                        "items": {"type": "STRING"},
                    },
                    "top_anchor": {
                        "type": "STRING",
                        "description": "Optional anchor near the top of the viewport.",
                    },
                    "bottom_anchor": {
                        "type": "STRING",
                        "description": "Optional anchor near the bottom of the viewport.",
                    },
                    "memory_updates": {
                        "type": "OBJECT",
                        "description": "Optional key-value pairs to update in persistent memory. Use this to track progress (e.g., 'visited_card1': 'true').",
                    },
                },
                "required": [
                    "assistant_message",
                    "action",
                    "goal_completed",
                    "sub_goal_completed",
                ],
            },
        }

    @staticmethod
    def __validate_state() -> Dict[str, Any]:
        """
        Definition for validate_state tool.
        """

        return {
            "name": "validate_state",
            "description": (
                "Verify if the screen state matches specific criteria by inspecting the screenshot. "
                "Use when the intent implies checking or verifying something: "
                "e.g., a toggle is green/on, a success toast is visible, a specific tab is highlighted, "
                "or an expected heading/text is displayed on screen."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "assistant_message": {
                        "type": "STRING",
                        "description": "Explanation of the verification result.",
                    },
                    "condition_to_verify": {
                        "type": "STRING",
                        "description": "The condition being verified (e.g., 'Settings screen is open').",
                    },
                    "condition_met": {
                        "type": "BOOLEAN",
                        "description": "True if the condition is met based on the visual evidence.",
                    },
                    "evidence": {
                        "type": "STRING",
                        "description": (
                            "Visual evidence from the screenshot supporting the conclusion "
                            "(e.g., 'Toggle is green and positioned right', 'Success banner shows Order Confirmed', "
                            "'Error text not visible anywhere on screen')."
                        ),
                    },
                    "goal_completed": {
                        "type": "BOOLEAN",
                        "description": "True only if the overall goal is complete.",
                    },
                    "sub_goal_completed": {
                        "type": "BOOLEAN",
                        "description": "True if the current sub-goal is complete.",
                    },
                },
                "required": [
                    "assistant_message",
                    "condition_to_verify",
                    "condition_met",
                    "evidence",
                    "goal_completed",
                    "sub_goal_completed",
                ],
            },
        }

    @staticmethod
    def __verify_goal() -> Dict[str, Any]:
        """
        Definition for verify_goal tool.
        """

        return {
            "name": "verify_goal",
            "description": "Verify if the user's overall goal has been fully completed by checking the current screen state.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "assistant_message": {
                        "type": "STRING",
                        "description": "Explanation of the goal completion status.",
                    },
                    "goal_completed": {
                        "type": "BOOLEAN",
                        "description": "True if the overall goal is FULLY completed based on screen state.",
                    },
                    "goal_completion_reason": {
                        "type": "STRING",
                        "description": "Explicit reason why the goal is complete (e.g., 'Order placed successfully', 'Login screen visible as requested'). Required when goal_completed=true.",
                    },
                    "sub_goal_completed": {
                        "type": "BOOLEAN",
                        "description": "True if the current sub-goal is completed based on screen state.",
                    },
                    "subgoal_completion_reason": {
                        "type": "STRING",
                        "description": "Explicit reason why the sub-goal is complete (e.g., 'Product page rendered after tapping result'). Required when sub_goal_completed=true.",
                    },
                    "completion_criteria_met": {
                        "type": "ARRAY",
                        "description": "List of observable criteria that triggered completion (e.g., ['title_visible', 'price_visible']). Optional; use for multi-condition completions.",
                        "items": {"type": "STRING"},
                    },
                    "current_screen": {
                        "type": "STRING",
                        "description": "The actual screen currently displayed.",
                    },
                    "evidence": {
                        "type": "STRING",
                        "description": "Visual evidence proving goal completion.",
                    },
                },
                "required": [
                    "assistant_message",
                    "goal_completed",
                    "sub_goal_completed",
                    "current_screen",
                    "evidence",
                ],
            },
        }

    @staticmethod
    def __store_memory() -> Dict[str, Any]:
        """
        Definition for store_memory tool.
        """

        return {
            "name": "store_memory",
            "description": "Store important information, progress, or state in memory to remember it later. Use this to track what you've already done (e.g., 'interacted with card 1 in carousel').",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "key": {
                        "type": "STRING",
                        "description": "The key under which to store the information.",
                    },
                    "value": {
                        "type": "STRING",
                        "description": "The information to store.",
                    },
                    "assistant_message": {
                        "type": "STRING",
                        "description": "Explanation of what is being saved.",
                    },
                },
                "required": ["key", "value", "assistant_message"],
            },
        }

    @staticmethod
    def __recall_memory() -> Dict[str, Any]:
        """
        Definition for recall_memory tool.
        """

        return {
            "name": "recall_memory",
            "description": "Retrieve previously stored information from memory. Use this to check your progress or recall specific details.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "key": {
                        "type": "STRING",
                        "description": "The key of the information to retrieve.",
                    },
                    "assistant_message": {
                        "type": "STRING",
                        "description": "Why this information is being retrieved.",
                    },
                },
                "required": ["key", "assistant_message"],
            },
        }

    @staticmethod
    def __ask_user() -> Dict[str, Any]:
        """
        Definition for ask_user tool.
        """

        return {
            "name": "ask_user",
            "description": "Ask the human user for clarification, guidance, or assistance when stuck or uncertain. Use this ONLY in interactive mode.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "question": {
                        "type": "STRING",
                        "description": "The specific question or request for the user.",
                    },
                    "context": {
                        "type": "STRING",
                        "description": "Context explaining what you were trying to do and why you need help.",
                    },
                    "goal_completed": {
                        "type": "BOOLEAN",
                        "description": "Must be false unless the overall goal is complete.",
                    },
                    "sub_goal_completed": {
                        "type": "BOOLEAN",
                        "description": "Must be false unless the current sub-goal is complete.",
                    },
                },
                "required": ["question", "goal_completed", "sub_goal_completed"],
            },
        }

    @staticmethod
    def __emit_script(
        action_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Definition for script export output tool.
        """

        # When action_ids are provided, constrain the schema so Gemini can
        # only output valid catalog IDs (prevents missing/extra/duplicated IDs).
        action_id_item: Dict[str, Any] = {"type": "STRING"}
        if action_ids:
            action_id_item["enum"] = list(action_ids)

        return {
            "name": "emit_script",
            "description": (
                "Return structured script sections derived only from allowed step action lines. "
                "Do not paraphrase executable actions. Rendered scripts use IF <condition> on one line "
                "and the opening { on the following line before indented block body lines."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "conditional_blocks": {
                        "type": "ARRAY",
                        "description": "Ordered IF blocks for condition-scoped actions using action IDs.",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "condition": {
                                    "type": "STRING",
                                    "description": "Condition text for IF block.",
                                },
                                "condition_type": {
                                    "type": "STRING",
                                    "enum": ["blocker", "transient", "error", "optional"],
                                    "description": (
                                        "Classification of this condition: blocker (popup/permission/consent "
                                        "that blocks progress), transient (loading/splash/spinner that will pass), "
                                        "error (error message that may appear), or optional (nice-to-have check)."
                                    ),
                                },
                                "action_ids": {
                                    "type": "ARRAY",
                                    "description": "Executable action IDs under this IF block. Must be selected from provided action catalog.",
                                    "items": action_id_item,
                                },
                            },
                            "required": ["condition", "action_ids"],
                        },
                    },
                    "remaining_action_ids": {
                        "type": "ARRAY",
                        "description": "Ordered executable action IDs outside IF blocks. Must be selected from provided action catalog.",
                        "items": action_id_item,
                    },
                    "action_validations": {
                        "type": "OBJECT",
                        "description": (
                            "Optional map of catalog action_id -> intermediate validation line after that action "
                            "(e.g. list or results visible right after a search or scroll). Each value must start "
                            "with 'Validate'. Use for mid-flow checks; do not put those in final_validation."
                        ),
                        "additionalProperties": {"type": "STRING"},
                    },
                    "final_validation": {
                        "type": "STRING",
                        "description": (
                            "Single terminal UI-state line after the last catalog action. Must start with 'Validate'. "
                            "State only: what screen/page/primary content is visible or displayed. One short clause—"
                            "no tap/click/type/select/navigate/search instructions (those belong in catalog actions "
                            "or action_validations). No chained 'and then' procedures."
                        ),
                    },
                },
                "required": ["remaining_action_ids", "final_validation"],
            },
        }
