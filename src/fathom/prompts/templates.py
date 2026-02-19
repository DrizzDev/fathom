from __future__ import annotations

# Coordinate system and confidence guidance
COORD_RULES = (
    "COORDINATES: Use NORMALIZED coords (0-1000 grid). "
    "bbox.x and bbox.y MUST be the TOP-LEFT corner of the element bounding box. "
    "width and height extend rightward and downward from that corner. "
    "Format: {x, y, width, height, coord_system:'normalized'}."
)

CONFIDENCE_RULES = "CONFIDENCE: 0.9+ clear match, 0.7-0.89 certain. Below 0.7 indicates ambiguity."

# Bbox precision rules
PRECISION_RULES = {
    "text": "TEXT: Bbox wraps ONLY visible text pixels. Exclude padding/margins/icons.",
    "icon": "ICONS/BUTTONS: Snap bbox TIGHTLY to visible edges. Exclude whitespace/containers.",
    "input": "INPUTS: Wrap editable area only (borders/background). Exclude labels/icons.",
    "list": "LIST ITEMS: Wrap ONLY the specific item's text (not entire list or row).",
}

# Action rules
ACTION_RULES = {
    "scroll": (
        "SCROLL/SWIPE: swipe_left (carousel), swipe_right, swipe_up (lists), swipe_down. "
        "Bbox wraps scrollable region only (exclude fixed headers/footers). "
        "If the screen does not change after swiping (same items visible, carousel at last dot, "
        "bounce effect), set content_exhausted=true to signal end of scrollable content."
    ),
    "wait": (
        "WAIT: ONLY for active loading (skeleton, spinner, 'Loading...' text). "
        "NOT for sparse screens with visible text/buttons. Include wait_duration_ms (default 2000ms)."
    ),
    "zoom": "",
    "type": (
        "CRITICAL - TAP BEFORE TYPE: Always tap input first to gain focus. "
        "Then generate 'type' with text_to_type. "
        "Both target SAME bbox. text_to_type is a literal value (no prefixes)."
    ),
}

# UI handling rules
UI_RULES = {
    "goal_lock": (
        "GOAL LOCK: Never change intent. Dismiss blockers (cookie consent, permission prompts, "
        "privacy popups, login walls, app update dialogs, survey popups, rating requests) FIRST, "
        "then proceed."
    ),
    "dropdown": "DROPDOWN DISMISS: Dismiss obstructions first: down arrows, X, 'Close'/'Done'.",
    "overlay": "OVERLAY: Ignore system overlays. Focus on actual app UI elements.",
    "icon_vs_page": (
        "ICON vs PAGE: Distinguish small icons vs full-page content. "
        "If menu/page is open, interact WITHIN it."
    ),
    "conditional": (
        "CONDITIONAL: When handling ANY optional/dynamic element (e.g. 'Close' on popup, 'Expand' on collapsed section, "
        "'Skip' on tutorial, 'Retry' on error), SET THE `condition` FIELD to the visible requirement "
        "(e.g. 'Promo popup is visible', 'Section is collapsed', 'Error message is displayed')."
    ),
}

# Backward-compatible aggregated blocks used by current builder
COMMON_RULES = f"""
{COORD_RULES}
{CONFIDENCE_RULES}

{UI_RULES["icon_vs_page"]}
{UI_RULES["dropdown"]}
{UI_RULES["overlay"]}
{UI_RULES["goal_lock"]}
{UI_RULES["conditional"]}

BBOX PRECISION:
- {PRECISION_RULES["text"]}
- {PRECISION_RULES["icon"]}
- {PRECISION_RULES["input"]}
- {PRECISION_RULES["list"]}

ACTIONS:
- {ACTION_RULES["type"]}
- {ACTION_RULES["scroll"]}
- {ACTION_RULES["wait"]}

STRICT FORMAT: Return only valid tool calls using provided schema fields.
"""

TOOL_GUIDANCE = """
TOOL SELECTION & VALIDATION:
- execute_ui: PRIMARY tool for physical UI interactions (tap, type, swipe, scroll, wait).
  * Use this for ALL device interactions. Do NOT include goal completion here.
  * ALWAYS set screen_description: goal-relevant screen state in ≤15 words.
  * Evaluate is_valid and validation_reason for EVERY action.
  * If action is risky/ambiguous, set is_valid=False and explain.
  * COMMAND NAMING: In 'target' and 'natural_language_target':
    - Use GENERIC, RELATIVE DESCRIPTIONS. Describe WHAT it is functionally.
    - DO NOT use IDs like 'edt_cvv' or 'button_23'.
    - LIST/COLLECTION ITEMS: When tapping an item in a list, grid, carousel, or search results,
      use POSITIONAL references: 'the 1st search result', 'the 2nd card', 'the 3rd item in the list'.
      Do NOT use the item's specific content text (e.g., avoid 'Optimum Nutrition protein powder').
    - UNIQUE UI ELEMENTS: For buttons, tabs, inputs, toggles, use their visible label
      (e.g., 'Submit button', 'Search tab', 'CVV input field').
  * TARGET CLASSIFICATION (action.target_type + action.script_target):
    Set these to help generate reusable test scripts that survive content changes.
    - target_type='stable': Permanent UI element (button, tab, input). Omit script_target.
    - target_type='positional': Item in a list/grid/carousel/search results.
      Set script_target to ordinal reference: 'the first search result', 'the second card'.
    - target_type='dynamic': Changing content not in a list (banner, notification, loading state).
      Set script_target to generic description: 'the promotional banner', 'app loading screen'.
    - WAIT ACTIONS: Always set target_type='dynamic' and script_target describing what you are
      waiting for (e.g. 'app loading screen', 'ad player', 'content to load'). Never leave
      target_name as 'UI Element' for wait actions.
    - Omit both fields if uncertain.
- complete_goal: DEDICATED completion signal. Call this ONLY when the screen proves the goal is done.
  * Do NOT call this while there are still actions to perform.
  * Provide visual evidence from the current screen.
- validate_state: Use for explicit state checks when no immediate UI action is required.
- verify_goal: Use for explicit completion checks with detailed evidence.
"""

TOOL_SCHEMAS = {
    "execute_ui": (
        "execute_ui: assistant_message (str), action (object with {action_type, rationale, is_valid}; "
        "optional: target_name, bbox {x,y,width,height,coord_system}, text_to_type, confidence 0.0-1.0, "
        "validation_reason, target_type (enum: stable/positional/dynamic), script_target (str)). "
        "screen_description (str, goal-relevant screen state ≤15 words). "
        "Optional: content_exhausted (bool), memory_updates (dict)."
    ),
    "complete_goal": "complete_goal: assistant_message (str), evidence (str).",
    "validate_state": (
        "validate_state: assistant_message (str), condition_to_verify (str), condition_met (bool), "
        "evidence (str), goal_completed (bool)."
    ),
    "verify_goal": (
        "verify_goal: assistant_message (str), goal_completed (bool), current_screen (str), evidence (str)."
    ),
    "store_memory": (
        "store_memory: category (enum: 'visited','progress','state','data'), "
        "item (str, snake_case identifier), value (str), assistant_message (str). "
        "Key is formed as category.item (e.g., visited.carousel_card_1)."
    ),
    "recall_memory": (
        "recall_memory: category (enum: 'visited','progress','state','data'), "
        "item (str, snake_case identifier), assistant_message (str). "
        "Use the EXACT same category and item from the corresponding store_memory call."
    ),
}

MODE_TOOLS = {
    "default": [
        "execute_ui",
        "complete_goal",
        "validate_state",
        "verify_goal",
        "store_memory",
        "recall_memory",
    ],
    "interaction": ["execute_ui", "complete_goal", "store_memory", "recall_memory"],
    "discovery": ["execute_ui", "complete_goal", "store_memory"],
    "verification": [
        "execute_ui",
        "complete_goal",
        "validate_state",
        "verify_goal",
        "store_memory",
        "recall_memory",
    ],
    "exploration": ["execute_ui"],
}


def build_output_schema(mode: str) -> str:
    """
    Assemble the output schema block for a given mode.
    Maps mode to available tools and returns schema-anchored instructions.
    """
    tool_names = MODE_TOOLS.get(mode, MODE_TOOLS["default"])
    schemas = [TOOL_SCHEMAS[t] for t in tool_names if t in TOOL_SCHEMAS]
    header = (
        "OUTPUT SCHEMA — You MUST respond with exactly ONE tool call. Never output plain text or markdown.\n"
        "Select the appropriate tool and provide ALL required fields:\n\n"
    )
    return header + "\n\n".join(schemas)
