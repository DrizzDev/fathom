from __future__ import annotations

# Coordinate system and confidence guidance
COORD_RULES = (
    "COORDINATES: Use NORMALIZED coords (0-1000 grid). x=0,y=0 is top-left. "
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
        "Bbox wraps scrollable region only (exclude fixed headers/footers)."
    ),
    "wait": (
        "WAIT: ONLY for active loading (skeleton, spinner, 'Loading...' text). "
        "NOT for sparse screens with visible text/buttons. Include wait_duration_ms (default 2000ms)."
    ),
    "zoom": "ZOOM: 'zoom_in' to enlarge, 'zoom_out' to shrink. Target the relevant region.",
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
}

# Backward-compatible aggregated blocks used by current builder
COMMON_RULES = f"""
{COORD_RULES}
{CONFIDENCE_RULES}

{UI_RULES["icon_vs_page"]}
{UI_RULES["dropdown"]}
{UI_RULES["overlay"]}
{UI_RULES["goal_lock"]}

BBOX PRECISION:
- {PRECISION_RULES["text"]}
- {PRECISION_RULES["icon"]}
- {PRECISION_RULES["input"]}
- {PRECISION_RULES["list"]}

ACTIONS:
- {ACTION_RULES["type"]}
- {ACTION_RULES["scroll"]}
- {ACTION_RULES["wait"]}
- {ACTION_RULES["zoom"]}

STRICT FORMAT: Return only valid tool calls using provided schema fields.
"""

TOOL_GUIDANCE = """
TOOL SELECTION & VALIDATION:
- execute_ui: PRIMARY tool for interactions (tap, type, swipe, scroll, wait, zoom).
  * Evaluate is_valid and validation_reason for EVERY action.
  * If action is risky/ambiguous, set is_valid=False and explain.
- validate_state: Use for explicit state checks when no immediate UI action is required.
- verify_goal: Use for explicit completion checks.
"""
