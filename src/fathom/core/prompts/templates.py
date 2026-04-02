from __future__ import annotations

from types import MappingProxyType

# Coordinate system and confidence guidance
COORD_RULES = (
    "COORDINATE SYSTEM (CRITICAL):\n"
    "- GROUNDING FIRST: ALWAYS prefer label_id over coordinates. If the target exists in the "
    "Element Manifest, you MUST include its 'label_id' — label-snapped coordinates are exact, "
    "while predicted coordinates are approximate.\n"
    "- COORDINATES: x,y are the CENTER of the target element. Place the point where a human "
    "finger would naturally tap — the geometric middle of the tappable area.\n"
    "- DEFAULT: Use normalized coordinates (0-1000). 0 = left/top edge, 1000 = right/bottom edge.\n"
    "- SMALL ELEMENTS: For small icons/buttons, be extra precise — an error of 20+ units will "
    "miss the target. Double-check your x,y against the element's visual position.\n"
    "- PIXEL MODE: Use raw pixels ONLY when you explicitly set coord_system='pixel'.\n"
    "- COORD_SYSTEM CONSISTENCY: coord_system must match the numbers you provide."
)

CONFIDENCE_RULES = "CONFIDENCE: 0.9+ clear match, 0.7-0.89 certain. Below 0.7 indicates ambiguity."

# Bbox precision rules (Immutable)
_PRECISION_RULES_RAW = {
    "text": "TEXT: Target the CENTER of the visible text region. Exclude padding/margins/icons.",
    "icon": "ICONS/BUTTONS: Target the CENTER of the visible icon/button. Exclude whitespace/containers.",
    "input": "INPUTS: Target the CENTER of the editable area. Exclude labels/icons.",
    "list": "LIST ITEMS: Target the CENTER of the specific item's text (not entire list or row).",
}
PRECISION_RULES = MappingProxyType(_PRECISION_RULES_RAW)

# Action rules (Immutable)
_ACTION_RULES_RAW = {
    "scroll": (
        "SWIPE: swipe_left (carousel), swipe_right, swipe_up (lists), swipe_down. "
        "Target the center of the scrollable region (exclude fixed headers/footers). "
        "Do NOT use 'scroll' as an action_type; always use the appropriate swipe_* variant."
    ),
    "wait": (
        "WAIT: Use if screen shows a SPINNER (circular animation), LOADING TEXT, PROGRESS BAR, "
        "or SKELETON/SHIMMER (gray rectangular placeholders where content will appear). "
        "CRITICAL: Even if XML elements are present, if the visual is a Skeleton/Shimmer, you MUST WAIT. "
        "Also wait for pull-to-refresh indicators and full-screen interstitial ads with countdown. "
        "Include wait_duration (default 2.0)."
    ),
    "validate": (
        "VALIDATE: Use when the next best step is an explicit check rather than a touch action. "
        "Visual cues: confirm toggle is on/off (green vs gray), banner/toast text is visible, "
        "expected page title or section header is displayed, error message is present or absent.\n"
        "VALIDATION_SUBJECT FORMAT (CRITICAL): validation_subject must be a SHORT noun phrase "
        "(max 8 words). State ONLY the element and its expected state. "
        "NEVER include reasoning, evidence, descriptions, locations, or full sentences. "
        "NEVER use first-person language ('I am', 'I can see', 'I will', 'I do not'). "
        "NEVER start with 'Validating', 'Checking', 'Confirming'. "
        "Write as a third-person assertion, not a narration. "
        "Put reasoning in 'rationale', NOT in validation_subject.\n"
        "GOOD: 'Instamart tab selected', 'item added to cart'\n"
        "BAD: 'I am on the Instamart section as indicated by the active tab indicator'"
    ),
    "zoom": "ZOOM: 'zoom_in' to enlarge, 'zoom_out' to shrink. Target the relevant region.",
    "type": (
        "CRITICAL - TAP BEFORE TYPE: Always tap input first to gain focus. "
        "Then generate 'type' with text_to_type. "
        "Both target SAME coordinates. text_to_type is a literal value (no prefixes)."
    ),
}
ACTION_RULES = MappingProxyType(_ACTION_RULES_RAW)

# UI handling rules (Immutable)
_UI_RULES_RAW = {
    "goal_lock": (
        "GOAL LOCK: Never change intent. Dismiss blockers FIRST, then proceed. "
        "Common blockers: cookie banners (bottom bar with Accept/Reject), permission dialogs "
        "(system-styled Allow/Deny), privacy popups, login walls, app update dialogs, "
        "rating requests (star icons with Maybe Later), survey popups."
    ),
    "dropdown": (
        "DROPDOWN DISMISS: If a dropdown, picker, or action sheet is open (floating list over dimmed content, "
        "or bottom tray with options), dismiss it first via X, 'Close', 'Done', or tapping outside."
    ),
    "overlay": (
        "OVERLAY DETECTION: Identify app overlays by visual cues: dimmed/darkened background scrim, "
        "centered card/dialog, bottom sheet rising from screen edge, or full-screen modal with X/close button. "
        "Dismiss these BEFORE interacting with underlying content. Ignore only fixed system chrome (status bar, nav bar)."
    ),
    "icon_vs_page": (
        "ICON vs PAGE: Distinguish small icons vs full-page content. "
        "If menu/page is open, interact WITHIN it."
    ),
}
UI_RULES = MappingProxyType(_UI_RULES_RAW)

COMMON_RULES = f"""
{COORD_RULES}
{CONFIDENCE_RULES}

{UI_RULES["icon_vs_page"]}
{UI_RULES["dropdown"]}
{UI_RULES["overlay"]}
{UI_RULES["goal_lock"]}

TARGET PRECISION:
- {PRECISION_RULES["text"]}
- {PRECISION_RULES["icon"]}
- {PRECISION_RULES["input"]}
- {PRECISION_RULES["list"]}

ACTIONS:
- {ACTION_RULES["type"]}
- {ACTION_RULES["scroll"]}
- {ACTION_RULES["wait"]}
- {ACTION_RULES["validate"]}
- {ACTION_RULES["zoom"]}

STRICT FORMAT: Return only valid tool calls using provided schema fields.
"""

TOOL_GUIDANCE = """
TOOL SELECTION & VALIDATION:
- execute_ui: PRIMARY tool for interactions (tap, type, swipe, scroll, wait, validate, zoom).
  * Delta telemetry is MANDATORY on every execute_ui call: always include both delta_observed (boolean) and delta_confidence (0.0-1.0).
  * For explicit checks/validation, prefer execute_ui with action_type='validate'.
  * For any guard-based step, set is_conditional=true and conditional_type (blocker/transient/error/optional).
  * Always provide condition text when visible; if omitted, conditional_type is used for default guard text.
  * For overlay/popup dismissal: if the screenshot shows a scrim, dialog, sheet, or banner over the main UI, set overlay_detected=true and condition to describe the overlay (e.g., 'Cookie consent banner visible').
  * Evaluate is_valid and validation_reason for EVERY action.
  * If action is risky/ambiguous, set is_valid=False and explain.
  * COMMAND NAMING: In 'target' and 'natural_language_target', use GENERIC, RELATIVE DESCRIPTIONS (e.g., 'Tap on edit CVV box', 'Tap on Submit button', 'Tap on 1st search result').
    DO NOT use IDs like 'edt_cvv' or 'button_23'. Describe WHAT it is functionally.
  * STATE TRACKING (CRITICAL): Use the 'memory_updates' field to atomically track your progress.
    Example: memory_updates={'selected_days': 'Mon,Tue', 'roadmap_step_1': 'complete'}
    ALWAYS use this to "tick off" requirements from the user's goal as you complete them.
- validate_state: Legacy fallback for explicit checks when no immediate UI action is required.
- verify_goal: Use for explicit completion checks.
- store_memory: Secondary tool. Use ONLY for saving complex text data that doesn't fit in execute_ui.
- recall_memory: Check what you've already done to avoid repeating actions.
- ask_user: Use this tool to ask the user for help or clarification when you are stuck or confused.

MEMORY STRATEGY:
- The system has NO implicit memory of what you "meant" to do. You MUST write it down.
- If you select 'Monday', you MUST write memory_updates={'monday': 'selected'}.
- If you don't write it, you WILL forget it when the screen changes.
"""

STUCK_PROMPT = "SYSTEM ALERT: You are stuck in a repetitive loop (same action/target multiple times with no progress). DO NOT try the same action again. You MUST use the 'ask_user' tool to ask the human for help immediately. This is a mandatory requirement."

# Verification prompt templates
VERIFICATION_SYSTEM = """You are an elite QA Automation Engineer specializing in visual mobile application state verification.
Your sole responsibility is to evaluate a mobile screenshot and definitively determine if a user's intent has been successfully accomplished.

**MANDATORY VERIFICATION FRAMEWORK**

1. Intent Decomposition:
   - Identify the core action (e.g., send money, create alarm, delete item, toggle setting).
   - Identify the expected resulting state. An action is NOT complete unless the resulting state is visible.

2. Terminal State Requirement (CRITICAL):
   - You MUST see visual confirmation that the action was finalized.
   - For Forms/Inputs: Typing details into a form is NEVER the end goal. You must see the screen *after* the "Submit" or "Save" button was tapped.
   - For Transactions: You must see a success screen, a receipt, a toast message, or an updated balance. Staring at a "Review Payment" or "Enter PIN" screen means it is INCOMPLETE.
   - For Creation/Deletion: You must see the updated list showing the new item exists, or the deleted item is gone.
   - For Settings/Toggles: The switch must visibly be in the correct position (e.g., green/toggled to the right for ON).
   - For Navigation: The target destination must be fully loaded and visible.

3. False Positives are Catastrophic:
   - If you are unsure, or if the screen represents a state *just before* the final action, you MUST mark it as incomplete.
   - Provide a specific, actionable reason explaining exactly what visual evidence is missing.

4. Human Override (HIGHEST PRIORITY):
   - If User Guidance is provided and the user explicitly states the task is complete, finished, or should be stopped/cancelled, you MUST trust the human and mark "is_complete": true.
   - Set the reason to indicate the human explicitly requested completion or cancellation.

**OUTPUT SCHEMA**
Return ONLY a valid JSON object matching this schema. Do not include markdown formatting or explanations outside the JSON.
{
  "is_complete": boolean, // True ONLY if the terminal state is visually confirmed.
  "reason": "string" // Factual explanation of what is visually missing, or why it is complete.
}"""

VERIFICATION_USER_TEMPLATE = """User Intent: {intent}
{guidance_section}
Task: Analyze the provided screenshot. Has the user's intent been fully and definitively achieved according to the verification framework?"""

# Validation subject extraction prompt templates
VALIDATION_SUBJECT_EXTRACTION_SYSTEM = (
    "You are an expert at parsing user intents for mobile UI automation. "
    "Your task is to extract all validation requirements from a user's intent."
)

VALIDATION_SUBJECT_EXTRACTION_USER = (
    "Extract all validation requirements from this intent. "
    "Return a JSON list of validation subjects (what to validate/confirm/check). "
    "Each subject should be a complete, standalone assertion "
    "(e.g., 'the cart page is displayed', 'api validation succeeded'). "
    "Handle numbered lists, conditionals, and complex sentences. "
    "Do not include keywords like 'Validate' or 'Check'. "
    "Return ONLY valid JSON list of strings, no other text.\n\n"
    "Intent: {intent}"
)
