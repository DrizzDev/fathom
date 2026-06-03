from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from fathom.constants.tools import ToolName
from fathom.schemas.tools import AllowedTools

# Coordinate system and confidence guidance
COORD_RULES = (
    "COORDINATE SYSTEM (CRITICAL):\n"
    "- GROUNDING: IF the target exists in the Element Manifest, you MUST include its 'label_id' (e.g., label_id='4').\n"
    "- BBOX SHAPE: x,y are TOP-LEFT; width,height extend right/down.\n"
    "- DEFAULT: Use normalized coordinates (0-1000) only for visually estimated regions.\n"
    "- MANIFEST REGIONS: If you are copying bounds from the manifest or from the screenshot resolution, you MUST set coordinate_system='pixel'.\n"
    "- PIXEL MODE: Use raw pixels ONLY when you explicitly set coordinate_system='pixel'.\n"
    "- COORD_SYSTEM CONSISTENCY: coordinate_system must match the numbers you provide."
)

CONFIDENCE_RULES = "CONFIDENCE: 0.9+ clear match, 0.7-0.89 certain. Below 0.7 indicates ambiguity."

# Bbox precision rules (Immutable)
_PRECISION_RULES_RAW = {
    "text": "TEXT: Bbox wraps ONLY visible text pixels. Exclude padding/margins/icons.",
    "icon": "ICONS/BUTTONS: Snap bbox TIGHTLY to visible edges. Exclude whitespace/containers.",
    "input": "INPUTS: Wrap editable area only (borders/background). Exclude labels/icons.",
    "list": "LIST ITEMS: Wrap ONLY the specific item's text (not entire list or row).",
}
PRECISION_RULES = MappingProxyType(_PRECISION_RULES_RAW)

# Action rules (Immutable)
_ACTION_RULES_RAW = {
    "scroll": (
        "SWIPE: swipe_left (carousel), swipe_right, swipe_up (lists), swipe_down. "
        "If a manifest-backed scrollable container exists, ground the swipe to that container via label_id first. "
        "Use bbox only when no manifest container matches the intended scroll surface. "
        "Bbox wraps scrollable region only (exclude fixed headers/footers). "
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
        "expected page title or section header is displayed, error message is present or absent."
    ),
    "zoom": "ZOOM: 'zoom_in' to enlarge, 'zoom_out' to shrink. Target the relevant region.",
    "type": (
        "CRITICAL - TAP BEFORE TYPE: Always tap input first to gain focus. "
        "Then generate 'type' with text_to_type. "
        "Both target SAME bbox. text_to_type is a literal value (no prefixes)."
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

BBOX PRECISION:
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

_EXECUTE_UI_GUIDANCE = (
    "- execute_ui: PRIMARY tool for interactions (tap, type, swipe, scroll, wait, validate, zoom).\n"
    "  * Delta telemetry is MANDATORY on every execute_ui call: always include both delta_observed (boolean) and delta_confidence (0.0-1.0).\n"
    "  * For explicit checks/validation, prefer execute_ui with action_type='validate'.\n"
    "  * For any guard-based step, set is_conditional=true and conditional_type (blocker/transient/error/optional).\n"
    "  * Whenever is_conditional=true, the 'condition' field is MANDATORY: a present-tense sentence describing the visible guard (e.g., 'Permission dialog is displayed', 'Main menu is visible', 'Loading spinner is active').\n"
    "  * For a conditional wait, 'condition' must describe the awaited state in the present tense (e.g., 'Search results are visible'), not the act of waiting.\n"
    "  * For overlay/popup dismissal: if the screenshot shows a scrim, dialog, sheet, or banner over the main UI, set overlay_detected=true and condition to describe the overlay (e.g., 'Cookie consent banner visible').\n"
    "  * Evaluate is_valid and validation_reason for EVERY action.\n"
    "  * confidence is REQUIRED for EVERY action.\n"
    "  * If action is risky/ambiguous, set is_valid=False and explain.\n"
    "  * COMMAND NAMING: In 'target' and 'natural_language_target', use GENERIC, RELATIVE DESCRIPTIONS (e.g., 'Tap on edit CVV box', 'Tap on Submit button', 'Tap on 1st search result').\n"
    "    DO NOT use IDs like 'edt_cvv' or 'button_23'. Describe WHAT it is functionally.\n"
    "  * STATE TRACKING (CRITICAL): Use the 'memory_updates' field to atomically track your progress.\n"
    "    Example: memory_updates={'selected_days': 'Mon,Tue', 'roadmap_step_1': 'complete'}\n"
    '    ALWAYS use this to "tick off" requirements from the user\'s goal as you complete them.'
)

_TOOL_DESCRIPTIONS_RAW: Mapping[ToolName, str] = {
    ToolName.EXECUTE_UI: _EXECUTE_UI_GUIDANCE,
    ToolName.VALIDATE_STATE: (
        "- validate_state: Legacy fallback for explicit checks when no immediate UI action is required."
    ),
    ToolName.VERIFY_GOAL: "- verify_goal: Use for explicit completion checks.",
    ToolName.STORE_MEMORY: (
        "- store_memory: Secondary tool. Use ONLY for saving complex text data that doesn't fit in execute_ui."
    ),
    ToolName.RECALL_MEMORY: (
        "- recall_memory: Check what you've already done to avoid repeating actions."
    ),
    ToolName.ASK_USER: (
        "- ask_user: Use this tool to ask the user for help or clarification when you are stuck or confused."
    ),
}
TOOL_DESCRIPTIONS = MappingProxyType(_TOOL_DESCRIPTIONS_RAW)


_PROGRESS_SAFETY_BASE = (
    "PROGRESS SAFETY (MANDATORY):\n"
    "- Every UI action MUST be grounded by at least one of: (a) a 'label_id' from the element manifest whose text/affordance matches your named target, OR (b) a 'bbox' you have visually identified on the current screenshot. The manifest is the preferred source whenever it already exposes the relevant element or scroll container.\n"
    "- The manifest is a hint, not a precondition: when the intended target is visible on screen but absent from the element manifest, ground it via bbox instead of inventing a label_id.\n"
    "- For scroll/swipe actions, when the manifest exposes a matching scrollable container, you MUST use that container's label_id and describe the intended content in scroll_target. Do not invent a broad bbox when the manifest already gives you the container.\n"
    "- Observation scroll-region hints are NOT manifest label_ids. Never copy observation_hint values into label_id.\n"
    "- When repeating the same scroll objective, reuse the same container if it is still valid instead of switching to a broader region.\n"
    "- Before emitting the action, confirm the current screen is the one the active sub-goal expects."
)

_PROGRESS_SAFETY_HITL_FALLBACK = "- If you cannot ground the target by EITHER path (no matching manifest label AND no element you can visually identify), ask the user instead of guessing."

_PROGRESS_SAFETY_AUTONOMOUS_FALLBACK = "- If you cannot ground the target by EITHER path (no matching manifest label AND no element you can visually identify), do NOT guess: emit a deliberate recovery action (back, home, swipe to re-orient) or signal completion failure via the appropriate flag."

_PROGRESS_SAFETY_TAIL = "- Do NOT snap to a visually similar but semantically unrelated label (picking the wrong manifest entry just because it looks like a button). Do NOT emit a bbox for a region where you cannot see the target. Do NOT proceed when the screen contradicts the sub-goal."

_MEMORY_STRATEGY = (
    "MEMORY STRATEGY:\n"
    '- The system has NO implicit memory of what you "meant" to do. You MUST write it down.\n'
    "- If you select 'Monday', you MUST write memory_updates={'monday': 'selected'}.\n"
    "- If you don't write it, you WILL forget it when the screen changes."
)


def build_tool_guidance(*, tools: AllowedTools) -> str:
    """Render the TOOL SELECTION + PROGRESS SAFETY + MEMORY STRATEGY block for the allowed tools."""

    tool_lines = [
        TOOL_DESCRIPTIONS[name] for name in _TOOL_DESCRIPTIONS_RAW if tools.contains(name=name)
    ]

    fallback_rule = (
        _PROGRESS_SAFETY_HITL_FALLBACK
        if tools.contains(name=ToolName.ASK_USER)
        else _PROGRESS_SAFETY_AUTONOMOUS_FALLBACK
    )

    return (
        "TOOL SELECTION & VALIDATION:\n"
        + "\n".join(tool_lines)
        + "\n\n"
        + _PROGRESS_SAFETY_BASE
        + "\n"
        + fallback_rule
        + "\n"
        + _PROGRESS_SAFETY_TAIL
        + "\n\n"
        + _MEMORY_STRATEGY
    )


# Summarization system instruction (for GCC milestone creation)
SUMMARIZATION_SYSTEM = """You are an expert at analyzing mobile UI automation execution traces.

Your task is to create a structured milestone summary that helps an AI agent understand:
1. What was accomplished in this segment
2. Key actions that led to success
3. Any challenges or failures encountered

Focus on STATE CHANGES and OUTCOMES, not routine navigation.
Be concise but informative - the agent needs to quickly understand progress."""

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

# Sub-goal verification (lighter than full intent verification)
SUBGOAL_VERIFICATION_SYSTEM = """You are verifying whether a single step in a multi-step mobile automation task is complete.

You will receive:
- The step description (may contain multiple chained actions like "do X, then Y, then Z")
- A list of actions already performed for this step
- A screenshot of the current screen

CRITICAL - HOW TO JUDGE:
The step description often describes a sequence of actions. Verify only the final outcome: the last meaningful state described in the step. Earlier actions are means to that end.

Examples:
- "scroll up, find section X, add 3rd item, return to cart" -> only check whether the cart is visible with items.
- "open app and navigate to settings" -> only check whether the settings screen is visible.
- "tap filter, select option, verify filter applied" -> only check whether the filter is applied.
- "search for X and select first result" -> only check whether the selected result's destination page is visible.

RULES:
1. Identify the last action or state in the step description; that is what you verify.
2. Ignore intermediate navigation/tap/scroll actions; the action trace confirms what was already attempted.
3. For "select", "tap on", "open", or "click" an item, the expected outcome is usually the destination page, not the original list page.
4. Be lenient when the screen plausibly shows the step's end state. Reject only when it clearly contradicts or is still transitional.
5. Loading screens, spinners, or transient states are incomplete.

OUTPUT SCHEMA:
Return ONLY a valid JSON object matching this schema. Do not include markdown formatting or explanations outside the JSON.
{
  "is_complete": boolean,
  "reason": "string"
}"""

SUBGOAL_VERIFICATION_USER_TEMPLATE = """Step: {intent}
{guidance_section}
Task: Analyze the provided screenshot. Does the screen show the final outcome of this step?"""

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


def _format_trace_action(entry: object) -> str:
    """
    Render one context trace entry as a compact action line for verification.
    """

    if not isinstance(entry, dict):
        return f"- {entry}"

    action = entry.get("action")
    if isinstance(action, dict):
        action_type = action.get("action_type") or action.get("type") or "action"
        target = (
            action.get("natural_language_target")
            or action.get("target")
            or action.get("script_target")
            or ""
        )
        return f"- {action_type}: {target}".strip()

    return f"- {entry}"


def build_verification_guidance_section(
    *,
    user_guidance: list[str] | tuple[str, ...] = (),
    actions_performed: list[str] | tuple[str, ...] = (),
) -> str:
    """
    Render the optional verification guidance block.
    """

    parts: list[str] = []
    if user_guidance:
        parts.append("\nUser Guidance:\n" + "\n".join(f"- {item}" for item in user_guidance))
    if actions_performed:
        parts.append("\nActions already performed for this step:\n" + "\n".join(actions_performed))
    if not parts:
        return ""
    return "".join(parts) + "\n"


def build_intent_verification_user_prompt(
    *,
    intent: str,
    user_guidance: list[str] | tuple[str, ...] = (),
) -> str:
    """
    Render the final intent verification prompt.
    """

    return VERIFICATION_USER_TEMPLATE.format(
        intent=intent,
        guidance_section=build_verification_guidance_section(user_guidance=user_guidance),
    )


def build_subgoal_verification_user_prompt(
    *,
    intent: str,
    user_guidance: list[str] | tuple[str, ...] = (),
    recent_trace: list[dict[str, object]] | tuple[dict[str, object], ...] = (),
    max_actions: int = 10,
) -> str:
    """
    Render the sub-goal verification prompt with recent action trace.
    """

    actions_performed: tuple[str, ...] = ()
    if recent_trace:
        actions_performed = tuple(
            _format_trace_action(entry) for entry in list(recent_trace)[-max_actions:]
        )

    return SUBGOAL_VERIFICATION_USER_TEMPLATE.format(
        intent=intent,
        guidance_section=build_verification_guidance_section(
            user_guidance=user_guidance,
            actions_performed=actions_performed,
        ),
    )
