"""Provider-neutral verification and extraction prompts.

These templates are used by the core agent to (a) verify whether a
user's intent has been completed, (b) verify whether a single sub-goal
has been completed, and (c) extract validation subjects from a user's
intent. They are pure product policy and are reused across providers.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from fathom.core.prompts.trace import format_trace_action_line

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
  "reason": "string", // Factual explanation of what is visually missing, or why it is complete.
  "next_action": "string" // If incomplete: a specific UI action using the correct verb for the interaction. Use "Tap" for buttons/icons/links, "Type 'text'" for text input, "Clear and type 'text'" if a field has wrong text, "Scroll/Swipe" for navigation, "Wait" for loading. MUST reference a VISIBLE element on the current screenshot. Do NOT suggest actions for off-screen elements. If complete: leave empty.
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

**CRITICAL — HOW TO JUDGE:**
The step description often describes a SEQUENCE of actions. You must ONLY verify the FINAL outcome — the last meaningful state described in the step. All earlier actions in the sequence have already been executed.

Examples:
- "scroll up, find section X, add 3rd item, return to cart" → ONLY check: is the cart visible with items?
- "open app and navigate to settings" → ONLY check: is the settings screen visible?
- "tap filter, select option, verify filter applied" → ONLY check: is the filter applied (tag/badge visible)?
- "search for X and select first result" → ONLY check: is the first result's detail page visible?

**RULES:**
1. Identify the LAST action or state in the step description — that is what you verify.
2. Ignore intermediate actions (scroll, navigate, tap) — they are means to an end.
3. The actions list confirms what was already done. Trust it.
4. NAVIGATION/SELECTION RULE (CRITICAL): When the step says "select", "tap on", "open", or "click" an item (e.g., a search result, a product, a restaurant), the expected outcome is the DESTINATION page — NOT the list page where the item was. If the screen shows a detail/content page that corresponds to the selected item, the step IS complete.
5. Be LENIENT: if the screen plausibly shows the end state, mark complete. Only reject if the screen clearly contradicts the expected outcome.
6. Loading screens, spinners, or transitional states → incomplete.

**OUTPUT:**
Return ONLY valid JSON:
{
  "is_complete": boolean,
  "reason": "string",
  "next_action": "string" // If incomplete: a specific UI action using the correct verb. Use "Tap" for buttons/icons, "Type 'text'" for input, "Clear and type 'text'" if wrong text is present, "Scroll/Swipe" for navigation, "Wait" for loading. MUST reference a VISIBLE element. If complete: leave empty.
}"""

SUBGOAL_VERIFICATION_USER_TEMPLATE = """Step: {intent}
{guidance_section}
Based on the screenshot, does the screen show the FINAL outcome of this step?"""

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


# ---------------------------------------------------------------------------
# Verification user-prompt builders
# ---------------------------------------------------------------------------


def _format_action_lines(trace: "Sequence[Mapping[str, Any]]") -> list[str]:
    """Render the most recent trace entries as ``"- kind: target"`` lines."""

    return [format_trace_action_line(entry) for entry in trace]


def build_verification_guidance_section(
    *,
    user_guidance: "Sequence[str]" = (),
    actions_performed: "Sequence[str]" = (),
) -> str:
    """Render the optional `{guidance_section}` block for verification prompts.

    Returns an empty string when neither user_guidance nor
    actions_performed is provided. Lives in core (not in the strategies
    layer) so the user-facing labels stay next to the templates they
    feed.
    """

    parts: list[str] = []
    if user_guidance:
        parts.append("\nUser Guidance:\n" + "\n".join(f"- {g}" for g in user_guidance))
    if actions_performed:
        parts.append("\nActions already performed for this step:\n" + "\n".join(actions_performed))
    if not parts:
        return ""
    return "".join(parts) + "\n"


def build_intent_verification_user_prompt(
    *,
    intent: str,
    user_guidance: "Sequence[str]" = (),
) -> str:
    """Render the user prompt for full intent verification."""

    return VERIFICATION_USER_TEMPLATE.format(
        intent=intent,
        guidance_section=build_verification_guidance_section(user_guidance=user_guidance),
    )


def build_subgoal_verification_user_prompt(
    *,
    intent: str,
    user_guidance: "Sequence[str]" = (),
    recent_trace: "Sequence[Mapping[str, Any]]" = (),
    max_actions: int = 10,
) -> str:
    """Render the user prompt for sub-goal verification.

    ``recent_trace`` is the agent's interaction history; the most recent
    ``max_actions`` entries are formatted as 'kind: target' lines and
    threaded into the prompt's `{guidance_section}` placeholder.
    """

    actions_performed: tuple[str, ...] = ()
    if recent_trace:
        actions_performed = tuple(_format_action_lines(list(recent_trace)[-max_actions:]))
    return SUBGOAL_VERIFICATION_USER_TEMPLATE.format(
        intent=intent,
        guidance_section=build_verification_guidance_section(
            user_guidance=user_guidance,
            actions_performed=actions_performed,
        ),
    )
