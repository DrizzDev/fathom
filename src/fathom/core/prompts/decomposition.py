"""Provider-neutral intent decomposition prompt policy.

Owns the system instruction, examples, and template used by any LLM
provider when breaking a high-level intent into sequential sub-goals.
Adapter layers (e.g. ``adapters/prompts/gemini_decomposition.py``) are
thin shims that satisfy the ``DecompositionPromptBuilder`` port and
delegate here.
"""

from __future__ import annotations

from fathom.interfaces.prompt import DecompositionPromptBuilder

__all__ = [
    "DecompositionPromptBuilder",
    "DECOMPOSITION_SYSTEM_INSTRUCTION",
    "DECOMPOSITION_USER_PROMPT_TEMPLATE",
    "DECOMPOSITION_REPLAN_SCREENSHOT_NOTE",
    "build_decomposition_user_prompt",
]


DECOMPOSITION_SYSTEM_INSTRUCTION = (
    "You are an expert task planner. Always preserve the user's exact wording.\n"
    "Return ONLY valid JSON (no markdown, no extra text)."
)


DECOMPOSITION_USER_PROMPT_TEMPLATE = (
    "You are an expert at breaking down user intents into executable micro-tasks.\n\n"
    "INTENT: {intent}\n\n"
    "INSTRUCTIONS:\n"
    "1. Break down the intent into sequential, non-skippable steps\n"
    "2. Each step must be atomic and testable\n"
    "3. Steps must be in execution order (no parallelization)\n"
    "4. Each step should be 1-2 sentences, action-oriented\n"
    "5. CRITICAL: Do not skip any steps required to achieve the intent\n"
    "6. CRITICAL: You MUST use the user's exact wording and terminology wherever possible\n"
    "   - Do NOT paraphrase, generalize, or rephrase their specific requests\n"
    "   - Preserve specific app names, button names, field names, and action verbs\n"
    "   - Keep technical terms and product names exactly as stated\n\n"
    "IMPORTANT - COMPOUND ACTIONS:\n"
    "Keep these patterns as SINGLE steps (do NOT split them):\n"
    '- "Scroll to X and select/tap/click Y" \u2192 ONE step (scroll is just navigation to reach Y)\n'
    '- "Navigate to X and do Y" \u2192 ONE step (navigate is means to reach Y)\n'
    '- "Find X and tap/click Y" \u2192 ONE step (find is means to reach Y)\n'
    '- "Go to X and verify/check Y" \u2192 ONE step (navigation + verification together)\n\n'
    "EXAMPLES:\n"
    '\u2713 GOOD: User says "Tap the login button" \u2192 Sub-goal: "Tap the login button"\n'
    '\u2717 BAD: User says "Tap the login button" \u2192 Sub-goal: "Authenticate with credentials"\n\n'
    "\u2713 GOOD: User says \"Enter password 'test123'\" \u2192 Sub-goal: \"Enter password 'test123'\"\n"
    '\u2717 BAD: User says "Enter password \'test123\'" \u2192 Sub-goal: "Input user credentials"\n\n'
    '\u2713 GOOD: User says "Open Settings app" \u2192 Sub-goal: "Open Settings app"\n'
    '\u2717 BAD: User says "Open Settings app" \u2192 Sub-goal: "Navigate to system configuration"\n\n'
    '\u2713 GOOD: User says "Scroll to labs section and select any category" \u2192 Sub-goal: "Scroll to labs section and select any category"\n'
    '\u2717 BAD: User says "Scroll to labs section and select any category" \u2192 Sub-goals: ["Scroll to labs section", "Select any category"]\n\n'
    '\u2713 GOOD: User says "Go to cart and verify total amount" \u2192 Sub-goal: "Go to cart and verify total amount"\n'
    '\u2717 BAD: User says "Go to cart and verify total amount" \u2192 Sub-goals: ["Go to cart", "Verify total amount"]\n\n'
    "Return ONLY a valid JSON with this structure:\n"
    "{{\n"
    '  "sub_goals": ["step 1", "step 2", "step 3"],\n'
    '  "confidence": 0.9\n'
    "}}\n"
)


DECOMPOSITION_REPLAN_SCREENSHOT_NOTE = (
    "\n\nA screenshot of the current screen is attached. "
    "Plan sub-goals starting from this screen. Do NOT include "
    "steps to reach this screen — the agent is already here."
)


def build_decomposition_user_prompt(*, intent: str) -> str:
    """Render the provider-neutral decomposition user prompt for an intent."""

    return DECOMPOSITION_USER_PROMPT_TEMPLATE.format(intent=intent)
