"""Provider-neutral intent decomposition prompt policy.

Owns the system instruction, examples, and template used by any LLM
provider when breaking a high-level intent into sequential sub-goals.
Adapter layers (e.g. ``adapters/prompts/gemini_decomposition.py``) are
thin shims that satisfy the ``DecompositionPromptBuilder`` port and
delegate here.
"""

from __future__ import annotations

from typing import Optional, Sequence

from fathom.interfaces.prompt import DecompositionPromptBuilder

__all__ = [
    "DecompositionPromptBuilder",
    "DECOMPOSITION_SYSTEM_INSTRUCTION",
    "DECOMPOSITION_USER_PROMPT_TEMPLATE",
    "DECOMPOSITION_REPLAN_SCREENSHOT_NOTE",
    "build_decomposition_user_prompt",
    "build_replan_context_section",
]


DECOMPOSITION_SYSTEM_INSTRUCTION = (
    "You are an expert task planner. Always preserve the user's exact wording.\n"
    "Return ONLY valid JSON (no markdown, no extra text)."
)


DECOMPOSITION_USER_PROMPT_TEMPLATE = (
    "You are an expert at breaking down user intents into executable micro-tasks.\n\n"
    "INTENT: {intent}\n"
    "{replan_context}"
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


def build_replan_context_section(
    *,
    stuck_sub_goal: Optional[str] = None,
    failure_reason: Optional[str] = None,
    suggested_next_action: Optional[str] = None,
    recent_actions: Sequence[str] = (),
) -> str:
    """Render the optional ``{replan_context}`` block for replanning.

    This block sits between the INTENT line and the INSTRUCTIONS in the
    decomposition template. It is empty by default for initial
    decomposition; populated only when the planner invokes replanning
    after the agent gets stuck, so the decomposer can steer the new plan
    AWAY from the failed approach instead of re-emitting the same steps.

    Returns an empty string when all inputs are empty — the surrounding
    template then collapses the placeholder to nothing.
    """

    parts: list[str] = []
    if stuck_sub_goal:
        parts.append(f"STUCK ON: {stuck_sub_goal}")
    if failure_reason:
        parts.append(f"WHY IT FAILED: {failure_reason}")
    if suggested_next_action:
        parts.append(f"VERIFIER SUGGESTED: {suggested_next_action}")
    if recent_actions:
        lines = "\n".join(f"- {line}" for line in recent_actions)
        parts.append(f"RECENTLY TRIED (do NOT repeat these):\n{lines}")

    if not parts:
        return ""

    body = "\n".join(parts)
    return (
        "\nREPLAN CONTEXT — the agent already tried to execute this intent and "
        "got stuck. Use the information below to propose a DIFFERENT approach; "
        "do not re-emit steps that were already tried without success.\n"
        f"{body}\n\n"
    )


def build_decomposition_user_prompt(
    *,
    intent: str,
    stuck_sub_goal: Optional[str] = None,
    failure_reason: Optional[str] = None,
    suggested_next_action: Optional[str] = None,
    recent_actions: Sequence[str] = (),
) -> str:
    """Render the provider-neutral decomposition user prompt for an intent.

    All replan-context parameters are optional; callers doing the initial
    decomposition pass nothing and the template renders exactly as before.
    Replanning callers pass the stuck sub-goal, failure reason, verifier
    suggestion, and recent action lines so the decomposer can avoid the
    dead-end path that triggered the replan.
    """

    replan_context = build_replan_context_section(
        stuck_sub_goal=stuck_sub_goal,
        failure_reason=failure_reason,
        suggested_next_action=suggested_next_action,
        recent_actions=recent_actions,
    )
    return DECOMPOSITION_USER_PROMPT_TEMPLATE.format(
        intent=intent,
        replan_context=replan_context,
    )
