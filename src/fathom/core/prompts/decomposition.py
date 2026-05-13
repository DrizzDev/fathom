from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from fathom.core.recovery.types import RecoveryTrigger


class DecompositionPromptBuilder(ABC):
    """
    Abstract builder for intent decomposition prompting.
    """

    @abstractmethod
    def build_system_instruction(self) -> str:
        """
        Build stable system instruction for decomposition generation.
        """

        raise NotImplementedError

    @abstractmethod
    def build_user_prompt(self, *, intent: str) -> str:
        """
        Build dynamic user prompt for decomposing an intent.
        """

        raise NotImplementedError

    @abstractmethod
    def build_replan_system_note(self) -> str:
        """
        Build the system-instruction addendum used when re-decomposing from a stuck state.
        """

        raise NotImplementedError

    @abstractmethod
    def build_replan_user_preamble(
        self,
        *,
        trigger: RecoveryTrigger,
        stuck_sub_goal: str,
        failure_reason: str,
        recent_actions: List[str],
        suggested_next_action: Optional[str],
    ) -> str:
        """
        Build the user-prompt preamble describing the stuck state.

        ``trigger`` selects a short framing sentence that names the stuck
        evidence — e.g. "looped without progress" vs "target not on screen"
        — so the model can reason about *why* a replan was requested
        instead of treating every replan as the same kind of failure.
        """

        raise NotImplementedError


# Keyed on RecoveryTrigger value strings so this module stays free of any
# direct import on ``fathom.core.recovery`` — that module imports prompts
# transitively, and a direct dependency here would cycle.
_TRIGGER_FRAMING: Dict[str, str] = {
    "ACTION_BLOCKED": (
        "The previously-planned action could not be reached from the current "
        "screen; the prior plan assumed UI affordances that are not present."
    ),
    "VERIFY_REJECTED": (
        "Final verification refused the run as complete. The remaining work "
        "must be reconsidered against what is actually on screen."
    ),
    "LOOP_DETECTED": (
        "The agent looped on the same screen without producing progress; the "
        "prior plan repeated an action that the screen does not respond to."
    ),
    "NO_PROGRESS": (
        "Recent actions produced no measurable visual progress; the prior "
        "plan's tactics are not advancing the goal on this screen."
    ),
    "TARGET_UNRESOLVED": (
        "The previously-named target could not be located on the current "
        "screen; the prior plan referenced an element that is not present."
    ),
    "SUBGOAL_BUDGET_EXCEEDED": (
        "The active sub-goal exhausted its step budget without its success "
        "criterion being met; the sub-goal as written is not reachable here."
    ),
    "REPORT_UNACTIONABLE": (
        "The agent reported that the active sub-goal does not match the "
        "current screen; rewrite the remaining plan around what IS on screen."
    ),
}


class GeminiDecompositionPromptBuilder(DecompositionPromptBuilder):
    """
    Gemini-focused prompt builder for sequential intent decomposition.
    """

    def build_system_instruction(self) -> str:
        return (
            "You are an expert task planner. Always preserve the user's exact wording.\n"
            "Return ONLY valid JSON (no markdown, no extra text)."
        )

    def build_user_prompt(self, *, intent: str) -> str:
        return (
            "You are an expert at breaking down user intents into executable micro-tasks.\n\n"
            f"INTENT: {intent}\n\n"
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
            "{\n"
            '  "sub_goals": ["step 1", "step 2", "step 3"],\n'
            '  "confidence": 0.9\n'
            "}\n"
        )

    def build_replan_system_note(self) -> str:
        """
        System-instruction addendum appended when decomposing in replan
        mode: tells the model a screenshot is attached and that failure
        evidence describes paths to avoid, not retry.
        """

        return (
            "\n\nA screenshot of the agent's current screen is attached. Plan sub-goals "
            "starting from this screen. Do NOT include steps to reach this screen — the "
            "agent is already here. Treat the supplied failure reason and recent actions "
            "as evidence of paths to avoid, not paths to retry."
        )

    def build_replan_user_preamble(
        self,
        *,
        trigger: RecoveryTrigger,
        stuck_sub_goal: str,
        failure_reason: str,
        recent_actions: List[str],
        suggested_next_action: Optional[str],
    ) -> str:
        """
        Render the stuck-state preamble prepended to the user prompt when
        re-decomposing from a stuck state.

        Per-trigger framing names the kind of evidence the system observed
        (loop, no-progress, unresolved target, ...) so the model treats the
        prior plan's failure mode appropriately rather than re-emitting
        equivalent steps.
        """

        framing = _TRIGGER_FRAMING.get(
            trigger.value,
            "The prior plan failed; reconsider the remaining work against the current screen.",
        )
        recent = "\n".join(f"- {entry}" for entry in recent_actions) or "- (none)"
        return (
            "REPLAN CONTEXT (the previous decomposition got stuck):\n"
            f"- Trigger: {trigger.value}\n"
            f"- What happened: {framing}\n"
            f"- Stuck sub-goal: {stuck_sub_goal}\n"
            f"- Failure reason: {failure_reason}\n"
            f"- Suggested next action: {suggested_next_action or '(none)'}\n"
            f"- Recent actions (most recent last):\n{recent}\n\n"
        )
