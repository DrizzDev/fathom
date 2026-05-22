from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

from fathom.schemas.escape import EscapeCategory
from fathom.schemas.subgoal import ExecutionContract

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
    def build_replan_system_note(
        self,
        *,
        strict_mode: bool,
        execution_contract: Optional[ExecutionContract],
    ) -> str:
        """
        Build the system-instruction addendum used when re-decomposing from a stuck state.
        """

        raise NotImplementedError

    @abstractmethod
    def build_replan_user_preamble(
        self,
        *,
        stuck_sub_goal: str,
        failure_reason: str,
        trigger: RecoveryTrigger,
        recent_actions: List[str],
        suggested_next_action: Optional[str],
        strict_mode: bool,
        execution_contract: Optional[ExecutionContract],
        escape_category: Optional[EscapeCategory] = None,
    ) -> str:
        """
        Build the user-prompt preamble describing the stuck state.

        ``trigger`` selects a short framing sentence that names the stuck evidence — e.g. "looped without progress" vs "target not on screen"
        — so the model can reason about *why* a replan was requested instead of treating every replan as the same kind of failure.

        When ``trigger`` is ``REQUEST_REPLAN`` the optional ``escape_category`` overrides the generic trigger framing with a category-specific sentence
        so the decomposer adapts to *why* the agent escaped (target missing, wrong screen, missing precondition) rather than treating every escape identically.
        """

        raise NotImplementedError


# Keyed on RecoveryTrigger value strings so this module stays free of any
# direct import on ``fathom.core.recovery`` — that module imports prompts
# transitively, and a direct dependency here would cycle.
_TRIGGER_FRAMING: Dict[str, str] = {
    "ACTION_BLOCKED": (
        "The previously-planned action could not be reached from the current "
        "screen; the prior plan assumed UI affordance's that are not present."
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
    "REQUEST_REPLAN": (
        "The agent emitted a structured escape report against the active "
        "sub-goal. Use the category framing below to drive the next plan."
    ),
}


# Per-:class:`EscapeCategory` framing for ``REQUEST_REPLAN`` replans.
# The category-specific sentence overrides the generic ``REQUEST_REPLAN`` trigger framing
# So the decomposer can adapt to *why* the agent escaped (target missing, wrong screen, missing precondition) rather than treating every escape identically.
_ESCAPE_CATEGORY_FRAMING: Dict[str, str] = {
    EscapeCategory.TARGET_NOT_AVAILABLE.value: (
        "The agent could not ground the named target by either path — "
        "no matching element in the manifest and nothing visible on the "
        "current screenshot that corresponds to it. Rewrite the remaining "
        "plan around elements that ARE reachable from this screen, or insert a sub-goal that brings the target into view."
    ),
    EscapeCategory.WRONG_SCREEN.value: (
        "The agent is on a different screen than the prior plan assumed. "
        "Rewrite the remaining plan starting from this screen — do not "
        "re-issue steps that depend on the expected (but absent) screen."
    ),
    EscapeCategory.PRECONDITION_NOT_MET.value: (
        "The agent reports that prior state required by the active sub-goal "
        "has not been reached. Insert the missing precondition step(s) at "
        "the head of the new plan before the previously-blocked work."
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
        parts = [
            "You are an expert at breaking down user intents into executable micro-tasks.\n\n",
            "INTENT: ",
            intent,
            "\n\n",
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
            "OBSERVABLE TERMINAL CRITERION (MANDATORY):\n"
            "Every task must declare a terminal screen state — what the agent will see when the task is complete.\n"
            "The criterion describes the post-condition, NOT the action used to reach it.\n"
            '- Task description names WHAT the agent does ("Tap on Continue").\n'
            '- Task criterion names WHAT THE SCREEN SHOWS afterwards ("Cart screen with items list visible").\n'
            "Without an observable criterion the runtime cannot tell the difference between a wrong tap that happened to change the screen and a correct tap that achieved the goal.\n\n"
            "EXECUTION CONTRACT (MANDATORY):\n"
            "Every task must also declare a structured execution_contract object.\n"
            '- required_action_family must be one of: "scroll", "tap", "input", "wait", "validate", "unspecified"\n'
            '- scroll_axis must be one of: "vertical", "horizontal", "unspecified"\n'
            '- surface must either be null or the exact section/container wording when the user names a specific area (for example "below Fast Delivery section").\n'
            "- Use scroll_axis='unspecified' unless the user or task text EXPLICITLY constrains the axis.\n"
            "- Do NOT guess horizontal or vertical from your own assumptions.\n"
            "- When the task names a specific section, rail, list, card row, or on-screen area, copy that exact wording into execution_contract.surface.\n"
            "- If the task is primarily a scroll/swipe task, required_action_family must be 'scroll'.\n"
            "- If the task is primarily a tap/press task, required_action_family must be 'tap'.\n"
            "- If the task is primarily a text-entry/search-entry task, required_action_family must be 'input'.\n"
            "- If the task is purely observational, required_action_family must be 'validate'.\n"
            "- If the task is only waiting for UI state, required_action_family must be 'wait'.\n\n"
            "Return ONLY a valid JSON with this structure:\n"
            "{\n"
            '  "sub_goals": [\n'
            "    {\n"
            '      "description": "step 1",\n'
            '      "criterion": "what the screen looks like after step 1",\n'
            '      "execution_contract": {\n'
            '        "required_action_family": "scroll",\n'
            '        "scroll_axis": "vertical",\n'
            '        "surface": "below Fast Delivery section"\n'
            "      }\n"
            "    },\n"
            "    {\n"
            '      "description": "step 2",\n'
            '      "criterion": "what the screen looks like after step 2",\n'
            '      "execution_contract": {\n'
            '        "required_action_family": "tap",\n'
            '        "scroll_axis": "unspecified",\n'
            '        "surface": null\n'
            "      }\n"
            "    }\n"
            "  ],\n"
            '  "confidence": 0.9\n'
            "}\n",
        ]
        return "".join(parts)

    def build_replan_system_note(
        self,
        *,
        strict_mode: bool,
        execution_contract: Optional[ExecutionContract],
    ) -> str:
        """
        System-instruction addendum appended when decomposing in replan
        mode: tells the model a screenshot is attached and that failure
        evidence describes paths to avoid, not retry.
        """
        note = (
            "\n\nA screenshot of the agent's current screen is attached. Plan sub-goals "
            "starting from this screen. Do NOT include steps to reach this screen — the "
            "agent is already here. Treat the supplied failure reason and recent actions "
            "as evidence of paths to avoid, not paths to retry."
        )
        if not strict_mode or execution_contract is None:
            return note

        return (
            f"{note}\n"
            "STRICT MODE IS ACTIVE.\n"
            "You MUST preserve the active sub-goal's execution contract exactly.\n"
            f"- required_action_family: {execution_contract.required_action_family.value}\n"
            f"- scroll_axis: {execution_contract.scroll_axis.value}\n"
            f"- surface: {execution_contract.surface or '(none)'}\n"
            "Do NOT change the task into a different workflow shape."
        )

    def build_replan_user_preamble(
        self,
        *,
        trigger: RecoveryTrigger,
        stuck_sub_goal: str,
        failure_reason: str,
        recent_actions: List[str],
        suggested_next_action: Optional[str],
        strict_mode: bool,
        execution_contract: Optional[ExecutionContract],
        escape_category: Optional[EscapeCategory] = None,
    ) -> str:
        """
        Render the stuck-state preamble prepended to the user prompt when re-decomposing from a stuck state.

        Per-trigger framing names the kind of evidence the system observed
        (loop, no-progress, unresolved target, ...). When an ``escape_category`` is supplied (REQUEST_REPLAN path)
        the category-specific framing overrides the generic trigger sentence so the decomposer sees the typed reason the agent escaped.
        """

        category_value = escape_category.value if escape_category is not None else None
        category_framing = (
            _ESCAPE_CATEGORY_FRAMING.get(category_value) if category_value is not None else None
        )
        framing = category_framing or _TRIGGER_FRAMING.get(
            trigger.value,
            "The prior plan failed; reconsider the remaining work against the current screen.",
        )
        recent = "\n".join(f"- {entry}" for entry in recent_actions) or "- (none)"
        strict_block = ""
        if strict_mode and execution_contract is not None:
            strict_block = (
                "STRICT EXECUTION CONTRACT:\n"
                f"- required_action_family: {execution_contract.required_action_family.value}\n"
                f"- scroll_axis: {execution_contract.scroll_axis.value}\n"
                f"- surface: {execution_contract.surface or '(none)'}\n"
                "- Preserve this contract exactly while replanning.\n\n"
            )

        return (
            "REPLAN CONTEXT (the previous decomposition got stuck):\n"
            f"- Trigger: {trigger.value}\n"
            f"- Escape category: {category_value or '(none)'}\n"
            f"- What happened: {framing}\n"
            f"- Stuck sub-goal: {stuck_sub_goal}\n"
            f"- Failure reason: {failure_reason}\n"
            f"- Suggested next action: {suggested_next_action or '(none)'}\n"
            f"- Recent actions (most recent last):\n{recent}\n\n"
            f"{strict_block}"
        )
