from __future__ import annotations

from typing import Optional, Sequence

from fathom.core.prompts.decomposition import (
    DECOMPOSITION_SYSTEM_INSTRUCTION,
    DecompositionPromptBuilder,
    build_decomposition_user_prompt,
)


class GeminiDecompositionPromptBuilder(DecompositionPromptBuilder):
    """
    Provider shim for sequential intent decomposition.

    Decomposition prompt content is provider-neutral product policy and
    lives in ``fathom.core.prompts.decomposition``. This class only exists
    so the PromptFactory can hand back a ``DecompositionPromptBuilder`` for
    the Gemini provider key — adding new providers reuses the same policy.
    """

    def build_system_instruction(self) -> str:
        return DECOMPOSITION_SYSTEM_INSTRUCTION

    def build_user_prompt(
        self,
        *,
        intent: str,
        stuck_sub_goal: Optional[str] = None,
        failure_reason: Optional[str] = None,
        suggested_next_action: Optional[str] = None,
        recent_actions: Sequence[str] = (),
    ) -> str:
        return build_decomposition_user_prompt(
            intent=intent,
            stuck_sub_goal=stuck_sub_goal,
            failure_reason=failure_reason,
            suggested_next_action=suggested_next_action,
            recent_actions=recent_actions,
        )
