"""
Pins for the per-:class:`EscapeCategory` framing dispatch in
``decomposition.py``. The replan preamble must surface a
category-specific sentence to the decomposer so a new category in
``REPLAN_ESCAPE_CATEGORIES`` cannot ship without a paired framing
entry (closes the Open/Closed gap that would otherwise let the
decomposer silently fall back to the generic REQUEST_REPLAN sentence).
"""

from __future__ import annotations

from fathom.core.prompts.decomposition import (
    _ESCAPE_CATEGORY_FRAMING,
    GeminiDecompositionPromptBuilder,
)
from fathom.core.recovery.types import RecoveryTrigger
from fathom.schemas.escape import REPLAN_ESCAPE_CATEGORIES, EscapeCategory
from fathom.schemas.subgoal import ExecutionContract


class TestDecompositionEscapeFraming:
    """
    Behavioural pins for category-aware decomposer preamble.
    """

    def test_every_replan_category_has_framing(self) -> None:
        """
        Each :class:`EscapeCategory` value in
        :data:`REPLAN_ESCAPE_CATEGORIES` must have a paired entry in
        :data:`_ESCAPE_CATEGORY_FRAMING` so the decomposer never falls
        back to the generic REQUEST_REPLAN framing for a known replan
        category.
        """

        missing = {
            category.value
            for category in REPLAN_ESCAPE_CATEGORIES
            if category.value not in _ESCAPE_CATEGORY_FRAMING
        }
        assert missing == set(), (
            f"REPLAN_ESCAPE_CATEGORIES missing decomposer framing: {sorted(missing)}"
        )

    def test_preamble_uses_category_specific_framing_when_provided(self) -> None:
        """
        The category sentence must appear verbatim in the preamble
        when ``escape_category`` is supplied, overriding the generic
        trigger framing.
        """

        builder = GeminiDecompositionPromptBuilder()
        preamble = builder.build_replan_user_preamble(
            trigger=RecoveryTrigger.REQUEST_REPLAN,
            stuck_sub_goal="Tap on See results for 'dosa'",
            failure_reason="no manifest label matches the named target",
            recent_actions=["Type 'dosa' in search bar"],
            suggested_next_action=None,
            escape_category=EscapeCategory.TARGET_NOT_AVAILABLE,
            strict_mode=False,
            execution_contract=ExecutionContract(),
        )

        category_framing = _ESCAPE_CATEGORY_FRAMING[EscapeCategory.TARGET_NOT_AVAILABLE.value]
        assert category_framing in preamble
        assert "Escape category: target_not_available" in preamble

    def test_preamble_falls_back_to_trigger_framing_without_category(self) -> None:
        """
        Legacy system-detected triggers (LOOP_DETECTED, NO_PROGRESS,
        ...) pass no ``escape_category`` and must still produce a
        trigger-shaped preamble.
        """

        builder = GeminiDecompositionPromptBuilder()
        preamble = builder.build_replan_user_preamble(
            trigger=RecoveryTrigger.LOOP_DETECTED,
            stuck_sub_goal="Tap on Alright, got it button",
            failure_reason="2-screen oscillation",
            recent_actions=["tap label 11", "tap label 12"],
            suggested_next_action=None,
            strict_mode=False,
            execution_contract=ExecutionContract(),
        )

        assert "Trigger: LOOP_DETECTED" in preamble
        assert "Escape category: (none)" in preamble
