from __future__ import annotations

from difflib import SequenceMatcher
from logging import getLogger
from typing import List, Optional, Set

from fathom.constants import ActionType
from fathom.schemas.actions import Action
from fathom.schemas.reasoning import CompletionSignal
from fathom.tools.vision import AnalysisResult

logger = getLogger(name=__name__)


class Reasoner:
    """
    High-speed intent reasoning engine.
    """

    def __init__(self, intent: str) -> None:
        self.__intent = intent.lower()

    def set_intent(self, intent: str) -> None:
        """Update the intent used for fuzzy matching."""

        self.__intent = intent.lower()

    def analyze_completion(
        self,
        analysis: AnalysisResult,
        screen_description: Optional[str] = None,
    ) -> CompletionSignal:
        """
        Determines completion using only local, fast signals.
        """

        evidence_list = []

        # 1. Primary Signal: LLM Flag (Zero Cost - already computed)
        if analysis.is_goal_complete:
            evidence_list.append("LLM explicitly flagged completion")

        # 2. Secondary Signal: Action Type (Zero Cost)
        if analysis.action.action_type == ActionType.COMPLETE:
            evidence_list.append("Agent recommended COMPLETE action")

        # 3. Tertiary Signal: Fast Fuzzy Match
        # We check if the reasoning text semantically overlaps with the intent
        context = f"{analysis.reasoning} {screen_description or ''}".lower()

        # Quick ratio check - O(N) but very fast for short strings
        similarity = SequenceMatcher(None, self.__intent, context).ratio()

        if similarity > 0.6:  # Threshold for "relevant context"
            evidence_list.append(f"Context alignment score: {similarity:.2f}")

        # Weighted Decision
        # If LLM says complete, we trust it unless the action is blatantly wrong.
        is_complete = analysis.is_goal_complete or (
            analysis.action.action_type == ActionType.COMPLETE
        )

        return CompletionSignal(
            success_indicator=is_complete,
            evidence="; ".join(evidence_list),
            expected_screen=analysis.is_goal_complete,
            llm_confidence=analysis.action.confidence if is_complete else 0.0,
        )

    def should_accept_action(
        self,
        action: Action,
        *,
        has_failed_before: bool = False,
    ) -> bool:
        """
        Fast safety check.
        """

        if action.confidence < 0.4:
            return False

        return not (has_failed_before and action.confidence < 0.8)

    def select_best_action(
        self,
        primary: Action,
        alternatives: List[Action],
        *,
        failed_actions: Set[str],
    ) -> Action:
        """
        Fast selection logic.
        """

        if primary.to_description() not in failed_actions:
            return primary

        for alternative in alternatives:
            if alternative.to_description() not in failed_actions and alternative.confidence > 0.5:
                return alternative

        return primary
