from __future__ import annotations

from difflib import SequenceMatcher
from logging import getLogger
from typing import List, Optional, Set, Tuple

from fathom.constants import (
    ActionType,
    StepEvent,
)
from fathom.constants.reasoning import (
    ACTION_MIN_CONFIDENCE,
    ACTION_NEXT_PHASE_CONFIDENCE,
    COMPLETION_KEYWORDS,
    NEXT_PHASE_KEYWORDS,
    OPENER_GOAL_WORDS,
    RATIONALE_CONTEXT_RELEVANCE_THRESHOLD,
    RATIONALE_KEYWORD_MATCH_THRESHOLD,
    RATIONALE_MIN_SIMILARITY_FLOOR,
)
from fathom.core.agent.opener import OpenerSignalPolicy
from fathom.core.exceptions import InvariantViolation
from fathom.schemas.actions import Action
from fathom.schemas.effect import ActionEffect, ActionEffectStatus
from fathom.schemas.reasoning import CompletionSignal
from fathom.schemas.results import AnalysisResult

logger = getLogger(name=__name__)


class Reasoner:
    """
    High-speed intent reasoning engine that derives completion signals from LLM output.
    """

    def __init__(self, intent: str, *, opener_policy: OpenerSignalPolicy) -> None:
        """
        Initialize with the intent string for alignment checks and the opener-completion policy.
        """
        self.__intent = intent.lower()
        self.__opener_policy = opener_policy

    def analyze_completion(
        self,
        analysis: AnalysisResult,
        screen_description: Optional[str] = None,
        current_sub_goal: Optional[str] = None,
    ) -> CompletionSignal:
        """
        Determines completion using only local, fast signals.

        Args:
            analysis: LLM analysis result.
            screen_description: Optional screen description for context.
            current_sub_goal: If provided, checks completion of this sub-goal instead of full intent.

        Returns:
            Completion signal with evidence.
        """

        evidence_list: List[str] = []
        action = self.__require_action(analysis=analysis)

        target_goal = (current_sub_goal or self.__intent).lower()
        goal_type = "sub-goal" if current_sub_goal else "intent"

        logger.info(
            f"[Reasoner] Checking {goal_type} completion: '{target_goal}' | "
            f"llm_complete={analysis.is_goal_complete} | "
            f"action_type={action.action_type}"
        )

        if analysis.is_goal_complete:
            evidence_list.append(f"LLM explicitly flagged {goal_type} completion")

        if action.action_type == ActionType.COMPLETE:
            evidence_list.append(f"Agent recommended COMPLETE action for {goal_type}")

        context = f"{analysis.reasoning} {screen_description or ''}".lower()

        similarity = SequenceMatcher(None, target_goal, context).ratio()

        if similarity > RATIONALE_CONTEXT_RELEVANCE_THRESHOLD:
            evidence_list.append(f"Context alignment score: {similarity:.2f}")

        keyword_match = similarity >= RATIONALE_KEYWORD_MATCH_THRESHOLD
        action_indicates_complete = action.action_type == ActionType.COMPLETE

        action_suggests_next_phase = False
        if (
            current_sub_goal
            and self.__opener_policy.advanced(action_type=action.action_type)
            and any(word in target_goal for word in OPENER_GOAL_WORDS)
        ):
            # An opener sub-goal ("open", "launch", "navigate to", "go to") is likely already past its opening
            # phase once the model performs a tap/type, so a next-phase keyword in the rationale confirms it.
            reasoning_lower = analysis.reasoning.lower()
            if any(keyword in reasoning_lower for keyword in NEXT_PHASE_KEYWORDS):
                evidence_list.append(
                    f"LLM performing next-phase action ({action.action_type.value})"
                )
                action_suggests_next_phase = True

        # For sub-goals, allow strong semantic alignment or next-phase actions to count as completion
        is_complete = (
            analysis.is_goal_complete
            or action_indicates_complete
            or keyword_match
            or action_suggests_next_phase
        )

        llm_confidence = 0.0

        if analysis.is_goal_complete:
            llm_confidence = max(llm_confidence, action.confidence)

        if action_indicates_complete:
            llm_confidence = max(llm_confidence, action.confidence)

        if keyword_match:
            llm_confidence = max(llm_confidence, similarity)

        if action_suggests_next_phase:
            llm_confidence = max(llm_confidence, ACTION_NEXT_PHASE_CONFIDENCE)

        logger.info(
            f"[Reasoner] {goal_type.capitalize()} completion: {is_complete} "
            f"(evidence: {'; '.join(evidence_list) if evidence_list else 'none'})"
        )

        evidence = (
            "; ".join(evidence_list)
            if evidence_list
            else f"No {goal_type} completion signals detected"
        )

        return CompletionSignal(
            evidence=evidence,
            keyword_match=keyword_match,
            llm_confidence=llm_confidence,
            success_indicator=is_complete,
            expected_screen=analysis.is_goal_complete,
        )

    @staticmethod
    def __validation_executed(*, action: Action, execution_success: bool) -> bool:
        """
        Return whether this turn recorded a validation-family examination.
        """

        return (
            execution_success
            and action.event_type == StepEvent.VALIDATION
            and bool((action.validation_subject or "").strip())
        )

    @staticmethod
    def __verify_rationale(
        *,
        target: str,
        flagged_complete: bool,
        analysis: AnalysisResult,
        screen_description: Optional[str],
    ) -> Tuple[bool, Optional[str], bool, float]:
        """
        Decide whether a claim carried a reason-like signal.
        Returns ``(present, evidence, keyword_match, similarity)``.
        """

        context = f"{analysis.reasoning} {screen_description or ''}".lower()

        similarity = SequenceMatcher(None, target, context).ratio()
        keyword_match = similarity >= RATIONALE_KEYWORD_MATCH_THRESHOLD
        keywords_found = any(kw in analysis.reasoning.lower() for kw in COMPLETION_KEYWORDS)

        explicit_reason = analysis.subgoal_completion_reason or analysis.goal_completion_reason
        if flagged_complete and explicit_reason:
            return (
                True,
                f"Completion reason provided by model: '{explicit_reason}'",
                keyword_match,
                similarity,
            )

        if keyword_match or (similarity >= RATIONALE_MIN_SIMILARITY_FLOOR and keywords_found):
            return (
                True,
                f"Completion reason inferred by legacy rationale matcher "
                f"(similarity={similarity:.2f})",
                keyword_match,
                similarity,
            )

        # Surface the rejection mode for observability: a completion keyword landed in the rationale but the
        # rationale text shares too little with the target to be trusted as evidence.
        if keywords_found and similarity < RATIONALE_MIN_SIMILARITY_FLOOR:
            logger.info(
                "Rationale rejected: completion keyword present but similarity below floor",
                extra={
                    "component": "reasoner",
                    "similarity": round(similarity, 3),
                    "floor": RATIONALE_MIN_SIMILARITY_FLOOR,
                    "event": "rationale.rejected.below_similarity_floor",
                },
            )

        return False, None, keyword_match, similarity

    @staticmethod
    def __verify_screen_change(
        *,
        screen_changed: bool,
        effect: Optional[ActionEffect] = None,
    ) -> Tuple[bool, str]:
        """
        Return (verified, evidence) for the screen-change verification; NO_PROGRESS effect short-circuits to false.
        """

        if effect is not None and effect.status is ActionEffectStatus.NO_PROGRESS:
            return (False, "effect.status=no_progress vetoed screen.evolved")

        if screen_changed:
            return True, "Screen changed after action execution"

        return False, "screen did not change after action"

    @staticmethod
    def __derive_llm_confidence(
        *, analysis: AnalysisResult, keyword_match: bool, similarity: float
    ) -> float:
        """
        Pick the strongest available confidence signal.
        """

        confidence = 0.0
        action = Reasoner.__require_action(analysis=analysis)

        if analysis.is_sub_goal_complete or analysis.is_goal_complete:
            confidence = max(confidence, action.confidence)

        if keyword_match:
            confidence = max(confidence, similarity)

        return confidence

    @staticmethod
    def __require_action(*, analysis: AnalysisResult) -> Action:
        """
        Return the executable action or fail on an invalid reasoner call.
        """

        if analysis.action is None:
            raise InvariantViolation("Reasoner requires an executable action analysis.")

        return analysis.action

    def should_accept_action(
        self,
        action: Action,
        *,
        has_failed_before: bool = False,
    ) -> bool:
        """
        Fast safety check — always accepts actions to avoid silently
        dropping planned steps before they reach the executor.
        """

        _ = action, has_failed_before
        return True

    def select_best_action(
        self,
        primary: Action,
        alternatives: List[Action],
        *,
        failed_actions: Set[str],
    ) -> Action:
        """
        Fast selection logic. Returns the highest-confidence non-failed action from candidates.
        """

        if primary.to_description() not in failed_actions:
            return primary

        for alternative in alternatives:
            if (
                alternative.to_description() not in failed_actions
                and alternative.confidence > ACTION_MIN_CONFIDENCE
            ):
                return alternative

        return primary
