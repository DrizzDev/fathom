from __future__ import annotations

from difflib import SequenceMatcher
from logging import getLogger
from typing import List, Optional, Set

from fathom.constants import (
    ACTION_EXECUTED_TYPES,
    NEXT_PHASE_ACTION_TYPES,
    ActionType,
)
from fathom.constants.reasoning import (
    ACTION_MIN_CONFIDENCE,
    ACTION_NEXT_PHASE_CONFIDENCE,
    COMPLETION_KEYWORDS,
    MEANINGFUL_SCREEN_DELTA_FLOOR,
    NEXT_PHASE_KEYWORDS,
    OPENER_GOAL_WORDS,
    RATIONALE_CONTEXT_RELEVANCE_THRESHOLD,
    RATIONALE_KEYWORD_MATCH_THRESHOLD,
    RATIONALE_MIN_SIMILARITY_FLOOR,
)
from fathom.schemas.actions import Action
from fathom.schemas.reasoning import CompletionSignal, SubGoalCompletionSignal
from fathom.schemas.results import AnalysisResult

logger = getLogger(name=__name__)


class Reasoner:
    """
    High-speed intent reasoning engine that derives completion signals from LLM output.
    """

    def __init__(self, intent: str) -> None:
        """
        Initialize with the full intent string used for semantic alignment checks.
        """
        self.__intent = intent.lower()

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

        # Determine what we're checking completion for
        target_goal = (current_sub_goal or self.__intent).lower()
        goal_type = "sub-goal" if current_sub_goal else "intent"

        logger.debug(
            f"[Reasoner] Checking {goal_type} completion: '{target_goal}' | "
            f"llm_complete={analysis.is_goal_complete} | "
            f"action_type={analysis.action.action_type}"
        )

        # 1. Primary Signal: LLM Flag (Zero Cost - already computed)
        if analysis.is_goal_complete:
            evidence_list.append(f"LLM explicitly flagged {goal_type} completion")

        # 2. Secondary Signal: Action Type (Zero Cost)
        if analysis.action.action_type == ActionType.COMPLETE:
            evidence_list.append(f"Agent recommended COMPLETE action for {goal_type}")

        # 3. Tertiary Signal: Fast Fuzzy Match
        # We check if the reasoning text semantically overlaps with the target goal
        context = f"{analysis.reasoning} {screen_description or ''}".lower()

        # Quick ratio check - O(N) but very fast for short strings
        similarity = SequenceMatcher(None, target_goal, context).ratio()

        if similarity > RATIONALE_CONTEXT_RELEVANCE_THRESHOLD:
            evidence_list.append(f"Context alignment score: {similarity:.2f}")

        keyword_match = similarity >= RATIONALE_KEYWORD_MATCH_THRESHOLD
        action_indicates_complete = analysis.action.action_type == ActionType.COMPLETE

        # 4. Additional Signal for Sub-Goals: Action Execution on Non-Opening Tasks
        # If we're checking a sub-goal like "Open X" and the LLM is DOING something
        # (not just planning), it means "Open X" is likely ALREADY complete
        action_suggests_next_phase = False
        if (
            current_sub_goal
            and analysis.action.action_type in NEXT_PHASE_ACTION_TYPES
            and any(word in target_goal for word in OPENER_GOAL_WORDS)
        ):
            # LLM is actively performing actions. If the current sub-goal is an opener
            # (contains "open", "launch", "navigate to", "go to"), then performing
            # a tap/type suggests we're past the opening phase.
            # Check if reasoning suggests we're at a next phase
            reasoning_lower = analysis.reasoning.lower()
            # More flexible keyword matching - check for partial matches
            if any(keyword in reasoning_lower for keyword in NEXT_PHASE_KEYWORDS):
                evidence_list.append(
                    f"LLM performing next-phase action ({analysis.action.action_type.value})"
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
            llm_confidence = max(llm_confidence, analysis.action.confidence)

        if action_indicates_complete:
            llm_confidence = max(llm_confidence, analysis.action.confidence)

        if keyword_match:
            llm_confidence = max(llm_confidence, similarity)

        if action_suggests_next_phase:
            llm_confidence = max(llm_confidence, ACTION_NEXT_PHASE_CONFIDENCE)

        logger.debug(
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

    def analyze_subgoal_completion(
        self,
        analysis: AnalysisResult,
        sub_goal_description: str,
        *,
        screen_changed: bool = False,
        delta_score: Optional[float] = None,
        screen_description: Optional[str] = None,
    ) -> SubGoalCompletionSignal:
        """
        Multi-signal verification for sub-goal completion.
        ``delta_score`` gates screen verification by magnitude; ``screen_changed`` is a boolean fallback when no magnitude is available.
        """

        evidence: List[str] = []
        target = sub_goal_description.lower()

        # Flag 1: model explicitly raised the completion flag via tool output
        # or recommended the COMPLETE action.
        if flagged_complete := (
            analysis.is_sub_goal_complete
            or analysis.is_goal_complete
            or analysis.action.action_type == ActionType.COMPLETE
        ):
            evidence.append("Model flagged sub-goal completion via tool output")

        # Flag 2: rationale-side verification. Prefer the model's explicit
        # completion reason; fall back to a keyword + similarity heuristic
        # so non-explicit narrations are not silently dropped.
        rationale_verified, rationale_evidence, keyword_match, similarity = self.__verify_rationale(
            target=target,
            analysis=analysis,
            flagged_complete=flagged_complete,
            screen_description=screen_description,
        )
        if rationale_evidence:
            evidence.append(rationale_evidence)

        # Flag 3: an action that actually ran (planning-only actions excluded).
        action_executed = analysis.action.action_type in ACTION_EXECUTED_TYPES

        if action_executed:
            evidence.append(f"Action executed: {analysis.action.action_type.value}")

        # Flag 4: post-action screen change exceeded the meaningful-delta floor.
        # Magnitude path rejects animation noise; boolean path is the fallback.
        screen_verified, screen_evidence = self.__verify_screen_change(
            delta_score=delta_score,
            screen_changed=screen_changed,
        )
        if screen_verified:
            evidence.append(screen_evidence)

        elif flagged_complete:
            evidence.append(f"WARNING: model flagged complete but {screen_evidence}")

        llm_confidence = self.__derive_llm_confidence(
            analysis=analysis, keyword_match=keyword_match, similarity=similarity
        )

        signal = SubGoalCompletionSignal(
            trace_verified=False,
            keyword_match=keyword_match,
            llm_confidence=llm_confidence,
            action_executed=action_executed,
            screen_verified=screen_verified,
            flagged_complete=flagged_complete,
            rationale_verified=rationale_verified,
            evidence="; ".join(evidence) if evidence else "No sub-goal completion signals detected",
        )

        logger.info(
            "[Reasoner] Sub-goal verdict",
            extra={
                "component": "reasoner",
                "event": "subgoal_signals",
                "sub_goal": sub_goal_description[:80],
                "claim_verified": signal.claim_verified,
                "action_effective": signal.action_effective,
                "flagged_complete": flagged_complete,
                "rationale_verified": rationale_verified,
                "action_executed": action_executed,
                "screen_verified": screen_verified,
                "delta_score": delta_score,
                "similarity": round(similarity, 3),
                "llm_confidence": round(llm_confidence, 3),
            },
        )
        return signal

    @staticmethod
    def __verify_rationale(
        *,
        target: str,
        flagged_complete: bool,
        analysis: AnalysisResult,
        screen_description: Optional[str],
    ) -> tuple[bool, Optional[str], bool, float]:
        """
        Decide rationale verification.
        Returns ``(verified, evidence, keyword_match, similarity)``.
        """

        context = f"{analysis.reasoning} {screen_description or ''}".lower()

        similarity = SequenceMatcher(None, target, context).ratio()
        keyword_match = similarity >= RATIONALE_KEYWORD_MATCH_THRESHOLD
        keywords_found = any(kw in analysis.reasoning.lower() for kw in COMPLETION_KEYWORDS)

        explicit_reason = analysis.subgoal_completion_reason or analysis.goal_completion_reason
        if flagged_complete and explicit_reason:
            return (
                True,
                f"Rationale verified via model reason: '{explicit_reason}'",
                keyword_match,
                similarity,
            )

        if keyword_match or (similarity >= RATIONALE_MIN_SIMILARITY_FLOOR and keywords_found):
            return (
                True,
                f"Rationale verified via heuristic (similarity={similarity:.2f})",
                keyword_match,
                similarity,
            )

        # Surface the rejection mode for observability: a completion
        # keyword landed in the rationale but the rationale text shares
        # too little with the target to be trusted as evidence. This is
        # the regression mode the similarity floor was raised to catch.
        if keywords_found and similarity < RATIONALE_MIN_SIMILARITY_FLOOR:
            logger.info(
                "Rationale rejected: completion keyword present but similarity below floor",
                extra={
                    "component": "reasoner",
                    "event": "rationale.rejected.below_similarity_floor",
                    "similarity": round(similarity, 3),
                    "floor": RATIONALE_MIN_SIMILARITY_FLOOR,
                },
            )

        return False, None, keyword_match, similarity

    @staticmethod
    def __verify_screen_change(
        *,
        delta_score: Optional[float],
        screen_changed: bool,
    ) -> tuple[bool, str]:
        """
        Decide screen verification. Returns ``(verified, evidence)``.
        Magnitude path takes precedence; boolean is the fallback.
        """

        if delta_score is not None:
            if delta_score >= MEANINGFUL_SCREEN_DELTA_FLOOR:
                return True, f"Screen changed meaningfully (delta={delta_score:.2f})"

            return False, f"delta {delta_score:.2f} below floor {MEANINGFUL_SCREEN_DELTA_FLOOR}"

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

        if analysis.is_sub_goal_complete or analysis.is_goal_complete:
            confidence = max(confidence, analysis.action.confidence)

        if keyword_match:
            confidence = max(confidence, similarity)

        return confidence

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
