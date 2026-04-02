from __future__ import annotations

from difflib import SequenceMatcher
from logging import getLogger
from typing import List, Optional, Set

from fathom.constants import ACTION_EXECUTED_TYPES, NEXT_PHASE_ACTION_TYPES
from fathom.constants.reasoning import (
    ACTION_MIN_CONFIDENCE,
    ACTION_NEXT_PHASE_CONFIDENCE,
    NEXT_PHASE_KEYWORDS,
    OPENER_GOAL_WORDS,
    RATIONALE_CONTEXT_RELEVANCE_THRESHOLD,
    RATIONALE_KEYWORD_MATCH_THRESHOLD,
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

        # 2. Tertiary Signal: Fast Fuzzy Match
        # We check if the reasoning text semantically overlaps with the target goal
        context = f"{analysis.reasoning} {screen_description or ''}".lower()

        # Quick ratio check - O(N) but very fast for short strings
        similarity = SequenceMatcher(None, target_goal, context).ratio()

        if similarity > RATIONALE_CONTEXT_RELEVANCE_THRESHOLD:
            evidence_list.append(f"Context alignment score: {similarity:.2f}")

        keyword_match = similarity >= RATIONALE_KEYWORD_MATCH_THRESHOLD

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

        # Completion requires the LLM's explicit flag, strong semantic alignment,
        # or next-phase action detection for opener sub-goals.
        is_complete = analysis.is_goal_complete or keyword_match or action_suggests_next_phase

        llm_confidence = 0.0

        if analysis.is_goal_complete:
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
        screen_description: Optional[str] = None,
        *,
        screen_changed: bool = False,
        pre_screen_hash: Optional[str] = None,
        post_screen_hash: Optional[str] = None,
    ) -> SubGoalCompletionSignal:
        """
        Multi-signal verification for sub-goal completion.

        Args:
            analysis: LLM analysis result
            sub_goal_description: Description of current sub-goal
            screen_description: Optional screen description for context

        Returns:
            SubGoalCompletionSignal with all verification flags
        """
        evidence_list: List[str] = []
        target_goal = sub_goal_description.lower()

        logger.debug(
            f"[Reasoner] Analyzing sub-goal completion: '{target_goal}' | "
            f"llm_flag={analysis.is_goal_complete} | "
            f"sub_goal_flag={analysis.is_sub_goal_complete} | "
            f"action_type={analysis.action.action_type}"
        )

        # Signal 1: LLM Explicit Signal (from tool output flags only).
        # action_type=COMPLETE is NOT treated as an independent completion
        # signal — it must be paired with an explicit sub_goal_completed=true
        # or goal_completed=true flag to count.
        llm_signaled = analysis.is_sub_goal_complete or analysis.is_goal_complete
        if llm_signaled:
            evidence_list.append("LLM signaled sub-goal completion via tool output")

        # Signal 2: Rationale verification — disabled.
        #
        # The LLM-backed rationale check added ~1-2s latency per step without
        # visual grounding (text-only comparison).  The VERIFY node with its
        # screenshot-based check is the sole completion gate for the overall
        # intent.  Sub-goal advancement now relies on llm_signaled +
        # effective_action (2 signals).
        rationale_verified = False

        # Lightweight similarity for confidence scoring only (not gating).
        context = f"{analysis.reasoning} {screen_description or ''}".lower()
        similarity = SequenceMatcher(None, target_goal, context).ratio()
        keyword_match = similarity >= RATIONALE_KEYWORD_MATCH_THRESHOLD

        # Signal 3: Action Execution (did we execute an action?)
        action_executed = analysis.action.action_type in ACTION_EXECUTED_TYPES
        if action_executed:
            evidence_list.append(f"Action executed: {analysis.action.action_type.value}")

        # Signal 4: Screen Verification — post-execution sanity check.
        # If the action didn't change the screen AND the LLM claimed sub-goal
        # completion, the action may have failed silently (e.g. tap on a blocking
        # overlay, wrong screen).  ``screen_verified`` gates ``action_executed``
        # in count_signals() to prevent premature sub-goal advancement.
        screen_verified = False
        if screen_changed:
            screen_verified = True
            evidence_list.append("Screen changed after action execution")
        elif pre_screen_hash and post_screen_hash and pre_screen_hash != post_screen_hash:
            screen_verified = True
            evidence_list.append("Screen hash changed after action execution")
        else:
            if llm_signaled:
                evidence_list.append("WARNING: LLM signaled completion but screen did not change")

        # Calculate LLM confidence
        llm_confidence = 0.0

        if analysis.is_sub_goal_complete or analysis.is_goal_complete:
            llm_confidence = max(llm_confidence, analysis.action.confidence)

        if keyword_match:
            llm_confidence = max(llm_confidence, similarity)

        logger.debug(
            f"[Reasoner] Sub-goal signals: llm={llm_signaled}, "
            f"rationale={rationale_verified}, action={action_executed}, "
            f"screen_verified={screen_verified}, "
            f"confidence={llm_confidence:.2f}"
        )

        evidence = (
            "; ".join(evidence_list) if evidence_list else "No sub-goal completion signals detected"
        )

        return SubGoalCompletionSignal(
            evidence=evidence,
            trace_verified=False,  # Will be set by state manager
            llm_signaled=llm_signaled,
            keyword_match=keyword_match,
            llm_confidence=llm_confidence,
            action_executed=action_executed,
            screen_verified=screen_verified,
            rationale_verified=rationale_verified,
        )

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
