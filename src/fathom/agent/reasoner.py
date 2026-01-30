from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Set

from fathom.constants import ActionType
from fathom.schemas.actions import Action
from fathom.tools.vision import AnalysisResult


@dataclass(frozen=True)
class CompletionSignal:
    """Signals that indicate intent completion.

    Captures evidence that the goal has been achieved.
    """

    keyword_match: bool = False
    success_indicator: bool = False
    expected_screen: bool = False
    llm_confidence: float = 0.0
    evidence: str = ""

    @property
    def is_complete(self) -> bool:
        """Determine if signals indicate completion.

        Uses weighted scoring:
        - LLM confidence > 0.8: high weight
        - Keyword match: medium weight
        - Success indicator: medium weight
        - Expected screen: low weight (may be intermediate)
        """
        score = 0.0
        if self.llm_confidence >= 0.8:
            score += 0.5
        elif self.llm_confidence >= 0.5:
            score += 0.25

        if self.keyword_match:
            score += 0.3
        if self.success_indicator:
            score += 0.15
        if self.expected_screen:
            score += 0.1

        return score >= 0.5


class CompletionMatcher(ABC):
    """
    Abstract matcher for completion detection.
    """

    @abstractmethod
    def matches(self, text: str) -> bool:
        """
        Check if text matches completion criteria.
        """
        raise NotImplementedError

    @abstractmethod
    def get_evidence(self, text: str) -> str:
        """
        Get evidence string if matched.
        """
        raise NotImplementedError


class KeywordMatcher(CompletionMatcher):
    """
    Matches completion based on keywords in screen content.
    """

    def __init__(self, keywords: Set[str]) -> None:
        self.__keywords = {k.lower() for k in keywords}
        self.__pattern = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in self.__keywords) + r")\b",
            re.IGNORECASE,
        )

    def matches(self, text: str) -> bool:
        return bool(self.__pattern.search(text))

    def get_evidence(self, text: str) -> str:
        match = self.__pattern.search(text)
        if match:
            return f"Found keyword: '{match.group(1)}'"
        return ""


class IntentMatcher(CompletionMatcher):
    """Matches completion based on intent-specific patterns.

    Parses the intent to extract expected outcomes.
    """

    __SUCCESS_PATTERNS: ClassVar[List[str]] = [
        r"success(ful(ly)?)?",
        r"complet(ed|ion)",
        r"confirm(ed|ation)",
        r"done",
        r"saved",
        r"submitted",
        r"logged\s*in",
        r"signed\s*in",
        r"welcome",
        r"thank\s*you",
        r"order\s*placed",
        r"payment\s*(successful|complete)",
    ]

    def __init__(self) -> None:
        self.__pattern = re.compile(
            "|".join(self.__SUCCESS_PATTERNS),
            re.IGNORECASE,
        )

    def matches(self, text: str) -> bool:
        return bool(self.__pattern.search(text))

    def get_evidence(self, text: str) -> str:
        match = self.__pattern.search(text)
        if match:
            return f"Success pattern: '{match.group(0)}'"
        return ""


class Reasoner:
    """Intent reasoning and completion detection.

    Combines multiple signals to determine:
    - Whether the intent is complete
    - What action to take next
    - Whether to abort due to errors

    Uses both rule-based matching and LLM confidence.
    """

    def __init__(
        self,
        intent: str,
        *,
        custom_keywords: Optional[Set[str]] = None,
    ) -> None:
        """Initialize reasoner.

        Args:
            intent: The goal being pursued.
            custom_keywords: Additional keywords to detect completion.
        """
        self.__intent = intent
        self.__keywords = self.__extract_keywords(intent)
        if custom_keywords:
            self.__keywords.update(custom_keywords)

        self.__keyword_matcher = KeywordMatcher(self.__keywords)
        self.__intent_matcher = IntentMatcher()
        self.__completion_evidence: List[str] = []

    def __extract_keywords(self, intent: str) -> Set[str]:
        """Extract completion keywords from intent.

        Parses intent to find expected outcomes.
        E.g., "Login with phone" -> looking for "logged in", "welcome"
        E.g., "Add to cart" -> looking for "added", "cart"
        """
        keywords: Set[str] = set()
        intent_lower = intent.lower()

        keyword_map = {
            "login": {"logged in", "welcome", "sign in", "dashboard"},
            "sign up": {"account created", "welcome", "verify"},
            "register": {"registered", "account created", "verify"},
            "add to cart": {"added to cart", "cart", "item added"},
            "checkout": {"order placed", "payment", "confirmation"},
            "search": {"results", "found"},
            "send": {"sent", "delivered"},
            "submit": {"submitted", "received"},
            "save": {"saved", "updated"},
            "delete": {"deleted", "removed"},
            "cancel": {"cancelled", "canceled"},
        }

        for trigger, words in keyword_map.items():
            if trigger in intent_lower:
                keywords.update(words)

        return keywords

    def analyze_completion(
        self,
        analysis: AnalysisResult,
        screen_description: Optional[str] = None,
    ) -> CompletionSignal:
        """Analyze whether the intent is complete.

        Args:
            analysis: Result from vision tool.
            screen_description: Optional screen content description.

        Returns:
            CompletionSignal with detailed evidence.
        """
        keyword_match = False
        success_indicator = False
        evidence_parts: List[str] = []

        if analysis.is_goal_complete:
            evidence_parts.append("LLM indicated goal complete")

        if analysis.action.action_type == ActionType.COMPLETE:
            evidence_parts.append("COMPLETE action recommended")

        text_to_check = analysis.reasoning or ""
        if screen_description:
            text_to_check += " " + screen_description
        if analysis.screen_description:
            text_to_check += " " + analysis.screen_description

        if self.__keyword_matcher.matches(text_to_check):
            keyword_match = True
            evidence_parts.append(self.__keyword_matcher.get_evidence(text_to_check))

        if self.__intent_matcher.matches(text_to_check):
            success_indicator = True
            evidence_parts.append(self.__intent_matcher.get_evidence(text_to_check))

        llm_confidence = analysis.action.confidence if analysis.is_goal_complete else 0.0

        return CompletionSignal(
            keyword_match=keyword_match,
            success_indicator=success_indicator,
            expected_screen=analysis.is_goal_complete,
            llm_confidence=llm_confidence,
            evidence="; ".join(evidence_parts),
        )

    def should_accept_action(
        self,
        action: Action,
        *,
        has_failed_before: bool = False,
    ) -> bool:
        """Determine if an action should be accepted.

        Args:
            action: Proposed action.
            has_failed_before: Whether this action has failed recently.

        Returns:
            True if action should be executed.
        """
        if action.confidence < 0.3:
            return False

        if has_failed_before and action.confidence < 0.7:
            return False

        if action.action_type in (ActionType.BACK, ActionType.HOME):
            return action.confidence >= 0.4

        return True

    def select_best_action(
        self,
        primary: Action,
        alternatives: List[Action],
        *,
        failed_actions: Optional[Set[str]] = None,
    ) -> Action:
        """Select the best action from candidates.

        Args:
            primary: Primary recommended action.
            alternatives: Alternative actions considered.
            failed_actions: Set of action descriptions that have failed.

        Returns:
            Best action to execute.
        """
        failed = failed_actions or set()
        primary_desc = primary.to_description()

        if primary_desc not in failed:
            return primary

        for alt in alternatives:
            alt_desc = alt.to_description()
            if alt_desc not in failed and alt.confidence >= 0.4:
                return alt

        return primary

    def build_prompt_context(
        self,
        recent_actions: List[str],
        recent_failures: List[str],
        step_count: int,
        max_steps: int,
    ) -> str:
        """Build context string for LLM prompt.

        Args:
            recent_actions: Recent action history.
            recent_failures: Recent failures.
            step_count: Current step number.
            max_steps: Maximum allowed steps.

        Returns:
            Formatted context string.
        """
        parts: List[str] = []

        parts.append(f"Goal: {self.__intent}")
        parts.append(f"Progress: Step {step_count}/{max_steps}")

        if recent_actions:
            parts.append("Recent actions:")
            for action in recent_actions[-5:]:
                parts.append(f"  - {action}")

        if recent_failures:
            parts.append("Recent failures (avoid repeating):")
            for failure in recent_failures[-3:]:
                parts.append(f"  - {failure}")

        steps_remaining = max_steps - step_count
        if steps_remaining <= 5:
            parts.append(f"⚠ Only {steps_remaining} steps remaining - prioritize completion")

        return "\n".join(parts)
