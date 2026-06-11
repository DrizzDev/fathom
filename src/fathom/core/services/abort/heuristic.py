from __future__ import annotations

import re
from logging import getLogger
from typing import Optional, Tuple

from rapidfuzz import fuzz, process

from fathom.interfaces.abort import AbortDetectorPort
from fathom.schemas.abort import AbortDecision, AbortFallbackConfiguration

logger = getLogger(__name__)


class HeuristicAbortDetector(AbortDetectorPort):
    """
    Rapidfuzz-backed fallback abort detector for use when the LLM classifier is unavailable.
    """

    __TOKEN_PATTERN = re.compile(r"[a-z]+")

    __UI_DIRECTIVE_VERBS = frozenset(
        {
            "go",
            "tap",
            "hit",
            "type",
            "open",
            "push",
            "pick",
            "input",
            "press",
            "swipe",
            "click",
            "enter",
            "scroll",
            "select",
            "choose",
            "navigate",
        }
    )

    __ANCHOR_PHRASES: Tuple[str, ...] = (
        "stop the run",
        "kill the run",
        "abort the run",
        "close the run",
        "cancel the run",
        "kill the agent",
        "end the test run",
        "end this test run",
        "stop the workflow",
        "stop this test run",
        "terminate the run",
        "mark as completed",
        "stop the execution",
        "abort the workflow",
        "cancel the workflow",
        "close the execution",
        "mark this as complete",
        "terminate the workflow",
    )

    def __init__(self, *, configuration: Optional[AbortFallbackConfiguration] = None) -> None:
        """
        Bind the detector to an optional fallback configuration; defaults when unbound.
        """

        self.__configuration = configuration or AbortFallbackConfiguration()

    async def aborted(self, *, response: str) -> AbortDecision:
        """
        Classify the response using rapidfuzz fuzzy matching against canonical anchors.
        """

        normalized = response.strip().lower()

        if not normalized:
            return self.__safe_decision()

        if self.__contains_ui_directive_verb(text=normalized):
            logger.info(
                "Heuristic abort detector blocked by UI-directive guard",
                extra={
                    "event": "abort.heuristic.ui_directive_guard",
                    "component": "core.services.abort.heuristic",
                    "response.preview": normalized[:120],
                },
            )
            return self.__safe_decision()

        match = process.extractOne(
            normalized,
            self.__ANCHOR_PHRASES,
            scorer=fuzz.token_set_ratio,
        )

        if match is None:
            return self.__safe_decision()

        _, score, _ = match
        normalized_score = score / 100.0

        floor = self.__configuration.similarity_floor
        aborted = normalized_score >= floor

        logger.info(
            "Heuristic abort detector verdict",
            extra={
                "event": "abort.heuristic.verdict",
                "component": "core.services.abort.heuristic",
                "verdict.floor": floor,
                "verdict.aborted": aborted,
                "response.preview": normalized[:120],
                "verdict.confidence": round(normalized_score, 4),
            },
        )

        return AbortDecision(aborted=aborted, confidence=normalized_score, fallback=True)

    async def warmup(self) -> None:
        """
        Pure-Python heuristic has no model to warm; no-op.
        """

        return

    @classmethod
    def __contains_ui_directive_verb(cls, *, text: str) -> bool:
        """
        Return True when any token in the text is a known UI-action verb.
        """

        tokens = cls.__TOKEN_PATTERN.findall(text)
        return any(token in cls.__UI_DIRECTIVE_VERBS for token in tokens)

    @staticmethod
    def __safe_decision() -> AbortDecision:
        """
        Build the safe non-abort decision used when input is empty or a UI directive.
        """

        return AbortDecision(aborted=False, confidence=0.0, fallback=True)
