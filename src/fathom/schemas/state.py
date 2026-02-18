from __future__ import annotations

from collections import deque
from logging import getLogger
from typing import Any, Deque, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenState

logger = getLogger(__name__)


class LoopDetector(BaseModel):
    """
    Detects when agent is stuck in a loop.

    Uses a sliding window of screen states to detect repeated states.
    Implements fuzzy matching via Hamming distance.
    """

    threshold: int = Field(default=3, description="Screen repetition threshold")
    window_size: int = Field(default=5, description="Size of the sliding window")

    __max_recovery: int = PrivateAttr(default=3)
    __recovery_attempts: int = PrivateAttr(default=0)
    __recent_actions: Deque[str] = PrivateAttr(default_factory=lambda: deque(maxlen=5))
    __recent_screens: Deque[ScreenState] = PrivateAttr(default_factory=lambda: deque(maxlen=5))

    def record(self, screen: ScreenState, action_description: Optional[str] = None) -> None:
        """
        Record a screen state and optionally an action description.
        """

        self.__recent_screens.append(screen)
        logger.debug(
            f"LoopDetector.record: {screen.visual_hash[:8]} ({screen.activity}) | deque_size={len(self.__recent_screens)}"
        )

        logger.debug(
            f"[H1] Recorded screen in loop detector | "
            f"activity={screen.activity} hash_prefix={screen.visual_hash[:8]} "
            f"recent_count={len(self.__recent_screens)} threshold={self.threshold}"
        )

        if action_description:
            self.__recent_actions.append(action_description)

    def is_stuck(self) -> bool:
        """
        Check if agent appears stuck in a loop.
        """

        screen_count = len(self.__recent_screens)
        logger.debug(
            f"[H4] Evaluating stuck status | "
            f"screen_count={screen_count} threshold={self.threshold} "
            f"recent_actions={len(self.__recent_actions)} can_recover={self.can_recover()}"
        )

        if screen_count < self.threshold:
            logger.debug(
                f"LoopDetector.is_stuck: False (only {screen_count} screens, need {self.threshold})"
            )
            return False

        # Check for repeated screens using fuzzy matching
        for index in range(len(self.__recent_screens)):
            count = 1
            current = self.__recent_screens[index]
            for __next_index in range(index + 1, len(self.__recent_screens)):
                if current.is_same_screen(self.__recent_screens[__next_index]):
                    count += 1

                    candidate = self.__recent_screens[__next_index]
                    distance = 64
                    if len(current.visual_hash) == len(candidate.visual_hash):
                        try:
                            distance = bin(
                                int(current.visual_hash, 16) ^ int(candidate.visual_hash, 16)
                            ).count("1")
                        except ValueError:
                            distance = 64

                    logger.debug(
                        f"[H6] Fuzzy screen match | "
                        f"base={current.activity} candidate={candidate.activity} "
                        f"dist={distance}"
                    )

            if count >= self.threshold:
                unique_recent_actions = len(set(self.__recent_actions))
                if (
                    len(self.__recent_actions) >= self.threshold
                    and unique_recent_actions >= self.threshold
                ):
                    logger.debug(
                        f"[H8] Bypassing stuck=true due to diverse recent actions | "
                        f"count={len(self.__recent_actions)} unique={unique_recent_actions} "
                        f"repeat_count={count}"
                    )
                    continue
                hashes = [s.visual_hash[:8] for s in self.__recent_screens]
                logger.debug(
                    f"LoopDetector.is_stuck: True (screen {current.visual_hash[:8]} repeated {count}x) | deque={hashes}"
                )

                logger.debug(
                    f"[H1] Stuck=true due to repeated screen | "
                    f"activity={current.activity} hash_prefix={current.visual_hash[:8]} "
                    f"repeat_count={count} unique_actions={len(set(self.__recent_actions))}"
                )
                return True

        # Check for repeated actions (exact match is fine for actions)
        if len(self.__recent_actions) >= self.threshold:
            action_counts: Dict[str, int] = {}

            for action_description in self.__recent_actions:
                action_counts[action_description] = action_counts.get(action_description, 0) + 1
                if action_counts[action_description] >= self.threshold:
                    logger.debug(
                        f"LoopDetector.is_stuck: True (action '{action_description}' repeated {action_counts[action_description]}x)"
                    )
                    return True

        logger.debug("LoopDetector.is_stuck: False (no repeats above threshold)")
        return False

    def can_recover(self) -> bool:
        """
        Check if recovery is still possible.
        """

        return self.__recovery_attempts < self.__max_recovery

    def record_recovery_attempt(self) -> int:
        """
        Record a recovery attempt and return the attempt number.
        """

        self.__recovery_attempts += 1
        return self.__recovery_attempts

    def reset(self) -> None:
        """
        Reset loop detection state.
        """

        prev_size = len(self.__recent_screens)
        self.__recent_screens.clear()
        self.__recent_actions.clear()
        self.__recovery_attempts = 0
        logger.info(f"LoopDetector.reset: cleared {prev_size} screens")

    @property
    def recovery_attempts(self) -> int:
        """Current number of recovery attempts."""
        return self.__recovery_attempts

    @recovery_attempts.setter
    def recovery_attempts(self, value: int) -> None:
        self.__recovery_attempts = value

    def signal_content_exhausted(self) -> None:
        """
        Model indicated scrollable content is exhausted.
        Reset loop state to prevent false stuck detection.
        """

        logger.info("LoopDetector: Content exhausted signal received, resetting.")
        self.__recent_screens.clear()
        self.__recent_actions.clear()
        self.__recovery_attempts = 0


class ActionHistory(BaseModel):
    """
    Tracks action history for context building with token optimization.

    Supports optional progressive summarization: when a
    ``StepSummarizer`` is attached via ``set_summarizer()``, items
    evicted from the bounded deque are fed into the summarizer
    rather than silently dropped.
    """

    max_size: int = Field(default=10, description="Maximum history size")

    __failure_count: int = PrivateAttr(default=0)
    __actions: Deque[Dict[str, Any]] = PrivateAttr(default_factory=lambda: deque(maxlen=10))
    __summarizer: Optional[Any] = PrivateAttr(default=None)

    model_config = ConfigDict(frozen=True)

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, "_ActionHistory__actions", deque(maxlen=self.max_size))

    def set_summarizer(self, summarizer: Any) -> None:
        """
        Attach a ``StepSummarizer`` to capture evicted items.

        Must be called after construction because Pydantic frozen
        models require ``object.__setattr__`` for mutation.
        """

        object.__setattr__(self, "_ActionHistory__summarizer", summarizer)

    def record_action(
        self, action: Action, success: bool, activity: str, screen_changed: bool = True
    ) -> None:
        """
        Record an action with its outcome and associated activity.

        When the deque is full, the oldest item is captured by the
        attached summarizer (if any) before it is evicted.
        """

        # Capture the item about to be evicted by the bounded deque
        if self.__summarizer is not None and len(self.__actions) == self.__actions.maxlen:
            evicted = self.__actions[0]
            self.__summarizer.ingest(evicted)

        self.__actions.append(
            {
                "success": success,
                "screen_changed": screen_changed,
                "activity": activity,
                "type": action.action_type.value.upper(),
                "full_description": action.to_description(),
                "target": action.natural_language_target or action.label_id or "UI",
            }
        )
        if not success:
            object.__setattr__(self, "_ActionHistory__failure_count", self.__failure_count + 1)

    def get_compact_history(self) -> str:
        """
        Returns a token-optimized representation of history.
        Format: TYPE:Target:Result
        """

        parts = []

        for action in self.__actions:
            result_indicator = "✓" if action["success"] else "✗"
            if action["success"] and not action.get("screen_changed", True):
                result_indicator += " (No Change)"

            parts.append(f"{action['type']}:{action['target']}:{result_indicator}")

        return " | ".join(parts) if parts else "None"

    def get_summarized_context(self) -> str:
        """
        Returns full tiered context: compressed older steps + recent raw steps.

        Structure (attention-aware placement):
            Middle: ``=== EARLIER STEPS (Summarized) ===`` — phase summaries
            End:    ``=== RECENT STEPS ===``              — raw compact history

        Falls back to ``get_compact_history()`` when no summarizer is
        attached, preserving backward compatibility.
        """

        parts: List[str] = []

        if self.__summarizer is not None:
            summary = self.__summarizer.history.format_context()
            if summary:
                parts.append(summary)

        compact = self.get_compact_history()
        if compact and compact != "None":
            parts.append(f"=== RECENT STEPS ===\n{compact}")

        return "\n".join(parts) if parts else "None"

    def get_context(self) -> List[str]:
        """
        Returns list of action descriptions.
        """

        return [action["full_description"] for action in self.__actions]

    def get_history_items(self) -> List[Dict[str, Any]]:
        """
        Returns raw list of history items for smart context.
        """
        return list(self.__actions)

    def get_activity_failures(self, current_activity: str) -> List[str]:
        """
        Returns only failures that occurred on the current activity.
        """

        return [
            f"{action['type']} on {action['target']}"
            for action in self.__actions
            if not action["success"] and action["activity"] == current_activity
        ]

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about action history.
        """

        total_count = len(self.__actions)

        return {
            "total": total_count,
            "failure": self.__failure_count,
            "success": total_count - self.__failure_count,
        }

    def get_raw_actions(self) -> List[Dict[str, Any]]:
        """
        Returns a copy of all raw action records for checkpoint serialization.
        """

        return [dict(action) for action in self.__actions]

    def restore_actions(self, records: List[Dict[str, Any]]) -> None:
        """
        Bulk-load action records into the deque during checkpoint restoration.

        Bypasses the summarizer eviction path because the summarizer
        state is restored separately from its own checkpoint data.
        """

        for record in records:
            self.__actions.append(record)
            if not record.get("success", True):
                object.__setattr__(self, "_ActionHistory__failure_count", self.__failure_count + 1)

    def has_repeated_failure(self, action: Action) -> bool:
        """
        Check if this exact action has failed recently.
        """

        description = action.to_description()

        return any(
            not historical["success"] and historical["full_description"] == description
            for historical in self.__actions
        )
