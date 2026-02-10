from __future__ import annotations

from collections import deque
import json
from logging import getLogger
import time
from typing import Any, Deque, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenState

logger = getLogger(__name__)
DEBUG_LOG_PATH = "/Users/mohnishbangaru/Fathom v1/fathom/.cursor/debug.log"


def _debug_log(hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    payload = {
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as debug_file:
            debug_file.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        logger.debug("Debug instrumentation write failed", exc_info=True)


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
        logger.debug(f"LoopDetector.record: {screen.visual_hash[:8]} ({screen.activity}) | deque_size={len(self.__recent_screens)}")
        # region agent log
        _debug_log(
            hypothesis_id="H1",
            location="src/fathom/schemas/state.py:record",
            message="Recorded screen in loop detector",
            data={
                "activity": screen.activity,
                "activity_hash": screen.activity_hash,
                "visual_hash_prefix": screen.visual_hash[:8],
                "recent_screen_count": len(self.__recent_screens),
                "threshold": self.threshold,
            },
        )
        # endregion

        if action_description:
            self.__recent_actions.append(action_description)

    def is_stuck(self) -> bool:
        """
        Check if agent appears stuck in a loop.
        """

        screen_count = len(self.__recent_screens)
        # region agent log
        _debug_log(
            hypothesis_id="H4",
            location="src/fathom/schemas/state.py:is_stuck",
            message="Evaluating stuck status",
            data={
                "screen_count": screen_count,
                "threshold": self.threshold,
                "recent_actions_count": len(self.__recent_actions),
                "can_recover": self.can_recover(),
            },
        )
        # endregion
        if screen_count < self.threshold:
            logger.debug(f"LoopDetector.is_stuck: False (only {screen_count} screens, need {self.threshold})")
            return False

        # Check for repeated screens using fuzzy matching
        for index in range(len(self.__recent_screens)):
            count = 1
            current = self.__recent_screens[index]
            for __next_index in range(index + 1, len(self.__recent_screens)):
                if current.is_same_screen(self.__recent_screens[__next_index]):
                    count += 1
                    # region agent log
                    candidate = self.__recent_screens[__next_index]
                    distance = 64
                    if len(current.visual_hash) == len(candidate.visual_hash):
                        try:
                            distance = bin(
                                int(current.visual_hash, 16) ^ int(candidate.visual_hash, 16)
                            ).count("1")
                        except ValueError:
                            distance = 64
                    _debug_log(
                        hypothesis_id="H6",
                        location="src/fathom/schemas/state.py:is_stuck",
                        message="Fuzzy screen match contributed to stuck count",
                        data={
                            "base_activity": current.activity,
                            "base_visual_hash": current.visual_hash,
                            "base_structural_hash": current.structural_hash,
                            "candidate_activity": candidate.activity,
                            "candidate_visual_hash": candidate.visual_hash,
                            "candidate_structural_hash": candidate.structural_hash,
                            "visual_hamming_distance": distance,
                        },
                    )
                    # endregion

            if count >= self.threshold:
                unique_recent_actions = len(set(self.__recent_actions))
                if (
                    len(self.__recent_actions) >= self.threshold
                    and unique_recent_actions >= self.threshold
                ):
                    # region agent log
                    _debug_log(
                        hypothesis_id="H8",
                        location="src/fathom/schemas/state.py:is_stuck",
                        message="Bypassing stuck=true due to diverse recent actions",
                        data={
                            "recent_actions_count": len(self.__recent_actions),
                            "unique_recent_actions": unique_recent_actions,
                            "recent_action_samples": list(self.__recent_actions),
                            "repeat_count": count,
                        },
                    )
                    # endregion
                    continue
                hashes = [s.visual_hash[:8] for s in self.__recent_screens]
                logger.debug(f"LoopDetector.is_stuck: True (screen {current.visual_hash[:8]} repeated {count}x) | deque={hashes}")
                # region agent log
                _debug_log(
                    hypothesis_id="H1",
                    location="src/fathom/schemas/state.py:is_stuck",
                    message="Stuck=true due to repeated screen",
                    data={
                        "matched_activity": current.activity,
                        "matched_activity_hash": current.activity_hash,
                        "matched_visual_hash_prefix": current.visual_hash[:8],
                        "matched_structural_hash": current.structural_hash,
                        "repeat_count": count,
                        "recent_visual_hashes": hashes,
                        "recent_action_samples": list(self.__recent_actions),
                        "recent_unique_actions": len(set(self.__recent_actions)),
                    },
                )
                # endregion
                return True

        # Check for repeated actions (exact match is fine for actions)
        if len(self.__recent_actions) >= self.threshold:
            action_counts: Dict[str, int] = {}

            for action_description in self.__recent_actions:
                action_counts[action_description] = action_counts.get(action_description, 0) + 1
                if action_counts[action_description] >= self.threshold:
                    logger.debug(f"LoopDetector.is_stuck: True (action '{action_description}' repeated {action_counts[action_description]}x)")
                    return True

        logger.debug(f"LoopDetector.is_stuck: False (no repeats above threshold)")
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


class ActionHistory(BaseModel):
    """
    Tracks action history for context building with token optimization.
    """

    max_size: int = Field(default=10, description="Maximum history size")

    __failure_count: int = PrivateAttr(default=0)
    __actions: Deque[Dict[str, Any]] = PrivateAttr(default_factory=lambda: deque(maxlen=10))

    model_config = ConfigDict(frozen=True)

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, "_ActionHistory__actions", deque(maxlen=self.max_size))

    def record_action(self, action: Action, success: bool, activity: str) -> None:
        """
        Record an action with its outcome and associated activity.
        """

        self.__actions.append(
            {
                "success": success,
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
            parts.append(f"{action['type']}:{action['target']}:{result_indicator}")

        return " | ".join(parts) if parts else "None"

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

    def has_repeated_failure(self, action: Action) -> bool:
        """
        Check if this exact action has failed recently.
        """

        description = action.to_description()

        return any(
            not historical["success"] and historical["full_description"] == description
            for historical in self.__actions
        )
