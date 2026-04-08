from __future__ import annotations

import time
from collections import deque
from logging import getLogger
from typing import Any, Deque, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from fathom.constants.screen import (
    DEFAULT_SAME_SCREEN_THRESHOLD,
    LOOP_ACTION_VELOCITY_INTERVAL_THRESHOLD_SECONDS,
    LOOP_OSCILLATION_AB_WINDOW,
    LOOP_OSCILLATION_ABC_WINDOW,
    LOOP_SCROLL_STALL_DISTANCE_THRESHOLD,
    LOOP_SCROLL_STALL_MIN_STREAK,
)
from fathom.schemas.actions import Action, resolve_action_target
from fathom.schemas.screens import ScreenState

logger = getLogger(__name__)


class LoopDetector(BaseModel):
    """
    Detects when agent is stuck in a loop using multi-strategy pattern analysis.
    """

    threshold: int = Field(default=3, description="Standard repetition threshold")
    window_size: int = Field(default=15, description="Size of the pattern analysis window")

    __max_recovery: int = PrivateAttr(default=3)
    __recovery_attempts: int = PrivateAttr(default=0)
    __recent_actions: Deque[str] = PrivateAttr(default_factory=lambda: deque(maxlen=15))
    __recent_types: Deque[str] = PrivateAttr(default_factory=lambda: deque(maxlen=15))
    __recent_hashes: Deque[str] = PrivateAttr(default_factory=lambda: deque(maxlen=15))
    __recent_screens: Deque[ScreenState] = PrivateAttr(default_factory=lambda: deque(maxlen=15))
    __recent_timestamps: Deque[float] = PrivateAttr(default_factory=lambda: deque(maxlen=15))

    def record(
        self,
        screen: ScreenState,
        action_description: Optional[str] = None,
        action_type: Optional[str] = None,
    ) -> None:
        """
        Record state and action data for pattern analysis.
        """

        self.__recent_screens.append(screen)
        self.__recent_hashes.append(screen.visual_hash)
        self.__recent_timestamps.append(time.time())

        # Store descriptions for semantic matching
        identifier = action_description or action_type or "None"
        self.__recent_actions.append(identifier)

        # Store raw types for velocity and scroll analysis
        self.__recent_types.append(str(action_type or "unknown").lower())

        logger.debug(
            f"LoopDetector.record: {screen.visual_hash[:8]} | "
            f"action={identifier} | type={action_type}"
        )

    def is_stuck(self) -> bool:
        """
        Evaluate if current interaction sequence indicates a loop.
        """

        has_enough_screens = len(self.__recent_screens) >= self.threshold
        has_enough_actions = len(self.__recent_actions) >= self.threshold

        if not has_enough_screens and not has_enough_actions:
            return False

        # Screen-based detectors require sufficient screen history.
        if has_enough_screens:
            # 1. Direct Repetition (screen + action counts)
            if self.__detect_repetition():
                return True

            # 2. State Oscillation (A-B-A-B or A-B-C-A)
            if self.__detect_oscillation():
                return True

            # 3. Scroll Stalling (Repetitive scrolling with minimal progress)
            if self.__detect_scroll_stall():
                return True

            # 4. Action Velocity (Rapid firing with no progress)
            if self.__detect_action_velocity_loop():
                return True

        # Action-based detection survives screen resets (advance).
        # Catches repeated actions across visually-different screens.
        return has_enough_actions and self.__detect_action_repetition()

    def __detect_repetition(self) -> bool:
        """
        Detect simple screen repetition.
        """

        def _is_scroll_navigation_sequence(window: int = 4) -> bool:
            recent_types = list(self.__recent_types)[-window:]
            if len(recent_types) < window:
                return False
            return all(
                any(token in t for token in ("scroll", "swipe", "flick")) for t in recent_types
            )

        for index in range(len(self.__recent_screens)):
            count = 1
            current = self.__recent_screens[index]
            for forward_index in range(index + 1, len(self.__recent_screens)):
                if current.is_same_screen(self.__recent_screens[forward_index]):
                    count += 1

            if count >= self.threshold:
                if _is_scroll_navigation_sequence():
                    continue
                if len(set(self.__recent_actions)) >= self.threshold:
                    continue

                logger.warning(f"LoopDetector: Stuck via screen repetition ({count}x)")
                return True

        return False

    def __detect_action_repetition(self) -> bool:
        """
        Detect repeated identical actions regardless of screen state.

        Survives screen resets (advance) so it can catch loops where each
        action produces a visually-different screen (e.g. incrementing a counter).
        """

        action_counts: Dict[str, int] = {}
        for action in self.__recent_actions:
            if action == "None":
                continue
            action_counts[action] = action_counts.get(action, 0) + 1
            if action_counts[action] >= self.threshold + 1:
                action_lower = action.lower()
                if any(token in action_lower for token in ("swipe", "scroll", "flick")):
                    continue
                logger.warning(
                    f"LoopDetector: Stuck via action repetition '{action}' ({action_counts[action]}x)"
                )
                return True

        return False

    def has_repeated_action_on_same_screen(
        self,
        action_description: str,
        screen_hash: str,
        *,
        repeat_threshold: int = 3,
    ) -> bool:
        """
        Check if the same tap/type action has been executed N+ times on the same screen.
        Excludes swipe/scroll actions which legitimately repeat on the same screen.

        Args:
            action_description: The action being proposed.
            screen_hash: Visual hash of the current screen.
            repeat_threshold: Number of repeats before triggering (default 3).

        Returns:
            True if the action has been repeated on this screen at or above threshold.
        """

        action_lower = action_description.lower()

        # Only track tap/type actions — swipes and scrolls legitimately repeat
        if any(token in action_lower for token in ("swipe", "scroll", "flick")):
            return False

        count = 0
        for i in range(len(self.__recent_actions)):
            if i >= len(self.__recent_hashes):
                break
            if (
                self.__recent_actions[i] == action_description
                and self.__recent_hashes[i] == screen_hash
            ):
                count += 1

        if count >= repeat_threshold:
            logger.warning(
                "LoopDetector: Same action '%s' repeated %dx on screen %s",
                action_description,
                count,
                screen_hash[:8],
            )
            return True

        return False

    def __detect_oscillation(self) -> bool:
        """
        Detect bouncing between 2 or 3 screens.
        """

        if len(self.__recent_screens) < LOOP_OSCILLATION_AB_WINDOW:
            return False

        screens = list(self.__recent_screens)

        # Pattern: A-B-A-B
        if (
            len(screens) >= LOOP_OSCILLATION_AB_WINDOW
            and self.__is_visually_same(left=screens[-1], right=screens[-3])
            and self.__is_visually_same(left=screens[-2], right=screens[-4])
        ):
            logger.warning("LoopDetector: Oscillation detected (A-B-A-B)")
            return True

        # Pattern: A-B-C-A-B-C
        if (
            len(screens) >= LOOP_OSCILLATION_ABC_WINDOW
            and self.__is_visually_same(left=screens[-1], right=screens[-4])
            and self.__is_visually_same(left=screens[-2], right=screens[-5])
            and self.__is_visually_same(left=screens[-3], right=screens[-6])
        ):
            logger.warning("LoopDetector: Oscillation detected (A-B-C-A-B-C)")
            return True

        return False

    def __is_visually_same(self, *, left: ScreenState, right: ScreenState) -> bool:
        """
        Return whether two screens are visually equivalent within tolerance.
        """

        return (
            ScreenState.hamming_distance(
                left_hash=left.visual_hash,
                right_hash=right.visual_hash,
            )
            <= DEFAULT_SAME_SCREEN_THRESHOLD
        )

    def __detect_scroll_stall(self) -> bool:
        """
        Detect repetitive scrolling that yields minimal visual progress.
        """

        recent_types = list(self.__recent_types)

        # Require a longer uninterrupted scroll streak before considering stall.
        trailing_scroll_streak = 0

        for action_type in reversed(recent_types):
            if any(token in action_type for token in ("scroll", "swipe", "flick")):
                trailing_scroll_streak += 1
            else:
                break
        if trailing_scroll_streak < LOOP_SCROLL_STALL_MIN_STREAK:
            return False

        # Evaluate over the trailing streak to avoid first/last hash aliasing.
        streak_start = len(recent_types) - trailing_scroll_streak

        first_hash = self.__recent_hashes[streak_start]
        last_hash = self.__recent_hashes[-1]

        streak_hashes = list(self.__recent_hashes)[streak_start:]
        unique_hash_count = len(set(streak_hashes))

        distance = ScreenState.hamming_distance(left_hash=first_hash, right_hash=last_hash)
        # Stall must show both low net movement and low diversity across streak.
        if distance < LOOP_SCROLL_STALL_DISTANCE_THRESHOLD and unique_hash_count <= 2:
            logger.warning(
                "LoopDetector: Scroll stall detected "
                f"(dist={distance}, streak={trailing_scroll_streak}, unique={unique_hash_count})"
            )
            return True

        return False

    def __detect_action_velocity_loop(self) -> bool:
        """
        Detect rapid-fire actions that fail to change the screen state.
        """

        if len(self.__recent_timestamps) < 3:
            return False

        times = list(self.__recent_timestamps)
        # Average interval between last 3 actions
        intervals = [times[i] - times[i - 1] for i in range(len(times) - 1, len(times) - 3, -1)]
        avg_interval = sum(intervals) / len(intervals)

        # If firing faster than 1.5s per action
        if avg_interval < LOOP_ACTION_VELOCITY_INTERVAL_THRESHOLD_SECONDS:
            # Check if state is actually changing
            recent_hashes = list(self.__recent_hashes)[-3:]
            if len(set(recent_hashes)) == 1:
                logger.warning(f"LoopDetector: Velocity loop detected (avg={avg_interval:.2f}s)")
                return True

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

    def advance(self) -> None:
        """
        Signal forward progress while preserving action history.

        Clears screen-based tracking (hashes, screens) so screen repetition
        and oscillation detectors reset on genuine visual progress. Preserves
        action descriptions, types, and timestamps so action-repeat and
        velocity detectors can catch loops across visually-different screens
        (e.g. tapping a counter button that changes the displayed number).
        """

        prev_size = len(self.__recent_screens)

        self.__recent_screens.clear()
        self.__recent_hashes.clear()

        self.__recovery_attempts = 0
        logger.info(
            f"LoopDetector.advance: cleared {prev_size} screens, preserved {len(self.__recent_actions)} actions"
        )

    def reset(self) -> None:
        """
        Full reset of all loop detection state.
        """

        prev_size = len(self.__recent_screens)

        self.__recent_screens.clear()
        self.__recent_actions.clear()

        self.__recent_types.clear()
        self.__recent_hashes.clear()
        self.__recent_timestamps.clear()

        self.__recovery_attempts = 0
        logger.info(f"LoopDetector.reset: cleared {prev_size} screens")

    def signal_content_exhausted(self) -> None:
        """
        Clear loop history after explicit end-of-content signal.
        """

        self.reset()


class InteractionTracker(BaseModel):
    """
    Elegantly tracks the cadence and repetition of agent interactions.

    Provides deterministic data on consecutive action sequences to enforce
    behavioral constraints (e.g., 'Don't swipe more than 3 times').
    """

    __consecutive_count: int = PrivateAttr(default=0)
    __last_action_type: Optional[str] = PrivateAttr(default=None)
    __total_counters: Dict[str, int] = PrivateAttr(default_factory=dict)

    def record(self, action_type: str) -> None:
        """
        Records an interaction and updates cadence metrics.
        """
        # Update Total
        self.__total_counters[action_type] = self.__total_counters.get(action_type, 0) + 1

        # Update Consecutive
        if action_type == self.__last_action_type:
            self.__consecutive_count += 1
        else:
            self.__last_action_type = action_type
            self.__consecutive_count = 1

    def get_cadence_note(self) -> Optional[str]:
        """
        Returns a semantic note about current interaction repetition if significant.
        Example: "Consecutive swipe_up: 3"
        """

        if self.__consecutive_count > 1 and self.__last_action_type:
            return f"Consecutive {self.__last_action_type}: {self.__consecutive_count}"

        return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns raw tracking data.
        """

        return {
            "last_type": self.__last_action_type,
            "totals": dict(self.__total_counters),
            "consecutive": self.__consecutive_count,
        }


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

        # Route the history entry through the canonical target
        # resolver so validate/wait/swipe steps surface their
        # canonical subject (validation_subject / wait_subject /
        # scroll_target) instead of leaking a placeholder like
        # "element" back into the agent's own context on the next
        # turn. resolve_action_target returns "unknown" as a
        # last-resort fallback; we then coerce that to "UI" to
        # preserve the historic display string.
        resolved_target = resolve_action_target(
            action_type=action.action_type,
            target_name=action.target,
            export_target=action.export_target,
            natural_language_target=action.natural_language_target,
            validation_subject=action.validation_subject,
            wait_subject=action.wait_subject,
            scroll_target=action.scroll_target,
            label_id=action.label_id,
        )
        if resolved_target == "unknown":
            resolved_target = "UI"

        self.__actions.append(
            {
                "success": success,
                "activity": activity,
                "type": action.action_type.value.upper(),
                "full_description": action.to_description(),
                "target": resolved_target,
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
