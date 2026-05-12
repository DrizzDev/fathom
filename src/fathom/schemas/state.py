from __future__ import annotations

import time
from collections import deque
from logging import getLogger
from typing import Any, Deque, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from fathom.constants.screen import (
    DEFAULT_SAME_SCREEN_THRESHOLD,
    LOOP_ACTION_VELOCITY_INTERVAL_THRESHOLD_SECONDS,
    LOOP_DETECTOR_WINDOW_SIZE,
    LOOP_MAX_AUTONOMOUS_RECOVERIES,
    LOOP_NEAR_DUPLICATE_HAMMING_THRESHOLD,
    LOOP_OSCILLATION_AB_WINDOW,
    LOOP_OSCILLATION_ABC_WINDOW,
    LOOP_REPETITION_THRESHOLD,
    LOOP_SCROLL_STALL_DISTANCE_THRESHOLD,
    LOOP_SCROLL_STALL_MIN_STREAK,
)
from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenState

logger = getLogger(__name__)


class LoopDetectorState(BaseModel):
    """
    Serializable snapshot of :class:`LoopDetector` internal deque's used
    to round-trip loop-detection evidence through graph checkpoints so accumulating signals survive iteration boundaries.
    """

    types: List[str] = Field(default_factory=list, description="Action type tokens window")
    actions: List[str] = Field(default_factory=list, description="Action descriptors window")
    hashes: List[str] = Field(default_factory=list, description="Screen visual hashes window")
    screens: List[ScreenState] = Field(
        default_factory=list, description="ScreenState entries window"
    )

    timestamps: List[float] = Field(default_factory=list, description="Record timestamps window")
    recovery_attempts: int = Field(default=0, description="Autonomous recovery attempts taken")

    model_config = ConfigDict(frozen=True)


class LoopDetector(BaseModel):
    """
    Detects when agent is stuck in a loop using multi-strategy pattern analysis.

    Note: stateful. Designed for single-threaded asyncio access — coordinator
    instances are scoped per agent run, so cross-task mutation does not occur under the current execution model.
    """

    max_recovery: int = Field(
        default=LOOP_MAX_AUTONOMOUS_RECOVERIES,
        description="Maximum autonomous recovery attempts permitted",
    )
    window_size: int = Field(
        default=LOOP_DETECTOR_WINDOW_SIZE, description="Size of the pattern analysis window"
    )
    threshold: int = Field(
        default=LOOP_REPETITION_THRESHOLD, description="Window occurrences before classifying stuck"
    )

    __recovery_attempts: int = PrivateAttr(default=0)
    __recent_types: Deque[str] = PrivateAttr(default_factory=deque)
    __recent_hashes: Deque[str] = PrivateAttr(default_factory=deque)
    __recent_actions: Deque[str] = PrivateAttr(default_factory=deque)
    __recent_timestamps: Deque[float] = PrivateAttr(default_factory=deque)
    __recent_screens: Deque[ScreenState] = PrivateAttr(default_factory=deque)

    def model_post_init(self, _context: Any) -> None:
        """
        Size all internal deque's from ``window_size`` so the maxlen tracks
        the configured field rather than the literal default.
        """

        self.__recent_types = deque(maxlen=self.window_size)
        self.__recent_hashes = deque(maxlen=self.window_size)
        self.__recent_actions = deque(maxlen=self.window_size)
        self.__recent_screens = deque(maxlen=self.window_size)
        self.__recent_timestamps = deque(maxlen=self.window_size)

    def to_state(self) -> LoopDetectorState:
        """
        Capture an immutable snapshot of the deque's for serialization.
        """

        return LoopDetectorState(
            types=list(self.__recent_types),
            hashes=list(self.__recent_hashes),
            screens=list(self.__recent_screens),
            actions=list(self.__recent_actions),
            timestamps=list(self.__recent_timestamps),
            recovery_attempts=self.__recovery_attempts,
        )

    def restore(self, *, state: LoopDetectorState) -> None:
        """
        Rehydrate the deque's from a snapshot. Replaces current contents.
        """

        self.__recovery_attempts = max(0, state.recovery_attempts)
        self.__recent_types = deque(state.types, maxlen=self.window_size)
        self.__recent_hashes = deque(state.hashes, maxlen=self.window_size)
        self.__recent_screens = deque(state.screens, maxlen=self.window_size)
        self.__recent_actions = deque(state.actions, maxlen=self.window_size)
        self.__recent_timestamps = deque(state.timestamps, maxlen=self.window_size)

    def record(
        self,
        screen: ScreenState,
        action_type: Optional[str] = None,
        action_description: Optional[str] = None,
    ) -> None:
        """
        Record state and action data for pattern analysis.
        """

        self.__recent_screens.append(screen)
        self.__recent_timestamps.append(time.time())
        self.__recent_hashes.append(screen.visual_hash)

        # Store descriptions for semantic matching
        identifier = action_description or action_type or "None"
        self.__recent_actions.append(identifier)

        # Store raw types for velocity and scroll analysis
        self.__recent_types.append(str(action_type or "unknown").lower())

        logger.debug(
            f"LoopDetector.record: {screen.visual_hash[:8]} | "
            f"action={identifier} | type={action_type}"
        )

    def observe_screen(self, *, previous: Optional[ScreenState], current: ScreenState) -> None:
        """
        Tell the detector a new screen was seen. Advances the loop-detection
        window only when the new screen is visually distinct from the
        previous one (hamming > near-duplicate threshold); otherwise keeps
        accumulating evidence so the near-duplicate detector can fire.
        """

        if current.has_visual_progress_from(
            previous=previous, threshold=LOOP_NEAR_DUPLICATE_HAMMING_THRESHOLD
        ):
            self.advance()

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

            # 2. Near-duplicate Visual Repetition (visual pHash only).
            # Catches the case where DOM micro-changes (overlay animation frames,
            # map redraws, transient spinners) flip ``xml_hash``/``interaction_hash``
            # and force ``is_same_screen`` to return False even though the screen
            # is visually identical. The standard repetition detector is bypassed
            # in that case; this complementary detector closes the gap.
            if self.__detect_near_duplicate_visual_repetition():
                return True

            # 3. State Oscillation (A-B-A-B or A-B-C-A)
            if self.__detect_oscillation():
                return True

            # 4. Scroll Stalling (Repetitive scrolling with minimal progress)
            if self.__detect_scroll_stall():
                return True

            # 5. Action Velocity (Rapid firing with no progress)
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

    def __detect_near_duplicate_visual_repetition(self) -> bool:
        """
        Detect screens whose visual pHash is within a tight hamming threshold
        of one another, ignoring structural and interaction hashes.

        Distinct from ``__detect_repetition`` (which uses ``is_same_screen``):
        that path returns False as soon as ``xml_hash`` or ``interaction_hash`` disagree, which masks overlay-animation and map-redraw loops.

        The visual-only check is intentionally narrow (threshold = pHash hamming)
        and only counts repetition when the same near-duplicate appears at least ``self.threshold`` times in the window.
        """

        hashes = [hash for hash in self.__recent_hashes if hash]

        if len(hashes) < self.threshold:
            return False

        for index, anchor in enumerate(hashes):
            count = 1
            for forward_index in range(index + 1, len(hashes)):
                distance = ScreenState.hamming_distance(
                    left_hash=anchor,
                    right_hash=hashes[forward_index],
                )
                if distance <= LOOP_NEAR_DUPLICATE_HAMMING_THRESHOLD:
                    count += 1

            if count >= self.threshold:
                logger.warning(
                    f"LoopDetector: Stuck via near-duplicate visual repetition "
                    f"({count}x within hamming {LOOP_NEAR_DUPLICATE_HAMMING_THRESHOLD})"
                )
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
        screen_hash: str,
        action_description: str,
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

        return self.__recovery_attempts < self.max_recovery

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
