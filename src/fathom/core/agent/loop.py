from __future__ import annotations

import time
from collections import deque
from logging import getLogger
from typing import Any, Deque, Dict, Optional

from pydantic import BaseModel, Field, PrivateAttr

from fathom.constants.runtime import DEFAULT_INERT_REPETITION_THRESHOLD
from fathom.constants.screen import (
    DEFAULT_SAME_SCREEN_THRESHOLD,
    LOOP_ACTION_VELOCITY_INTERVAL_THRESHOLD_SECONDS,
    LOOP_DETECTOR_WINDOW_SIZE,
    LOOP_HASH_CLUSTER_HAMMING_THRESHOLD,
    LOOP_MAX_AUTONOMOUS_RECOVERIES,
    LOOP_NEAR_DUPLICATE_HAMMING_THRESHOLD,
    LOOP_OSCILLATION_AB_WINDOW,
    LOOP_OSCILLATION_ABC_WINDOW,
    LOOP_REPETITION_THRESHOLD,
    LOOP_SCROLL_STALL_DISTANCE_THRESHOLD,
    LOOP_SCROLL_STALL_MIN_STREAK,
    SCREEN_PROGRESS_HAMMING_THRESHOLD,
)
from fathom.schemas.effect import ActionEffectStatus
from fathom.schemas.loop import LoopEvidence, LoopReason, LoopTurn
from fathom.schemas.screens import ScreenState
from fathom.schemas.state import LoopDetectorState
from fathom.schemas.vision import ActionKindResolver

logger = getLogger(__name__)


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
    inert_repetition_threshold: int = Field(
        ge=2,
        default=DEFAULT_INERT_REPETITION_THRESHOLD,
        description=(
            "Identical-action + NO_PROGRESS effect streak length that trips the inert-repetition detector. "
            "Intentionally smaller than ``threshold`` so the planner can pivot after one wasted action instead of three; "
            "safe because the NO_PROGRESS classifier already requires every available no-progress metric to agree."
        ),
    )

    __recovery_attempts: int = PrivateAttr(default=0)
    __recent_types: Deque[str] = PrivateAttr(default_factory=deque)
    __recent_hashes: Deque[str] = PrivateAttr(default_factory=deque)
    __recent_actions: Deque[str] = PrivateAttr(default_factory=deque)
    __recent_timestamps: Deque[float] = PrivateAttr(default_factory=deque)
    __recent_screens: Deque[ScreenState] = PrivateAttr(default_factory=deque)
    __recent_effect_statuses: Deque[str] = PrivateAttr(default_factory=deque)

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
        self.__recent_effect_statuses = deque(maxlen=self.window_size)

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
            effect_statuses=list(self.__recent_effect_statuses),
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
        self.__recent_effect_statuses = deque(state.effect_statuses, maxlen=self.window_size)

    def record(
        self,
        screen: ScreenState,
        action_type: Optional[str] = None,
        action_description: Optional[str] = None,
        effect_status: Optional[ActionEffectStatus] = None,
    ) -> None:
        """
        Record state, action, and effect data for pattern analysis.
        """

        self.__recent_screens.append(screen)
        self.__recent_timestamps.append(time.time())
        self.__recent_hashes.append(screen.visual_hash)

        # Store descriptions for semantic matching
        identifier = action_description or action_type or "None"
        self.__recent_actions.append(identifier)

        # Store raw types for velocity and scroll analysis
        self.__recent_types.append(str(action_type or "unknown").lower())

        # Effect status decorates the action; empty string means "not
        # recorded for this slot" (e.g. first turn before any effect).
        self.__recent_effect_statuses.append(
            effect_status.value if effect_status is not None else "",
        )

        logger.info(
            "LoopDetector recorded turn",
            extra={
                "component": "schemas.state.loop_detector",
                "event": "loop_detector.record",
                "action.type": action_type,
                "action.identifier": identifier,
                "screen.visual_hash": screen.visual_hash[:8],
                "action.effect": effect_status.value if effect_status is not None else None,
            },
        )

        if action_type is None and action_description is None:
            logger.info(
                "LoopDetector recorded turn with empty descriptor — "
                "action_repetition will skip this slot",
                extra={
                    "component": "schemas.state.loop_detector",
                    "event": "loop_detector.record.empty_descriptor",
                    "screen.visual_hash": screen.visual_hash[:8],
                },
            )

    def observe_screen(self, *, previous: Optional[ScreenState], current: ScreenState) -> None:
        """
        Tell the detector a new screen was seen.

        Advances the loop-detection window only when the new screen is
        *genuinely* distinct from the previous one — hamming greater than
        :data:`SCREEN_PROGRESS_HAMMING_THRESHOLD`. The progress threshold
        is deliberately much higher than the near-duplicate threshold
        used by the stuck detectors below, so cosmetic differences
        (status-bar tick, suggestion-count increment, anti-aliasing
        noise) do not trip ``advance()`` and wipe accumulating evidence.
        That distinction is what enables the scroll-loop detection to
        actually fire on long sequences of near-identical screens.
        """

        progressed = current.has_visual_progress_from(
            previous=previous, threshold=SCREEN_PROGRESS_HAMMING_THRESHOLD
        )
        logger.info(
            "LoopDetector.observe_screen evaluated",
            extra={
                "progressed": progressed,
                "component": "loop.detector",
                "event": "observe_screen.evaluated",
                "threshold": SCREEN_PROGRESS_HAMMING_THRESHOLD,
                "previous.visual_hash": (
                    previous.visual_hash[:16] if previous is not None else None
                ),
                "current.visual_hash": current.visual_hash[:16],
                "recovery_attempts.before": self.__recovery_attempts,
                "recent_screens.count_before": len(self.__recent_screens),
            },
        )
        if progressed:
            self.advance()

    def is_stuck(self) -> bool:
        """
        Evaluate if current interaction sequence indicates a loop.
        """

        snapshot = {
            "component": "loop.detector",
            "event": "is_stuck.evaluate",
            "threshold": self.threshold,
            "window_size": self.window_size,
            "recovery_attempts": self.__recovery_attempts,
            "recent_hashes.count": len(self.__recent_hashes),
            "recent_screens.count": len(self.__recent_screens),
            "recent_actions.count": len(self.__recent_actions),
            "recent_actions.unique": len(set(self.__recent_actions)),
        }
        logger.info("LoopDetector.is_stuck evaluating", extra=snapshot)

        # Inert-action repetition fires at the tightest threshold so the planner can pivot after one wasted
        # action — independent of how much screen / action history has accumulated.
        if self.__detect_inert_repetition():
            logger.info(
                "LoopDetector.is_stuck=True via inert_repetition",
                extra={**snapshot, "event": "is_stuck.fired", "detector": "inert_repetition"},
            )
            return True

        has_enough_screens = len(self.__recent_screens) >= self.threshold
        has_enough_actions = len(self.__recent_actions) >= self.threshold

        if not has_enough_screens and not has_enough_actions:
            logger.info(
                "LoopDetector.is_stuck=False insufficient_history",
                extra={
                    **snapshot,
                    "event": "is_stuck.skipped",
                    "reason": "insufficient_history",
                    "has_enough_screens": has_enough_screens,
                    "has_enough_actions": has_enough_actions,
                },
            )
            return False

        if has_enough_screens:
            if self.__detect_repetition():
                logger.info(
                    "LoopDetector.is_stuck=True via screen_repetition",
                    extra={**snapshot, "event": "is_stuck.fired", "detector": "screen_repetition"},
                )
                return True

            # Near-duplicate visual repetition (visual pHash only).
            # Catches the case where DOM micro-changes (overlay animation frames,
            # map redraws, transient spinners) flip ``xml_hash``/``interaction_hash``
            # and force ``is_same_screen`` to return False even though the screen
            # is visually identical. The standard repetition detector is bypassed
            # in that case; this complementary detector closes the gap.
            if self.__detect_near_duplicate_visual_repetition():
                logger.info(
                    "LoopDetector.is_stuck=True via near_duplicate_visual",
                    extra={
                        **snapshot,
                        "event": "is_stuck.fired",
                        "detector": "near_duplicate_visual",
                    },
                )
                return True

            if self.__detect_oscillation():
                logger.info(
                    "LoopDetector.is_stuck=True via oscillation",
                    extra={**snapshot, "event": "is_stuck.fired", "detector": "oscillation"},
                )
                return True

            if self.__detect_scroll_stall():
                logger.info(
                    "LoopDetector.is_stuck=True via scroll_stall",
                    extra={**snapshot, "event": "is_stuck.fired", "detector": "scroll_stall"},
                )
                return True

            if self.__detect_action_velocity_loop():
                logger.info(
                    "LoopDetector.is_stuck=True via action_velocity",
                    extra={**snapshot, "event": "is_stuck.fired", "detector": "action_velocity"},
                )
                return True

        # Action-based detection survives screen resets (advance).
        # Catches repeated actions across visually-different screens.
        action_repetition = has_enough_actions and self.__detect_action_repetition()
        if action_repetition:
            logger.info(
                "LoopDetector.is_stuck=True via action_repetition",
                extra={**snapshot, "event": "is_stuck.fired", "detector": "action_repetition"},
            )
            return True

        logger.info(
            "LoopDetector.is_stuck=False no_detector_fired",
            extra={
                **snapshot,
                "event": "is_stuck.not_stuck",
                "evaluated_detectors": [
                    "inert_repetition",
                    *(
                        [
                            "screen_repetition",
                            "near_duplicate_visual",
                            "oscillation",
                            "scroll_stall",
                            "action_velocity",
                        ]
                        if has_enough_screens
                        else []
                    ),
                    *(["action_repetition"] if has_enough_actions else []),
                ],
            },
        )
        return False

    def __detect_inert_repetition(self) -> bool:
        """
        Detect identical action descriptors paired with trailing NO_PROGRESS effects.

        Fires when the last ``inert_repetition_threshold`` action
        descriptors are identical AND the matching trailing effect
        statuses are all ``NO_PROGRESS``. Both conditions must hold
        so cosmetic same-action retries on a screen that *did* change
        don't false-fire (the planner explores during real scrolling and that's not stuck).
        """

        if len(self.__recent_actions) < self.inert_repetition_threshold:
            return False

        if len(self.__recent_effect_statuses) < self.inert_repetition_threshold:
            return False

        trailing_actions = list(self.__recent_actions)[-self.inert_repetition_threshold :]
        trailing_statuses = list(self.__recent_effect_statuses)[-self.inert_repetition_threshold :]

        if len(set(trailing_actions)) > 1:
            return False

        if any(status != ActionEffectStatus.NO_PROGRESS.value for status in trailing_statuses):
            return False

        logger.warning(
            "LoopDetector: stuck via inert action repetition '%s' (%dx)",
            trailing_actions[-1],
            self.inert_repetition_threshold,
            extra={
                "component": "loop.detector",
                "action": trailing_actions[-1],
                "event": "stuck.inert.repetition",
                "count": self.inert_repetition_threshold,
            },
        )
        return True

    def __detect_repetition(self) -> bool:
        """
        Detect simple screen repetition.

        High action diversity is treated as legitimate exploration; identical screens with similar actions
        and converging visual hashes are flagged as stuck regardless of action kind.
        """

        for index in range(len(self.__recent_screens)):
            count = 1
            current = self.__recent_screens[index]
            for forward_index in range(index + 1, len(self.__recent_screens)):
                if current.is_same_screen(self.__recent_screens[forward_index]):
                    count += 1

            if count >= self.threshold:
                unique_actions = len(set(self.__recent_actions))
                if unique_actions >= self.threshold:
                    logger.info(
                        "LoopDetector.detect_repetition: screen-repeat detected but action diversity high; not flagging stuck",
                        extra={
                            "component": "loop.detector",
                            "screen.repeat.count": count,
                            "actions.unique": unique_actions,
                            "actions.threshold": self.threshold,
                            "event": "detect_repetition.skipped",
                            "reason": "action_diversity_above_threshold",
                            "actions.recent": list(self.__recent_actions),
                        },
                    )
                    continue

                logger.warning(
                    "LoopDetector: stuck via screen repetition (%dx)",
                    count,
                    extra={
                        "count": count,
                        "component": "loop.detector",
                        "actions.unique": unique_actions,
                        "event": "stuck.screen.repetition",
                    },
                )
                return True

        logger.info(
            "LoopDetector.detect_repetition=False no_screen_reached_threshold",
            extra={
                "threshold": self.threshold,
                "component": "loop.detector",
                "event": "detect_repetition.no_match",
                "recent_screens.count": len(self.__recent_screens),
            },
        )
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

        Survives screen resets (advance) so it catches loops where each action produces a visually-different
        screen (e.g. tapping a counter button that increments a number). Scroll-like actions suppress detection
        only while the screens they produce keep diverging (productive scroll); once those screens converge into
        a near-duplicate cluster (stuck scroll) they trip stuck like any other repeated action.
        """

        action_counts: Dict[str, int] = {}

        for action in self.__recent_actions:
            if action == "None":
                continue

            action_counts[action] = action_counts.get(action, 0) + 1
            if action_counts[action] >= self.threshold + 1:
                action_lower = action.lower()
                is_scroll_like = any(
                    token in action_lower for token in ("swipe", "scroll", "flick")
                )
                if is_scroll_like and not self.__recent_hashes_are_converging():
                    continue

                logger.warning(
                    "LoopDetector: stuck via action repetition '%s' (%dx)",
                    action,
                    action_counts[action],
                    extra={
                        "component": "loop_detector",
                        "event": "stuck_action_repetition",
                        "action": action,
                        "count": action_counts[action],
                    },
                )
                return True

        return False

    def __recent_hashes_are_converging(self) -> bool:
        """
        Return True when the most recent ``self.threshold`` visual
        hashes all lie within :data:`LOOP_HASH_CLUSTER_HAMMING_THRESHOLD`
        of one another.

        Used by the scroll-repetition guard to distinguish productive
        scrolling (screens diverging through fresh content) from stuck
        scrolling (screens converging into a near-duplicate cluster).
        """

        recent_hashes = [hash_ for hash_ in self.__recent_hashes if hash_]
        tail = recent_hashes[-self.threshold :]
        if len(tail) < self.threshold:
            return False

        anchor = tail[0]
        for hash_ in tail[1:]:
            distance = ScreenState.hamming_distance(left_hash=anchor, right_hash=hash_)
            if distance > LOOP_HASH_CLUSTER_HAMMING_THRESHOLD:
                return False

        return True

    def has_repeated_action_on_same_screen(
        self,
        screen_hash: str,
        action_description: str,
        *,
        repeat_threshold: int = 3,
    ) -> bool:
        """
        Whether the same tap/type action has run at or above ``repeat_threshold`` times on this screen.

        Swipe/scroll actions are excluded — they legitimately repeat on the same screen.
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

        streak_hashes = [hash_ for hash_ in list(self.__recent_hashes)[streak_start:] if hash_]
        distance = ScreenState.hamming_distance(left_hash=first_hash, right_hash=last_hash)

        # Stall must show both low net movement AND all streak hashes clustered tightly around the anchor.
        # The cluster-hamming check is jitter-tolerant: pHash jitter routinely produces 3+ unique short hashes
        # even when the screen is visually identical.
        all_clustered = True
        if streak_hashes:
            anchor = streak_hashes[0]
            for hash_ in streak_hashes[1:]:
                if (
                    ScreenState.hamming_distance(left_hash=anchor, right_hash=hash_)
                    > LOOP_HASH_CLUSTER_HAMMING_THRESHOLD
                ):
                    all_clustered = False
                    break

        if distance < LOOP_SCROLL_STALL_DISTANCE_THRESHOLD and all_clustered:
            logger.warning(
                "LoopDetector: scroll stall detected (dist=%d, streak=%d, clustered=%s)",
                distance,
                trailing_scroll_streak,
                all_clustered,
                extra={
                    "component": "loop_detector",
                    "event": "stuck_scroll_stall",
                    "net_distance": distance,
                    "streak_length": trailing_scroll_streak,
                    "all_within_cluster": all_clustered,
                },
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

    @property
    def last_action_type(self) -> Optional[str]:
        """
        Return the most recent recorded action type.
        """

        if not self.__recent_types:
            return None

        return self.__recent_types[-1]

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

    def evidence(self) -> LoopEvidence:
        """
        Typed read-only snapshot consumed by the escalation gate.

        ``reason`` identifies which detection strategy classified the window as
        stuck (or ``NOT_STUCK``). ``since_progress`` is the trailing slice of
        turns starting after the most recent PROGRESS effect — that is the
        only span the gate is allowed to consider, because older turns belong
        to a prior recovery cycle and must not unlock escalation on their own.
        """

        reason = self.__classify_reason()
        recent = self.__snapshot_recent()
        since_progress = self.__compute_since_progress(recent=recent)

        return LoopEvidence(
            stuck=reason is not LoopReason.NOT_STUCK,
            reason=reason,
            recent=recent,
            since_progress=since_progress,
        )

    def __classify_reason(self) -> LoopReason:
        """
        Return the first matching detection strategy, mirroring ``is_stuck`` order.
        """

        if self.__detect_inert_repetition():
            return LoopReason.INERT_REPETITION

        has_enough_screens = len(self.__recent_screens) >= self.threshold
        has_enough_actions = len(self.__recent_actions) >= self.threshold

        if has_enough_screens:
            if self.__detect_repetition():
                return LoopReason.SCREEN_REPETITION

            if self.__detect_near_duplicate_visual_repetition():
                return LoopReason.NEAR_DUPLICATE_VISUAL

            if self.__detect_oscillation():
                return LoopReason.STATE_OSCILLATION

            if self.__detect_scroll_stall():
                return LoopReason.SCROLL_STALL

            if self.__detect_action_velocity_loop():
                return LoopReason.ACTION_VELOCITY

        if has_enough_actions and self.__detect_action_repetition():
            return LoopReason.ACTION_REPETITION

        return LoopReason.NOT_STUCK

    def __snapshot_recent(self) -> tuple[LoopTurn, ...]:
        """
        Build the oldest-first turn snapshot aligned to the trailing deque slice.

        :meth:`advance` clears screens and hashes but preserves actions, types,
        and effect statuses so action-pattern detectors survive screen resets.
        That leaves the deques temporally misaligned — actions accumulate
        across advances while screens only cover the current window. The
        snapshot must therefore align from the tail: the most recent ``size``
        entries from each deque represent the same temporal slice.
        """

        types = list(self.__recent_types)
        hashes = list(self.__recent_hashes)
        effects = list(self.__recent_effect_statuses)

        size = min(len(types), len(effects), len(hashes))
        if size == 0:
            return ()

        types = types[-size:]
        effects = effects[-size:]
        hashes = hashes[-size:]

        turns: list[LoopTurn] = []

        for index in range(size):
            action_token = types[index]
            effect_token = effects[index]

            try:
                effect_status = (
                    ActionEffectStatus(effect_token)
                    if effect_token
                    else ActionEffectStatus.UNCERTAIN
                )

            except ValueError:
                effect_status = ActionEffectStatus.UNCERTAIN

            turns.append(
                LoopTurn(
                    action_type=action_token,
                    effect_status=effect_status,
                    screen_hash_prefix=hashes[index][:8] if hashes[index] else "",
                    action_kind=ActionKindResolver.resolve_token(token=action_token),
                )
            )

        return tuple(turns)

    @staticmethod
    def __compute_since_progress(*, recent: tuple[LoopTurn, ...]) -> tuple[LoopTurn, ...]:
        """
        Trailing turns after the most recent PROGRESS effect.

        Walks oldest-first to find the index of the last PROGRESS turn; the
        slice ``since_progress`` is every turn that follows it. PROGRESS turns
        themselves are excluded — they bound the live window. UNCERTAIN turns
        are pass-through. When the window contains no PROGRESS turn the slice
        is the entire window.
        """

        last_progress_index = -1
        for index, turn in enumerate(recent):
            if turn.effect_status is ActionEffectStatus.PROGRESS:
                last_progress_index = index

        return tuple(recent[last_progress_index + 1 :])
