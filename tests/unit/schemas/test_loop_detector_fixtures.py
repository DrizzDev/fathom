from __future__ import annotations

from typing import List

from fathom.constants.execution import VISUAL_HASH_LENGTH
from fathom.constants.screen import LOOP_DETECTOR_WINDOW_SIZE, LOOP_REPETITION_THRESHOLD
from fathom.schemas.screens import ScreenState
from fathom.schemas.state import LoopDetector


def _screen(visual_hash: str, *, activity: str = "com.example.app") -> ScreenState:
    """
    Build a minimal :class:`ScreenState` whose visual hash is padded to the
    canonical length, so hamming distance returns production-equivalent values.
    """

    padded = (
        visual_hash
        if len(visual_hash) >= VISUAL_HASH_LENGTH
        else visual_hash + "0" * (VISUAL_HASH_LENGTH - len(visual_hash))
    )
    return ScreenState(
        activity=activity,
        timestamp=0,
        activity_hash="a" * VISUAL_HASH_LENGTH,
        visual_hash=padded,
    )


def _replay(
    detector: LoopDetector,
    *,
    visual_hashes: List[str],
    actions: List[str],
    action_types: List[str],
) -> None:
    """
    Replay a (hash, action, type) sequence through ``observe_screen`` +
    ``record`` the same way the live GROUND node does. Callers assert on
    ``detector.is_stuck()`` after the replay completes.
    """

    assert len(visual_hashes) == len(actions) == len(action_types)

    previous: ScreenState | None = None
    for visual_hash, action, action_type in zip(visual_hashes, actions, action_types, strict=True):
        current = _screen(visual_hash=visual_hash)
        detector.observe_screen(previous=previous, current=current)
        detector.record(screen=current, action_type=action_type, action_description=action)
        previous = current


class TestLoopDetectorFixtures:
    """
    Pinned-trace behavioural tests for :class:`LoopDetector`.
    """

    SCROLL_LOOP_HASHES: List[str] = [
        "87a77272",
        "87667072",
        "a7677072",
        "a7677072",
        "a7677076",
        "a7677072",
        "a7677072",
        "a7677072",
        "a7677076",
        "a7677076",
        "a7677072",
    ]

    def test_repeated_scroll_action_with_clustered_hashes_is_detected_as_stuck(self) -> None:
        """
        Repeated scrolls on near-duplicate hashes within the progress threshold
        must trip ``is_stuck`` — micro-jitter is not progress.
        """

        detector = LoopDetector(
            threshold=LOOP_REPETITION_THRESHOLD, window_size=LOOP_DETECTOR_WINDOW_SIZE
        )
        _replay(
            detector,
            visual_hashes=self.SCROLL_LOOP_HASHES,
            actions=["Swipe up on results page"] * len(self.SCROLL_LOOP_HASHES),
            action_types=["swipe_up"] * len(self.SCROLL_LOOP_HASHES),
        )

        assert detector.is_stuck() is True

    def test_two_screen_oscillation_with_single_action_is_detected_as_stuck(self) -> None:
        """
        Alternating between two near-duplicate screens with the same dismissal
        action must fire stuck — ``observe_screen`` cannot treat alternation
        as progress.
        """

        detector = LoopDetector(
            threshold=LOOP_REPETITION_THRESHOLD, window_size=LOOP_DETECTOR_WINDOW_SIZE
        )
        oscillating = ["98e8a527", "98e8a526"] * 6
        _replay(
            detector,
            visual_hashes=oscillating,
            actions=["Tap on dismiss overlay"] * len(oscillating),
            action_types=["tap"] * len(oscillating),
        )

        assert detector.is_stuck() is True

    def test_repeated_scroll_action_with_diverging_hashes_is_not_stuck(self) -> None:
        """
        Repeated scroll across genuinely diverging hashes (new content per
        swipe) must NOT fire stuck — long-feed scrolling must remain feasible.
        """

        detector = LoopDetector(
            threshold=LOOP_REPETITION_THRESHOLD, window_size=LOOP_DETECTOR_WINDOW_SIZE
        )
        diverging = [
            "0000000000000000",
            "ffff000000000000",
            "00ff00ff00ff00ff",
            "ffffffff00000000",
            "0123456789abcdef",
            "fedcba9876543210",
        ]
        _replay(
            detector,
            visual_hashes=diverging,
            actions=["Swipe up on feed"] * len(diverging),
            action_types=["swipe_up"] * len(diverging),
        )

        assert detector.is_stuck() is False

    def test_mixed_actions_on_converging_screens_is_detected_as_stuck(self) -> None:
        """
        Diverse actions on near-duplicate screens must still fire stuck — the
        agent is cycling through tactics against an unchanging screen.
        """

        detector = LoopDetector(
            threshold=LOOP_REPETITION_THRESHOLD, window_size=LOOP_DETECTOR_WINDOW_SIZE
        )
        clustered = ["d8e8a12f", "d8e8a12e", "d8e8a12d", "d8e8a12f"]
        _replay(
            detector,
            visual_hashes=clustered,
            actions=[
                "Tap on dropdown",
                "Tap on header",
                "Tap on X button",
                "Tap on dropdown",
            ],
            action_types=["tap", "tap", "tap", "tap"],
        )

        assert detector.is_stuck() is True

    def test_observe_screen_does_not_reset_buffer_on_sub_threshold_hash_delta(self) -> None:
        """
        ``observe_screen.advance`` must NOT fire when the hash delta is below
        the progress threshold; the accumulating evidence must survive.
        """

        detector = LoopDetector(
            threshold=LOOP_REPETITION_THRESHOLD, window_size=LOOP_DETECTOR_WINDOW_SIZE
        )
        jittery = ["a7677072", "a7677073", "a7677076", "a7677072"]
        _replay(
            detector,
            visual_hashes=jittery,
            actions=["Swipe up"] * len(jittery),
            action_types=["swipe_up"] * len(jittery),
        )

        snapshot = detector.to_state()
        assert len(snapshot.hashes) == len(jittery)
