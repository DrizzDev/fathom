"""
Fixture-replay tests for :class:`LoopDetector`.

These tests pin behaviour against real failure traces so threshold
constants cannot regress silently. Each fixture replays the
``observe_screen`` → ``record`` pattern the live RECORD/GROUND nodes
follow, then asserts ``is_stuck`` returns the expected verdict.

The fixtures cover four shapes:

1. **3.txt scroll loop** — 11 ``swipe_up`` actions on the Swiggy
   auto-suggest page; pHashes oscillate inside the ``a767707x`` cluster
   with one-bit jitter. Must fire stuck.
2. **yVKnb-style coachmark oscillation** — agent alternates between
   two near-duplicate screens (overlay visible / overlay redrawn) and
   keeps emitting the same dismissal action. Must fire stuck.
3. **Productive scroll** — same scroll action across diverging hashes
   (real new content revealed each swipe). Must NOT fire stuck.
4. **Mixed actions on stable screen** — diverse actions on a tightly
   clustered set of screens. Must fire stuck because the screen-set is
   converging.
"""

from __future__ import annotations

from typing import List

from fathom.schemas.screens import ScreenState
from fathom.schemas.state import LoopDetector


def _screen(visual_hash: str, *, activity: str = "bundl.swiggy.production") -> ScreenState:
    """
    Build a minimal :class:`ScreenState` for fixture replay.

    The visual hash is padded to the canonical 16 hex-char form so
    :meth:`ScreenState.hamming_distance` returns the same values as it
    would for a real production capture.
    """

    padded = visual_hash if len(visual_hash) >= 16 else visual_hash + "0" * (16 - len(visual_hash))
    return ScreenState(
        activity=activity,
        timestamp=0,
        activity_hash="aaaaaaaaaaaaaaaa",
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
    ``record`` the same way the live GROUND node does. Caller asserts
    on ``detector.is_stuck()`` afterwards.
    """

    assert len(visual_hashes) == len(actions) == len(action_types)

    previous: ScreenState | None = None
    for visual_hash, action, action_type in zip(visual_hashes, actions, action_types):
        current = _screen(visual_hash=visual_hash)
        detector.observe_screen(previous=previous, current=current)
        detector.record(screen=current, action_type=action_type, action_description=action)
        previous = current


class TestLoopDetectorFixtures:
    """
    Trace-pinned behavioural tests for :class:`LoopDetector`.
    """

    # Short-hash sequence from the actual 3.txt run (workflow 883f12f6),
    # steps 14 through 24 — the 11-swipe auto-suggest scroll loop.
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

    def test_scroll_loop_from_three_dot_txt_is_detected_as_stuck(self) -> None:
        """
        The 3.txt scroll loop must trip ``is_stuck`` — this is the run
        that motivated the threshold split and the carve-out removal.
        """

        detector = LoopDetector(threshold=3, window_size=15)
        _replay(
            detector,
            visual_hashes=self.SCROLL_LOOP_HASHES,
            actions=["Swipe up on Auto suggest page"] * len(self.SCROLL_LOOP_HASHES),
            action_types=["swipe_up"] * len(self.SCROLL_LOOP_HASHES),
        )

        assert detector.is_stuck() is True

    def test_coachmark_two_screen_oscillation_is_detected_as_stuck(self) -> None:
        """
        A yVKnb-style coachmark loop alternates between two
        near-duplicate screens (overlay visible vs overlay re-rendered)
        with the same dismissal action. ``observe_screen`` must NOT
        treat the alternation as progress, and the oscillation /
        repetition detectors must converge on stuck.
        """

        detector = LoopDetector(threshold=3, window_size=15)
        # Two near-duplicate screens within hamming-cluster threshold
        oscillating = ["98e8a527", "98e8a526"] * 6
        _replay(
            detector,
            visual_hashes=oscillating,
            actions=["Tap on Alright, got it button"] * len(oscillating),
            action_types=["tap"] * len(oscillating),
        )

        assert detector.is_stuck() is True

    def test_productive_scroll_with_diverging_hashes_is_not_stuck(self) -> None:
        """
        Repeated scroll actions across genuinely diverging hashes
        (real new content revealed each swipe) must NOT fire stuck,
        otherwise long-feed scrolling becomes impossible.
        """

        detector = LoopDetector(threshold=3, window_size=15)
        # Hashes diverge by >> SCREEN_PROGRESS_HAMMING_THRESHOLD (16)
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
        Even with action diversity, a sequence of near-duplicate
        screens (hashes inside the cluster threshold) must fire stuck
        — the agent is cycling through different tactics against an
        unchanging screen.
        """

        detector = LoopDetector(threshold=3, window_size=15)
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

    def test_observe_screen_does_not_clear_buffer_on_micro_changes(self) -> None:
        """
        Direct pin against the bug 3.txt surfaced:
        :meth:`observe_screen.advance` must NOT fire when the hash
        delta is below the progress threshold, so accumulating
        evidence survives.
        """

        detector = LoopDetector(threshold=3, window_size=15)
        # One-bit jitter between consecutive frames is below the
        # SCREEN_PROGRESS_HAMMING_THRESHOLD (16); buffer must accumulate
        # rather than reset on every step.
        jittery = ["a7677072", "a7677073", "a7677076", "a7677072"]
        _replay(
            detector,
            visual_hashes=jittery,
            actions=["Swipe up"] * len(jittery),
            action_types=["swipe_up"] * len(jittery),
        )

        snapshot = detector.to_state()
        # All four screens must be retained — none were treated as
        # "progress" by observe_screen.
        assert len(snapshot.hashes) == len(jittery)
