from fathom.schemas.state import LoopDetector, ScreenState


def test_loop_detector_exhaustion_signal():
    detector = LoopDetector(threshold=3)

    # Simulate stuck state
    screen = ScreenState(
        activity="TestActivity",
        timestamp=1000,
        activity_hash="deadbeef",
        structural_hash="c0ffee",
        visual_hash="a1b2c3d4",
    )

    # Record same screen 3 times
    detector.record(screen, "swipe_left")
    detector.record(screen, "swipe_left")
    detector.record(screen, "swipe_left")

    # Should be stuck
    assert detector.is_stuck() is True

    # Signal exhaustion
    detector.signal_content_exhausted()

    # Should no longer be stuck
    assert detector.is_stuck() is False


def test_loop_detector_exhaustion_clears_history():
    detector = LoopDetector(threshold=3)
    screen = ScreenState(
        activity="TestActivity",
        timestamp=1000,
        activity_hash="deadbeef",
        structural_hash="c0ffee",
        visual_hash="a1b2c3d4",
    )

    detector.record(screen, "swipe_left")
    detector.signal_content_exhausted()

    # Verify history is cleared by checking internal state (if accessible) or behavior
    # After clear, recording 2 more identical screens shouldn't trigger stuck (needs 3)
    detector.record(screen, "swipe_left")
    detector.record(screen, "swipe_left")

    assert detector.is_stuck() is False


def test_loop_detector_repeated_action_triggers_stuck():
    detector = LoopDetector(threshold=3)
    screens = [
        ScreenState(
            activity=f"TestActivity{i}",
            timestamp=1000 + i,
            activity_hash=f"deadbe0{i}",
            structural_hash="c0ffee",
            visual_hash=f"a1b2c3d{i}",
        )
        for i in range(3)
    ]

    for screen in screens:
        detector.record(screen, "tap:retry-button")

    assert detector.is_stuck() is True


def test_loop_detector_diverse_actions_bypass_repeated_screen_false_positive():
    detector = LoopDetector(threshold=3)
    screen = ScreenState(
        activity="TestActivity",
        timestamp=1000,
        activity_hash="deadbeef",
        structural_hash="c0ffee",
        visual_hash="a1b2c3d4",
    )

    detector.record(screen, "tap:menu")
    detector.record(screen, "scroll:down")
    detector.record(screen, "back")

    assert detector.is_stuck() is False
