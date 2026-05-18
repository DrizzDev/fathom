from __future__ import annotations

import unittest

from fathom.core.runtime.screen import ScreenRuntimeState
from fathom.schemas.observation import KeyboardObservation, ScreenObservation
from fathom.schemas.screens import ScreenHashBundle, ScreenState


def _state(*, activity: str, visual: str = "0" * 16, xml: str = "a" * 16) -> ScreenState:
    """
    Build a minimal ScreenState for runtime-screen fixtures.
    """

    return ScreenState(
        activity=activity,
        timestamp=0,
        activity_hash=activity,
        visual_hash=visual,
        xml_hash=xml,
        interaction_hash="b" * 16,
    )


def _observation(*, activity: str = "screen") -> ScreenObservation:
    """
    Build a minimal ScreenObservation for fixtures.
    """

    return ScreenObservation(
        activity=activity,
        hashes=ScreenHashBundle(
            visual_hash="0" * 16,
            xml_hash="a" * 16,
            interaction_hash="b" * 16,
        ),
        elements=(),
        keyboard=KeyboardObservation(visible=False),
    )


class ScreenRuntimeStateTest(unittest.TestCase):
    """
    Pins for the ScreenRuntimeState observation history and seen-screens tracking.
    """

    def test_update_advances_current_and_previous(self) -> None:
        """
        update() rolls the current screen into the previous slot and stores the new one.
        """

        runtime = ScreenRuntimeState()
        first = _state(activity="first")
        second = _state(activity="second", xml="c" * 16)

        runtime.update(screen=first, observation=_observation(activity="first"))
        self.assertEqual(runtime.current, first)
        self.assertIsNone(runtime.previous)

        runtime.update(screen=second, observation=_observation(activity="second"))
        self.assertEqual(runtime.current, second)
        self.assertEqual(runtime.previous, first)

    def test_is_new_true_for_unseen_screen(self) -> None:
        """
        is_new() must be true for a screen not yet observed in this run.
        """

        runtime = ScreenRuntimeState()

        self.assertTrue(runtime.is_new(screen=_state(activity="first")))

    def test_is_new_false_after_remember(self) -> None:
        """
        is_new() must be false after the same screen has been remembered.
        """

        runtime = ScreenRuntimeState()
        screen = _state(activity="first")
        runtime.remember(screen=screen)

        self.assertFalse(runtime.is_new(screen=screen))

    def test_remember_only_appends_new_screens(self) -> None:
        """
        remember() must not append a screen that is already in the seen set.
        """

        runtime = ScreenRuntimeState()
        first = _state(activity="first")
        runtime.remember(screen=first)
        runtime.remember(screen=first)

        self.assertEqual(len(runtime.seen), 1)

    def test_load_seen_replaces_in_memory_list(self) -> None:
        """
        load_seen() replaces the seen-screens history; later is_new follows the new list.
        """

        runtime = ScreenRuntimeState()
        original = _state(activity="first")
        replacement = _state(activity="second", xml="c" * 16)

        runtime.remember(screen=original)
        runtime.load_seen(screens=[replacement])

        self.assertTrue(runtime.is_new(screen=original))
        self.assertFalse(runtime.is_new(screen=replacement))

    def test_seen_returns_a_copy(self) -> None:
        """
        The seen accessor must return a copy that callers cannot mutate.
        """

        runtime = ScreenRuntimeState()
        runtime.remember(screen=_state(activity="first"))

        snapshot = runtime.seen
        snapshot.clear()

        self.assertEqual(len(runtime.seen), 1)

    def test_reset_loop_history_delegates_to_detector(self) -> None:
        """
        reset_loop_history() delegates to the loop detector reset method.
        """

        runtime = ScreenRuntimeState()
        runtime.detector.record_recovery_attempt()
        runtime.reset_loop_history()

        self.assertTrue(runtime.detector.can_recover())
