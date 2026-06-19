from __future__ import annotations

import unittest

from fathom.constants import SWIPE_ACTIONS
from fathom.core.services.exporter.step_record import swipe_direction_label


class SwipeDirectionLabelTest(unittest.TestCase):
    """
    Pins the gesture-to-script-label mapping used when emitting scroll lines.

    A production incident traced to this mapping silently flipped the
    vertical direction: the planner correctly executed ``swipe_up`` against
    the device but the exported script line said ``"Scroll down"``. The
    script-replay engine interpreted the label literally and scrolled in
    the opposite direction, so the intended target never came into view.
    The mapping must therefore preserve the direction the agent actually
    executed.
    """

    def test_swipe_up_preserves_direction(self) -> None:
        """
        Swipe-up gestures must surface as ``"Scroll up"`` so the script-line
        the replay engine receives matches the direction Fathom executed.
        Previously this row inverted to ``"Scroll down"`` and broke replays.
        """

        self.assertEqual(swipe_direction_label(action_type="swipe_up"), "Scroll up")

    def test_swipe_down_preserves_direction(self) -> None:
        """
        Swipe-down gestures must surface as ``"Scroll down"`` for the same
        direction-preservation reason as :meth:`test_swipe_up_preserves_direction`.
        Previously this row inverted to ``"Scroll up"``.
        """

        self.assertEqual(swipe_direction_label(action_type="swipe_down"), "Scroll down")

    def test_bare_scroll_defaults_to_scroll_up(self) -> None:
        """A direction-less ``"scroll"`` action resolves to ``"Scroll up"``."""

        self.assertEqual(swipe_direction_label(action_type="scroll"), "Scroll up")

    def test_swipe_left_unchanged(self) -> None:
        """
        Horizontal gestures are labeled ``"Swipe left/right"`` rather than
        scrolls. The original mapping was already correct for this row;
        the test exists so a future refactor cannot accidentally rename it.
        """

        self.assertEqual(swipe_direction_label(action_type="swipe_left"), "Swipe left")

    def test_swipe_right_unchanged(self) -> None:
        """
        Mirror of :meth:`test_swipe_left_unchanged` for the right direction.
        """

        self.assertEqual(swipe_direction_label(action_type="swipe_right"), "Swipe right")

    def test_unknown_action_type_defaults_to_scroll_up(self) -> None:
        """Any action type the mapping does not enumerate must still produce a valid script line; falling back to ``"Scroll up"`` keeps the default aligned with the bare-``scroll`` convention rather than emitting a direction-less ``"Scroll"`` token the replay engine would reject."""

        self.assertEqual(swipe_direction_label(action_type="unknown_gesture"), "Scroll up")

    def test_every_swipe_action_constant_is_mapped(self) -> None:
        """``swipe_direction_label`` must explicitly map every member of :data:`SWIPE_ACTIONS` so that adding a new gesture to the constants without updating this mapping cannot silently fall back to the generic default and ship a wrong direction into a script."""

        for action_type in SWIPE_ACTIONS:
            with self.subTest(action_type=action_type):
                label = swipe_direction_label(action_type=action_type)
                self.assertTrue(
                    label.startswith("Scroll ") or label.startswith("Swipe "),
                    f"unexpected label for {action_type!r}: {label!r}",
                )


if __name__ == "__main__":
    unittest.main()
