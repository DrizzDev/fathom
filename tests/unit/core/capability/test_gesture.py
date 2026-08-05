from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.constants.flow import SwipeDirection
from fathom.core.capability.gesture import GestureNormalizer


class GestureNormalizerTest(unittest.TestCase):
    """
    Pins deterministic swipe-alias normalization: finger direction preserved, generic swipe fails closed.
    """

    def setUp(self) -> None:
        """
        Build the gesture normalizer under test.
        """

        self.__normalizer = GestureNormalizer()

    def test_flattened_aliases_map_to_finger_direction_without_inversion(self) -> None:
        """
        Each directional alias maps to the same-named finger direction, never inverted.
        """

        cases = {
            ActionType.SWIPE_UP: SwipeDirection.UP,
            ActionType.SWIPE_DOWN: SwipeDirection.DOWN,
            ActionType.SWIPE_LEFT: SwipeDirection.LEFT,
            ActionType.SWIPE_RIGHT: SwipeDirection.RIGHT,
        }

        for alias, direction in cases.items():
            requirement = self.__normalizer.canonical(operation=alias)
            self.assertEqual(requirement.operation, ActionType.SWIPE)
            self.assertEqual(requirement.direction, direction)

    def test_target_is_preserved(self) -> None:
        """
        A supplied gesture surface is carried onto the canonical requirement.
        """

        requirement = self.__normalizer.canonical(operation=ActionType.SWIPE_UP, target="carousel")

        self.assertEqual(requirement.target, "carousel")

    def test_generic_swipe_alias_fails_closed(self) -> None:
        """
        A directionless swipe alias cannot be normalized and fails closed.
        """

        with self.assertRaises(ValueError):
            self.__normalizer.canonical(operation=ActionType.SWIPE)

    def test_non_swipe_operation_fails_closed(self) -> None:
        """
        A non-swipe operation is not a gesture alias and fails closed.
        """

        with self.assertRaises(ValueError):
            self.__normalizer.canonical(operation=ActionType.TAP)
