from __future__ import annotations

import unittest

from pydantic import TypeAdapter, ValidationError

from fathom.constants import ActionType
from fathom.constants.flow import ScrollDirection
from fathom.schemas.requirement import (
    CommandRequirement,
    NavigationRequirement,
    PressRequirement,
    ScrollRequirement,
    SwipeRequirement,
    TypeRequirement,
    WaitRequirement,
)


class CommandRequirementTest(unittest.TestCase):
    """
    Pins the canonical command-requirement union: valid variants only, invalid combinations impossible.
    """

    __adapter: TypeAdapter[CommandRequirement] = TypeAdapter(CommandRequirement)

    def test_press_variant_round_trips(self) -> None:
        """
        Tap and long-press deserialize to the press variant with a target.
        """

        for operation in ("tap", "long_press"):
            requirement = self.__adapter.validate_python({"operation": operation, "target": "Login"})
            self.assertIsInstance(requirement, PressRequirement)

    def test_type_variant_requires_target_and_text(self) -> None:
        """
        Type deserializes only with both target and text.
        """

        requirement = self.__adapter.validate_python(
            {"operation": "type", "target": "Search", "text": "Ghar soaps"}
        )
        self.assertIsInstance(requirement, TypeRequirement)

    def test_type_without_text_is_rejected(self) -> None:
        """
        A type requirement missing its text is invalid.
        """

        with self.assertRaises(ValidationError):
            self.__adapter.validate_python({"operation": "type", "target": "Search"})

    def test_scroll_variant_requires_direction(self) -> None:
        """
        Scroll deserializes with a typed content direction and an optional target.
        """

        requirement = self.__adapter.validate_python({"operation": "scroll", "direction": "DOWN"})
        self.assertIsInstance(requirement, ScrollRequirement)

    def test_scroll_without_direction_is_rejected(self) -> None:
        """
        A scroll requirement missing its direction is invalid.
        """

        with self.assertRaises(ValidationError):
            self.__adapter.validate_python({"operation": "scroll"})

    def test_swipe_variant_requires_direction(self) -> None:
        """
        Swipe deserializes to the swipe variant with a finger direction, distinct from scroll.
        """

        requirement = self.__adapter.validate_python({"operation": "swipe", "direction": "UP"})
        self.assertIsInstance(requirement, SwipeRequirement)

    def test_swipe_without_direction_fails_closed(self) -> None:
        """
        A generic swipe with no direction is invalid; there is no hidden default.
        """

        with self.assertRaises(ValidationError):
            SwipeRequirement(operation=ActionType.SWIPE)

    def test_wait_variant_requires_condition_and_bound(self) -> None:
        """
        Wait deserializes with an observable condition and a positive bound.
        """

        requirement = self.__adapter.validate_python(
            {"operation": "wait", "condition": "The results load", "bound": 5.0}
        )
        self.assertIsInstance(requirement, WaitRequirement)

    def test_navigation_variant_carries_no_parameters(self) -> None:
        """
        Back, home, and hide-keyboard deserialize to navigation with no target or payload.
        """

        for operation in ("back", "home", "hide_keyboard"):
            requirement = self.__adapter.validate_python({"operation": operation})
            self.assertIsInstance(requirement, NavigationRequirement)

    def test_tap_with_text_is_rejected(self) -> None:
        """
        A press requirement cannot carry text; the field is impossible for that operation.
        """

        with self.assertRaises(ValidationError):
            self.__adapter.validate_python({"operation": "tap", "target": "Login", "text": "x"})

    def test_back_with_target_is_rejected(self) -> None:
        """
        A navigation requirement cannot carry a target.
        """

        with self.assertRaises(ValidationError):
            self.__adapter.validate_python({"operation": "back", "target": "Login"})

    def test_control_memory_and_terminal_operations_are_rejected(self) -> None:
        """
        Non-primitive operations are not command requirements.
        """

        for operation in (
            ActionType.VALIDATE,
            ActionType.COMPLETE,
            ActionType.ASK_USER,
            ActionType.STORE,
            ActionType.SAVE_MEMORY,
            ActionType.RETRIEVE_MEMORY,
            ActionType.INFER,
            ActionType.UNKNOWN,
        ):
            with self.assertRaises(ValidationError):
                self.__adapter.validate_python({"operation": operation.value})

    def test_invalid_scroll_direction_is_rejected(self) -> None:
        """
        The scroll direction must be a typed content direction.
        """

        with self.assertRaises(ValidationError):
            self.__adapter.validate_python({"operation": "scroll", "direction": "sideways"})

    def test_scroll_direction_is_the_typed_enum(self) -> None:
        """
        A valid scroll requirement exposes the direction as the typed enum.
        """

        requirement = ScrollRequirement(operation=ActionType.SCROLL, direction=ScrollDirection.UP)

        self.assertEqual(requirement.direction, ScrollDirection.UP)
