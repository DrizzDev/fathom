from __future__ import annotations

import unittest

from fathom.constants import (
    CONTROL_ACTION_TYPES,
    DEVICE_ACTION_TYPES,
    GESTURE_ACTION_TYPES,
    SPATIAL_ACTION_TYPES,
    ActionType,
)
from fathom.core.capability.catalog import (
    CommandAvailabilityResolver,
    CommandCatalogProvider,
)
from fathom.core.exceptions import InvariantViolation
from fathom.schemas.capability import CommandAvailabilityConfig


class CommandCatalogParityTest(unittest.TestCase):
    """
    Proves the full catalog reproduces today's command behaviour exactly, asserting each
    catalog view against the actual current frozensets as the oracle rather than against
    hand-typed literals. Routing/retry (ActionExecutor) and the directive projection (decomposer)
    are now catalog-backed, so asserting catalog-vs-consumer here would be tautological; their
    golden masters live in tests/unit/core/command/test_baseline.py.
    """

    def setUp(self) -> None:
        """
        Build the full catalog.
        """

        self.__catalog = CommandCatalogProvider().build()

    def test_catalog_is_exhaustive_over_action_types(self) -> None:
        """
        The full catalog declares a profile for every command Fathom can emit.
        """

        self.assertEqual(self.__catalog.action_types(), frozenset(ActionType))

    def test_spatial_and_gesture_match_current_frozensets(self) -> None:
        """
        Catalog spatial/gesture views equal the current SPATIAL/GESTURE frozensets.
        """

        for action_type in ActionType:
            with self.subTest(action_type=action_type):
                self.assertEqual(
                    self.__catalog.is_spatial(action_type=action_type),
                    action_type in SPATIAL_ACTION_TYPES,
                )
                self.assertEqual(
                    self.__catalog.is_gesture(action_type=action_type),
                    action_type in GESTURE_ACTION_TYPES,
                )

    def test_control_and_device_match_current_frozensets(self) -> None:
        """
        Catalog control/device views equal the current CONTROL/DEVICE frozensets.
        """

        for action_type in ActionType:
            with self.subTest(action_type=action_type):
                self.assertEqual(
                    self.__catalog.is_control(action_type=action_type),
                    action_type in CONTROL_ACTION_TYPES,
                )
                self.assertEqual(
                    self.__catalog.is_device(action_type=action_type),
                    action_type in DEVICE_ACTION_TYPES,
                )


class CommandAvailabilityResolverTest(unittest.TestCase):
    """
    Proves a disabled command is absent from the resolved catalog entirely.
    """

    def setUp(self) -> None:
        """
        Build the full catalog and the resolver.
        """

        self.__catalog = CommandCatalogProvider().build()
        self.__resolver = CommandAvailabilityResolver()

    def test_disabled_command_is_absent_from_enabled_catalog(self) -> None:
        """
        A disabled command is not present in the resolved catalog's known commands.
        """

        enabled = self.__resolver.resolve(
            catalog=self.__catalog,
            config=CommandAvailabilityConfig(disabled=frozenset({ActionType.WAIT})),
        )

        self.assertNotIn(ActionType.WAIT, enabled.action_types())
        self.assertIn(ActionType.TAP, enabled.action_types())

    def test_querying_a_disabled_command_fails_fast(self) -> None:
        """
        Looking up a disabled command raises, behaving as if Fathom does not know it exists.
        """

        enabled = self.__resolver.resolve(
            catalog=self.__catalog,
            config=CommandAvailabilityConfig(disabled=frozenset({ActionType.WAIT})),
        )

        with self.assertRaises(InvariantViolation):
            enabled.profile(action_type=ActionType.WAIT)

    def test_no_disabled_commands_preserves_full_catalog(self) -> None:
        """
        With an empty disabled set the resolved catalog matches the full catalog.
        """

        enabled = self.__resolver.resolve(
            catalog=self.__catalog, config=CommandAvailabilityConfig()
        )

        self.assertEqual(enabled.action_types(), self.__catalog.action_types())
