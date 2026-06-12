from __future__ import annotations

import sys
import unittest
from typing import List, cast
from unittest.mock import AsyncMock, Mock, patch

from fathom.constants.exploration import DEFAULT_EXPLORATION_INTENT
from fathom.constants.platform import DeviceConnectionType, DevicePlatform
from fathom.runtime.command.application import CommandApplication
from fathom.schemas.configuration import DeviceConfiguration
from fathom.schemas.run import ExplorationRunRequest


class CommandApplicationExploreTest(unittest.TestCase):
    """
    The explore command threads --package and --focus into the exploration request.
    """

    @staticmethod
    def __exploration_request(*, argv: List[str]) -> ExplorationRunRequest:
        """
        Run the CLI with the given argv and return the dispatched exploration request.
        """

        resolver = Mock()
        resolver.resolve = Mock(
            return_value=DeviceConfiguration(
                type=DeviceConnectionType.LOCAL, platform=DevicePlatform.ANDROID
            )
        )
        executor = Mock()
        executor.explore = AsyncMock(return_value=0)

        with (
            patch.object(sys, "argv", argv),
            patch("fathom.runtime.command.application.BaseLogger"),
            patch("fathom.runtime.command.application.CommandExecutor", return_value=executor),
        ):
            CommandApplication(local_device_resolver=resolver).run()

        return cast("ExplorationRunRequest", executor.explore.await_args.kwargs["request"])

    def test_package_flag_flows_into_exploration_request(self) -> None:
        """
        `explore --package <id>` builds a request whose objective targets that package.
        """

        request = self.__exploration_request(
            argv=["fathom", "explore", "--package", "ai.hangjam.app", "--max-steps", "3"]
        )

        self.assertEqual(request.objective.package_name, "ai.hangjam.app")
        self.assertEqual(request.objective.max_steps, 3)

    def test_focus_flag_flows_into_exploration_request(self) -> None:
        """
        `explore --focus <text>` sets the objective intent (the prompt GOAL).
        """

        request = self.__exploration_request(
            argv=["fathom", "explore", "--focus", "Focus on the checkout flow"]
        )

        self.assertEqual(request.objective.intent, "Focus on the checkout flow")

    def test_absent_focus_uses_default_intent(self) -> None:
        """
        Without --focus the objective falls back to the default exploration intent.
        """

        request = self.__exploration_request(argv=["fathom", "explore"])

        self.assertEqual(request.objective.intent, DEFAULT_EXPLORATION_INTENT)
