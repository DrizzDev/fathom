from __future__ import annotations

import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fathom.constants.platform import DeviceConnectionType, DevicePlatform
from fathom.runtime.command.application import CommandApplication
from fathom.schemas.configuration import DeviceConfiguration


class CommandApplicationExploreTest(unittest.TestCase):
    """
    The explore command threads --package into the exploration request.
    """

    def test_package_flag_flows_into_exploration_request(self) -> None:
        """
        `explore --package <id>` builds a request whose objective targets that package.
        """

        resolver = Mock()
        resolver.resolve = Mock(
            return_value=DeviceConfiguration(
                type=DeviceConnectionType.LOCAL, platform=DevicePlatform.ANDROID
            )
        )
        application = CommandApplication(local_device_resolver=resolver)

        executor = Mock()
        executor.explore = AsyncMock(return_value=0)

        argv = ["fathom", "explore", "--package", "ai.hangjam.app", "--max-steps", "3"]
        with (
            patch.object(sys, "argv", argv),
            patch("fathom.runtime.command.application.BaseLogger"),
            patch("fathom.runtime.command.application.CommandExecutor", return_value=executor),
        ):
            exit_code = application.run()

        self.assertEqual(exit_code, 0)
        request = executor.explore.await_args.kwargs["request"]
        self.assertEqual(request.objective.package_name, "ai.hangjam.app")
        self.assertEqual(request.objective.max_steps, 3)
