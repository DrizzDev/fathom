from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.schemas.cli import ExploreCommandInput


class ExploreCommandInputTest(unittest.TestCase):
    """
    Validate the package identifier accepted by `fathom explore --package`.
    """

    def test_accepts_dotted_package_identifier(self) -> None:
        """
        A dotted reverse-DNS identifier is accepted and trimmed.
        """

        command_input = ExploreCommandInput.model_validate({"package_name": "  ai.hangjam.app  "})

        self.assertEqual(command_input.package_name, "ai.hangjam.app")

    def test_defaults_to_none_when_absent(self) -> None:
        """
        Omitting the package leaves the target unset for foreground exploration.
        """

        self.assertIsNone(ExploreCommandInput().package_name)

    def test_rejects_shell_unsafe_identifier(self) -> None:
        """
        A value carrying shell metacharacters is rejected at the boundary.
        """

        with self.assertRaises(ValidationError):
            ExploreCommandInput.model_validate({"package_name": "com.evil; rm -rf /"})
