from __future__ import annotations

import unittest

from pydantic import TypeAdapter, ValidationError

from fathom.schemas.proposal import (
    CaptureProposal,
    CommandProposal,
    DecompositionProposal,
    ObservedProposal,
)


class DecompositionProposalTest(unittest.TestCase):
    """
    Pins the untrusted proposal union: valid kinds parse, navigation can never be a command.
    """

    __adapter: TypeAdapter[DecompositionProposal] = TypeAdapter(DecompositionProposal)

    def test_observed_proposal_parses(self) -> None:
        """
        An observable outcome parses to the observed kind.
        """

        proposal = self.__adapter.validate_python(
            {"kind": "OBSERVED", "assertion": "the home screen is displayed"}
        )

        self.assertIsInstance(proposal, ObservedProposal)

    def test_command_proposal_parses_for_a_targeted_primitive(self) -> None:
        """
        A user-named targeted primitive parses to the command kind.
        """

        proposal = self.__adapter.validate_python(
            {
                "kind": "COMMAND",
                "requirement": {"operation": "tap", "target": "Login"},
                "quote": "tap Login",
            }
        )

        self.assertIsInstance(proposal, CommandProposal)

    def test_capture_proposal_parses(self) -> None:
        """
        A store clause parses to the capture kind.
        """

        proposal = self.__adapter.validate_python(
            {
                "kind": "CAPTURE",
                "subject": "the verification code",
                "name": "otp_code",
                "provenance": "USER",
            }
        )

        self.assertIsInstance(proposal, CaptureProposal)

    def test_navigation_cannot_be_a_command_proposal(self) -> None:
        """
        Back, home, and hide-keyboard name a destination, not a success, so no command proposal admits them.
        """

        for operation in ("back", "home", "hide_keyboard"):
            with self.subTest(operation=operation), self.assertRaises(ValidationError):
                self.__adapter.validate_python(
                    {
                        "kind": "COMMAND",
                        "requirement": {"operation": operation},
                        "quote": "come back to the home screen",
                    }
                )


if __name__ == "__main__":
    unittest.main()
