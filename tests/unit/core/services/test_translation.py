from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.constants.success import CaptureNameProvenance
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.core.services.translation import ProposalTranslator
from fathom.schemas.proposal import CaptureProposal, CommandProposal, ObservedProposal
from fathom.schemas.requirement import PressRequirement
from fathom.schemas.success import CaptureSuccess, CommandSuccess, ObservedSuccess


class ProposalTranslatorTest(unittest.TestCase):
    """
    Pins trusted translation: proposals become canonical Success or fail closed.
    """

    def setUp(self) -> None:
        """
        Build a translator over the full command catalog.
        """

        self.__translator = ProposalTranslator(catalog=CommandCatalogProvider().build())

    def test_observed_proposal_becomes_observed_success(self) -> None:
        """
        An observed proposal is boundary-validated into an observed success.
        """

        success = self.__translator.translate(
            intent="Search for soap",
            proposal=ObservedProposal(assertion="Search results are displayed"),
        )

        self.assertIsInstance(success, ObservedSuccess)
        assert isinstance(success, ObservedSuccess)
        self.assertEqual(success.observation.assertion, "Search results are displayed")

    def test_capture_proposal_carries_model_authored_name_verbatim(self) -> None:
        """
        A model-proposed name is preserved verbatim and labelled model-authored; the host never derives it.
        """

        success = self.__translator.translate(
            intent="Capture the balance",
            proposal=CaptureProposal(
                subject="account balance",
                name="account_balance",
                provenance=CaptureNameProvenance.MODEL,
            ),
        )

        self.assertIsInstance(success, CaptureSuccess)
        assert isinstance(success, CaptureSuccess)
        self.assertEqual(success.target.name, "account_balance")
        self.assertIs(success.target.provenance, CaptureNameProvenance.MODEL)
        self.assertEqual(success.subject, "account balance")

    def test_capture_proposal_preserves_explicit_user_name(self) -> None:
        """
        An explicit user-supplied capture name is preserved exactly and labelled user-authored.
        """

        success = self.__translator.translate(
            intent="Capture the code as otp",
            proposal=CaptureProposal(
                subject="the code", name="otp", provenance=CaptureNameProvenance.USER
            ),
        )

        assert isinstance(success, CaptureSuccess)
        self.assertEqual(success.target.name, "otp")
        self.assertIs(success.target.provenance, CaptureNameProvenance.USER)

    def test_command_proposal_admits_on_catalog_structure(self) -> None:
        """
        A command proposal admits on catalog structure; a locatable quote is attached as diagnostic provenance.
        """

        success = self.__translator.translate(
            intent="First Tap Login then continue",
            proposal=CommandProposal(
                quote="Tap Login",
                requirement=PressRequirement(operation=ActionType.TAP, target="Login"),
            ),
        )

        self.assertIsInstance(success, CommandSuccess)
        assert isinstance(success, CommandSuccess)
        self.assertIsNotNone(success.source)
        assert success.source is not None
        self.assertEqual(success.source.quote, "Tap Login")

    def test_command_proposal_admits_even_when_quote_absent(self) -> None:
        """
        The cited quote is no longer authorization: a command whose quote is absent still admits, uncited.
        """

        success = self.__translator.translate(
            intent="Open the app",
            proposal=CommandProposal(
                quote="Tap Login",
                requirement=PressRequirement(operation=ActionType.TAP, target="Login"),
            ),
        )

        self.assertIsInstance(success, CommandSuccess)
        assert isinstance(success, CommandSuccess)
        self.assertIsNone(success.source)
