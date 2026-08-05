from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.constants.capability import (
    COMMAND_REQUIREMENT_CHANNELS,
    CompletionMode,
    ExecutionChannel,
    RecordMode,
    RetryMode,
)
from fathom.constants.command import CommandRejection
from fathom.constants.flow import ScrollDirection, SwipeDirection
from fathom.core.capability.binder import CommandBinder
from fathom.core.capability.catalog import CommandCatalog, CommandCatalogProvider
from fathom.schemas.capability import CommandProfile
from fathom.schemas.proposal import BoundCommand, CommandProposal, RejectedCommand
from fathom.schemas.requirement import (
    CommandRequirement,
    NavigationRequirement,
    PressRequirement,
    ScrollRequirement,
    SwipeRequirement,
    TypeRequirement,
    WaitRequirement,
)


class CommandBinderTest(unittest.TestCase):
    """
    Pins structural command binding: exact host-computed provenance and catalog admission only.
    """

    def setUp(self) -> None:
        """
        Build a binder over the full command catalog.
        """

        self.__binder = CommandBinder(catalog=CommandCatalogProvider().build())

    @staticmethod
    def __press() -> PressRequirement:
        """
        Build a press requirement on a fixed target.
        """

        return PressRequirement(operation=ActionType.TAP, target="Login")

    @staticmethod
    def __proposal(*, requirement: CommandRequirement, quote: str) -> CommandProposal:
        """
        Build a command proposal for the given requirement and cited quote.
        """

        return CommandProposal(requirement=requirement, quote=quote)

    def test_unique_quote_is_cited_as_diagnostic_provenance(self) -> None:
        """
        A single exact occurrence is cited with host-computed offsets; this provenance never gates admission.
        """

        intent = "Open the app then Tap Login now"
        result = self.__binder.bind(
            intent=intent, proposal=self.__proposal(requirement=self.__press(), quote="Tap Login")
        )

        self.assertIsInstance(result, BoundCommand)
        assert isinstance(result, BoundCommand)
        self.assertIsNotNone(result.success.source)
        assert result.success.source is not None
        start = intent.find("Tap Login")
        self.assertEqual(result.success.source.location.start, start)
        self.assertEqual(result.success.source.location.end, start + len("Tap Login"))
        self.assertEqual(result.success.source.quote, "Tap Login")

    def test_absent_quote_still_binds_without_citation(self) -> None:
        """
        A quote absent from the intent no longer rejects; the command binds on catalog structure, uncited.
        """

        result = self.__binder.bind(
            intent="Open the app",
            proposal=self.__proposal(requirement=self.__press(), quote="Tap Login"),
        )

        self.assertIsInstance(result, BoundCommand)
        assert isinstance(result, BoundCommand)
        self.assertIsNone(result.success.source)

    def test_ambiguous_quote_still_binds_without_citation(self) -> None:
        """
        A quote occurring more than once no longer rejects; ambiguity only withholds the diagnostic citation.
        """

        result = self.__binder.bind(
            intent="Tap Login then Tap Login",
            proposal=self.__proposal(requirement=self.__press(), quote="Tap Login"),
        )

        self.assertIsInstance(result, BoundCommand)
        assert isinstance(result, BoundCommand)
        self.assertIsNone(result.success.source)

    def test_proposal_cannot_carry_offsets(self) -> None:
        """
        The untrusted proposal has no location or span field, so offsets can never be caller-supplied.
        """

        self.assertNotIn("location", CommandProposal.model_fields)
        self.assertNotIn("source", CommandProposal.model_fields)

    def test_device_and_wait_requirements_are_admitted(self) -> None:
        """
        Press, type, scroll, swipe, wait, and navigation requirements bind on admissible channels.
        """

        cases = [
            (PressRequirement(operation=ActionType.TAP, target="Login"), "Tap Login"),
            (TypeRequirement(operation=ActionType.TYPE, target="Search", text="soap"), "Type soap"),
            (
                ScrollRequirement(operation=ActionType.SCROLL, direction=ScrollDirection.DOWN),
                "Scroll down",
            ),
            (
                SwipeRequirement(operation=ActionType.SWIPE, direction=SwipeDirection.LEFT),
                "Swipe left",
            ),
            (
                WaitRequirement(operation=ActionType.WAIT, condition="results load", bound=5.0),
                "Wait for results",
            ),
            (NavigationRequirement(operation=ActionType.BACK), "Go back"),
        ]

        for requirement, quote in cases:
            intent = f"please {quote} thanks"
            result = self.__binder.bind(
                intent=intent, proposal=self.__proposal(requirement=requirement, quote=quote)
            )
            self.assertIsInstance(result, BoundCommand, msg=requirement.operation.value)

    def test_non_primitive_operations_are_not_admissible_channels(self) -> None:
        """
        Store, validate, complete, ask-user, and memory run on channels the requirement set excludes.
        """

        catalog = CommandCatalogProvider().build()
        for operation in (
            ActionType.STORE,
            ActionType.VALIDATE,
            ActionType.COMPLETE,
            ActionType.ASK_USER,
            ActionType.SAVE_MEMORY,
        ):
            self.assertNotIn(
                catalog.profile(action_type=operation).channel, COMMAND_REQUIREMENT_CHANNELS
            )

    def test_unknown_operation_fails_closed(self) -> None:
        """
        A catalog that does not support the operation rejects the proposal fail-closed.
        """

        binder = CommandBinder(catalog=CommandCatalog(profiles={}))
        result = binder.bind(
            intent="Tap Login",
            proposal=self.__proposal(requirement=self.__press(), quote="Tap Login"),
        )

        self.assertIsInstance(result, RejectedCommand)
        assert isinstance(result, RejectedCommand)
        self.assertEqual(result.reason, CommandRejection.OPERATION_UNSUPPORTED)

    def test_admission_comes_from_the_catalog_not_an_internal_table(self) -> None:
        """
        The same requirement is rejected when the injected catalog maps its operation to a barred channel.
        """

        catalog = CommandCatalog(
            profiles={
                ActionType.TAP: CommandProfile(
                    channel=ExecutionChannel.CONTROL,
                    completion=CompletionMode.SCREEN_VERIFIED,
                    records=RecordMode.ACTION,
                    retry=RetryMode.NONE,
                )
            }
        )
        result = CommandBinder(catalog=catalog).bind(
            intent="Tap Login",
            proposal=self.__proposal(requirement=self.__press(), quote="Tap Login"),
        )

        self.assertIsInstance(result, RejectedCommand)
        assert isinstance(result, RejectedCommand)
        self.assertEqual(result.reason, CommandRejection.CHANNEL_NOT_ADMITTED)
