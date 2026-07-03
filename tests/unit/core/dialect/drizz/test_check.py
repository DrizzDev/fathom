from __future__ import annotations

import unittest
from typing import FrozenSet

from fathom.adapters.dialect.drizz.parser import DrizzLarkParser
from fathom.constants.flow import IssueCode
from fathom.core.dialect.drizz.check import Checker
from fathom.core.dialect.drizz.print import CanonicalPrinter


class CheckerTest(unittest.TestCase):
    """
    Cover grammar-driven syntax and round-trip validation of rendered Drizz.
    """

    def setUp(self) -> None:
        """
        Build a checker backed by the Lark parser and the canonical printer.
        """

        self.__checker = Checker(parser=DrizzLarkParser(), printer=CanonicalPrinter())

    def __codes(self, *, text: str) -> FrozenSet[IssueCode]:
        """
        Check text and return the set of raised issue codes.
        """

        return frozenset(issue.code for issue in self.__checker.check(text=text).issues)

    def test_valid_script_passes(self) -> None:
        """
        A canonical script raises no issues.
        """

        text = "OPEN_APP: com.example\nTap on Login CTA\nValidate home is visible\n"
        self.assertEqual(self.__codes(text=text), frozenset())

    def test_well_formed_wait_passes(self) -> None:
        """
        A well-formed Wait command is accepted.
        """

        text = "OPEN_APP: com.example\nWait for 5 seconds\nValidate home is visible\n"

        self.assertEqual(self.__codes(text=text), frozenset())

    def test_branch_and_numbered_validation_pass(self) -> None:
        """
        An IF block and a numbered validation are accepted.
        """

        text = (
            "IF Overlay is visible\n{\n    Tap on Skip\n}\n"
            'Validate the following are visible: 1. "home" 2. "cart"\n'
        )

        self.assertEqual(self.__codes(text=text), frozenset())

    def test_unknown_keyword_is_rejected(self) -> None:
        """
        A line that does not begin with a whitelisted keyword is a syntax error.
        """

        self.assertIn(IssueCode.SYNTAX_ERROR, self.__codes(text="Frobnicate the widget\n"))

    def test_unbalanced_braces_are_rejected(self) -> None:
        """
        A dangling opening brace is a syntax error.
        """

        text = 'IF Overlay is visible\n{\n    Tap on "Skip"\n'
        self.assertIn(IssueCode.SYNTAX_ERROR, self.__codes(text=text))

    def test_quoted_clean_tap_target_is_non_canonical(self) -> None:
        """
        A clean tap target needs no quotes, so a quoted one round-trips to the bare form and is flagged.
        """

        self.assertIn(IssueCode.ROUND_TRIP_MISMATCH, self.__codes(text='Tap on "Login"\n'))

    def test_scroll_without_direction_is_rejected(self) -> None:
        """
        A scroll command lacking a direction is a syntax error.
        """

        self.assertIn(IssueCode.SYNTAX_ERROR, self.__codes(text="Scroll\n"))

    def test_empty_numbered_validation_is_rejected(self) -> None:
        """
        A numbered validation with no items is a syntax error.
        """

        self.assertIn(
            IssueCode.SYNTAX_ERROR, self.__codes(text="Validate the following are visible:\n")
        )

    def test_multiple_commands_on_one_line_are_rejected(self) -> None:
        """
        A line carrying more than one command is a syntax error.
        """

        self.assertIn(IssueCode.SYNTAX_ERROR, self.__codes(text='Tap on "A"; Tap on "B"\n'))

    def test_non_canonical_spacing_is_a_round_trip_mismatch(self) -> None:
        """
        Parseable but non-canonical spacing fails the round-trip gate.
        """

        self.assertIn(IssueCode.ROUND_TRIP_MISMATCH, self.__codes(text="Tap on  Login\n"))

    def test_round_trip_mismatch_reports_line_expected_and_actual(self) -> None:
        """
        The mismatch diagnostic pinpoints the diverging line with both the expected and actual text.
        """

        text = "OPEN_APP: com.example\nTap on  Login\nValidate home is visible\n"
        issue = next(
            issue
            for issue in self.__checker.check(text=text).issues
            if issue.code is IssueCode.ROUND_TRIP_MISMATCH
        )

        self.assertIn("line 2", issue.message)
        self.assertIn("Tap on  Login", issue.message)
        self.assertIn("Tap on Login", issue.message)

    def test_single_quoted_value_round_trips(self) -> None:
        """
        A typed value forced to single quotes round-trips canonically.
        """

        self.assertEqual(self.__codes(text="Type 'He said \"Hi\"' into search bar\n"), frozenset())

    def test_empty_type_value_is_rejected(self) -> None:
        """
        An empty Type value is a syntax error.
        """

        self.assertIn(IssueCode.SYNTAX_ERROR, self.__codes(text='Type "" into search bar\n'))
