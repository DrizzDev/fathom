from __future__ import annotations

import unittest

from fathom.adapters.dialect.drizz.parser import DrizzLarkParser
from fathom.constants.flow import CheckKind, ScrollDirection
from fathom.core.exceptions import LanguageParseError
from fathom.schemas.dialect.drizz.command import (
    IfCommand,
    OpenAppCommand,
    ScrollCommand,
    SetGpsCommand,
    StoreCommand,
    TapCommand,
    TypeCommand,
    ValidateCommand,
    WaitCommand,
)


class DrizzLarkParserTest(unittest.TestCase):
    """
    Cover parsing of rendered Drizz into the typed command AST.
    """

    def setUp(self) -> None:
        """
        Build a shared Lark parser.
        """

        self.__parser = DrizzLarkParser()

    def test_flat_script_builds_typed_commands(self) -> None:
        """
        A flat script parses into launch, tap, and validate commands.
        """

        script = self.__parser.parse(
            text="OPEN_APP: com.example\nTap on Login CTA\nValidate home is visible\n"
        )
        launch, tap, validate = script.commands

        assert isinstance(tap, TapCommand)
        assert isinstance(launch, OpenAppCommand)
        assert isinstance(validate, ValidateCommand)

        self.assertEqual(tap.target.text, "Login CTA")
        self.assertEqual(launch.package, "com.example")
        self.assertEqual(validate.assertions[0].subject, "home")
        self.assertEqual(validate.assertions[0].state, CheckKind.VISIBLE)

    def test_if_block_builds_conditional(self) -> None:
        """
        An IF block parses into a conditional with a leaf body and its verbatim condition.
        """

        script = self.__parser.parse(text="IF Overlay is visible\n{\n    Tap on Skip\n}\n")
        branch = script.commands[0]

        assert isinstance(branch, IfCommand)
        self.assertEqual(branch.condition, "Overlay is visible")

        body = branch.body[0]
        assert isinstance(body, TapCommand)
        self.assertEqual(body.target.text, "Skip")

    def test_tap_docs_variants_parse(self) -> None:
        """
        The documented Tap forms (on / the / bare) all parse to a tap command.
        """

        for text, expected in (
            ("Tap on Login CTA\n", "Login CTA"),
            ("Tap the cart icon\n", "cart icon"),
            ("Tap Add under Snacks header\n", "Add under Snacks header"),
        ):
            command = self.__parser.parse(text=text).commands[0]
            assert isinstance(command, TapCommand)
            self.assertEqual(command.target.text, expected)

    def test_scroll_inside_until_is_parsed(self) -> None:
        """
        A container-scoped scroll-until parses its container and quoted target.
        """

        scroll = self.__parser.parse(
            text='Scroll down inside product list until "Add to Cart"\n'
        ).commands[0]
        assert isinstance(scroll, ScrollCommand)
        self.assertEqual(scroll.container, "product list")
        assert scroll.until is not None
        self.assertEqual(scroll.until.text, "Add to Cart")

    def test_natural_language_target_is_parsed(self) -> None:
        """
        An unquoted tap target with a folded ordinal parses as one natural-language phrase.
        """

        script = self.__parser.parse(text="Tap on the first Result\n")
        tap = script.commands[0]

        assert isinstance(tap, TapCommand)
        self.assertEqual(tap.target.text, "the first Result")

    def test_container_target_is_parsed(self) -> None:
        """
        A tap target with a folded container parses as one natural-language phrase.
        """

        script = self.__parser.parse(text="Tap on Add under Cart\n")
        tap = script.commands[0]

        assert isinstance(tap, TapCommand)
        self.assertEqual(tap.target.text, "Add under Cart")

    def test_numbered_validation_builds_assertions(self) -> None:
        """
        A numbered validation parses into one assertion per item.
        """

        script = self.__parser.parse(
            text='Validate the following are present: 1. "home" 2. "cart icon"\n'
        )
        validate = script.commands[0]

        assert isinstance(validate, ValidateCommand)
        self.assertEqual(len(validate.assertions), 2)
        self.assertEqual(validate.assertions[1].subject, "cart icon")
        self.assertEqual(validate.assertions[1].state, CheckKind.PRESENT)

    def test_wait_forms_are_parsed(self) -> None:
        """
        All three wait forms parse into the matching duration and/or subject.
        """

        duration = self.__parser.parse(text="Wait for 5 seconds\n").commands[0]
        subject = self.__parser.parse(text='Wait until "Spinner"\n').commands[0]
        combined = self.__parser.parse(text="Wait 5 seconds for page content to load\n").commands[0]

        assert isinstance(subject, WaitCommand)
        assert isinstance(duration, WaitCommand)
        assert isinstance(combined, WaitCommand)

        self.assertEqual(duration.duration, 5)
        self.assertEqual(subject.subject, "Spinner")
        self.assertEqual(combined.duration, 5)
        self.assertEqual(combined.subject, "page content to load")

    def test_store_captures_value_and_name(self) -> None:
        """
        A store command captures its multi-word value and variable name.
        """

        store = self.__parser.parse(text="Store cart total as savedTotal\n").commands[0]

        assert isinstance(store, StoreCommand)
        self.assertEqual(store.value, "cart total")
        self.assertEqual(store.name, "savedTotal")

    def test_scroll_until_is_parsed(self) -> None:
        """
        A scroll-until command captures its direction and quoted target.
        """

        scroll = self.__parser.parse(text='Scroll down until "Load more"\n').commands[0]

        assert isinstance(scroll, ScrollCommand)
        self.assertEqual(scroll.direction, ScrollDirection.DOWN)
        assert scroll.until is not None
        self.assertEqual(scroll.until.text, "Load more")

    def test_set_gps_parses_coordinates(self) -> None:
        """
        A GPS command parses signed decimal coordinates.
        """

        gps = self.__parser.parse(text="SET_GPS(latitude=12.34, longitude=-56.78)\n").commands[0]
        assert isinstance(gps, SetGpsCommand)
        self.assertEqual(gps.latitude, 12.34)
        self.assertEqual(gps.longitude, -56.78)

    def test_open_app_accepts_both_colon_and_space_forms(self) -> None:
        """
        Both the colon and the space OPEN_APP forms parse to the same launch command.
        """

        colon = self.__parser.parse(text="OPEN_APP:com.example\n").commands[0]
        space = self.__parser.parse(text="OPEN_APP com.example\n").commands[0]
        assert isinstance(colon, OpenAppCommand)
        assert isinstance(space, OpenAppCommand)
        self.assertEqual(colon.package, "com.example")
        self.assertEqual(space.package, "com.example")

    def test_single_quoted_type_value_is_parsed(self) -> None:
        """
        A single-quoted type value is parsed with its quotes stripped.
        """

        typed = self.__parser.parse(text="Type 'Login' into name field\n").commands[0]
        assert isinstance(typed, TypeCommand)
        self.assertEqual(typed.value, "Login")
        self.assertEqual(typed.field.text, "name field")

    def test_backtick_type_value_preserving_inner_double_quote_is_parsed(self) -> None:
        """
        A backtick-delimited type value retains an inner double quote verbatim.
        """

        typed = self.__parser.parse(text='Type `He said "Hi"` into note field\n').commands[0]
        assert isinstance(typed, TypeCommand)
        self.assertEqual(typed.value, 'He said "Hi"')

    def test_quoted_tap_target_unwraps_to_bare_text(self) -> None:
        """
        A quoted tap target parses, unwrapping to the bare phrase so reserved-word targets round-trip.
        """

        tap = self.__parser.parse(text='Tap on "Login as guest"\n').commands[0]

        assert isinstance(tap, TapCommand)
        self.assertEqual(tap.target.text, "Login as guest")

    def test_empty_type_value_raises(self) -> None:
        """
        An empty Type value is rejected during parsing.
        """

        with self.assertRaises(LanguageParseError):
            self.__parser.parse(text='Type "" into Search\n')
