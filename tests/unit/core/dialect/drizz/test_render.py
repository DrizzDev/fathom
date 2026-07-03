from __future__ import annotations

import unittest
from typing import Tuple

from fathom.constants.flow import CheckKind, ScrollDirection
from fathom.core.dialect.drizz.render import Renderer
from fathom.core.exceptions import LanguageComplianceError
from fathom.schemas.flow import (
    BackNode,
    BranchNode,
    Check,
    CheckNode,
    ClearNode,
    Flow,
    FlowNode,
    Guard,
    KillNode,
    LaunchNode,
    LocationNode,
    MapNode,
    MinimizeNode,
    ScrollNode,
    ScrollUntilNode,
    Selector,
    StoreNode,
    TapNode,
    TypeNode,
    WaitNode,
)


class RendererTest(unittest.TestCase):
    """
    Cover Flow-to-Drizz rendering of each command form.
    """

    def setUp(self) -> None:
        """
        Build a shared renderer.
        """

        self.__renderer = Renderer()

    def __render(self, *, nodes: Tuple[FlowNode, ...]) -> str:
        """
        Render a single-package flow into text.
        """

        return self.__renderer.render(flow=Flow(intent="t", package="com.example", nodes=nodes))

    def __target(self, *, text: str) -> Selector:
        """
        Build a plain text selector.
        """

        return Selector(text=text)

    def test_open_app_uses_colon_space(self) -> None:
        """
        Launch renders with the docs colon-space form.
        """

        text = self.__render(nodes=(LaunchNode(package="com.android.chrome", source_steps=(0,)),))
        self.assertIn("OPEN_APP: com.android.chrome", text)

    def test_tap_renders_unquoted_target(self) -> None:
        """
        Tap renders the target as unquoted natural-language text.
        """

        text = self.__render(
            nodes=(TapNode(selector=self.__target(text="Login CTA"), source_steps=(0,)),)
        )
        self.assertIn("Tap on Login CTA\n", text)

    def test_tap_renders_positional_and_container(self) -> None:
        """
        Tap folds an ordinal and container into the unquoted target phrase.
        """

        node = TapNode(
            selector=Selector(text="Add", position="first", container="Snacks header"),
            source_steps=(0,),
        )
        text = self.__render(nodes=(node,))
        self.assertIn("Tap on the first Add under Snacks header", text)

    def test_scroll_until_quotes_target(self) -> None:
        """
        Scroll-until renders direction and a quoted target.
        """

        text = self.__render(
            nodes=(
                ScrollUntilNode(
                    direction=ScrollDirection.DOWN, target="Jars & containers", source_steps=(0,)
                ),
            )
        )
        self.assertIn('Scroll down until "Jars & containers"', text)

    def test_scroll_renders_percentage(self) -> None:
        """
        Scroll renders the percentage form when a percentage is present.
        """

        text = self.__render(
            nodes=(ScrollNode(direction=ScrollDirection.DOWN, percentage=30, source_steps=(0,)),)
        )
        self.assertIn("Scroll down by 30%", text)

    def test_scroll_renders_container(self) -> None:
        """
        Scroll renders the inside-container form when a container is present.
        """

        text = self.__render(
            nodes=(
                ScrollNode(
                    direction=ScrollDirection.DOWN, container="product list", source_steps=(0,)
                ),
            )
        )
        self.assertIn("Scroll down inside product list", text)

    def test_wait_prefers_seconds(self) -> None:
        """
        Wait renders the time form when a duration is present.
        """

        text = self.__render(nodes=(WaitNode(duration=5, source_steps=(0,)),))
        self.assertIn("Wait for 5 seconds", text)

    def test_wait_falls_back_to_subject(self) -> None:
        """
        Wait renders the element form when no duration is present.
        """

        text = self.__render(nodes=(WaitNode(subject="Home screen", source_steps=(0,)),))
        self.assertIn('Wait until "Home screen"', text)

    def test_wait_renders_combined_form(self) -> None:
        """
        Wait renders the combined form when both a duration and a subject are present.
        """

        node = WaitNode(duration=5, subject="page content to load", source_steps=(0,))
        text = self.__render(nodes=(node,))
        self.assertIn("Wait 5 seconds for page content to load", text)

    def test_back_renders_device_keyword(self) -> None:
        """
        Back renders as the device back-button keyword.
        """

        text = self.__render(nodes=(BackNode(source_steps=(0,)),))
        self.assertIn("PRESS_DEVICE_BACK_BUTTON", text)

    def test_multiple_validations_render_grouped_state_once(self) -> None:
        """
        A check with multiple assertions renders the docs grouped form with quoted items.
        """

        text = self.__render(
            nodes=(
                CheckNode(
                    checks=(
                        Check(kind=CheckKind.VISIBLE, subject="cart"),
                        Check(kind=CheckKind.VISIBLE, subject="total"),
                    ),
                    source_steps=(0,),
                ),
            )
        )
        self.assertIn('Validate the following are visible: 1. "cart" 2. "total"', text)

    def test_branch_renders_indented_body(self) -> None:
        """
        A branch renders an IF header, braces, and an indented body.
        """

        text = self.__render(
            nodes=(
                BranchNode(
                    guard=Guard(condition="Overlay is visible", source_step=0),
                    body=(TapNode(selector=self.__target(text="Skip"), source_steps=(0,)),),
                    source_steps=(0,),
                ),
            )
        )
        self.assertIn("IF Overlay is visible", text)
        self.assertIn("\n    Tap on Skip", text)
        self.assertIn("{", text)
        self.assertIn("}", text)

    def test_kill_renders_keyword(self) -> None:
        """
        Kill renders the exact KILL_APP keyword.
        """

        text = self.__render(nodes=(KillNode(source_steps=(0,)),))
        self.assertIn("KILL_APP", text)

    def test_clear_renders_keyword(self) -> None:
        """
        Clear renders the exact CLEAR_APP keyword.
        """

        text = self.__render(nodes=(ClearNode(source_steps=(0,)),))
        self.assertIn("CLEAR_APP", text)

    def test_minimize_renders_keyword(self) -> None:
        """
        Minimize renders the canonical MINIMISE_APP keyword.
        """

        text = self.__render(nodes=(MinimizeNode(source_steps=(0,)),))
        self.assertIn("MINIMISE_APP", text)

    def __typed(self, *, value: str) -> TypeNode:
        """
        Build a type node entering the given quoted value into a field.
        """

        return TypeNode(text=value, field=self.__target(text="search bar"), source_steps=(0,))

    def test_value_with_double_quote_uses_single_quotes(self) -> None:
        """
        A typed value containing a double quote is delimited with single quotes.
        """

        text = self.__render(nodes=(self.__typed(value='He said "Hi"'),))
        self.assertIn("'He said \"Hi\"'", text)

    def test_value_with_double_and_single_uses_backtick(self) -> None:
        """
        A typed value containing both double and single quotes is delimited with backticks.
        """

        text = self.__render(nodes=(self.__typed(value='it\'s a "trap"'),))
        self.assertIn('`it\'s a "trap"`', text)

    def test_value_with_every_quote_type_fails(self) -> None:
        """
        A typed value that contains all three delimiters cannot be rendered and fails explicitly.
        """

        with self.assertRaises(LanguageComplianceError):
            self.__render(nodes=(self.__typed(value="\"'`"),))

    def test_location_renders_set_gps(self) -> None:
        """
        Location renders the SET_GPS coordinate form.
        """

        text = self.__render(
            nodes=(LocationNode(latitude=12.97, longitude=77.59, source_steps=(0,)),)
        )
        self.assertIn("SET_GPS(latitude=12.97, longitude=77.59)", text)

    def test_store_renders_memory_form(self) -> None:
        """
        Store renders the captured value under the variable name.
        """

        text = self.__render(nodes=(StoreNode(value="OTP", name="otp_login", source_steps=(0,)),))
        self.assertIn("Store OTP as otp_login", text)

    def test_map_action_renders_tap(self) -> None:
        """
        Map renders the MAP_ACTION Tap form with an unquoted target.
        """

        text = self.__render(
            nodes=(MapNode(selector=self.__target(text="red location pin"), source_steps=(0,)),)
        )
        self.assertIn("MAP_ACTION Tap on red location pin", text)
