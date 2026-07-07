from __future__ import annotations

import unittest
from typing import Tuple

from fathom.adapters.dialect.drizz.parser import DrizzLarkParser
from fathom.constants.flow import CheckKind, ScrollDirection
from fathom.core.dialect.drizz.check import Checker
from fathom.core.dialect.drizz.print import CanonicalPrinter
from fathom.core.dialect.drizz.render import Renderer
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


class RoundTripTest(unittest.TestCase):
    """
    Cover that real renderer output round-trips through the parser and printer.
    """

    def setUp(self) -> None:
        """
        Build the renderer and a checker backed by the parser and printer.
        """

        self.__renderer = Renderer()
        self.__checker = Checker(parser=DrizzLarkParser(), printer=CanonicalPrinter())

    def __check(self, *, nodes: Tuple[FlowNode, ...]) -> Tuple[str, ...]:
        """
        Render a flow then return the issue codes raised when checking it.
        """

        text = self.__renderer.render(flow=Flow(intent="t", package="com.example", nodes=nodes))
        return tuple(str(issue.code) for issue in self.__checker.check(text=text).issues)

    def test_every_command_form_round_trips(self) -> None:
        """
        A flow exercising every command form renders to text that re-parses cleanly.
        """

        steps = (0,)
        nodes: Tuple[FlowNode, ...] = (
            BackNode(source_steps=steps),
            KillNode(source_steps=steps),
            ClearNode(source_steps=steps),
            MinimizeNode(source_steps=steps),
            WaitNode(duration=5, source_steps=steps),
            WaitNode(subject="Spinner", source_steps=steps),
            WaitNode(duration=5, subject="page content to load", source_steps=steps),
            LaunchNode(package="com.example", source_steps=steps),
            MapNode(selector=Selector(text="Pin"), source_steps=steps),
            TapNode(selector=Selector(text="Login"), source_steps=steps),
            ScrollNode(direction=ScrollDirection.DOWN, source_steps=steps),
            ScrollNode(direction=ScrollDirection.DOWN, percentage=30, source_steps=steps),
            ScrollNode(direction=ScrollDirection.UP, container="product list", source_steps=steps),
            LocationNode(latitude=12.34, longitude=-56.78, source_steps=steps),
            StoreNode(value="cart total", name="savedTotal", source_steps=steps),
            TapNode(selector=Selector(text="Add", container="Cart"), source_steps=steps),
            TapNode(selector=Selector(text="Result", position="first"), source_steps=steps),
            TypeNode(text="hello world", field=Selector(text="Search box"), source_steps=steps),
            ScrollUntilNode(direction=ScrollDirection.DOWN, target="Load more", source_steps=steps),
            BranchNode(
                body=(TapNode(selector=Selector(text="Skip"), source_steps=steps),),
                guard=Guard(condition="Overlay is visible", source_step=0),
                source_steps=steps,
            ),
            CheckNode(
                checks=(Check(kind=CheckKind.VISIBLE, subject="home screen"),), source_steps=steps
            ),
            CheckNode(
                checks=(
                    Check(kind=CheckKind.PRESENT, subject="home"),
                    Check(kind=CheckKind.PRESENT, subject="cart icon"),
                ),
                source_steps=steps,
            ),
        )
        self.assertEqual(self.__check(nodes=nodes), ())

    def test_mixed_state_validation_preserves_each_state(self) -> None:
        """
        A grouped check with differing states renders separate lines keeping every state.
        """

        renderer = Renderer()
        nodes: Tuple[FlowNode, ...] = (
            CheckNode(
                checks=(
                    Check(kind=CheckKind.VISIBLE, subject="Home"),
                    Check(kind=CheckKind.ENABLED, subject="Apply CTA"),
                ),
                source_steps=(0,),
            ),
        )
        text = renderer.render(flow=Flow(intent="t", package="com.example", nodes=nodes))

        self.assertIn("Validate Home is visible", text)
        self.assertIn("Validate Apply CTA is enabled", text)
        self.assertEqual(self.__check(nodes=nodes), ())

    def test_validation_subject_with_embedded_single_quotes_round_trips(self) -> None:
        """
        A validation subject containing single quotes renders in a parser-safe form.
        """

        nodes: Tuple[FlowNode, ...] = (
            CheckNode(
                checks=(
                    Check(
                        kind=CheckKind.VISIBLE,
                        subject="'Continue with' phone number selection dialog",
                    ),
                ),
                source_steps=(0,),
            ),
        )
        text = self.__renderer.render(flow=Flow(intent="t", package="com.example", nodes=nodes))

        self.assertIn(
            "Validate \"'Continue with' phone number selection dialog\" is visible",
            text,
        )
        self.assertEqual(self.__check(nodes=nodes), ())

    def test_validation_subject_with_state_phrase_round_trips_once(self) -> None:
        """
        A validation subject carrying the state phrase prints the state only once.
        """

        nodes: Tuple[FlowNode, ...] = (
            CheckNode(
                checks=(Check(kind=CheckKind.VISIBLE, subject="cart is visible"),),
                source_steps=(0,),
            ),
        )
        text = self.__renderer.render(flow=Flow(intent="t", package="com.example", nodes=nodes))

        self.assertIn("Validate cart is visible", text)
        self.assertNotIn('"cart is visible" is visible', text)
        self.assertEqual(self.__check(nodes=nodes), ())

    def test_wait_subject_with_embedded_single_quotes_round_trips(self) -> None:
        """
        A wait subject containing single quotes renders in a parser-safe form.
        """

        nodes: Tuple[FlowNode, ...] = (
            WaitNode(subject="'Continue with' dialog", source_steps=(0,)),
        )
        text = self.__renderer.render(flow=Flow(intent="t", package="com.example", nodes=nodes))

        self.assertIn("Wait until \"'Continue with' dialog\"", text)
        self.assertEqual(self.__check(nodes=nodes), ())

    def test_typed_value_with_embedded_quotes_round_trips(self) -> None:
        """
        A typed value containing double and single quotes renders and re-parses cleanly.
        """

        nodes: Tuple[FlowNode, ...] = (
            TypeNode(text='it\'s a "trap"', field=Selector(text="note field"), source_steps=(0,)),
            CheckNode(checks=(Check(kind=CheckKind.VISIBLE, subject="home"),), source_steps=(0,)),
        )
        self.assertEqual(self.__check(nodes=nodes), ())

    def test_typed_value_with_every_delimiter_round_trips(self) -> None:
        """
        A typed value containing every string delimiter renders with escapes and parses cleanly.
        """

        nodes: Tuple[FlowNode, ...] = (
            TypeNode(text="\"'`", field=Selector(text="note field"), source_steps=(0,)),
            CheckNode(checks=(Check(kind=CheckKind.VISIBLE, subject="home"),), source_steps=(0,)),
        )
        self.assertEqual(self.__check(nodes=nodes), ())

    def test_wait_subject_with_line_break_round_trips(self) -> None:
        """
        A wait subject containing line breaks renders as one escaped Drizz line.
        """

        nodes: Tuple[FlowNode, ...] = (
            WaitNode(subject="results\nare visible", source_steps=(0,)),
            CheckNode(checks=(Check(kind=CheckKind.VISIBLE, subject="home"),), source_steps=(0,)),
        )
        self.assertEqual(self.__check(nodes=nodes), ())

    def test_branch_condition_with_line_break_round_trips(self) -> None:
        """
        A branch guard containing line breaks renders as one parser-safe IF header.
        """

        nodes: Tuple[FlowNode, ...] = (
            BranchNode(
                guard=Guard(condition="overlay\nis visible", source_step=0),
                body=(TapNode(selector=Selector(text="Close"), source_steps=(0,)),),
                source_steps=(0,),
            ),
        )
        text = self.__renderer.render(flow=Flow(intent="t", package="com.example", nodes=nodes))

        self.assertIn("IF overlay is visible", text)
        self.assertEqual(self.__check(nodes=nodes), ())
