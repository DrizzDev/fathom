from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.constants.flow import CheckKind, IssueCode, ScrollDirection
from fathom.schemas.flow import (
    BackNode,
    BranchNode,
    Check,
    CheckNode,
    Evidence,
    EvidenceStep,
    Flow,
    Issue,
    LaunchNode,
    Report,
    ScrollUntilNode,
    Selector,
    TapNode,
    TypeNode,
    WaitNode,
)


class NodeTest(unittest.TestCase):
    """
    Cover node provenance defaults and immutability.
    """

    def test_source_steps_required(self) -> None:
        """
        A node without evidence provenance is rejected at construction.
        """

        with self.assertRaises(ValidationError):
            BackNode()  # type: ignore[call-arg]

    def test_node_is_frozen(self) -> None:
        """
        Nodes reject mutation after construction.
        """

        node = BackNode(source_steps=(0,))
        with self.assertRaises(ValidationError):
            node.source_steps = (1,)

    def test_extra_fields_are_forbidden(self) -> None:
        """
        Unknown fields are rejected at construction.
        """

        with self.assertRaises(ValidationError):
            BackNode(unexpected="x")  # type: ignore[call-arg]


class NodeKindValidationTest(unittest.TestCase):
    """
    Cover per-kind required-field validation on typed nodes.
    """

    def test_tap_requires_selector(self) -> None:
        """
        A tap node without a selector is invalid.
        """

        with self.assertRaises(ValidationError):
            TapNode()  # type: ignore[call-arg]

    def test_launch_requires_package(self) -> None:
        """
        A launch node without a package is invalid.
        """

        with self.assertRaises(ValidationError):
            LaunchNode()  # type: ignore[call-arg]

    def test_scroll_until_requires_target(self) -> None:
        """
        A scroll-until node without a target is invalid.
        """

        with self.assertRaises(ValidationError):
            ScrollUntilNode(direction=ScrollDirection.DOWN)  # type: ignore[call-arg]

    def test_wait_keeps_both_duration_and_subject(self) -> None:
        """
        A wait node given both a duration and a subject keeps both.
        """

        node = WaitNode(duration=5, subject="home", source_steps=(1,))
        self.assertEqual(node.duration, 5)
        self.assertEqual(node.subject, "home")

    def test_wait_requires_at_least_one_form(self) -> None:
        """
        A wait node with neither a duration nor a subject is invalid.
        """

        with self.assertRaises(ValidationError):
            WaitNode(source_steps=(1,))

    def test_wait_rejects_empty_subject(self) -> None:
        """
        A wait node with an empty subject is invalid.
        """

        with self.assertRaises(ValidationError):
            WaitNode(subject="")

    def test_type_requires_non_empty_text(self) -> None:
        """
        A type node with empty text is invalid.
        """

        with self.assertRaises(ValidationError):
            TypeNode(text="", field=Selector(text="Search"))

    def test_check_requires_assertions(self) -> None:
        """
        A check node with no assertions is invalid.
        """

        with self.assertRaises(ValidationError):
            CheckNode()  # type: ignore[call-arg]

    def test_branch_requires_guard_and_body(self) -> None:
        """
        A branch node without a guard and body is invalid.
        """

        with self.assertRaises(ValidationError):
            BranchNode()  # type: ignore[call-arg]


class FlowTest(unittest.TestCase):
    """
    Cover flow construction with typed nodes.
    """

    def test_flow_holds_ordered_nodes(self) -> None:
        """
        A flow preserves the order of its nodes.
        """

        flow = Flow(
            intent="open and verify",
            package="com.example",
            nodes=(
                LaunchNode(package="com.example", source_steps=(0,)),
                CheckNode(
                    checks=(Check(kind=CheckKind.VISIBLE, subject="home screen"),),
                    source_steps=(1,),
                ),
            ),
        )
        self.assertIsInstance(flow.nodes[0], LaunchNode)
        self.assertIsInstance(flow.nodes[-1], CheckNode)


class ReportTest(unittest.TestCase):
    """
    Cover the report pass/fail property.
    """

    def test_empty_report_is_ok(self) -> None:
        """
        A report with no issues passes.
        """

        self.assertTrue(Report().ok)

    def test_report_with_issue_is_not_ok(self) -> None:
        """
        A report carrying any issue fails.
        """

        report = Report(issues=(Issue(code=IssueCode.SYNTAX_ERROR, message="bad"),))
        self.assertFalse(report.ok)


class EvidenceTest(unittest.TestCase):
    """
    Cover evidence aggregate construction.
    """

    def test_steps_preserve_order_and_count(self) -> None:
        """
        Evidence preserves the order and count of its recorded steps.
        """

        evidence = Evidence(
            intent="search",
            goal="results visible",
            package="com.example",
            steps=(
                EvidenceStep(index=0, event="action", action="tap"),
                EvidenceStep(index=1, event="validation", action="validate"),
            ),
        )
        self.assertEqual(len(evidence.steps), 2)
        self.assertEqual(evidence.steps[1].action, "validate")

    def test_duplicate_step_numbers_are_rejected(self) -> None:
        """
        Evidence rejects steps that reuse a step number.
        """

        with self.assertRaises(ValidationError):
            Evidence(
                intent="search",
                goal="results visible",
                package="com.example",
                steps=(
                    EvidenceStep(index=0, event="action", action="tap"),
                    EvidenceStep(index=0, event="action", action="tap"),
                ),
            )


class SelectorTest(unittest.TestCase):
    """
    Cover selector validation.
    """

    def test_text_is_required_non_empty(self) -> None:
        """
        A selector rejects empty target text.
        """

        with self.assertRaises(ValidationError):
            Selector(text="")
