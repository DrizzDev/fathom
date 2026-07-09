from __future__ import annotations

import unittest

from fathom.authoring.application.materializer import AuthoringMaterializer
from fathom.constants.flow import AssertionSource, CheckKind, LaunchProvenance
from fathom.schemas.flow import (
    BranchNode,
    Check,
    CheckNode,
    CompletionAssertion,
    Evidence,
    EvidenceStep,
    Flow,
    LaunchNode,
    Selector,
    StepGuard,
    StepLaunch,
    TapNode,
)


class AuthoringMaterializerTest(unittest.TestCase):
    """
    Cover deterministic metadata repair for authored flows.
    """

    def test_assertion_check_uses_valid_fallback_source_step(self) -> None:
        """
        Assertion-backed checks replace verifier-only step indexes with executable provenance.
        """

        flow = Flow(
            intent="checkout",
            package="com.example",
            nodes=(
                CheckNode(
                    source_steps=(22,),
                    assertion_ids=("terminal.cart",),
                    checks=(Check(kind=CheckKind.VISIBLE, subject="Cart screen"),),
                ),
            ),
        )
        evidence = Evidence(
            intent="checkout",
            goal="cart visible",
            package="com.example",
            steps=(
                EvidenceStep(index=20, event="validation", action="validate"),
                EvidenceStep(index=21, event="action", action="store"),
            ),
            assertions=(
                CompletionAssertion(
                    id="terminal.cart",
                    kind=CheckKind.VISIBLE,
                    source=AssertionSource.VERIFICATION,
                    subject="Cart screen",
                    step_index=22,
                ),
            ),
        )

        result = AuthoringMaterializer().materialize(flow=flow, evidence=evidence)
        node = result.nodes[0]

        self.assertIsInstance(node, CheckNode)
        assert isinstance(node, CheckNode)
        self.assertEqual(node.source_steps, (21,))

    def test_launch_uses_single_matching_evidence_marker(self) -> None:
        """
        Launch metadata is canonicalized from normalized evidence when the package is unambiguous.
        """

        flow = Flow(
            intent="login",
            package="com.example",
            nodes=(
                LaunchNode(
                    package="com.example",
                    provenance=LaunchProvenance.LAUNCHER_TRANSITION,
                ),
            ),
        )
        evidence = Evidence(
            intent="login",
            goal="login",
            package="com.example",
            steps=(
                EvidenceStep(
                    index=0,
                    event="launch",
                    action="open_app",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                    ),
                ),
            ),
        )

        result = AuthoringMaterializer().materialize(flow=flow, evidence=evidence)
        node = result.nodes[0]

        self.assertIsInstance(node, LaunchNode)
        assert isinstance(node, LaunchNode)
        self.assertIs(node.provenance, LaunchProvenance.SYNTHETIC_WARM_START)
        self.assertEqual(node.source_steps, ())

    def test_repeated_package_launches_use_ordered_evidence_markers(self) -> None:
        """
        Multi-app flows can return to a package and still materialize each launch by order.
        """

        flow = Flow(
            intent="switch apps",
            package="com.example",
            nodes=(
                LaunchNode(package="com.example"),
                LaunchNode(package="com.other"),
                LaunchNode(package="com.example"),
            ),
        )
        evidence = Evidence(
            intent="switch apps",
            goal="switch apps",
            package="com.example",
            steps=(
                EvidenceStep(
                    index=0,
                    event="launch",
                    action="open_app",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                    ),
                ),
                EvidenceStep(
                    index=4,
                    event="launch",
                    action="open_app",
                    launch=StepLaunch(
                        package="com.other",
                        provenance=LaunchProvenance.LAUNCHER_TRANSITION,
                        source_steps=(4,),
                    ),
                ),
                EvidenceStep(
                    index=9,
                    event="launch",
                    action="open_app",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.LAUNCHER_TRANSITION,
                        source_steps=(9,),
                    ),
                ),
            ),
        )

        result = AuthoringMaterializer().materialize(flow=flow, evidence=evidence)

        self.assertEqual(
            tuple(node.source_steps for node in result.nodes if isinstance(node, LaunchNode)),
            ((), (4,), (9,)),
        )

    def test_extra_duplicate_launch_without_marker_is_not_materialized(self) -> None:
        """
        A duplicate authored launch is left ungrounded when evidence has no matching marker.
        """

        flow = Flow(
            intent="open app",
            package="com.example",
            nodes=(
                LaunchNode(package="com.example"),
                LaunchNode(
                    package="com.example",
                    provenance=LaunchProvenance.LAUNCHER_TRANSITION,
                ),
            ),
        )
        evidence = Evidence(
            intent="open app",
            goal="open app",
            package="com.example",
            steps=(
                EvidenceStep(
                    index=0,
                    event="launch",
                    action="open_app",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                    ),
                ),
            ),
        )

        result = AuthoringMaterializer().materialize(flow=flow, evidence=evidence)
        launches = tuple(node for node in result.nodes if isinstance(node, LaunchNode))

        self.assertEqual(launches[0].source_steps, ())
        self.assertIs(launches[1].provenance, LaunchProvenance.LAUNCHER_TRANSITION)
        self.assertEqual(launches[1].source_steps, ())

    def test_unmatched_launch_does_not_consume_next_marker(self) -> None:
        """
        A wrong authored launch cannot steal provenance from the next matching launch.
        """

        flow = Flow(
            intent="switch apps",
            package="com.example",
            nodes=(
                LaunchNode(package="com.unrelated"),
                LaunchNode(package="com.example"),
            ),
        )
        evidence = Evidence(
            intent="switch apps",
            goal="switch apps",
            package="com.example",
            steps=(
                EvidenceStep(
                    index=3,
                    event="launch",
                    action="open_app",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.LAUNCHER_TRANSITION,
                        source_steps=(3,),
                    ),
                ),
            ),
        )

        result = AuthoringMaterializer().materialize(flow=flow, evidence=evidence)
        launches = tuple(node for node in result.nodes if isinstance(node, LaunchNode))

        self.assertEqual(launches[0].source_steps, ())
        self.assertEqual(launches[1].source_steps, (3,))

    def test_conditional_nodes_are_wrapped_from_evidence_guard(self) -> None:
        """
        Bare nodes citing a conditional step are wrapped before policy review.
        """

        flow = Flow(
            intent="login",
            package="com.example",
            nodes=(
                TapNode(
                    source_steps=(5,),
                    selector=Selector(text="Not now button"),
                ),
                CheckNode(
                    source_steps=(5,),
                    checks=(Check(kind=CheckKind.VISIBLE, subject="Login screen"),),
                ),
            ),
        )
        evidence = Evidence(
            intent="login",
            goal="login",
            package="com.example",
            steps=(
                EvidenceStep(
                    index=5,
                    event="action",
                    action="tap",
                    guard=StepGuard(
                        conditional=True,
                        condition="Not now popup is visible",
                    ),
                ),
            ),
        )

        result = AuthoringMaterializer().materialize(flow=flow, evidence=evidence)
        branch = result.nodes[0]

        self.assertEqual(len(result.nodes), 1)
        self.assertIsInstance(branch, BranchNode)
        assert isinstance(branch, BranchNode)
        self.assertEqual(branch.guard.condition, "Not now popup is visible")
        self.assertEqual(branch.guard.source_step, 5)
        self.assertEqual(len(branch.body), 2)

    def test_non_conditional_nodes_are_not_wrapped(self) -> None:
        """
        Ordinary authored nodes stay top-level.
        """

        flow = Flow(
            intent="login",
            package="com.example",
            nodes=(
                TapNode(
                    source_steps=(1,),
                    selector=Selector(text="Continue button"),
                ),
            ),
        )
        evidence = Evidence(
            intent="login",
            goal="login",
            package="com.example",
            steps=(EvidenceStep(index=1, event="action", action="tap"),),
        )

        result = AuthoringMaterializer().materialize(flow=flow, evidence=evidence)

        self.assertIsInstance(result.nodes[0], TapNode)
