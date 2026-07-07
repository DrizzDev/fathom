from __future__ import annotations

import unittest

from fathom.authoring.application.materializer import AuthoringMaterializer
from fathom.constants.flow import AssertionSource, CheckKind
from fathom.schemas.flow import (
    Check,
    CheckNode,
    CompletionAssertion,
    Evidence,
    EvidenceStep,
    Flow,
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
