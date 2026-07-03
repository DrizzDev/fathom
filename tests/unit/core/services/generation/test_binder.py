from __future__ import annotations

import unittest

from fathom.constants.flow import CheckKind, LaunchProvenance
from fathom.core.dialect.policy import Policy
from fathom.core.services.generation.binder import LaunchBinder
from fathom.schemas.flow import (
    Check,
    CheckNode,
    Evidence,
    EvidenceStep,
    Flow,
    LaunchNode,
    Selector,
    StepLaunch,
    StepOutcome,
    StepTarget,
    TapNode,
)

PACKAGE = "in.swiggy.android"


class LaunchBinderTest(unittest.TestCase):
    """
    Pins the deterministic launch binder: provenance comes from evidence markers, never the model.
    """

    @staticmethod
    def __evidence() -> Evidence:
        """
        Build a launcher-transition run: grounded launch, one tap, a terminal validation.
        """

        return Evidence(
            intent="order a burger",
            goal="burger added",
            package=PACKAGE,
            steps=(
                EvidenceStep(
                    index=0,
                    event="launch",
                    action="launch",
                    launch=StepLaunch(
                        package=PACKAGE,
                        provenance=LaunchProvenance.LAUNCHER_TRANSITION,
                        source_steps=(0,),
                    ),
                ),
                EvidenceStep(
                    index=1,
                    event="action",
                    action="tap",
                    target=StepTarget(export="Search box"),
                    outcome=StepOutcome(success=True),
                ),
                EvidenceStep(
                    index=2,
                    event="validation",
                    action="complete",
                    target=StepTarget(export="burger"),
                    outcome=StepOutcome(success=True),
                ),
            ),
        )

    @staticmethod
    def __flow_with_ungrounded_launch() -> Flow:
        """
        Build the model's flawed flow: a warm-start launch with no grounding steps.
        """

        return Flow(
            intent="order a burger",
            package=PACKAGE,
            nodes=(
                LaunchNode(package=PACKAGE, source_steps=()),
                TapNode(selector=Selector(text="Search box"), source_steps=(1,)),
                CheckNode(
                    checks=(Check(kind=CheckKind.VISIBLE, subject="burger"),), source_steps=(2,)
                ),
            ),
        )

    def test_binds_launch_provenance_and_steps_from_marker(self) -> None:
        """
        The bound launch node mirrors the marker's package, provenance, and grounding steps.
        """

        bound = LaunchBinder().bind(
            flow=self.__flow_with_ungrounded_launch(), evidence=self.__evidence()
        )

        launch = bound.nodes[0]
        assert isinstance(launch, LaunchNode)
        self.assertEqual(launch.package, PACKAGE)
        self.assertIs(launch.provenance, LaunchProvenance.LAUNCHER_TRANSITION)
        self.assertEqual(launch.source_steps, (0,))

    def test_ungrounded_launch_fails_policy_but_bound_flow_is_clean(self) -> None:
        """
        Swiggy regression: a launch with empty source_steps is Policy-dirty, and clean after binding.
        """

        policy = Policy()
        evidence = self.__evidence()
        flawed = self.__flow_with_ungrounded_launch()

        self.assertTrue(policy.evaluate(flow=flawed, evidence=evidence).issues)

        bound = LaunchBinder().bind(flow=flawed, evidence=evidence)
        self.assertEqual(policy.evaluate(flow=bound, evidence=evidence).issues, ())

    def test_markerless_evidence_leaves_flow_unchanged(self) -> None:
        """
        With no launch markers, the binder returns the flow untouched.
        """

        evidence = Evidence(
            intent="x",
            goal="y",
            package=PACKAGE,
            steps=(
                EvidenceStep(
                    index=0,
                    event="action",
                    action="tap",
                    target=StepTarget(export="Search box"),
                    outcome=StepOutcome(success=True),
                ),
            ),
        )
        flow = Flow(
            intent="x",
            package=PACKAGE,
            nodes=(TapNode(selector=Selector(text="Search box"), source_steps=(0,)),),
        )

        self.assertEqual(LaunchBinder().bind(flow=flow, evidence=evidence), flow)
