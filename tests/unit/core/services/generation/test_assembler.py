from __future__ import annotations

import unittest

from fathom.constants import StepEvent
from fathom.constants.flow import LaunchProvenance
from fathom.core.exceptions import InvariantViolation
from fathom.core.services.generation.assembler import EvidenceAssembler
from fathom.schemas.artifacts import ScreenArtifact, ScreenArtifactBundle, StepArtifacts
from fathom.schemas.capture import Capture, CaptureRequest
from fathom.schemas.generation import LaunchMarker, NormalizedEntry, NormalizedTrace
from fathom.schemas.steps import StepGoal, StepRecord


class EvidenceAssemblerTest(unittest.TestCase):
    """
    Cover mapping a normalized trace into the evidence aggregate.
    """

    def setUp(self) -> None:
        """
        Build a shared assembler.
        """

        self.__assembler = EvidenceAssembler()

    def __trace(self, *entries: NormalizedEntry) -> NormalizedTrace:
        """
        Wrap normalized entries into a trace.
        """

        return NormalizedTrace(entries=tuple(entries))

    def __record(self, **overrides: object) -> StepRecord:
        """
        Build a step record with sensible defaults overridden per case.
        """

        base = {
            "duration": 0,
            "target": "App",
            "success": True,
            "step_number": 0,
            "action_type": "tap",
            "screen_changed": True,
        }
        base.update(overrides)
        return StepRecord(**base)  # type: ignore[arg-type]

    def test_metadata_and_step_count_are_carried(self) -> None:
        """
        Run metadata and the recorded step order are preserved.
        """

        evidence = self.__assembler.assemble(
            goal="home visible",
            package="com.example",
            intent="open and verify",
            trace=self.__trace(
                NormalizedEntry(record=self.__record(step_number=0)),
                NormalizedEntry(record=self.__record(step_number=1, target="Login")),
            ),
        )

        self.assertEqual(len(evidence.steps), 2)
        self.assertEqual(evidence.goal, "home visible")
        self.assertEqual(evidence.package, "com.example")
        self.assertEqual(evidence.intent, "open and verify")

    def test_distillation_outcome_is_carried(self) -> None:
        """
        The partial flag, discarded step numbers, and reason propagate onto the evidence.
        """

        evidence = self.__assembler.assemble(
            intent="i",
            goal="g",
            package="com.example",
            trace=self.__trace(NormalizedEntry(record=self.__record(step_number=0))),
            partial=True,
            discarded=(3, 5),
            reason="loop thrash distilled",
        )

        self.assertTrue(evidence.partial)
        self.assertEqual(evidence.discarded, (3, 5))
        self.assertEqual(evidence.reason, "loop thrash distilled")

    def test_goal_context_is_carried(self) -> None:
        """
        Persisted sub-goal context propagates to evidence for episode-aware authoring.
        """

        goal = StepGoal(
            index=2,
            description="Check whether customer rating is >= 4.2",
            directive="validate",
        )
        evidence = self.__assembler.assemble(
            intent="i",
            goal="g",
            package="com.example",
            trace=self.__trace(NormalizedEntry(record=self.__record(step_number=5, goal=goal))),
        )

        self.assertEqual(evidence.steps[0].goal, goal)

    def test_conditional_fields_are_carried(self) -> None:
        """
        Conditional markers map verbatim onto the evidence step.
        """

        evidence = self.__assembler.assemble(
            intent="i",
            goal="g",
            package="com.example",
            trace=self.__trace(
                NormalizedEntry(
                    record=self.__record(
                        step_number=2,
                        is_conditional=True,
                        conditional_type="blocker",
                        condition="Overlay is visible",
                    )
                )
            ),
        )
        step = evidence.steps[0]

        self.assertTrue(step.guard.conditional)
        self.assertEqual(step.guard.kind, "blocker")
        self.assertEqual(step.guard.condition, "Overlay is visible")

    def test_validation_subject_becomes_script_assertion_target(self) -> None:
        """
        A structured validation subject outranks visible anchor text for authoring.
        """

        evidence = self.__assembler.assemble(
            intent="i",
            goal="g",
            package="com.example",
            trace=self.__trace(
                NormalizedEntry(
                    record=self.__record(
                        step_number=3,
                        action_type="validate",
                        target="Phone Number",
                        natural_language_target="Phone Number",
                        export_target="Phone Number input field",
                        validation_subject="Login screen",
                    )
                )
            ),
        )

        self.assertEqual(evidence.steps[0].target.export, "Login screen")
        self.assertEqual(evidence.steps[0].target.name, "Phone Number")

    def test_validation_event_subject_becomes_script_assertion_target(self) -> None:
        """
        A validation event keeps its assertion subject even when the action is COMPLETE.
        """

        evidence = self.__assembler.assemble(
            intent="i",
            goal="g",
            package="com.example",
            trace=self.__trace(
                NormalizedEntry(
                    record=self.__record(
                        event_type=StepEvent.VALIDATION,
                        step_number=3,
                        action_type="complete",
                        target="Home screen",
                        validation_subject="Home screen is visible",
                    )
                )
            ),
        )

        self.assertEqual(evidence.steps[0].event, StepEvent.VALIDATION)
        self.assertEqual(evidence.steps[0].action, "complete")
        self.assertEqual(evidence.steps[0].target.export, "Home screen is visible")

    def test_validation_description_does_not_backfill_assertion_target(self) -> None:
        """
        Validation targets must come from the structured validation subject or export field.
        """

        evidence = self.__assembler.assemble(
            intent="i",
            goal="g",
            package="com.example",
            trace=self.__trace(
                NormalizedEntry(
                    record=self.__record(
                        step_number=3,
                        action_type="validate",
                        target="Phone Number",
                        natural_language_target="Phone Number",
                        export_target="Phone Number input field",
                        action_description="Validate login screen visibility",
                    )
                )
            ),
        )

        self.assertEqual(evidence.steps[0].target.export, "Phone Number input field")

    def test_planner_target_claim_is_not_verified_without_anchor_match(self) -> None:
        """
        Planner-authored target text remains a claim when it does not match a structured anchor.
        """

        evidence = self.__assembler.assemble(
            intent="i",
            goal="g",
            package="com.example",
            trace=self.__trace(
                NormalizedEntry(
                    record=self.__record(
                        target="Product wrapper text",
                        step_number=4,
                        action_type="tap",
                        export_target="product card",
                        natural_language_target="Product wrapper text",
                        target_element_type="product card",
                    )
                )
            ),
        )
        target = evidence.steps[0].target

        self.assertEqual(target.anchors.accessibility, ("product card",))
        self.assertEqual(target.claim.text, "Product wrapper text")
        self.assertFalse(target.claim.verified)

    def test_planner_target_claim_is_verified_by_structured_anchor_match(self) -> None:
        """
        Planner-authored target text is verified only when it matches a structured anchor.
        """

        evidence = self.__assembler.assemble(
            intent="i",
            goal="g",
            package="com.example",
            trace=self.__trace(
                NormalizedEntry(
                    record=self.__record(
                        target="Search box",
                        step_number=2,
                        action_type="tap",
                        export_target="Search box",
                        natural_language_target="Search box",
                        target_element_type="search field",
                    )
                )
            ),
        )
        target = evidence.steps[0].target

        self.assertEqual(target.anchors.accessibility, ("Search box",))
        self.assertEqual(target.claim.text, "Search box")
        self.assertTrue(target.claim.verified)

    def test_successful_capture_value_is_a_visual_anchor(self) -> None:
        """
        Successful STORE values are visual anchors for value-bearing authoring.
        """

        evidence = self.__assembler.assemble(
            intent="i",
            goal="g",
            package="com.example",
            trace=self.__trace(
                NormalizedEntry(
                    record=self.__record(
                        target="price",
                        step_number=7,
                        action_type="store",
                        capture=Capture.succeeded(name="item_price", value="₹87", step=7),
                        capture_request=CaptureRequest(
                            name="item_price",
                            value="₹87",
                            subject="selected product price",
                        ),
                    )
                )
            ),
        )

        self.assertEqual(evidence.steps[0].target.anchors.visual, ("₹87",))

    def test_structured_step_artifacts_are_carried(self) -> None:
        """
        Persisted step artifacts propagate to evidence for authoring.
        """

        artifacts = StepArtifacts(
            screen=ScreenArtifactBundle(
                before=ScreenArtifact(uri="history://step-1-before.png"),
                after=ScreenArtifact(uri="history://step-1-after.png"),
                annotated=ScreenArtifact(uri="history://step-1-annotated.png"),
                traces=(ScreenArtifact(uri="history://step-1-trace.png"),),
            )
        )

        evidence = self.__assembler.assemble(
            intent="i",
            goal="g",
            package="com.example",
            trace=self.__trace(
                NormalizedEntry(record=self.__record(step_number=1, artifacts=artifacts))
            ),
        )

        self.assertEqual(evidence.steps[0].artifacts, artifacts)

    def test_launch_marker_becomes_a_launch_step(self) -> None:
        """
        A launch entry maps to a launch evidence step carrying the marker's provenance.
        """

        evidence = self.__assembler.assemble(
            intent="i",
            goal="g",
            package="com.shopping.supply",
            trace=self.__trace(
                NormalizedEntry(
                    launch=LaunchMarker(
                        package="com.shopping.supply",
                        provenance=LaunchProvenance.LAUNCHER_TRANSITION,
                        source_steps=(0,),
                    )
                ),
                NormalizedEntry(record=self.__record(step_number=1, target="Login")),
            ),
        )
        launch = evidence.steps[0]

        self.assertEqual(launch.event, "launch")
        self.assertIsNotNone(launch.launch)
        assert launch.launch is not None
        self.assertEqual(launch.launch.package, "com.shopping.supply")
        self.assertEqual(launch.launch.provenance, LaunchProvenance.LAUNCHER_TRANSITION)
        self.assertEqual(launch.launch.source_steps, (0,))

    def test_warm_start_launch_coexists_with_kept_step_zero(self) -> None:
        """
        A synthetic warm-start launch and the kept step-0 record assemble without an index clash.
        """

        evidence = self.__assembler.assemble(
            intent="i",
            goal="g",
            package="com.example",
            trace=self.__trace(
                NormalizedEntry(
                    launch=LaunchMarker(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                        source_steps=(),
                    )
                ),
                NormalizedEntry(record=self.__record(step_number=0)),
                NormalizedEntry(record=self.__record(step_number=1, target="Login")),
            ),
        )

        self.assertEqual(evidence.steps[0].event, "launch")
        self.assertEqual([step.index for step in evidence.steps if step.launch is None], [0, 1])

    def test_evidence_package_is_taken_from_the_first_launch_marker(self) -> None:
        """
        Evidence reports the entered app from the first launch marker, not the passed fallback.
        """

        evidence = self.__assembler.assemble(
            intent="i",
            goal="g",
            package="workflow",
            trace=self.__trace(
                NormalizedEntry(
                    launch=LaunchMarker(
                        package="com.shopping.supply",
                        provenance=LaunchProvenance.LAUNCHER_TRANSITION,
                        source_steps=(0,),
                    )
                ),
                NormalizedEntry(record=self.__record(step_number=1, target="Login")),
            ),
        )

        self.assertEqual(evidence.package, "com.shopping.supply")

    def test_store_record_maps_to_capture_evidence(self) -> None:
        """
        A STORE record's persisted request and outcome surface as combined capture evidence.
        """

        evidence = self.__assembler.assemble(
            intent="i",
            goal="g",
            package="com.example",
            trace=self.__trace(
                NormalizedEntry(
                    record=self.__record(
                        step_number=4,
                        target="store",
                        action_type="store",
                        capture_request=CaptureRequest(name="abc", subject="price", value="₹499"),
                        capture=Capture.succeeded(name="abc", value="₹499", step=4),
                    )
                )
            ),
        )
        step = evidence.steps[0]

        self.assertIsNotNone(step.capture)
        assert step.capture is not None
        self.assertEqual(step.capture.name, "abc")
        self.assertEqual(step.capture.subject, "price")
        self.assertTrue(step.capture.success)
        self.assertEqual(step.capture.value, "₹499")

    def test_failed_store_record_maps_to_failed_capture_evidence(self) -> None:
        """
        A failed STORE outcome remains first-class capture evidence with its failure reason.
        """

        evidence = self.__assembler.assemble(
            intent="i",
            goal="g",
            package="com.example",
            trace=self.__trace(
                NormalizedEntry(
                    record=self.__record(
                        step_number=4,
                        target="store",
                        action_type="store",
                        capture_request=CaptureRequest(name="abc", subject="price", value="₹499"),
                        capture=Capture.failed(name="abc", reason="not visible", step=4),
                    )
                )
            ),
        )
        step = evidence.steps[0]

        self.assertIsNotNone(step.capture)
        assert step.capture is not None
        self.assertEqual(step.capture.name, "abc")
        self.assertEqual(step.capture.subject, "price")
        self.assertFalse(step.capture.success)
        self.assertIsNone(step.capture.value)
        self.assertEqual(step.capture.reason, "not visible")

    def test_capture_name_mismatch_is_rejected(self) -> None:
        """
        Capture request and outcome names must agree before evidence is exposed.
        """

        with self.assertRaises(InvariantViolation):
            self.__assembler.assemble(
                intent="i",
                goal="g",
                package="com.example",
                trace=self.__trace(
                    NormalizedEntry(
                        record=self.__record(
                            step_number=4,
                            target="store",
                            action_type="store",
                            capture_request=CaptureRequest(
                                name="item_price", subject="price", value="₹499"
                            ),
                            capture=Capture.succeeded(name="coupon", value="₹499", step=4),
                        )
                    )
                ),
            )

    def test_capture_step_mismatch_is_rejected(self) -> None:
        """
        Capture outcome provenance must point at the record that carries it.
        """

        with self.assertRaises(InvariantViolation):
            self.__assembler.assemble(
                intent="i",
                goal="g",
                package="com.example",
                trace=self.__trace(
                    NormalizedEntry(
                        record=self.__record(
                            step_number=4,
                            target="store",
                            action_type="store",
                            capture_request=CaptureRequest(
                                name="item_price", subject="price", value="₹499"
                            ),
                            capture=Capture.succeeded(name="item_price", value="₹499", step=3),
                        )
                    )
                ),
            )

    def test_capture_value_mismatch_is_rejected(self) -> None:
        """
        Successful STORE evidence must preserve the exact requested captured value.
        """

        with self.assertRaises(InvariantViolation):
            self.__assembler.assemble(
                intent="i",
                goal="g",
                package="com.example",
                trace=self.__trace(
                    NormalizedEntry(
                        record=self.__record(
                            step_number=4,
                            target="store",
                            action_type="store",
                            capture_request=CaptureRequest(
                                name="item_price", subject="price", value="₹499"
                            ),
                            capture=Capture.succeeded(name="item_price", value="₹599", step=4),
                        )
                    )
                ),
            )

    def test_capture_on_non_store_record_is_rejected(self) -> None:
        """
        Capture evidence may only appear on real STORE records.
        """

        with self.assertRaises(InvariantViolation):
            self.__assembler.assemble(
                intent="i",
                goal="g",
                package="com.example",
                trace=self.__trace(
                    NormalizedEntry(
                        record=self.__record(
                            step_number=4,
                            target="tap",
                            action_type="tap",
                            capture_request=CaptureRequest(
                                name="item_price", subject="price", value="₹499"
                            ),
                            capture=Capture.succeeded(name="item_price", value="₹499", step=4),
                        )
                    )
                ),
            )

    def test_capture_outcome_without_request_is_rejected(self) -> None:
        """
        A capture outcome without the matching STORE request is malformed evidence.
        """

        with self.assertRaises(InvariantViolation):
            self.__assembler.assemble(
                intent="i",
                goal="g",
                package="com.example",
                trace=self.__trace(
                    NormalizedEntry(
                        record=self.__record(
                            step_number=4,
                            target="store",
                            action_type="store",
                            capture=Capture.succeeded(name="item_price", value="₹499", step=4),
                        )
                    )
                ),
            )

    def test_non_store_record_has_no_capture_evidence(self) -> None:
        """
        A record without a capture request produces no capture evidence.
        """

        evidence = self.__assembler.assemble(
            intent="i",
            goal="g",
            package="com.example",
            trace=self.__trace(NormalizedEntry(record=self.__record(step_number=0))),
        )

        self.assertIsNone(evidence.steps[0].capture)

    def test_capture_request_without_outcome_is_rejected(self) -> None:
        """
        A STORE request without a recorded outcome is malformed evidence.
        """

        with self.assertRaises(InvariantViolation):
            self.__assembler.assemble(
                intent="i",
                goal="g",
                package="com.example",
                trace=self.__trace(
                    NormalizedEntry(
                        record=self.__record(
                            step_number=4,
                            target="store",
                            action_type="store",
                            capture_request=CaptureRequest(
                                name="abc", subject="price", value="₹499"
                            ),
                        )
                    )
                ),
            )
