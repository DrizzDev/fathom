from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Optional, Tuple

from fathom.constants import StepEvent
from fathom.core.services.generation.distiller import Distiller
from fathom.schemas.steps import StepGoal, StepRecord


class DistillerTest(unittest.TestCase):
    """
    Cover recovery removal, region-scoped collapse, and success-aware partial flagging.
    """

    def setUp(self) -> None:
        """
        Build a shared distiller.
        """

        self.__distiller = Distiller()

    def __fixture(self) -> Path:
        """
        Return the committed Shopping loop run fixture path.
        """

        return Path("assets/history/2026-06-09/loop-run/history__com.example.shop.json")

    def __recovery_steps(self) -> Tuple[int, ...]:
        """
        Return the recorded recovery step numbers in the Shopping loop fixture.
        """

        return (10, 14, 15, 27, 28)

    def __record(
        self,
        *,
        number: int,
        action: str = "tap",
        target: str = "Element",
        event: StepEvent = StepEvent.ACTION,
        condition: Optional[str] = None,
        rationale: Optional[str] = None,
        goal: Optional[StepGoal] = None,
        success: bool = True,
    ) -> StepRecord:
        """
        Build a minimal step record with the fields the distiller reads.
        """

        return StepRecord(
            step_number=number,
            event_type=event,
            action_type=action,
            target=target,
            natural_language_target=target,
            condition=condition,
            rationale=rationale,
            success=success,
            screen_changed=True,
            duration=0,
            goal=goal,
        )

    def __recovery(self, *, number: int) -> StepRecord:
        """
        Build a recovery step the planner took to break a loop.
        """

        return self.__record(
            number=number,
            action="back",
            target="system: back",
            condition="recovery",
            rationale="Loop detected (screen repeating). Forcing BACK.",
        )

    def __numbers(self, *, records: Tuple[StepRecord, ...]) -> Tuple[int, ...]:
        """
        Return the step numbers of the given records.
        """

        return tuple(record.step_number for record in records)

    def test_drops_recovery_steps_by_condition_and_rationale(self) -> None:
        """
        Steps marked recovery by condition or a loop rationale are removed.
        """

        records = (
            self.__record(number=0, target="Categories"),
            self.__recovery(number=1),
            self.__record(
                number=2,
                action="scroll",
                target="system: scroll",
                rationale="Loop detected (screen repeating). Forcing SCROLL.",
            ),
            self.__record(number=3, target="Home"),
        )

        result = self.__distiller.distill(records=records)

        self.assertEqual(self.__numbers(records=result.records), (0, 3))
        self.assertEqual(result.discarded, (1, 2))

    def test_drops_incidental_surface_dismissal(self) -> None:
        """
        A planner close/dismiss action outside the active goal is recovery, not replayable script.
        """

        records = (
            self.__record(number=0, target="Add"),
            self.__record(
                number=1,
                target="Close",
                rationale="Closing the quantity sheet to go back to search results.",
                goal=StepGoal(
                    index=2,
                    description="Search and add 1 diet coke to cart",
                    directive="tap",
                ),
            ),
            self.__record(number=2, target="Cart"),
        )

        result = self.__distiller.distill(records=records)

        self.assertEqual(self.__numbers(records=result.records), (0, 2))
        self.assertEqual(result.discarded, (1,))

    def test_keeps_intentional_surface_dismissal(self) -> None:
        """
        A close action remains when closing is itself the active goal.
        """

        records = (
            self.__record(
                number=0,
                target="Close",
                rationale="Closing the dialog requested by the user.",
                goal=StepGoal(index=0, description="Close the dialog", directive="tap"),
            ),
        )

        result = self.__distiller.distill(records=records)

        self.assertEqual(self.__numbers(records=result.records), (0,))
        self.assertEqual(result.discarded, ())

    def test_does_not_collapse_repeats_across_a_single_recovery(self) -> None:
        """
        Repeats straddling one recovery are legitimate and must be preserved.
        """

        records = (
            self.__record(number=0, target="Plus"),
            self.__recovery(number=1),
            self.__record(number=2, target="Plus"),
            self.__record(number=3, target="Plus"),
        )

        result = self.__distiller.distill(records=records)

        self.assertEqual(self.__numbers(records=result.records), (0, 2, 3))

    def test_does_not_collapse_repeated_taps_inside_recovery_region(self) -> None:
        """
        Identical taps between two recoveries are semantic actions and must be preserved.
        """

        records = (
            self.__recovery(number=0),
            self.__record(number=1, target="Plus"),
            self.__record(number=2, target="Plus"),
            self.__recovery(number=3),
        )

        result = self.__distiller.distill(records=records)

        self.assertEqual(self.__numbers(records=result.records), (1, 2))

    def test_preserves_repeats_when_no_loop_occurred(self) -> None:
        """
        Without any recovery marker, legitimate repeats are kept intact.
        """

        records = (
            self.__record(number=0, target="Plus"),
            self.__record(number=1, target="Plus"),
            self.__record(number=2, target="Plus"),
        )

        result = self.__distiller.distill(records=records)

        self.assertEqual(self.__numbers(records=result.records), (0, 1, 2))

    def test_preserves_trailing_no_screen_change_action(self) -> None:
        """
        A normal executed action remains scriptable even when it records no screen change.
        """

        trailing = self.__record(
            number=1,
            condition="Overlay is visible",
            target="Dismiss",
        ).model_copy(update={"screen_changed": False})
        records = (
            self.__record(number=0, target="Home"),
            trailing,
        )

        result = self.__distiller.distill(records=records)

        self.assertEqual(self.__numbers(records=result.records), (0, 1))

    def test_collapses_only_consecutive_scrolls_inside_recovery_region(self) -> None:
        """
        Only back-to-back scrolls between two recoveries collapse; a tap breaks the run.
        """

        records = (
            self.__recovery(number=0),
            self.__record(number=1, action="swipe_up", target="grid one"),
            self.__record(number=2, action="swipe_up", target="grid two"),
            self.__record(number=3, target="Product"),
            self.__record(number=4, action="swipe_up", target="grid three"),
            self.__recovery(number=5),
        )

        result = self.__distiller.distill(records=records)

        self.assertEqual(self.__numbers(records=result.records), (1, 3, 4))

    def test_does_not_merge_regions_through_repeated_clean_work(self) -> None:
        """
        Repeated targets in separate recovery intervals are not collapsed across them.
        """

        records = (
            self.__recovery(number=0),
            self.__record(number=1, target="Loop A"),
            self.__record(number=2, target="Plus"),
            self.__record(number=3, target="Plus"),
            self.__recovery(number=4),
            self.__record(number=5, target="Loop B"),
            self.__record(number=6, target="Loop B"),
            self.__recovery(number=7),
        )

        result = self.__distiller.distill(records=records)

        self.assertEqual(self.__numbers(records=result.records), (1, 2, 3, 5, 6))

    def test_does_not_collapse_same_target_on_different_screens(self) -> None:
        """
        The same target across different recovery intervals stays as distinct actions.
        """

        records = (
            self.__recovery(number=0),
            self.__record(number=1, target="Continue"),
            self.__record(number=2, target="Payment"),
            self.__recovery(number=3),
            self.__record(number=4, target="Continue"),
            self.__record(number=5, target="Confirmation"),
            self.__recovery(number=6),
        )

        result = self.__distiller.distill(records=records)

        self.assertEqual(self.__numbers(records=result.records), (1, 2, 4, 5))

    def test_never_collapses_typed_input_inside_the_region(self) -> None:
        """
        Typed input is protected from collapse even inside a loop region.
        """

        records = (
            self.__recovery(number=0),
            self.__record(number=1, action="type", target="Search"),
            self.__record(number=2, action="type", target="Search"),
            self.__recovery(number=3),
        )

        result = self.__distiller.distill(records=records)

        self.assertEqual(self.__numbers(records=result.records), (1, 2))

    def test_flags_partial_when_no_validation_survives(self) -> None:
        """
        A run with no recorded validation step is marked partial with a reason.
        """

        result = self.__distiller.distill(records=(self.__record(number=0, target="Categories"),))

        self.assertTrue(result.partial)
        self.assertIsNotNone(result.reason)

    def test_flags_partial_when_validation_failed(self) -> None:
        """
        A run whose only validation did not succeed is partial.
        """

        records = (
            self.__record(number=0, target="A"),
            self.__record(
                number=1,
                event=StepEvent.VALIDATION,
                action="complete",
                success=False,
            ),
        )

        result = self.__distiller.distill(records=records)

        self.assertTrue(result.partial)

    def test_not_partial_when_validation_succeeded(self) -> None:
        """
        A run that ends in a successful recorded validation is not partial.
        """

        records = (
            self.__record(number=0, target="A"),
            self.__record(
                number=1,
                event=StepEvent.VALIDATION,
                action="complete",
                success=True,
            ),
        )

        result = self.__distiller.distill(records=records)

        self.assertFalse(result.partial)
        self.assertIsNone(result.reason)

    def test_shopping_loop_fixture_is_distilled(self) -> None:
        """
        The recorded Shopping loop drops recovery, collapses inside the region, and flags partial.
        """

        if not self.__fixture().exists():
            self.skipTest("Shopping loop history fixture absent (fixtures are gitignored).")

        payload = json.loads(self.__fixture().read_text())
        records = tuple(StepRecord.model_validate(item) for item in payload["history"])

        result = self.__distiller.distill(records=records)
        kept = self.__numbers(records=result.records)
        recovery_steps = self.__recovery_steps()

        self.assertTrue(result.partial)
        for step in recovery_steps:
            self.assertIn(step, result.discarded)
        self.assertTrue(all(record.condition != "recovery" for record in result.records))
        self.assertLess(len(result.records), len(records))
        self.assertIn(2, kept)
        self.assertGreater(len(result.discarded), len(recovery_steps))
