from __future__ import annotations

import unittest

from tests.integration.healing._fixtures import FixtureLoader


class HealingFixtureContractTest(unittest.TestCase):
    """
    Contract-level pins for the five Phase 12 fixtures.

    These assertions verify that each fixture's frozen ``expected.json``
    encodes the regression class it is meant to pin. They are NOT a
    replay through the runtime graph — that work lives in
    :class:`HealingReplayAcceptanceTest` and is gated until the
    full IntentNodeProvider stub harness lands (audit entries 036-046).
    """

    def test_overlay_scrim_pins_pixel_overlay_block(self) -> None:
        """
        overlay_scrim fixture must pin the overlay-still-present block + overlay strategy.
        """

        trace = FixtureLoader.load(name="overlay_scrim")
        self.assertIn("overlay_still_present", trace.expected.block_reasons)
        self.assertIn("overlay", trace.expected.recoveries_invoked)

    def test_overlay_thrash_pins_bounded_failure_under_budget(self) -> None:
        """
        overlay_thrash fixture must terminate BOUNDED_FAILURE within the per-task budget.
        """

        trace = FixtureLoader.load(name="overlay_thrash")
        self.assertEqual(trace.expected.terminal_status.value, "BOUNDED_FAILURE")
        self.assertLessEqual(trace.expected.max_repeated_no_effect, 5)

    def test_scroll_loop_pins_repeated_no_effect_block(self) -> None:
        """
        scroll_loop fixture must pin REPEATED_NO_EFFECT block + scroll recovery.
        """

        trace = FixtureLoader.load(name="scroll_loop")
        self.assertIn("repeated_no_effect", trace.expected.block_reasons)
        self.assertIn("scroll", trace.expected.recoveries_invoked)

    def test_scroll_loop_pins_non_succeeded_termination(self) -> None:
        """
        scroll_loop must not terminate SUCCEEDED; verifier rejection blocks force-close.
        """

        trace = FixtureLoader.load(name="scroll_loop")
        self.assertNotEqual(trace.expected.terminal_status.value, "SUCCEEDED")

    def test_coachmark_pins_escalation_path(self) -> None:
        """
        coachmark fixture must pin the escalation recovery and ESCALATED terminal.
        """

        trace = FixtureLoader.load(name="coachmark")
        self.assertIn("escalation", trace.expected.recoveries_invoked)
        self.assertEqual(trace.expected.terminal_status.value, "ESCALATED")

    def test_cosmetic_replan_pins_happy_path(self) -> None:
        """
        cosmetic_replan must succeed within the bounded step budget via replan.
        """

        trace = FixtureLoader.load(name="cosmetic_replan")
        self.assertEqual(trace.expected.terminal_status.value, "SUCCEEDED")
        self.assertIn("replan", trace.expected.recoveries_invoked)

    def test_every_fixture_forbids_legacy_raw_coordinate_execution(self) -> None:
        """
        Every fixture pins raw-LLM-coordinate executions at zero (legacy regression).
        """

        for name in (
            "coachmark",
            "cosmetic_replan",
            "overlay_scrim",
            "overlay_thrash",
            "scroll_loop",
        ):
            with self.subTest(scenario=name):
                trace = FixtureLoader.load(name=name)
                self.assertEqual(trace.expected.raw_llm_coordinates_executed, 0)


@unittest.skip(
    "Replay harness pending: stubbing every adapter port on IntentNodeProvider "
    "so a fixture trace drives the full GROUND→VERIFY cycle deterministically "
    "is the remainder of audit entries 036-046. The eleven acceptance cases "
    "from plan §17 Phase 12 are tracked here so they fail loudly once the "
    "harness lands and this `@skip` is removed."
)
class HealingReplayAcceptanceTest(unittest.IsolatedAsyncioTestCase):
    """
    Eleven Phase 12 acceptance cases from healing_runtime_completion_plan.md §17.

    Each case will replay one fixture through a stubbed IntentNodeProvider
    and assert real runtime behaviour (BlockReason emitted, RecoveryStrategy
    dispatched, terminal status reached). Today the test bodies are stubbed
    placeholders — fill them in when the harness lands.
    """

    async def test_overlay_primary_button_detected_without_xml(self) -> None:
        """Overlay primary dismiss button is detected with no manifest dialog."""

        raise NotImplementedError("Pending replay harness")

    async def test_overlay_failed_dismiss_does_not_repeat_beyond_budget(self) -> None:
        """Overlay-dismiss thrash is bounded by per-task healing budget."""

        raise NotImplementedError("Pending replay harness")

    async def test_same_action_same_target_no_effect_blocked(self) -> None:
        """REPEATED_NO_EFFECT supervision policy blocks redundant retries."""

        raise NotImplementedError("Pending replay harness")

    async def test_scroll_on_unchanged_screen_is_blocked(self) -> None:
        """Scroll on an unchanged screen invokes the scroll recovery strategy."""

        raise NotImplementedError("Pending replay harness")

    async def test_scroll_with_keyboard_visible_routes_to_keyboard_strategy(self) -> None:
        """Keyboard-occluding scroll routes through the keyboard recovery strategy."""

        raise NotImplementedError("Pending replay harness")

    async def test_visible_terminal_cta_prevents_blind_scroll(self) -> None:
        """A visible terminal CTA caps continued blind scrolling."""

        raise NotImplementedError("Pending replay harness")

    async def test_target_ambiguity_returns_candidates(self) -> None:
        """Ambiguous target localization returns candidate elements rather than guessing."""

        raise NotImplementedError("Pending replay harness")

    async def test_target_unresolved_routes_to_healing_or_ask_user(self) -> None:
        """Unresolved target routes through healing first, then ASK_USER on exhaustion."""

        raise NotImplementedError("Pending replay harness")

    async def test_happy_path_does_not_call_healing_llm(self) -> None:
        """A clean replay never dispatches healing."""

        raise NotImplementedError("Pending replay harness")

    async def test_paid_vision_localizer_obeys_budget(self) -> None:
        """Paid-vision localizer attempts stay within the configured per-run budget."""

        raise NotImplementedError("Pending replay harness")

    async def test_final_verification_rejection_is_not_force_closed(self) -> None:
        """Verifier rejection terminates BOUNDED_FAILURE or routes recovery, never SUCCEEDED."""

        raise NotImplementedError("Pending replay harness")
