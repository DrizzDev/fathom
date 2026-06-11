from __future__ import annotations

import unittest

from fathom.core.services.abort.heuristic import HeuristicAbortDetector
from fathom.schemas.abort import AbortFallbackConfiguration


class HeuristicAbortDetectorAbortPhrasesTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins canonical abort phrases that must be classified as aborted by the heuristic.
    """

    def setUp(self) -> None:
        """
        Build a detector with the default fallback configuration.
        """

        self.__detector = HeuristicAbortDetector()

    async def test_close_the_execution_is_abort(self) -> None:
        """
        Canonical production phrase from 9.txt must be classified as aborted.
        """

        decision = await self.__detector.aborted(response="close the execution")

        self.assertTrue(decision.aborted)
        self.assertTrue(decision.fallback)
        self.assertGreaterEqual(decision.confidence, 0.85)

    async def test_stop_the_run_is_abort(self) -> None:
        """
        Variant 'stop the run' is classified as aborted.
        """

        decision = await self.__detector.aborted(response="stop the run")

        self.assertTrue(decision.aborted)

    async def test_end_this_test_run_is_abort(self) -> None:
        """
        Production injected-context phrase is classified as aborted.
        """

        decision = await self.__detector.aborted(response="end this test run")

        self.assertTrue(decision.aborted)

    async def test_stop_this_test_run_is_abort(self) -> None:
        """
        Manual pause variant is classified as aborted.
        """

        decision = await self.__detector.aborted(response="stop this test run")

        self.assertTrue(decision.aborted)

    async def test_terminate_the_workflow_is_abort(self) -> None:
        """
        Variant 'terminate the workflow' is classified as aborted.
        """

        decision = await self.__detector.aborted(response="terminate the workflow")

        self.assertTrue(decision.aborted)

    async def test_response_with_surrounding_clause_is_abort(self) -> None:
        """
        Operator phrase embedded inside a longer sentence is still classified as aborted.
        """

        decision = await self.__detector.aborted(response="I told you to close the execution")

        self.assertTrue(decision.aborted)


class HeuristicAbortDetectorUiDirectiveGuardTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the safety-critical UI-directive guard. UI directives must never abort.
    """

    def setUp(self) -> None:
        """
        Build a detector with the default fallback configuration.
        """

        self.__detector = HeuristicAbortDetector()

    async def test_tap_on_stop_is_not_abort(self) -> None:
        """
        Canonical blunder case: 'tap on stop' must never be classified as aborted.
        """

        decision = await self.__detector.aborted(response="tap on stop")

        self.assertFalse(decision.aborted)

    async def test_click_on_cancel_is_not_abort(self) -> None:
        """
        UI-action verb 'click' disqualifies the response from abort.
        """

        decision = await self.__detector.aborted(response="click on cancel")

        self.assertFalse(decision.aborted)

    async def test_press_the_close_button_is_not_abort(self) -> None:
        """
        UI-action verb 'press' disqualifies the response from abort.
        """

        decision = await self.__detector.aborted(response="press the close button")

        self.assertFalse(decision.aborted)

    async def test_tap_the_x_is_not_abort(self) -> None:
        """
        Short directive 'tap the X' is not aborted.
        """

        decision = await self.__detector.aborted(response="tap the X")

        self.assertFalse(decision.aborted)

    async def test_i_want_to_tap_stop_is_not_abort(self) -> None:
        """
        UI-action verb anywhere in the text blocks the abort decision.
        """

        decision = await self.__detector.aborted(response="I want to tap stop")

        self.assertFalse(decision.aborted)


class HeuristicAbortDetectorEdgeCaseTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins edge-case input handling (empty, whitespace, unrelated text).
    """

    def setUp(self) -> None:
        """
        Build a detector with the default fallback configuration.
        """

        self.__detector = HeuristicAbortDetector()

    async def test_empty_string_is_not_abort(self) -> None:
        """
        Empty input is never an abort.
        """

        decision = await self.__detector.aborted(response="")

        self.assertFalse(decision.aborted)

    async def test_whitespace_only_is_not_abort(self) -> None:
        """
        Whitespace-only input is treated like empty input.
        """

        decision = await self.__detector.aborted(response="   \n\t")

        self.assertFalse(decision.aborted)

    async def test_unrelated_text_is_not_abort(self) -> None:
        """
        Unrelated guidance does not match any anchor strongly enough.
        """

        decision = await self.__detector.aborted(response="the button is in the top right")

        self.assertFalse(decision.aborted)

    async def test_done_here_continuation_is_not_abort(self) -> None:
        """
        Ambiguous continuation guidance must not be classified as an abort.
        """

        decision = await self.__detector.aborted(response="we are done here, let's move to step 3")

        self.assertFalse(decision.aborted)


class HeuristicAbortDetectorThresholdTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins behavior of the configurable similarity floor.
    """

    async def test_lower_floor_admits_marginal_matches(self) -> None:
        """
        A relaxed similarity floor classifies marginal matches as aborted.
        """

        detector = HeuristicAbortDetector(
            configuration=AbortFallbackConfiguration(similarity_floor=0.3)
        )

        decision = await detector.aborted(response="finish please")

        self.assertTrue(decision.aborted)

    async def test_high_floor_rejects_marginal_matches(self) -> None:
        """
        A strict similarity floor rejects marginal matches.
        """

        detector = HeuristicAbortDetector(
            configuration=AbortFallbackConfiguration(similarity_floor=0.99)
        )

        decision = await detector.aborted(response="finish please")

        self.assertFalse(decision.aborted)


class HeuristicAbortDetectorWarmupTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the warmup contract.
    """

    async def test_warmup_is_a_no_op(self) -> None:
        """
        Heuristic has no model to prime so warmup completes silently.
        """

        await HeuristicAbortDetector().warmup()
