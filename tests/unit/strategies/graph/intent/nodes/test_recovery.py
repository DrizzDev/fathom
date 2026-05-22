from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from fathom.constants.command import CommandExecutionMode
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.core.recovery import BoundedFailureOutcome, ReplanOutcome
from fathom.schemas.escape import EscapeCategory, EscapeReport
from fathom.schemas.subgoal import ExecutionContract, RequiredActionFamily, ScrollAxis, SubGoal
from fathom.strategies.graph.intent.nodes.recovery import (
    RecoveryDispatcher,
    RecoveryOutcomeApplier,
)


class RecoveryDispatcherExtractEscapeReportTest(unittest.TestCase):
    """
    Pins :meth:`RecoveryDispatcher.extract_escape_report`.

    The static decoder turns planner-emitted metadata into a typed
    :class:`EscapeReport` and routes the run accordingly. Any malformed
    payload — missing key, non-dict value, unknown category — must
    return ``None`` rather than raising, so the planner can keep
    producing turns while a deeper bug is investigated. Valid payloads
    must route correctly to either the replan or the human path.
    """

    def test_returns_none_for_empty_metadata(self) -> None:
        """
        Empty or None metadata yields None and never raises.
        """

        self.assertIsNone(RecoveryDispatcher.extract_escape_report(metadata=None))
        self.assertIsNone(RecoveryDispatcher.extract_escape_report(metadata={}))

    def test_returns_none_when_key_missing(self) -> None:
        """
        Metadata without the escape_report key yields None.
        """

        self.assertIsNone(
            RecoveryDispatcher.extract_escape_report(metadata={"other": "value"}),
        )

    def test_returns_none_when_payload_not_dict(self) -> None:
        """
        Non-dict payloads under the escape_report key yield None.
        """

        result = RecoveryDispatcher.extract_escape_report(
            metadata={"escape_report": "not-a-dict"},
        )

        self.assertIsNone(result)

    def test_returns_none_for_invalid_category(self) -> None:
        """
        An EscapeReport with an unknown category fails validation and yields None.
        """

        result = RecoveryDispatcher.extract_escape_report(
            metadata={"escape_report": {"category": "bogus", "detail": "x"}},
        )

        self.assertIsNone(result)

    def test_decodes_valid_replan_payload(self) -> None:
        """
        A well-formed payload decodes into an EscapeReport that routes to replan.
        """

        result = RecoveryDispatcher.extract_escape_report(
            metadata={
                "escape_report": {
                    "category": EscapeCategory.WRONG_SCREEN.value,
                    "detail": "Stuck on permissions sheet.",
                }
            }
        )

        self.assertIsInstance(result, EscapeReport)
        assert result is not None
        self.assertEqual(result.category, EscapeCategory.WRONG_SCREEN)
        self.assertTrue(result.routes_to_replan())
        self.assertFalse(result.routes_to_human())

    def test_decodes_valid_human_payload(self) -> None:
        """
        A well-formed human-routing payload routes to the HITL path.
        """

        result = RecoveryDispatcher.extract_escape_report(
            metadata={
                "escape_report": {
                    "category": EscapeCategory.UNSAFE_ACTION.value,
                    "detail": "Confirm before factory reset.",
                }
            }
        )

        self.assertIsInstance(result, EscapeReport)
        assert result is not None
        self.assertTrue(result.routes_to_human())
        self.assertFalse(result.routes_to_replan())


class RecoveryDispatcherDescriptorsFromTraceTest(unittest.TestCase):
    """
    Pins :meth:`RecoveryDispatcher.descriptors_from_trace`.

    The static helper renders the most-recent N trace entries as one-
    line descriptors handed to recovery strategies. Two invariants
    matter most: (1) Python's ``list[-0:]`` returns the whole list, so
    a ``recent_window=0`` policy must be floored to one and (2) trace
    entries that are not dicts must be stringified rather than dropped
    so synthetic recovery steps stay observable.
    """

    @staticmethod
    def __trace():  # type: ignore[no-untyped-def]
        """
        Three-entry deterministic trace shared by the slicing tests.
        Each entry uses the dict shape the dispatcher expects in
        production; the string-entry test inserts a raw string ad-hoc.
        """

        return [
            {"action": "tap", "target": "Home"},
            {"action": "swipe_up", "target": "Auto suggest"},
            {"action": "tap", "target": "Search"},
        ]

    def test_window_zero_is_floored_to_one(self) -> None:
        """
        recent_window=0 must yield exactly one descriptor, not the whole trace.

        Python's ``list[-0:]`` returns the entire list; the floor exists so a
        misconfigured policy cannot silently surface every recent action.
        """

        descriptors = RecoveryDispatcher.descriptors_from_trace(
            trace=self.__trace(),
            window=0,
        )

        self.assertEqual(len(descriptors), 1)
        self.assertIn("Search", descriptors[0])

    def test_window_smaller_than_trace_returns_tail(self) -> None:
        """
        A small window returns the most-recent entries in insertion order.
        """

        descriptors = RecoveryDispatcher.descriptors_from_trace(
            trace=self.__trace(),
            window=2,
        )

        self.assertEqual(len(descriptors), 2)
        self.assertIn("Auto suggest", descriptors[0])
        self.assertIn("Search", descriptors[1])

    def test_window_larger_than_trace_returns_full_trace(self) -> None:
        """
        A window larger than the trace returns every entry.
        """

        descriptors = RecoveryDispatcher.descriptors_from_trace(
            trace=self.__trace(),
            window=99,
        )

        self.assertEqual(len(descriptors), 3)

    def test_empty_trace_returns_empty_list(self) -> None:
        """
        An empty trace yields no descriptors regardless of the window.
        """

        self.assertEqual(
            RecoveryDispatcher.descriptors_from_trace(trace=[], window=5),
            [],
        )

    def test_non_list_trace_returns_empty_list(self) -> None:
        """
        A non-list trace (e.g. None or a dict) yields no descriptors.
        """

        self.assertEqual(
            RecoveryDispatcher.descriptors_from_trace(trace=None, window=5),
            [],
        )

    def test_string_entries_are_stringified_not_dropped(self) -> None:
        """
        Non-dict trace entries are preserved by ``str(entry)`` rather than skipped.
        """

        descriptors = RecoveryDispatcher.descriptors_from_trace(
            trace=[{"action": "tap", "target": "Home"}, "raw-string-entry"],
            window=5,
        )

        self.assertEqual(len(descriptors), 2)
        self.assertEqual(descriptors[1], "raw-string-entry")


class RecoveryOutcomeApplierStrictExecutionContractTest(unittest.TestCase):
    """
    Pins the strict-mode semantic mission guard on recovery replans.
    """

    def test_strict_scroll_mission_rejects_search_workflow_replan(self) -> None:
        """
        Strict scroll discovery must not be rewritten into tap/type search steps.
        """

        context = MagicMock(name="GraphContext")
        context.workflow_id = "run-test"
        context.configuration.intent.command_mode = CommandExecutionMode.STRICT
        context.agent_state.get_current_sub_goal.return_value = SubGoal(
            index=0,
            description="Scroll vertically until you find Asha Tiffin on the screen",
            execution_contract=ExecutionContract(
                required_action_family=RequiredActionFamily.SCROLL,
                scroll_axis=ScrollAxis.VERTICAL,
            ),
        )
        context.agent_state.mark_complete = MagicMock()
        builder = MagicMock(name="RecoveryRequestBuilder")
        persistence = MagicMock(name="GraphStatePersistence")

        applier = RecoveryOutcomeApplier(
            context=context,
            builder=builder,
            persistence=persistence,
        )

        result = applier.apply(
            outcome=ReplanOutcome(
                summary="Switch to search workflow.",
                new_sub_goals=[
                    SubGoal(
                        index=0,
                        description="Tap the Search button at the bottom of the screen",
                        execution_contract=ExecutionContract(
                            required_action_family=RequiredActionFamily.TAP,
                        ),
                    ),
                    SubGoal(
                        index=1,
                        description="Search for Asha Tiffin in the restaurant list",
                        execution_contract=ExecutionContract(
                            required_action_family=RequiredActionFamily.INPUT,
                        ),
                    ),
                    SubGoal(
                        index=2,
                        description="Select Asha Tiffin from the search results",
                        execution_contract=ExecutionContract(
                            required_action_family=RequiredActionFamily.TAP,
                        ),
                    ),
                ],
            )
        )

        assert result is not None
        self.assertFalse(result.get(CommonStateKey.IS_COMPLETE))
        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))
        self.assertEqual(result.get(IntentStateKey.LAST_BLOCK_REASON), "strict_replan_mismatch")
        context.agent_state.replan_pending_sub_goals.assert_not_called()

    def test_strict_replan_rejects_surface_drift(self) -> None:
        """
        Strict replans must preserve the declared surface contract.
        """

        context = MagicMock(name="GraphContext")
        context.workflow_id = "run-test"
        context.configuration.intent.command_mode = CommandExecutionMode.STRICT
        context.agent_state.get_current_sub_goal.return_value = SubGoal(
            index=0,
            description="Scroll horizontally below Fast Delivery until Millet Express is visible",
            execution_contract=ExecutionContract(
                required_action_family=RequiredActionFamily.SCROLL,
                scroll_axis=ScrollAxis.HORIZONTAL,
                surface="below Fast Delivery section",
            ),
        )
        builder = MagicMock(name="RecoveryRequestBuilder")
        persistence = MagicMock(name="GraphStatePersistence")

        applier = RecoveryOutcomeApplier(
            context=context,
            builder=builder,
            persistence=persistence,
        )

        result = applier.apply(
            outcome=ReplanOutcome(
                summary="Keep scrolling in another row.",
                new_sub_goals=[
                    SubGoal(
                        index=0,
                        description="Scroll horizontally below Top Rated until Millet Express is visible",
                        execution_contract=ExecutionContract(
                            required_action_family=RequiredActionFamily.SCROLL,
                            scroll_axis=ScrollAxis.HORIZONTAL,
                            surface="below Top Rated section",
                        ),
                    )
                ],
            )
        )

        assert result is not None
        self.assertFalse(result.get(CommonStateKey.IS_COMPLETE))
        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))
        self.assertEqual(result.get(IntentStateKey.LAST_BLOCK_REASON), "strict_replan_mismatch")

    def test_strict_scroll_mission_rejects_validate_only_replan(self) -> None:
        """
        Strict scroll recovery must not degrade into validation-only churn.
        """

        context = MagicMock(name="GraphContext")
        context.workflow_id = "run-test"
        context.configuration.intent.command_mode = CommandExecutionMode.STRICT
        context.agent_state.get_current_sub_goal.return_value = SubGoal(
            index=0,
            description="Scroll horizontally below Fast Delivery until Millet Express is visible",
            execution_contract=ExecutionContract(
                required_action_family=RequiredActionFamily.SCROLL,
                scroll_axis=ScrollAxis.HORIZONTAL,
                surface="below Fast Delivery section",
            ),
        )
        builder = MagicMock(name="RecoveryRequestBuilder")
        persistence = MagicMock(name="GraphStatePersistence")

        applier = RecoveryOutcomeApplier(
            context=context,
            builder=builder,
            persistence=persistence,
        )

        result = applier.apply(
            outcome=ReplanOutcome(
                summary="Just validate whether Millet Express is already visible.",
                new_sub_goals=[
                    SubGoal(
                        index=0,
                        description="Validate Millet Express is visible below Fast Delivery",
                        execution_contract=ExecutionContract(
                            required_action_family=RequiredActionFamily.VALIDATE,
                            surface="below Fast Delivery section",
                        ),
                    )
                ],
            )
        )

        assert result is not None
        self.assertFalse(result.get(CommonStateKey.IS_COMPLETE))
        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))
        self.assertEqual(result.get(IntentStateKey.LAST_BLOCK_REASON), "strict_replan_mismatch")

    def test_bounded_failure_marks_failed_and_preserves_diagnostic(self) -> None:
        """
        Bounded failure must terminate as FAILED, not as a faux-success completion.
        """

        context = MagicMock(name="GraphContext")
        context.workflow_id = "run-test"
        context.agent_state.mark_complete = MagicMock()
        builder = MagicMock(name="RecoveryRequestBuilder")
        persistence = MagicMock(name="GraphStatePersistence")

        applier = RecoveryOutcomeApplier(
            context=context,
            builder=builder,
            persistence=persistence,
        )

        result = applier.apply(
            outcome=BoundedFailureOutcome(
                summary="stop",
                diagnostic="sub-goal exhausted step budget",
            )
        )

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON), CompletionReason.FAILED.value
        )
        self.assertEqual(
            result.get(CommonStateKey.FAILURE_DIAGNOSTIC),
            "sub-goal exhausted step budget",
        )
        context.agent_state.mark_complete.assert_called_once_with(
            reason=CompletionReason.FAILED.value
        )
