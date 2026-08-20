from __future__ import annotations

import unittest
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from fathom.constants import ActionType, StepEvent
from fathom.constants.tools import StateNamespace
from fathom.constants.turn.validation import ValidationSource
from fathom.core.exceptions import ToolValidationError
from fathom.core.services.parsing import ToolResponseParser
from fathom.schemas.results import GenerateResult


class _Call(BaseModel):
    """
    Minimal tool-call stand-in consumed by the parser.
    """

    name: str = Field(description="Tool name")
    args: Dict[str, Any] = Field(description="Tool arguments")


class ToolResponseParserExecuteUiTest(unittest.TestCase):
    """
    Covers execute_ui command parsing.
    """

    @staticmethod
    def __response(*, calls: List[_Call]) -> GenerateResult:
        """
        Wrap tool calls in a generate result.
        """

        return GenerateResult(content="", tool_calls=list(calls), metrics={})

    def test_emits_command_for_malformed_normalized_bbox(self) -> None:
        """
        Parser preserves the external bbox payload for post-catalog materialization.
        """

        parser = ToolResponseParser()
        response = self.__response(
            calls=[
                _Call(
                    name="execute_ui",
                    args={
                        "assistant_message": "scroll down",
                        "goal_completed": False,
                        "sub_goal_completed": False,
                        "action": {
                            "action_type": "swipe_up",
                            "target_name": "Restaurant list area",
                            "scroll_target": "Restaurant list area",
                            "confidence": 0.84,
                            "bbox": {
                                "x": 0,
                                "y": 858,
                                "width": 1206,
                                "height": 1396,
                                "coordinate_system": "normalized",
                            },
                        },
                    },
                )
            ]
        )

        result = parser.parse(response=response)

        self.assertIsNone(result.action)
        assert result.tool_response is not None
        assert result.tool_response.command is not None
        self.assertEqual(result.tool_response.command.action_type, ActionType.SWIPE_UP)
        assert result.tool_response.command.payload.bbox is not None
        self.assertEqual(result.tool_response.command.payload.bbox.coordinate_system, "normalized")

    def test_emits_command_for_valid_normalized_bbox(self) -> None:
        """
        Parser emits the command without building an executable action.
        """

        parser = ToolResponseParser()
        response = self.__response(
            calls=[
                _Call(
                    name="execute_ui",
                    args={
                        "assistant_message": "scroll down",
                        "goal_completed": False,
                        "sub_goal_completed": False,
                        "action": {
                            "action_type": "swipe_up",
                            "target_name": "Restaurant list area",
                            "scroll_target": "Restaurant list area",
                            "confidence": 0.84,
                            "bbox": {
                                "x": 100,
                                "y": 200,
                                "width": 800,
                                "height": 500,
                                "coordinate_system": "normalized",
                            },
                        },
                    },
                )
            ]
        )

        result = parser.parse(response=response)

        self.assertIsNone(result.action)
        assert result.tool_response is not None
        assert result.tool_response.command is not None
        self.assertEqual(result.tool_response.command.action_type, ActionType.SWIPE_UP)
        assert result.tool_response.command.payload.bbox is not None
        self.assertEqual(result.tool_response.command.payload.bbox.coordinate_system, "normalized")

    def test_missing_confidence_fails_validation(self) -> None:
        """
        Missing planner confidence must fail validation instead of being guessed.
        """

        parser = ToolResponseParser()
        response = self.__response(
            calls=[
                _Call(
                    name="execute_ui",
                    args={
                        "assistant_message": "tap restaurant",
                        "goal_completed": False,
                        "sub_goal_completed": False,
                        "action": {
                            "action_type": "tap",
                            "target_name": "Asha Tiffin",
                            "label_id": "12",
                            "bbox": {
                                "x": 100,
                                "y": 200,
                                "width": 300,
                                "height": 120,
                                "coordinate_system": "normalized",
                            },
                        },
                    },
                )
            ]
        )

        with self.assertRaises(ToolValidationError):
            parser.parse(response=response)

    def test_scroll_objective_stays_on_command_payload(self) -> None:
        """
        scroll_target stays on the command payload for materialization.
        """

        parser = ToolResponseParser()
        response = self.__response(
            calls=[
                _Call(
                    name="execute_ui",
                    args={
                        "assistant_message": "scroll down",
                        "goal_completed": False,
                        "sub_goal_completed": False,
                        "action": {
                            "action_type": "swipe_up",
                            "scroll_target": "Asha Tiffin",
                            "confidence": 0.84,
                            "bbox": {
                                "x": 0,
                                "y": 858,
                                "width": 1206,
                                "height": 1396,
                                "coordinate_system": "pixel",
                            },
                        },
                    },
                )
            ]
        )

        result = parser.parse(response=response)

        self.assertIsNone(result.action)
        assert result.tool_response is not None
        assert result.tool_response.command is not None
        self.assertEqual(result.tool_response.command.action_type, ActionType.SWIPE_UP)
        self.assertEqual(result.tool_response.command.payload.scroll_target, "Asha Tiffin")

    def test_rejects_legacy_enter_action_with_tool_validation_error(self) -> None:
        """
        The deprecated 'enter' action_type from VLM output must raise ToolValidationError
        with neutral feedback pointing the planner back at the execute_ui schema.
        """

        parser = ToolResponseParser()
        response = self.__response(
            calls=[
                _Call(
                    name="execute_ui",
                    args={
                        "assistant_message": "submit search",
                        "goal_completed": False,
                        "sub_goal_completed": False,
                        "action": {
                            "action_type": "enter",
                            "target_name": "Search button on keyboard",
                            "confidence": 0.84,
                        },
                    },
                )
            ]
        )

        with self.assertRaises(ToolValidationError) as context:
            parser.parse(response=response)

        feedback = context.exception.feedback
        self.assertEqual(feedback.tool_name, "execute_ui")
        self.assertEqual(feedback.error_kind, "validation")
        self.assertIn("action_type='enter' is not a supported", feedback.message)
        self.assertIn("execute_ui tool schema", feedback.message)
        for biased_term in ("hide_keyboard", "Search/Done/Submit", "keyboard", "tap "):
            with self.subTest(term=biased_term):
                self.assertNotIn(biased_term, feedback.message)

    def test_rejects_unknown_action_type_without_defaulting_to_wait(self) -> None:
        """
        Unknown action types must fail validation instead of becoming WAIT.
        """

        parser = ToolResponseParser()
        response = self.__response(
            calls=[
                _Call(
                    name="execute_ui",
                    args={
                        "assistant_message": "do unsupported thing",
                        "goal_completed": False,
                        "sub_goal_completed": False,
                        "action": {
                            "action_type": "secret_₹86",
                            "target_name": "Checkout",
                            "confidence": 0.84,
                        },
                    },
                )
            ]
        )

        with self.assertRaises(ToolValidationError) as context:
            parser.parse(response=response)

        feedback = context.exception.feedback
        self.assertEqual(feedback.tool_name, "execute_ui")
        self.assertEqual(feedback.error_kind, "validation")
        self.assertIn("action.action_type is not a supported", feedback.message)
        self.assertNotIn("secret_₹86", feedback.message)


class ToolResponseParserCompletionReasonAutofillTest(unittest.TestCase):
    """
    Pins the parser's autofill of missing ``subgoal_completion_reason`` /
    ``goal_completion_reason`` fields so the downstream completion gate
    can verify the claim without falling back to fuzzy matching.
    """

    @staticmethod
    def __response(*, args: Dict[str, Any]) -> GenerateResult:
        """
        Wrap an ``execute_ui`` tool call in a generate result.
        """

        return GenerateResult(
            content="",
            metrics={},
            tool_calls=[_Call(name="execute_ui", args=args)],
        )

    @staticmethod
    def __completion_args(
        *,
        rationale: str,
        sub_goal_completed: bool,
        goal_completed: bool = False,
        subgoal_completion_reason: Any = None,
        goal_completion_reason: Any = None,
    ) -> Dict[str, Any]:
        """
        Build a baseline ``execute_ui`` argument set that flags completion.
        """

        return {
            "assistant_message": "OK",
            "goal_completed": goal_completed,
            "sub_goal_completed": sub_goal_completed,
            "goal_completion_reason": goal_completion_reason,
            "subgoal_completion_reason": subgoal_completion_reason,
            "action": {
                "action_type": "complete",
                "target_name": "Sub-goal verification",
                "rationale": rationale,
                "is_valid": True,
                "confidence": 1.0,
            },
        }

    def test_autofills_subgoal_reason_from_rationale_when_missing(self) -> None:
        """
        Missing ``subgoal_completion_reason`` is filled from ``action.rationale``
        when the model flagged the sub-goal as complete.
        """

        parser = ToolResponseParser()
        response = self.__response(
            args=self.__completion_args(
                sub_goal_completed=True,
                rationale="Offerwall menu has been dismissed successfully",
            )
        )

        result = parser.parse(response=response)

        self.assertTrue(result.is_sub_goal_complete)
        self.assertEqual(
            result.subgoal_completion_reason,
            "Offerwall menu has been dismissed successfully",
        )

    def test_preserves_model_supplied_subgoal_reason(self) -> None:
        """
        A non-empty ``subgoal_completion_reason`` from the model is preserved verbatim.
        """

        parser = ToolResponseParser()
        response = self.__response(
            args=self.__completion_args(
                sub_goal_completed=True,
                rationale="Backup rationale not used",
                subgoal_completion_reason="Order placed successfully",
            )
        )

        result = parser.parse(response=response)

        self.assertEqual(result.subgoal_completion_reason, "Order placed successfully")

    def test_does_not_autofill_when_completion_flag_false(self) -> None:
        """
        Autofill only triggers when the model claims completion.
        """

        parser = ToolResponseParser()
        args = self.__completion_args(
            sub_goal_completed=False,
            rationale="Just a regular tap on the search box",
        )
        args["action"]["action_type"] = "tap"
        args["action"]["target_name"] = "Search box"
        args["action"]["export_target"] = "Search box"
        response = self.__response(args=args)

        result = parser.parse(response=response)

        self.assertFalse(result.is_sub_goal_complete)
        self.assertIsNone(result.subgoal_completion_reason)

    def test_autofills_goal_reason_from_rationale(self) -> None:
        """
        Missing ``goal_completion_reason`` is filled from rationale when the
        model claims overall goal completion.
        """

        parser = ToolResponseParser()
        response = self.__response(
            args=self.__completion_args(
                sub_goal_completed=True,
                goal_completed=True,
                rationale="Workflow finished as per operator instruction",
            )
        )

        result = parser.parse(response=response)

        self.assertTrue(result.is_goal_complete)
        self.assertEqual(
            result.goal_completion_reason,
            "Workflow finished as per operator instruction",
        )

    def test_leaves_reason_null_when_rationale_is_whitespace_only(self) -> None:
        """
        With both the reason field empty AND the rationale collapsing to
        whitespace, the autofill must leave the reason None — never guess.
        """

        parser = ToolResponseParser()
        response = self.__response(
            args=self.__completion_args(
                sub_goal_completed=True,
                rationale="   \t\n  ",
            )
        )

        result = parser.parse(response=response)
        self.assertTrue(result.is_sub_goal_complete)
        self.assertIsNone(result.subgoal_completion_reason)

    def test_complete_action_with_false_sub_goal_flag_autofills_after_normalization(
        self,
    ) -> None:
        """When ``action_type=COMPLETE`` arrives with ``sub_goal_completed=False``, the normalizer must force the flag to True FIRST, then the autofill must see the True flag and populate ``subgoal_completion_reason`` from the rationale — pinning the COMPLETE-flag → autofill ordering."""

        parser = ToolResponseParser()
        response = self.__response(
            args=self.__completion_args(
                sub_goal_completed=False,
                rationale="Offerwall confirmation visible",
            )
        )

        result = parser.parse(response=response)

        self.assertTrue(result.is_sub_goal_complete)
        self.assertEqual(
            result.subgoal_completion_reason,
            "Offerwall confirmation visible",
        )


class ToolResponseParserValidationSubjectTest(unittest.TestCase):
    """
    Covers canonical validation subject normalization across validation tools.
    """

    @staticmethod
    def __response(*, call: _Call) -> GenerateResult:
        """
        Wrap one tool call in a generate result.
        """

        return GenerateResult(content="", tool_calls=[call], metrics={})

    def test_validate_state_sets_validation_subject_from_condition(self) -> None:
        """
        validate_state must produce a validate action completion can recognize.
        """

        parser = ToolResponseParser()
        result = parser.parse(
            response=self.__response(
                call=_Call(
                    name="validate_state",
                    args={
                        "assistant_message": "Both items are visible in the cart.",
                        "condition_to_verify": (
                            "Coca-Cola Diet Coke and Sunfeast Dark fantasy are present"
                        ),
                        "condition_met": True,
                        "evidence": "The cart list shows both products with quantity 1.",
                        "goal_completed": False,
                        "sub_goal_completed": True,
                    },
                )
            )
        )

        assert result.action is not None
        self.assertEqual(result.action.event_type, StepEvent.VALIDATION)
        self.assertEqual(result.action.action_type, ActionType.VALIDATE)
        self.assertEqual(
            result.action.validation_subject,
            "Coca-Cola Diet Coke and Sunfeast Dark fantasy are present",
        )
        assert result.validation is not None
        self.assertEqual(result.validation.source, ValidationSource.STATE)
        self.assertEqual(
            result.validation.subject,
            "Coca-Cola Diet Coke and Sunfeast Dark fantasy are present",
        )

    def test_verify_goal_sets_validation_subject_when_goal_is_not_complete(self) -> None:
        """
        verify_goal must prefer explicit sub-goal reason as validation evidence.
        """

        parser = ToolResponseParser()
        result = parser.parse(
            response=self.__response(
                call=_Call(
                    name="verify_goal",
                    args={
                        "assistant_message": "The cart screen is visible.",
                        "goal_completed": False,
                        "sub_goal_completed": True,
                        "subgoal_completion_reason": "Cart screen is visible.",
                        "current_screen": "Cart screen with both requested items",
                        "evidence": "The cart has Diet Coke and Dark Fantasy.",
                    },
                )
            )
        )

        assert result.action is not None
        self.assertEqual(result.action.event_type, StepEvent.VALIDATION)
        self.assertEqual(result.action.action_type, ActionType.VALIDATE)
        self.assertEqual(
            result.action.validation_subject,
            "Cart screen is visible.",
        )
        assert result.validation is not None
        self.assertEqual(result.validation.source, ValidationSource.GOAL)
        self.assertEqual(result.validation.subject, "Cart screen is visible.")

    def test_verify_goal_falls_back_to_current_screen_for_validation_subject(self) -> None:
        """
        verify_goal uses current screen only when no explicit validation reason exists.
        """

        parser = ToolResponseParser()
        result = parser.parse(
            response=self.__response(
                call=_Call(
                    name="verify_goal",
                    args={
                        "assistant_message": "The cart screen is visible.",
                        "goal_completed": False,
                        "sub_goal_completed": True,
                        "subgoal_completion_reason": "",
                        "current_screen": "Cart screen with both requested items",
                        "evidence": "The cart has Diet Coke and Dark Fantasy.",
                    },
                )
            )
        )

        assert result.action is not None
        self.assertEqual(result.action.event_type, StepEvent.VALIDATION)
        self.assertEqual(result.action.action_type, ActionType.VALIDATE)
        self.assertEqual(
            result.action.validation_subject,
            "Cart screen with both requested items",
        )

    def test_verify_goal_complete_preserves_validation_subject(self) -> None:
        """
        verify_goal terminal completion remains COMPLETE while carrying validation evidence.
        """

        parser = ToolResponseParser()
        result = parser.parse(
            response=self.__response(
                call=_Call(
                    name="verify_goal",
                    args={
                        "assistant_message": "The full task is complete.",
                        "goal_completed": True,
                        "goal_completion_reason": "All requested items are in the cart.",
                        "sub_goal_completed": True,
                        "subgoal_completion_reason": "The final step is complete.",
                        "current_screen": "Cart screen with total amount",
                        "evidence": "The cart has both items and the total price.",
                    },
                )
            )
        )

        assert result.action is not None
        self.assertEqual(result.action.event_type, StepEvent.VALIDATION)
        self.assertEqual(result.action.action_type, ActionType.COMPLETE)
        self.assertEqual(result.action.validation_subject, "The final step is complete.")

    def test_verify_goal_prefers_explicit_assertion_for_validation_subject(self) -> None:
        """
        A crisp assertion beats completion reasons and the screen description.
        """

        parser = ToolResponseParser()
        result = parser.parse(
            response=self.__response(
                call=_Call(
                    name="verify_goal",
                    args={
                        "assistant_message": "The cart shows the item.",
                        "goal_completed": False,
                        "sub_goal_completed": True,
                        "subgoal_completion_reason": "Item added.",
                        "current_screen": "Cart screen",
                        "assertion": "Cart contains Diet Coke x1",
                        "evidence": "Cart badge shows 1.",
                    },
                )
            )
        )

        assert result.validation is not None
        assert result.action is not None
        self.assertEqual(result.validation.subject, "Cart contains Diet Coke x1")
        self.assertEqual(result.action.validation_subject, "Cart contains Diet Coke x1")


class ToolResponseParserBoundaryTest(unittest.TestCase):
    """
    Covers the model-tool response boundary.
    """

    @staticmethod
    def __response(*, calls: List[_Call]) -> GenerateResult:
        """
        Wrap tool calls in a generate result.
        """

        return GenerateResult(content="", tool_calls=list(calls), metrics={})

    def test_store_memory_is_state_change_not_executable_action(self) -> None:
        """
        store_memory must not synthesize WAIT/STORE or a Memory Store target.
        """

        parser = ToolResponseParser()
        result = parser.parse(
            response=self.__response(
                calls=[
                    _Call(
                        name="store_memory",
                        args={
                            "key": "item_price",
                            "value": "94",
                            "assistant_message": "Remember selected price",
                        },
                    )
                ]
            )
        )

        self.assertIsNone(result.action)
        assert result.tool_response is not None
        self.assertIsNone(result.tool_response.command)
        self.assertEqual(len(result.tool_response.updates), 1)
        change = result.tool_response.updates[0]
        self.assertEqual(change.namespace, StateNamespace.MEMORY)
        self.assertEqual(change.key, "item_price")
        self.assertEqual(change.value, "94")
        self.assertNotIn("Memory Store", result.reasoning)
        self.assertEqual(result.metadata["tool_args"]["value"], "<redacted>")

    def test_execute_ui_memory_updates_are_updates_not_action_payload(self) -> None:
        """
        execute_ui memory_updates must ride the ToolResponse, never Action.
        """

        parser = ToolResponseParser()
        result = parser.parse(
            response=self.__response(
                calls=[
                    _Call(
                        name="execute_ui",
                        args={
                            "assistant_message": "tap filter",
                            "goal_completed": False,
                            "sub_goal_completed": False,
                            "memory_updates": {"filter_opened": "true"},
                            "action": {
                                "action_type": "tap",
                                "target_name": "Filter",
                                "export_target": "Filter button",
                                "confidence": 0.9,
                            },
                        },
                    )
                ]
            )
        )

        self.assertIsNone(result.action)
        assert result.tool_response is not None
        self.assertIsNotNone(result.tool_response.command)
        self.assertEqual(len(result.tool_response.updates), 1)
        change = result.tool_response.updates[0]
        self.assertEqual(change.namespace, StateNamespace.MEMORY)
        self.assertEqual(change.key, "filter_opened")
        self.assertEqual(change.value, "true")
        self.assertEqual(
            result.metadata["tool_args"]["memory_updates"],
            {"filter_opened": "<redacted>"},
        )

    def test_mixed_execute_ui_and_store_memory_keeps_one_command_and_update(self) -> None:
        """
        Parallel execute_ui + store_memory creates one command and one update.
        """

        parser = ToolResponseParser()
        result = parser.parse(
            response=self.__response(
                calls=[
                    _Call(
                        name="execute_ui",
                        args={
                            "assistant_message": "tap product",
                            "goal_completed": False,
                            "sub_goal_completed": False,
                            "action": {
                                "action_type": "tap",
                                "target_name": "Product card",
                                "export_target": "Product card",
                                "confidence": 0.9,
                            },
                        },
                    ),
                    _Call(
                        name="store_memory",
                        args={
                            "key": "candidate",
                            "value": "soap",
                            "assistant_message": "Track selected candidate",
                        },
                    ),
                ]
            )
        )

        self.assertIsNone(result.action)
        assert result.tool_response is not None
        command = result.tool_response.command
        assert command is not None
        self.assertEqual(command.action_type, ActionType.TAP)
        self.assertEqual(len(result.tool_response.updates), 1)
        self.assertEqual(result.tool_response.updates[0].key, "candidate")

    def test_capture_value_is_redacted_from_tool_metadata(self) -> None:
        """
        STORE capture values are not retained in parser metadata.
        """

        parser = ToolResponseParser()
        result = parser.parse(
            response=self.__response(
                calls=[
                    _Call(
                        name="execute_ui",
                        args={
                            "assistant_message": "store price",
                            "goal_completed": False,
                            "sub_goal_completed": False,
                            "action": {
                                "action_type": "store",
                                "confidence": 0.9,
                                "capture": {
                                    "name": "item_price",
                                    "subject": "product price",
                                    "value": "₹86",
                                },
                            },
                        },
                    )
                ]
            )
        )

        action = result.metadata["tool_args"]["action"]
        self.assertEqual(action["capture"]["value"], "<redacted:length=3>")
        self.assertNotIn("₹86", str(action))
        assert result.tool_response is not None
        command = result.tool_response.command
        assert command is not None
        assert command.payload.capture is not None
        self.assertEqual(command.payload.capture.value, "₹86")

    def test_tool_turn_logs_are_value_safe(self) -> None:
        """
        Tool-turn observability keeps counts and keys without leaking memory or capture values.
        """

        parser = ToolResponseParser()

        with self.assertLogs("fathom.core.services.parsing", level="INFO") as captured:
            parser.parse(
                response=self.__response(
                    calls=[
                        _Call(
                            name="execute_ui",
                            args={
                                "assistant_message": "store price",
                                "goal_completed": False,
                                "sub_goal_completed": False,
                                "memory_updates": {"filter_opened": "secret"},
                                "action": {
                                    "action_type": "store",
                                    "confidence": 0.9,
                                    "capture": {
                                        "name": "item_price",
                                        "subject": "product price",
                                        "value": "₹86",
                                    },
                                },
                            },
                        )
                    ]
                )
            )

        rendered = "\n".join(captured.output)
        events = [getattr(record, "event", None) for record in captured.records]
        parsed = next(
            record
            for record in captured.records
            if getattr(record, "event", None) == "tool.turn.parsed"
        )

        self.assertIn("tool.command.parsed", events)
        self.assertIn("tool.turn.parsed", events)
        self.assertEqual(parsed.__dict__["update.keys"], ["filter_opened"])
        self.assertEqual(parsed.__dict__["capture.value.length"], len("₹86"))
        self.assertNotIn("secret", rendered)
        self.assertNotIn("₹86", rendered)

    def test_validation_failure_feedback_and_logs_do_not_leak_capture_value(self) -> None:
        """
        Malformed STORE payloads expose exact validation messages without payload values.
        """

        parser = ToolResponseParser()

        with (
            self.assertLogs("fathom.core.services.parsing", level="WARNING") as captured,
            self.assertRaises(ToolValidationError) as context,
        ):
            parser.parse(
                response=self.__response(
                    calls=[
                        _Call(
                            name="execute_ui",
                            args={
                                "assistant_message": "store price",
                                "goal_completed": False,
                                "sub_goal_completed": False,
                                "action": {
                                    "action_type": "store",
                                    "capture": {
                                        "name": "item_price",
                                        "subject": "product price",
                                        "value": "₹86",
                                    },
                                },
                            },
                        )
                    ]
                )
            )

        rendered = "\n".join(captured.output)
        invalid = next(
            record
            for record in captured.records
            if getattr(record, "event", None) == "tool.schema.invalid"
        )

        self.assertEqual(invalid.__dict__["tool.name"], "execute_ui")
        self.assertIn("action.confidence", invalid.__dict__["tool.error.locations"])
        self.assertIn("Field required", invalid.__dict__["tool.error.messages"])
        self.assertEqual(
            invalid.__dict__["tool.action.keys"],
            ("action_type", "capture"),
        )
        self.assertNotIn("₹86", rendered)
        self.assertNotIn("₹86", context.exception.feedback.message)
        self.assertIn("execute_ui arguments validation failed", context.exception.feedback.message)
        self.assertIn("Field required", context.exception.feedback.message)

    def test_validation_failure_feedback_names_missing_validate_subject(self) -> None:
        """
        validate actions missing validation_subject produce actionable retry feedback.
        """

        parser = ToolResponseParser()

        with self.assertRaises(ToolValidationError) as context:
            parser.parse(
                response=self.__response(
                    calls=[
                        _Call(
                            name="execute_ui",
                            args={
                                "assistant_message": "check state",
                                "goal_completed": False,
                                "sub_goal_completed": False,
                                "action": {
                                    "action_type": "validate",
                                    "target_name": "HSR Layout",
                                    "confidence": 0.9,
                                },
                            },
                        )
                    ]
                )
            )

        self.assertIn(
            "validation_subject is required for action_type='validate'",
            context.exception.feedback.message,
        )
        self.assertTrue(context.exception.retryable)

    def test_recall_memory_does_not_invent_tool_data(self) -> None:
        """
        recall_memory must not fabricate a value before a real memory read exists.
        """

        parser = ToolResponseParser()
        result = parser.parse(
            response=self.__response(
                calls=[
                    _Call(
                        name="recall_memory",
                        args={
                            "key": "item_price",
                            "assistant_message": "Recall selected price",
                        },
                    )
                ]
            )
        )

        self.assertIsNone(result.action)
        assert result.tool_response is not None
        self.assertEqual(result.tool_response.data, ())
        self.assertEqual(len(result.tool_response.diagnostics), 1)
        self.assertEqual(
            result.tool_response.diagnostics[0].code,
            "MEMORY_RECALL_REQUESTED",
        )


class ToolResponseParserVisualAssessmentTest(unittest.TestCase):
    """
    Covers the shadow visual assessment riding the same primary response as the live action.
    """

    __ACTION = {
        "action_type": "swipe_up",
        "target_name": "Restaurant list area",
        "scroll_target": "Restaurant list area",
        "confidence": 0.84,
        "bbox": {
            "x": 0,
            "y": 858,
            "width": 1206,
            "height": 1396,
            "coordinate_system": "normalized",
        },
    }

    @staticmethod
    def __response(*, args: Dict[str, Any]) -> GenerateResult:
        return GenerateResult(
            content="", tool_calls=[_Call(name="execute_ui", args=args)], metrics={}
        )

    def test_assessment_and_action_decode_from_same_response(self) -> None:
        result = ToolResponseParser().parse(
            response=self.__response(
                args={
                    "assistant_message": "scroll",
                    "goal_completed": False,
                    "sub_goal_completed": False,
                    "action": self.__ACTION,
                    "visual_assessment": {
                        "verdict": "SATISFIED",
                        "confidence": 0.9,
                        "evidence": "results visible",
                    },
                }
            )
        )

        assert result.tool_response is not None
        assert result.tool_response.command is not None  # live action preserved
        assert result.visual_assessment is not None
        self.assertEqual(result.visual_assessment.verdict.value, "SATISFIED")
        self.assertFalse(result.assessment_malformed)

    def test_malformed_assessment_preserves_live_action(self) -> None:
        # Missing required 'evidence' -> malformed. The live action must still parse; no fabrication.
        result = ToolResponseParser().parse(
            response=self.__response(
                args={
                    "assistant_message": "scroll",
                    "goal_completed": False,
                    "sub_goal_completed": False,
                    "action": self.__ACTION,
                    "visual_assessment": {"verdict": "SATISFIED", "confidence": 0.9},
                }
            )
        )

        assert result.tool_response is not None
        assert result.tool_response.command is not None  # live action preserved
        self.assertIsNone(result.visual_assessment)  # never fabricated
        self.assertTrue(result.assessment_malformed)

    def test_absent_assessment_is_clean(self) -> None:
        result = ToolResponseParser().parse(
            response=self.__response(
                args={
                    "assistant_message": "scroll",
                    "goal_completed": False,
                    "sub_goal_completed": False,
                    "action": self.__ACTION,
                }
            )
        )

        self.assertIsNone(result.visual_assessment)
        self.assertFalse(result.assessment_malformed)


if __name__ == "__main__":
    unittest.main()
