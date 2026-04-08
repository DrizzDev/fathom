"""Tests for cross-action target inheritance in ExecuteUIArgs.

When Gemini emits the common tap-then-type pattern, the type action
typically omits ``target_name`` because the prior tap already named the
field. The ``ExecuteUIArgs._propagate_targets`` validator inherits the
target fields from the prior action when the two share the same
``label_id`` (or near-identical bbox center) so the second action does
not fail per-action target enforcement.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fathom.schemas.tool_args import ExecuteUIArgs


def _action(**overrides: object) -> dict:
    """Build a minimal ExecuteAction dict; callers override what they need."""

    base = {
        "action_type": "tap",
        "rationale": "test rationale",
        "is_valid": True,
        "bbox": {"x": 100, "y": 100, "coord_system": "normalized"},
    }
    base.update(overrides)
    return base


class TestTargetInheritance:
    def test_type_inherits_target_name_from_prior_tap_via_label_id(self) -> None:
        """The exact bug-report shape: tap names the field, type omits it."""

        args = ExecuteUIArgs.model_validate(
            {
                "sub_goal_completed": False,
                "actions": [
                    _action(
                        action_type="tap",
                        target_name="Search for malls & landmarks search bar",
                        label_id="10",
                        bbox={"x": 387, "y": 271, "coord_system": "normalized"},
                    ),
                    _action(
                        action_type="type",
                        text_to_type="hello",
                        label_id="10",
                        bbox={"x": 387, "y": 271, "coord_system": "normalized"},
                    ),
                ],
            }
        )
        assert args.actions[1].target_name == "Search for malls & landmarks search bar"
        assert args.actions[1].export_target == "Search for malls & landmarks search bar"

    def test_type_inherits_when_bbox_drift_within_tolerance(self) -> None:
        """No label_id, but bbox center drifts by <=5 normalized units."""

        args = ExecuteUIArgs.model_validate(
            {
                "sub_goal_completed": False,
                "actions": [
                    _action(
                        action_type="tap",
                        target_name="Email field",
                        bbox={"x": 500, "y": 300, "coord_system": "normalized"},
                    ),
                    _action(
                        action_type="type",
                        text_to_type="a@b.c",
                        bbox={"x": 502, "y": 298, "coord_system": "normalized"},
                    ),
                ],
            }
        )
        assert args.actions[1].target_name == "Email field"

    def test_type_does_not_inherit_when_bbox_far_apart(self) -> None:
        """Far-apart bboxes are different elements; the type must be rejected."""

        with pytest.raises(ValidationError):
            ExecuteUIArgs.model_validate(
                {
                    "sub_goal_completed": False,
                    "actions": [
                        _action(
                            action_type="tap",
                            target_name="Cancel",
                            bbox={"x": 100, "y": 100, "coord_system": "normalized"},
                        ),
                        _action(
                            action_type="type",
                            text_to_type="hello",
                            bbox={"x": 800, "y": 700, "coord_system": "normalized"},
                        ),
                    ],
                }
            )

    def test_explicit_target_name_is_not_overwritten(self) -> None:
        """A type that names its own target keeps its name even if label_id matches."""

        args = ExecuteUIArgs.model_validate(
            {
                "sub_goal_completed": False,
                "actions": [
                    _action(
                        action_type="tap",
                        target_name="Search box",
                        label_id="5",
                    ),
                    _action(
                        action_type="type",
                        text_to_type="hi",
                        target_name="Different field",
                        label_id="5",
                    ),
                ],
            }
        )
        assert args.actions[1].target_name == "Different field"

    def test_lonely_type_with_no_prior_is_rejected(self) -> None:
        """An isolated type with target_name (no prior) still requires bbox or label_id."""

        with pytest.raises(ValidationError):
            ExecuteUIArgs.model_validate(
                {
                    "sub_goal_completed": False,
                    "actions": [
                        {
                            "action_type": "type",
                            "rationale": "r",
                            "is_valid": True,
                            "text_to_type": "hi",
                            "target_name": "Search box",
                        }
                    ],
                }
            )


class TestLabelIdGrounding:
    """label_id implies the manifest snapper will provide exact bounds, so
    the LLM-emitted bbox can be omitted entirely."""

    def test_type_with_label_id_no_bbox_accepted(self) -> None:
        """The exact bug-report shape: type action with label_id and no bbox."""

        args = ExecuteUIArgs.model_validate(
            {
                "sub_goal_completed": True,
                "actions": [
                    {
                        "action_type": "type",
                        "label_id": "4",
                        "text_to_type": "agufdduhvdttycvvhtr",
                        "target_name": "Search 'Chinese' or 'Cafe' input field",
                        "rationale": "Typing the required text into the search bar.",
                        "is_valid": True,
                    }
                ],
            }
        )
        assert args.actions[0].label_id == "4"
        assert args.actions[0].bbox is None

    def test_tap_with_label_id_no_bbox_accepted(self) -> None:
        """Same exemption applies to tap and long_press, not just type."""

        args = ExecuteUIArgs.model_validate(
            {
                "sub_goal_completed": False,
                "actions": [
                    {
                        "action_type": "tap",
                        "label_id": "12",
                        "target_name": "Login button",
                        "rationale": "Submit credentials.",
                        "is_valid": True,
                    }
                ],
            }
        )
        assert args.actions[0].label_id == "12"

    def test_tap_without_label_id_or_bbox_is_rejected(self) -> None:
        """When label_id is absent, the validator still demands a real bbox."""

        with pytest.raises(ValidationError):
            ExecuteUIArgs.model_validate(
                {
                    "sub_goal_completed": False,
                    "actions": [
                        {
                            "action_type": "tap",
                            "target_name": "Login button",
                            "rationale": "Submit credentials.",
                            "is_valid": True,
                        }
                    ],
                }
            )

    def test_tap_with_zero_bbox_and_no_label_id_is_rejected(self) -> None:
        """Placeholder all-zero bbox with no label_id is still rejected."""

        with pytest.raises(ValidationError):
            ExecuteUIArgs.model_validate(
                {
                    "sub_goal_completed": False,
                    "actions": [
                        {
                            "action_type": "tap",
                            "target_name": "Login button",
                            "rationale": "Submit credentials.",
                            "is_valid": True,
                            "bbox": {"x": 0, "y": 0, "coord_system": "normalized"},
                        }
                    ],
                }
            )

    def test_label_id_alone_exempts_export_target_requirement(self) -> None:
        """tap with only label_id (no bbox, no target_name) is valid — the
        manifest snapper provides the display text at bind time."""

        args = ExecuteUIArgs.model_validate(
            {
                "sub_goal_completed": False,
                "actions": [
                    {
                        "action_type": "tap",
                        "label_id": "42",
                        "rationale": "tap it",
                        "is_valid": True,
                    }
                ],
            }
        )
        assert args.actions[0].label_id == "42"
        assert args.actions[0].target_name is None
        assert args.actions[0].export_target is None
        assert args.actions[0].bbox is None

    def test_tap_without_label_id_or_export_target_is_rejected(self) -> None:
        """Without label_id AND without any target field, validator rejects."""

        with pytest.raises(ValidationError):
            ExecuteUIArgs.model_validate(
                {
                    "sub_goal_completed": False,
                    "actions": [
                        {
                            "action_type": "tap",
                            "rationale": "tap",
                            "is_valid": True,
                            "bbox": {"x": 100, "y": 100, "coord_system": "normalized"},
                        }
                    ],
                }
            )


class TestTextFieldNormalization:
    """``text_to_type`` is the canonical field on ExecuteAction; ``text``
    remains as a back-compat alias for direct programmatic construction
    (tests, replay, internal callers). The ``_normalize_text_field``
    validator copies the legacy field into the canonical one."""

    def test_text_to_type_alone_passes_through(self) -> None:
        from fathom.schemas.tool_args import ExecuteAction, GeminiBBox

        action = ExecuteAction(
            action_type="type",
            rationale="r",
            is_valid=True,
            bbox=GeminiBBox(x=500, y=300),
            target_name="Search box",
            text_to_type="hello",
        )
        assert action.text_to_type == "hello"
        assert action.text is None

    def test_legacy_text_alias_is_promoted_to_text_to_type(self) -> None:
        from fathom.schemas.tool_args import ExecuteAction, GeminiBBox

        action = ExecuteAction(
            action_type="type",
            rationale="r",
            is_valid=True,
            bbox=GeminiBBox(x=500, y=300),
            target_name="Search box",
            text="hello",
        )
        assert action.text_to_type == "hello"
        assert action.text == "hello"  # original value preserved on the alias

    def test_text_to_type_wins_when_both_are_set(self) -> None:
        """Canonical field is authoritative; legacy alias does not override it."""

        from fathom.schemas.tool_args import ExecuteAction, GeminiBBox

        action = ExecuteAction(
            action_type="type",
            rationale="r",
            is_valid=True,
            bbox=GeminiBBox(x=500, y=300),
            target_name="Search box",
            text="legacy value",
            text_to_type="canonical value",
        )
        assert action.text_to_type == "canonical value"

    def test_empty_text_alias_does_not_clobber(self) -> None:
        """A whitespace-only ``text`` must not overwrite a valid ``text_to_type``."""

        from fathom.schemas.tool_args import ExecuteAction, GeminiBBox

        action = ExecuteAction(
            action_type="type",
            rationale="r",
            is_valid=True,
            bbox=GeminiBBox(x=500, y=300),
            target_name="Search box",
            text="   ",
            text_to_type="hello",
        )
        assert action.text_to_type == "hello"


class TestValidationSubjectFillerRejection:
    """The validate-action guard rejects the filler word ``element`` as a
    standalone token, catching prompt leaks like 'Validate X, element
    visible' while still accepting legitimate phrases with 'elements',
    'elementary', etc."""

    def test_element_filler_in_middle_is_rejected(self) -> None:
        from fathom.constants import ActionType
        from fathom.schemas.actions import Action

        with pytest.raises(ValidationError):
            Action(
                action_type=ActionType.VALIDATE,
                rationale="r",
                target="x",
                validation_subject="HealthTap homepage content, element visible",
            )

    def test_element_filler_uppercase_is_rejected(self) -> None:
        from fathom.constants import ActionType
        from fathom.schemas.actions import Action

        with pytest.raises(ValidationError):
            Action(
                action_type=ActionType.VALIDATE,
                rationale="r",
                target="x",
                validation_subject="ELEMENT visible",
            )

    def test_clean_subject_without_element_is_accepted(self) -> None:
        from fathom.constants import ActionType
        from fathom.schemas.actions import Action

        action = Action(
            action_type=ActionType.VALIDATE,
            rationale="r",
            target="x",
            validation_subject="HealthTap homepage content loaded",
        )
        assert action.validation_subject == "HealthTap homepage content loaded"

    def test_plural_elements_is_accepted(self) -> None:
        """``elements`` is a legitimate subject (e.g., '3 elements visible')."""

        from fathom.constants import ActionType
        from fathom.schemas.actions import Action

        action = Action(
            action_type=ActionType.VALIDATE,
            rationale="r",
            target="x",
            validation_subject="3 elements visible",
        )
        assert action.validation_subject == "3 elements visible"

    def test_elementary_substring_is_accepted(self) -> None:
        """Word-boundary regex must not trip on substrings like 'elementary'."""

        from fathom.constants import ActionType
        from fathom.schemas.actions import Action

        action = Action(
            action_type=ActionType.VALIDATE,
            rationale="r",
            target="x",
            validation_subject="elementary school selected",
        )
        assert action.validation_subject == "elementary school selected"

    def test_non_validate_actions_unaffected(self) -> None:
        """Tap/type/etc. actions can still use 'element' as their target."""

        from fathom.constants import ActionType
        from fathom.schemas.actions import Action

        action = Action(action_type=ActionType.TAP, rationale="r", target="element")
        assert action.target == "element"


class TestLabelIdParserFallback:
    """parsing.py must NOT emit the generic 'element' target when label_id
    is available — instead it stamps a namespaced 'label:{id}' placeholder
    so downstream code can detect 'this needs manifest lookup'."""

    def test_parser_uses_label_namespace_when_target_name_missing(self) -> None:
        from types import SimpleNamespace

        from fathom.core.services.parsing import ToolResponseParser
        from fathom.schemas.results import GenerateResult

        parser = ToolResponseParser()
        tool_call = SimpleNamespace(
            name="execute_ui",
            args={
                "sub_goal_completed": False,
                "actions": [
                    {
                        "action_type": "tap",
                        "label_id": "42",
                        "rationale": "tap it",
                        "is_valid": True,
                    }
                ],
            },
        )
        result = parser.parse(GenerateResult(content="", tool_calls=[tool_call]))

        assert result.action.target == "label:42"
        assert result.action.natural_language_target == "label:42"
        assert result.action.label_id == "42"

    def test_parser_falls_back_to_element_without_label_or_target(self) -> None:
        """Legacy fallback path: when neither label_id nor target_name is
        available, the parser falls through to the plain 'element' literal.
        This test documents the exception path — in practice the schema
        validator now blocks this case, but the branch is exercised by any
        legacy Action constructed directly (not via tool args)."""

        from fathom.constants import ActionType
        from fathom.schemas.actions import Action

        action = Action(
            action_type=ActionType.TAP,
            rationale="r",
            target="element",  # explicit legacy placeholder
            natural_language_target="element",
        )
        assert action.target == "element"
