"""Tests for the canonical target resolver in ``fathom.schemas.actions``.

``resolve_action_target`` is the single source of truth for "what does
this action point at?". Three pipelines used to implement this routing
independently (``trace_payload._resolve_target``,
``prompts.trace.extract_action_fields``, and
``Action.to_description``) with subtly different fallback orders; this
test suite locks down the unified behavior so the three delegates
stay in lock-step.
"""

from __future__ import annotations

import pytest

from fathom.constants import ActionType
from fathom.schemas.actions import (
    GENERIC_TARGET_PLACEHOLDERS,
    is_resolved_target,
    resolve_action_target,
)


class TestIsResolvedTarget:
    """Placeholder strings must be rejected so the chain keeps walking."""

    def test_none_is_unresolved(self) -> None:
        assert is_resolved_target(None) is False

    def test_empty_string_is_unresolved(self) -> None:
        assert is_resolved_target("") is False
        assert is_resolved_target("   ") is False

    def test_concrete_name_is_resolved(self) -> None:
        assert is_resolved_target("Search box") is True
        assert is_resolved_target("the 1st product") is True

    @pytest.mark.parametrize("placeholder", sorted(GENERIC_TARGET_PLACEHOLDERS))
    def test_every_placeholder_is_unresolved(self, placeholder: str) -> None:
        """Every member of GENERIC_TARGET_PLACEHOLDERS must be rejected.

        This includes 'unknown', which is the default fallback returned
        by ``resolve_action_target``; treating it as unresolved lets
        downstream consumers correctly skip steps where the LLM gave
        us nothing usable.
        """

        assert is_resolved_target(placeholder) is False
        assert is_resolved_target(placeholder.upper()) is False

    def test_case_and_whitespace_are_normalized(self) -> None:
        assert is_resolved_target("  Element  ") is False
        assert is_resolved_target("ELEMENT") is False


class TestResolveActionTargetRouting:
    """Each action kind has a canonical subject field that must win."""

    def test_validate_routes_to_validation_subject(self) -> None:
        result = resolve_action_target(
            action_type="validate",
            target_name="",
            validation_subject="cart is empty",
        )
        assert result == "cart is empty"

    def test_wait_routes_to_wait_subject(self) -> None:
        result = resolve_action_target(
            action_type="wait",
            target_name="",
            wait_subject="search results to appear",
        )
        assert result == "search results to appear"

    @pytest.mark.parametrize(
        "kind",
        ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "scroll"],
    )
    def test_swipe_and_scroll_route_to_scroll_target(self, kind: str) -> None:
        result = resolve_action_target(
            action_type=kind,
            target_name="",
            scroll_target="Vitamins and supplements",
        )
        assert result == "Vitamins and supplements"

    def test_tap_ignores_canonical_subject_fields(self) -> None:
        """Validation/wait/scroll subjects belong to other action kinds.

        A ``tap`` action with a bogus ``validation_subject`` must still
        resolve via the general chain — we don't want stale subject
        data from prior steps leaking into a tap's display name.
        """

        result = resolve_action_target(
            action_type="tap",
            target_name="Search box",
            validation_subject="cart is empty",
        )
        assert result == "Search box"


class TestResolveActionTargetGeneralChain:
    """After the kind-specific subject, the chain walks target_name → export_target → natural_language_target."""

    def test_target_name_wins_over_export_target(self) -> None:
        result = resolve_action_target(
            action_type="tap",
            target_name="Search box",
            export_target="Search box (export)",
            natural_language_target="Search box (legacy)",
        )
        assert result == "Search box"

    def test_export_target_wins_when_target_name_missing(self) -> None:
        result = resolve_action_target(
            action_type="tap",
            target_name=None,
            export_target="the 1st search result",
            natural_language_target="First result",
        )
        assert result == "the 1st search result"

    def test_natural_language_target_is_last_resort_before_label(self) -> None:
        result = resolve_action_target(
            action_type="tap",
            target_name=None,
            export_target=None,
            natural_language_target="Settings tab",
        )
        assert result == "Settings tab"

    def test_label_id_placeholder_when_nothing_else_resolves(self) -> None:
        """A bare label_id becomes 'label:N' — better than 'unknown'."""

        result = resolve_action_target(
            action_type="tap",
            label_id="7",
        )
        assert result == "label:7"

    def test_fallback_returned_when_everything_is_empty(self) -> None:
        result = resolve_action_target(action_type="tap")
        assert result == "unknown"

    def test_custom_fallback_respected(self) -> None:
        result = resolve_action_target(
            action_type="tap",
            fallback="no target available",
        )
        assert result == "no target available"


class TestResolveActionTargetPlaceholderSkipping:
    """Placeholder strings must be treated as if the field were blank."""

    def test_element_in_target_name_is_skipped(self) -> None:
        """The filler word 'element' must never win — fall through."""

        result = resolve_action_target(
            action_type="tap",
            target_name="element",
            export_target="Add to cart button",
        )
        assert result == "Add to cart button"

    def test_placeholder_chain_eventually_falls_through_to_fallback(self) -> None:
        result = resolve_action_target(
            action_type="tap",
            target_name="element",
            export_target="button",
            natural_language_target="icon",
        )
        assert result == "unknown"

    def test_validate_with_empty_subject_falls_through_to_target_name(self) -> None:
        """If the canonical subject is empty, fall back to the general chain.

        This matches the invariant that ``Action._enforce_validation_subject``
        would reject an empty subject at construction time — but the
        resolver is used for many non-Action shapes (StepResult dicts,
        ExecuteAction instances, history entries) where the subject
        may legitimately be missing.
        """

        result = resolve_action_target(
            action_type="validate",
            validation_subject=None,
            target_name="Home tab",
        )
        assert result == "Home tab"


class TestResolveActionTargetActionTypeNormalization:
    """Accepts both raw strings and ActionType enum members."""

    def test_enum_member_routes_correctly(self) -> None:
        result = resolve_action_target(
            action_type=ActionType.VALIDATE,
            validation_subject="Settings screen open",
        )
        assert result == "Settings screen open"

    def test_uppercase_string_routes_correctly(self) -> None:
        result = resolve_action_target(
            action_type="VALIDATE",
            validation_subject="Settings screen open",
        )
        assert result == "Settings screen open"

    def test_none_action_type_still_resolves_via_general_chain(self) -> None:
        result = resolve_action_target(
            action_type=None,
            target_name="Search box",
        )
        assert result == "Search box"
