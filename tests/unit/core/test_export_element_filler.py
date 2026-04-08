"""Tests guarding the exported script against 'Validate element' leaks.

Two layers must cooperate to prevent the filler word from reaching
the exported script:

1. ``trace_payload.py`` routes validate/wait/swipe targets to their
   canonical subject fields instead of falling back to the literal
   ``"element"`` string.
2. ``ScriptExportStructuredPayload.enforce_policy`` rejects any LLM-
   emitted ``action_validations`` or ``final_validation`` line that
   contains the 'element' filler token as a last-line defense.
"""

from __future__ import annotations

import pytest

from fathom.core.services.exporter.trace_payload import build_export_payload
from fathom.schemas.export import (
    ScriptExportStructuredPayload,
    ScriptExportStructuredPayloadShape,
)


class TestTracePayloadRouting:
    """The JSON sent to the export LLM must surface each action kind's
    canonical subject instead of falling back to ``"element"``."""

    def test_validate_target_uses_validation_subject(self) -> None:
        payload = build_export_payload(
            step_results=[
                {
                    "action_type": "validate",
                    "validation_subject": "Popular Chains section visible",
                }
            ]
        )
        assert len(payload) == 1
        assert payload[0]["target"] == "Popular Chains section visible"
        assert payload[0]["validation_subject"] == "Popular Chains section visible"
        assert payload[0]["target"] != "element"

    def test_wait_target_uses_wait_subject(self) -> None:
        payload = build_export_payload(
            step_results=[{"action_type": "wait", "wait_subject": "app to load"}]
        )
        assert payload[0]["target"] == "app to load"
        assert payload[0]["wait_subject"] == "app to load"

    def test_swipe_target_uses_scroll_target(self) -> None:
        payload = build_export_payload(
            step_results=[{"action_type": "swipe_up", "scroll_target": "Popular Chains"}]
        )
        assert payload[0]["target"] == "Popular Chains"
        assert payload[0]["scroll_target"] == "Popular Chains"

    def test_tap_target_still_uses_export_target(self) -> None:
        """Regular tap actions unchanged — use export_target fallback chain."""

        payload = build_export_payload(
            step_results=[
                {
                    "action_type": "tap",
                    "export_target": "Login button",
                    "natural_language_target": "Login button",
                }
            ]
        )
        assert payload[0]["target"] == "Login button"

    def test_validate_with_missing_subject_falls_back_to_element(self) -> None:
        """Defense-in-depth: if validation_subject is somehow None, the
        'element' literal is the last-resort placeholder. The exporter's
        enforce_policy guard catches it downstream."""

        payload = build_export_payload(step_results=[{"action_type": "validate"}])
        assert payload[0]["target"] == "element"


def _policy_payload(
    *,
    final_validation: str = "Validate that the cart page is visible.",
    action_validations: dict | None = None,
) -> ScriptExportStructuredPayload:
    shape = ScriptExportStructuredPayloadShape(
        final_validation=final_validation,
        remaining_action_ids=["A1"],
        action_validations=action_validations or {},
    )
    return ScriptExportStructuredPayload.enforce_policy(
        shape=shape,
        action_catalog={"A1": "Tap on Cart button"},
        required_action_ids=["A1"],
        required_open_app_id=None,
        require_if_block=False,
        expected_validation_count=1,
    )


class TestExporterFillerGuard:
    def test_exporter_rejects_validate_element_in_action_validations(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _policy_payload(action_validations={"A1": "Validate element"})
        assert "element" in str(exc_info.value).lower()

    def test_exporter_rejects_element_visible_in_action_validations(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _policy_payload(action_validations={"A1": "Validate that the cart element is visible"})
        assert "element" in str(exc_info.value).lower()

    def test_exporter_accepts_clean_validation_line(self) -> None:
        # Should not raise.
        _policy_payload(
            action_validations={"A1": "Validate that the Popular Chains section is visible"}
        )

    def test_exporter_rejects_element_in_final_validation(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _policy_payload(final_validation="Validate that the element is visible.")
        assert "element" in str(exc_info.value).lower()

    def test_exporter_accepts_elements_plural_in_validation(self) -> None:
        """Word boundary lets legitimate plurals through."""

        _policy_payload(
            action_validations={"A1": "Validate that 3 elements are visible in the list"}
        )


def _policy_payload_with_catalog(
    catalog: dict[str, str],
    *,
    required_action_ids: list[str] | None = None,
) -> ScriptExportStructuredPayload:
    ids = required_action_ids or list(catalog.keys())
    shape = ScriptExportStructuredPayloadShape(
        final_validation="Validate that the cart page is visible.",
        remaining_action_ids=ids,
        action_validations={},
    )
    return ScriptExportStructuredPayload.enforce_policy(
        shape=shape,
        action_catalog=catalog,
        required_action_ids=ids,
        required_open_app_id=None,
        require_if_block=False,
        expected_validation_count=1,
    )


class TestExporterCatalogFillerGuard:
    """Same filler guard also applies to catalog-emitted lines so wait /
    scroll / tap / type actions cannot bleed the 'element' placeholder
    via their target-resolution path."""

    def test_exporter_rejects_wait_for_element(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _policy_payload_with_catalog({"A1": "Wait for element to appear"})
        assert "element" in str(exc_info.value).lower()

    def test_exporter_rejects_scroll_until_element(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _policy_payload_with_catalog({"A1": "Scroll down until element is visible"})
        assert "element" in str(exc_info.value).lower()

    def test_exporter_rejects_tap_on_element(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _policy_payload_with_catalog({"A1": "Tap on element"})
        assert "element" in str(exc_info.value).lower()

    def test_exporter_rejects_type_into_element(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _policy_payload_with_catalog({"A1": "Type 'hello' into element"})
        assert "element" in str(exc_info.value).lower()

    def test_exporter_accepts_clean_catalog_lines(self) -> None:
        _policy_payload_with_catalog(
            {
                "A1": "Tap on the Login button",
                "A2": "Type 'user@example.com' into the Email field",
                "A3": "Wait for the home screen to appear",
                "A4": "Scroll down until the Popular Chains section is visible",
            }
        )

    def test_exporter_accepts_elements_plural_in_catalog(self) -> None:
        _policy_payload_with_catalog({"A1": "Scroll until 3 elements are visible"})
