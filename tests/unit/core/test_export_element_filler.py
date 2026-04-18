"""Tests guarding the exported script against 'Validate element' leaks.

Three layers must cooperate to prevent the filler word from reaching
the exported script:

1. ``action_catalog.build_action_catalog_from_steps`` routes targets per
   action kind and skips entries whose subject is missing or generic.
2. ``trace_payload.py`` routes validate/wait/swipe targets to their
   canonical subject fields instead of falling back to placeholders.
3. ``ScriptExportStructuredPayload.enforce_policy`` rejects any LLM-
   emitted catalog/``action_validations``/``final_validation`` line that
   contains the 'element' filler token as a last-line defense.
"""

from __future__ import annotations

import pytest

from fathom.core.services.exporter.action_catalog import (
    build_action_catalog_from_steps,
)
from fathom.core.services.exporter.trace_payload import build_export_payload
from fathom.core.services.normalizer import Normalizer
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

    def test_validate_with_missing_subject_falls_back_to_unknown(self) -> None:
        """Defense-in-depth: if validation_subject is somehow None, the
        canonical resolver returns ``"unknown"`` (never the historic
        filler ``"element"``). ``"unknown"`` is itself in
        ``GENERIC_TARGET_PLACEHOLDERS`` so downstream consumers that
        run ``is_resolved_target`` on it will still skip the line, and
        the exporter's ``enforce_policy`` guard rejects any validation
        string containing the filler word."""

        payload = build_export_payload(step_results=[{"action_type": "validate"}])
        assert payload[0]["target"] == "unknown"
        assert payload[0]["target"] != "element"


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

    def test_exporter_rejects_unreferenced_validate_catalog_entry(self) -> None:
        """``to_script`` auto-injects every validate catalog entry regardless
        of whether the LLM referenced it in ``ordered_action_ids``. The
        guard must scan the ENTIRE catalog, not just referenced IDs, so
        auto-injected "Validate element" lines can't bypass the check."""

        shape = ScriptExportStructuredPayloadShape(
            final_validation="Validate that the cart page is visible.",
            remaining_action_ids=["A1"],
            action_validations={},
        )
        with pytest.raises(ValueError) as exc_info:
            ScriptExportStructuredPayload.enforce_policy(
                shape=shape,
                action_catalog={
                    "A1": "Tap on the Cart button",
                    # Auto-injected validate entry the LLM never referenced.
                    "A2": "Validate element",
                },
                required_action_ids=["A1"],
                required_open_app_id=None,
                require_if_block=False,
                expected_validation_count=1,
            )
        assert "element" in str(exc_info.value).lower()


class TestNormalizerRejectsEmptyTarget:
    """Normalizer.action must raise rather than silently render the
    'element' filler when callers pass an empty target."""

    def test_validate_with_empty_target_and_subject_raises(self) -> None:
        with pytest.raises(ValueError, match="empty target"):
            Normalizer.action(action_type="validate", target="", validation_subject=None)

    def test_validate_with_empty_target_uses_subject(self) -> None:
        result = Normalizer.action(
            action_type="validate",
            target="",
            validation_subject="Popular Chains section visible",
        )
        assert result == "Validate Popular Chains section visible"

    def test_tap_with_empty_target_raises(self) -> None:
        with pytest.raises(ValueError, match="empty target"):
            Normalizer.action(action_type="tap", target="")

    def test_wait_with_empty_target_raises(self) -> None:
        with pytest.raises(ValueError, match="empty target"):
            Normalizer.action(action_type="wait", target="")


class TestActionCatalogSkipsUnresolvedEntries:
    """The catalog builder must drop steps whose target field zoo
    resolves to nothing or to a generic placeholder so the rendered
    catalog never contains 'Validate element' / 'Tap on element'."""

    def test_validate_step_without_subject_is_skipped(self) -> None:
        catalog, _required, _open_app = build_action_catalog_from_steps(
            step_results=[
                {
                    "action_type": "validate",
                    "validation_subject": None,
                    "export_target": None,
                    "natural_language_target": None,
                },
                {
                    "action_type": "tap",
                    "export_target": "Login button",
                    "natural_language_target": "Login button",
                },
            ],
            package_name="",
            intent="",
        )
        descriptions = [entry.description for entry in catalog.values()]
        assert any("Login button" in desc for desc in descriptions)
        assert not any("element" in desc.lower() for desc in descriptions)

    def test_validate_step_with_element_placeholder_is_skipped(self) -> None:
        catalog, _required, _open_app = build_action_catalog_from_steps(
            step_results=[
                {
                    "action_type": "validate",
                    "validation_subject": None,
                    "export_target": "element",
                    "natural_language_target": "element",
                }
            ],
            package_name="",
            intent="",
        )
        assert all("element" not in entry.description.lower() for entry in catalog.values())

    def test_validate_step_with_subject_is_kept(self) -> None:
        catalog, _required, _open_app = build_action_catalog_from_steps(
            step_results=[
                {
                    "action_type": "validate",
                    "validation_subject": "Popular Chains section visible",
                }
            ],
            package_name="",
            intent="",
        )
        descriptions = [entry.description for entry in catalog.values()]
        assert any("Popular Chains" in desc for desc in descriptions)

    def test_tap_step_with_only_placeholder_target_is_skipped(self) -> None:
        catalog, _required, _open_app = build_action_catalog_from_steps(
            step_results=[
                {
                    "action_type": "tap",
                    "export_target": "element",
                    "natural_language_target": "button",
                }
            ],
            package_name="",
            intent="",
        )
        assert all("element" not in entry.description.lower() for entry in catalog.values())


class TestTraceHistoryFormatting:
    """The CURRENT_TRACE block sent to the planning LLM is rendered via
    extract_action_fields. That helper used to always read action.target,
    which for validate actions fell back to the 'element' placeholder —
    surfacing 'VALIDATE:element' in the prompt even when the underlying
    action had a clean validation_subject. Lock in the per-action-kind
    routing."""

    def test_validate_entry_uses_validation_subject(self) -> None:
        from fathom.core.prompts.trace import extract_action_fields

        entry = {
            "action": {
                "action_type": "validate",
                "target": "element",
                "validation_subject": "Dineout section active and Koramangala location visible",
            }
        }
        action_type, desc = extract_action_fields(entry)
        assert action_type == "validate"
        assert desc == "Dineout section active and Koramangala location visible"

    def test_wait_entry_uses_wait_subject(self) -> None:
        from fathom.core.prompts.trace import extract_action_fields

        entry = {
            "action": {
                "action_type": "wait",
                "target": "element",
                "wait_subject": "home screen to appear",
            }
        }
        _, desc = extract_action_fields(entry)
        assert desc == "home screen to appear"

    def test_swipe_entry_uses_scroll_target(self) -> None:
        from fathom.core.prompts.trace import extract_action_fields

        entry = {
            "action": {
                "action_type": "swipe_up",
                "target": "element",
                "scroll_target": "Popular Chains section",
            }
        }
        _, desc = extract_action_fields(entry)
        assert desc == "Popular Chains section"

    def test_tap_entry_falls_back_through_target_chain(self) -> None:
        from fathom.core.prompts.trace import extract_action_fields

        entry = {
            "action": {
                "action_type": "tap",
                "target": "Login button",
            }
        }
        _, desc = extract_action_fields(entry)
        assert desc == "Login button"

    def test_generic_placeholder_target_is_skipped(self) -> None:
        from fathom.core.prompts.trace import extract_action_fields

        entry = {
            "action": {
                "action_type": "tap",
                "target": "element",
                "export_target": "Submit button",
            }
        }
        _, desc = extract_action_fields(entry)
        assert desc == "Submit button"


class TestStepRecordPersistsValidationSubject:
    """StepResult.to_record() used to drop validation_subject when
    serializing runtime steps to history.json. The exporter then
    rebuilt step dicts from disk, saw no validation_subject field,
    and fell back to the 'element' placeholder. Lock in that the
    serializer round-trips the subject so scripts never regress to
    'Validate element' for checkpointed traces."""

    def test_validate_step_round_trips_validation_subject(self) -> None:
        from fathom.schemas.actions import Action
        from fathom.schemas.steps import Step, StepResult

        subject = "Dineout section active and Koramangala location visible"
        step = Step(
            action=Action(
                action_type="validate",
                target=subject,
                validation_subject=subject,
                confidence=1.0,
                rationale="goal check",
            ),
            screen_hash="abc123",
            step_number=8,
            event_type="validation",
        )
        result = StepResult(
            step=step,
            success=True,
            pre_hash="abc123",
            post_hash="abc123",
            screen_changed=False,
            duration=0,
        )

        record = result.to_record().model_dump()
        assert record["validation_subject"] == subject

        # Catalog builder consumes the dict shape persisted to disk.
        catalog, _required, _open_app = build_action_catalog_from_steps(
            step_results=[record], package_name="", intent=""
        )
        descriptions = [entry.description for entry in catalog.values()]
        assert any(subject in desc for desc in descriptions)
        assert not any("element" in desc.lower() for desc in descriptions)
