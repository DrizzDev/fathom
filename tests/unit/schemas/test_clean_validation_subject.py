"""Tests for :func:`fathom.schemas.actions.clean_validation_subject`.

The sanitizer is called from several non-Action code paths (legacy
parsing fallbacks, replay shims, dict-shaped payload construction)
that bypass the ``Action._enforce_validation_subject`` model
validator. That makes ``clean_validation_subject`` the only guard
those paths have against the ``"element"`` filler word leaking into
the exported script.
"""

from __future__ import annotations

import pytest

from fathom.schemas.actions import (
    GENERIC_TARGET_PLACEHOLDERS,
    clean_validation_subject,
)


class TestCleanValidationSubjectHappyPath:
    """Baseline: real subjects pass through untouched."""

    def test_concrete_subject_is_returned_verbatim(self) -> None:
        assert clean_validation_subject("cart is empty", fallback="x") == "cart is empty"

    def test_multiword_subject_under_cap_is_returned(self) -> None:
        out = clean_validation_subject("Add to cart button visible", fallback="x")
        assert out == "Add to cart button visible"

    def test_first_sentence_only(self) -> None:
        out = clean_validation_subject("home tab selected. other stuff", fallback="x")
        assert out == "home tab selected"

    def test_word_cap_kicks_in(self) -> None:
        out = clean_validation_subject(
            "one two three four five six seven eight nine ten", fallback="x"
        )
        assert out == "one two three four five six seven eight"


class TestCleanValidationSubjectPrefixStripping:
    """First-person / narrative prefixes are stripped before length capping."""

    def test_i_can_prefix_stripped(self) -> None:
        assert (
            clean_validation_subject("I can see the Home tab", fallback="x") == "see the Home tab"
        )

    def test_validating_prefix_stripped(self) -> None:
        assert clean_validation_subject("validating cart is empty", fallback="x") == "cart is empty"

    def test_the_presence_of_prefix_stripped(self) -> None:
        assert (
            clean_validation_subject("the presence of Submit button", fallback="x")
            == "Submit button"
        )


class TestCleanValidationSubjectEmptyInput:
    """Empty / whitespace inputs return the fallback."""

    def test_none_returns_fallback(self) -> None:
        assert clean_validation_subject(None, fallback="screen state") == "screen state"

    def test_empty_string_returns_fallback(self) -> None:
        assert clean_validation_subject("", fallback="screen state") == "screen state"

    def test_whitespace_only_returns_fallback(self) -> None:
        assert clean_validation_subject("   \t\n ", fallback="screen state") == "screen state"


class TestCleanValidationSubjectPlaceholderRejection:
    """Every member of ``GENERIC_TARGET_PLACEHOLDERS`` must be rejected."""

    @pytest.mark.parametrize("placeholder", sorted(GENERIC_TARGET_PLACEHOLDERS))
    def test_exact_placeholder_is_rejected(self, placeholder: str) -> None:
        assert clean_validation_subject(placeholder, fallback="screen state") == "screen state"

    @pytest.mark.parametrize("placeholder", sorted(GENERIC_TARGET_PLACEHOLDERS))
    def test_uppercase_placeholder_is_rejected(self, placeholder: str) -> None:
        assert (
            clean_validation_subject(placeholder.upper(), fallback="screen state") == "screen state"
        )

    def test_element_with_leading_whitespace_is_rejected(self) -> None:
        assert clean_validation_subject("  element  ", fallback="screen state") == "screen state"

    def test_unknown_is_rejected_not_returned_as_valid(self) -> None:
        """`unknown` is in the placeholder set — it must not sneak through."""

        assert clean_validation_subject("unknown", fallback="screen state") == "screen state"


class TestCleanValidationSubjectEmbeddedElementToken:
    """Subjects containing the standalone word ``element`` are rejected."""

    def test_element_at_end_is_rejected(self) -> None:
        out = clean_validation_subject("search box element", fallback="screen state")
        assert out == "screen state"

    def test_element_in_middle_is_rejected(self) -> None:
        out = clean_validation_subject("home element visible", fallback="screen state")
        assert out == "screen state"

    def test_element_is_matched_case_insensitively(self) -> None:
        out = clean_validation_subject("Home ELEMENT visible", fallback="screen state")
        assert out == "screen state"

    def test_elements_plural_is_not_rejected(self) -> None:
        """``elements`` (with the s) is not the forbidden token."""

        out = clean_validation_subject("three elements visible", fallback="screen state")
        assert out == "three elements visible"

    def test_elementary_substring_is_not_rejected(self) -> None:
        """Regex uses word boundary — substrings don't trip the guard."""

        out = clean_validation_subject("elementary school badge", fallback="screen state")
        assert out == "elementary school badge"


class TestCleanValidationSubjectInteractionWithStripping:
    """Prefix stripping then placeholder check — stripped placeholder still rejected."""

    def test_stripping_leaves_pure_placeholder_then_rejects(self) -> None:
        """`validating element` → strips prefix → lands on `element` → fallback."""

        out = clean_validation_subject("validating element", fallback="screen state")
        assert out == "screen state"

    def test_stripping_leaves_embedded_filler_then_rejects(self) -> None:
        """`checking the element at top` → strips prefix → still contains 'element'."""

        out = clean_validation_subject("checking the element at top", fallback="screen state")
        assert out == "screen state"
