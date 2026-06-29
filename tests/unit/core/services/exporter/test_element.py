from __future__ import annotations

import unittest

from fathom.core.services.exporter.element import ElementText


class TestElementText(unittest.TestCase):
    """A link target is reduced to the element's exact visible text for grounding."""

    def test_strips_a_trailing_generic_descriptor(self) -> None:
        # Real targets observed during a crawl: the model appends the element type.
        self.assertEqual(ElementText.visible(target="Continue button"), "Continue")
        self.assertEqual(ElementText.visible(target="Female button"), "Female")
        self.assertEqual(ElementText.visible(target="Gallery icon"), "Gallery")
        self.assertEqual(ElementText.visible(target="Someone else option"), "Someone else")
        self.assertEqual(ElementText.visible(target="Bolt tab"), "Bolt")

    def test_prefers_the_longest_matching_descriptor(self) -> None:
        # "input field" must be stripped whole, not just its "field" tail.
        self.assertEqual(ElementText.visible(target="Email input field"), "Email")
        self.assertEqual(
            ElementText.visible(target="Search conditions or doctors search bar"),
            "Search conditions or doctors",
        )

    def test_strips_only_one_descriptor(self) -> None:
        # "Back arrow button" keeps "arrow"; only the trailing descriptor goes.
        self.assertEqual(ElementText.visible(target="Back arrow button"), "Back arrow")

    def test_leaves_targets_without_a_trailing_descriptor_untouched(self) -> None:
        self.assertEqual(ElementText.visible(target="Restaurant card"), "Restaurant card")
        self.assertEqual(ElementText.visible(target="Open settings"), "Open settings")
        self.assertEqual(
            ElementText.visible(target="2:00pm appointment slot"), "2:00pm appointment slot"
        )

    def test_keeps_a_bare_descriptor_with_no_visible_text(self) -> None:
        # Nothing meaningful remains, so the original target is preserved.
        self.assertEqual(ElementText.visible(target="button"), "button")
        self.assertEqual(ElementText.visible(target="a field"), "a field")

    def test_trims_surrounding_whitespace(self) -> None:
        self.assertEqual(ElementText.visible(target="  Save button  "), "Save")
