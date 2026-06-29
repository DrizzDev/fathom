from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.constants.document import SCREEN_DOCUMENT_SCHEMA_VERSION
from fathom.constants.exploration import ExpectedOutcome
from fathom.constants.screen import ScreenCategory
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.document import (
    DocumentIndex,
    LinkSemantics,
    ScreenDocument,
    ScreenFlow,
    ScreenLink,
)
from fathom.schemas.report import ReportMetadata


class TestScreenDocumentModels(unittest.TestCase):
    """The per-screen document models default to safe empty structures."""

    def test_screen_link_defaults_count_to_one(self) -> None:
        link = ScreenLink(action="tap", screen="Cart")

        self.assertEqual(link.count, 1)
        self.assertIsNone(link.element)

    def test_screen_flow_defaults_to_empty_lists(self) -> None:
        flow = ScreenFlow()

        self.assertEqual(flow.inbound, [])
        self.assertEqual(flow.outbound, [])

    def test_screen_document_minimal_construction(self) -> None:
        document = ScreenDocument(
            slug="home",
            title="Home",
            category=ScreenCategory.HOME,
            activity="com.app/.MainActivity",
        )

        self.assertEqual(document.purpose, "")
        self.assertEqual(document.narrative, "")
        self.assertEqual(document.elements, [])
        self.assertEqual(document.actions, [])
        self.assertEqual(document.interactions, [])
        self.assertEqual(document.defects, [])
        self.assertEqual(document.fingerprints, 1)
        self.assertEqual(document.flow.outbound, [])

    def test_screen_document_carries_structured_elements_and_actions(self) -> None:
        document = ScreenDocument(
            slug="cart",
            title="Cart",
            category=ScreenCategory.OTHER,
            activity="com.app/.CartActivity",
            elements=["'Checkout' button", "Item list"],
            actions=["Proceed to checkout", "Remove an item"],
        )

        self.assertEqual(document.elements, ["'Checkout' button", "Item list"])
        self.assertEqual(document.actions, ["Proceed to checkout", "Remove an item"])

    def test_screen_link_defaults_value_and_semantics_to_none(self) -> None:
        link = ScreenLink(action="tap", screen="Cart")

        self.assertIsNone(link.value)
        self.assertIsNone(link.semantics)

    def test_document_index_holds_metadata_and_documents(self) -> None:
        index = DocumentIndex(
            metadata=ReportMetadata(workflow="wf", package="com.app", generated_at="t"),
            documents=[],
        )

        self.assertEqual(index.metadata.workflow, "wf")
        self.assertEqual(index.documents, [])
        self.assertEqual(index.schema_version, SCREEN_DOCUMENT_SCHEMA_VERSION)


class TestLinkSemantics(unittest.TestCase):
    """Link semantics are derived from the action that drove the transition."""

    def test_of_maps_action_intent_and_classification(self) -> None:
        action = Action(
            action_type=ActionType.TAP,
            rationale="open the cart",
            natural_language_target="Cart",
            region="top_bar",
            element_category="global_navigation",
            overlay_detected=True,
            expected_outcome=ExpectedOutcome.NEW_SCREEN,
            bounds=Bounds(x=10, y=20, width=5, height=5),
        )

        semantics = LinkSemantics.of(action=action)

        self.assertEqual(semantics.outcome, "new_screen")
        self.assertEqual(semantics.category, "global_navigation")
        self.assertEqual(semantics.region, "top_bar")
        self.assertTrue(semantics.overlay)
        self.assertEqual(semantics.rationale, "open the cart")

    def test_of_leaves_absent_fields_none(self) -> None:
        action = Action(action_type=ActionType.BACK, rationale="")

        semantics = LinkSemantics.of(action=action)

        self.assertIsNone(semantics.outcome)
        self.assertIsNone(semantics.region)
        self.assertIsNone(semantics.rationale)
        self.assertFalse(semantics.overlay)


if __name__ == "__main__":
    unittest.main()
