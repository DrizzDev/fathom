from __future__ import annotations

import unittest

from fathom.constants.screen import ScreenCategory
from fathom.schemas.document import DocumentIndex, ScreenDocument, ScreenFlow, ScreenLink
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
        self.assertEqual(document.defects, [])
        self.assertEqual(document.fingerprints, 1)
        self.assertEqual(document.flow.outbound, [])

    def test_document_index_holds_metadata_and_documents(self) -> None:
        index = DocumentIndex(
            metadata=ReportMetadata(workflow="wf", package="com.app", generated_at="t"),
            documents=[],
        )

        self.assertEqual(index.metadata.workflow, "wf")
        self.assertEqual(index.documents, [])


if __name__ == "__main__":
    unittest.main()
