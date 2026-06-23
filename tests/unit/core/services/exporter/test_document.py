from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional

from fathom.constants.defect import DefectSignal, DefectSource
from fathom.constants.screen import ScreenCategory
from fathom.core.services.exporter.document import (
    ScreenDocumentExporter,
    ScreenDocumentRenderer,
)
from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph
from fathom.schemas.defect import Defect, DefectEvidence
from fathom.schemas.document import DocumentIndex, ScreenDocument, ScreenFlow, ScreenLink
from fathom.schemas.report import ReportMetadata

_HOME_A = "0000000000000000"
_HOME_B = "1111111111111111"
_DETAIL = "ffffffffffffffff"
_ACCOUNT = "aaaaaaaaaaaaaaaa"


def _screen_row(
    *,
    visual_hash: str,
    activity: str,
    category: str,
    description: Optional[str] = None,
    rich: Optional[str] = None,
    visits: int = 1,
    structure: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "visual_hash": visual_hash,
        "activity": activity,
        "description": description,
        "first_seen": 1,
        "last_seen": 2,
        "visit_count": visits,
        "rich_description": rich,
        "activity_hash": None,
        "xml_hash": None,
        "interaction_hash": None,
        "structure_hash": structure,
        "exhausted": False,
        "relevance": "unscoped",
        "category": category,
    }


def _transition_row(
    *, source: str, destination: str, action_type: str = "tap", action_target: str = ""
) -> Dict[str, Any]:
    return {
        "source_hash": source,
        "destination_hash": destination,
        "action_type": action_type,
        "action_target": action_target,
        "coord_bucket": None,
        "coord_region": None,
        "element_category": None,
        "count": 1,
        "first_seen": 1,
        "last_seen": 2,
    }


class _Provider:
    """Minimal memory provider serving canned screen and transition rows to load()."""

    def __init__(self, *, screens: List[Dict[str, Any]], transitions: List[Dict[str, Any]]) -> None:
        self.__screens = screens
        self.__transitions = transitions

    async def get_all_screens(self) -> List[Dict[str, Any]]:
        return self.__screens

    async def get_all_transitions(self) -> List[Dict[str, Any]]:
        return self.__transitions


def _metadata() -> ReportMetadata:
    return ReportMetadata(workflow="wf", package="com.app", generated_at="2026-06-23T00:00:00")


class TestScreenDocumentExporter(unittest.IsolatedAsyncioTestCase):
    """The exporter groups fingerprints into logical screens and resolves their flow."""

    async def asyncSetUp(self) -> None:
        screens = [
            _screen_row(
                visual_hash=_HOME_A,
                activity="com.app/.MainActivity",
                category="home",
                description="Home feed",
                rich="Home prose",
                visits=5,
            ),
            _screen_row(
                visual_hash=_HOME_B,
                activity="com.app/.MainActivity",
                category="home",
                description="Home feed scrolled further",
                visits=3,
            ),
            _screen_row(
                visual_hash=_DETAIL,
                activity="com.app/.MainActivity",
                category="detail",
                description="Restaurant detail",
                rich="Detail prose",
                visits=2,
            ),
            _screen_row(
                visual_hash=_ACCOUNT,
                activity="com.app/.AccountActivity",
                category="settings",
                description="Account settings",
                rich="Account prose",
                visits=1,
            ),
        ]
        transitions = [
            _transition_row(source=_HOME_A, destination=_DETAIL, action_target="Restaurant card"),
            _transition_row(source=_HOME_A, destination=_HOME_B, action_type="swipe_up"),
            _transition_row(source=_DETAIL, destination=_ACCOUNT, action_target="Open settings"),
        ]
        self.__graph = KnowledgeGraph(provider=_Provider(screens=screens, transitions=transitions))
        await self.__graph.load()
        self.__exporter = ScreenDocumentExporter()

    def __build(self, *, defects: Optional[List[Defect]] = None) -> DocumentIndex:
        return self.__exporter.build(
            graph=self.__graph, defects=defects or [], metadata=_metadata()
        )

    def __by_slug(self, index: DocumentIndex) -> Dict[str, ScreenDocument]:
        return {document.slug: document for document in index.documents}

    async def test_one_document_per_canonical_node(self) -> None:
        index = self.__build()

        # Four distinct nodes -> four docs (no activity/category collapse).
        self.assertEqual(len(index.documents), 4)

    async def test_title_and_slug_come_from_the_node_description(self) -> None:
        home = self.__by_slug(self.__build())["home-feed"]

        self.assertEqual(home.title, "Home feed")
        self.assertEqual(home.visits, 5)
        self.assertEqual(home.category, ScreenCategory.HOME)

    async def test_orders_documents_by_visits_descending(self) -> None:
        index = self.__build()

        self.assertEqual(
            [document.slug for document in index.documents],
            ["home-feed", "home-feed-scrolled-further", "restaurant-detail", "account-settings"],
        )

    async def test_outbound_links_resolve_cross_node_targets(self) -> None:
        home = self.__by_slug(self.__build())["home-feed"]

        # Both edges out of HOME_A now cross to distinct nodes; self-loops only are skipped.
        targets = {link.screen for link in home.flow.outbound}
        self.assertEqual(targets, {"Restaurant detail", "Home feed scrolled further"})

    async def test_inbound_links_name_the_source_screen(self) -> None:
        detail = self.__by_slug(self.__build())["restaurant-detail"]

        self.assertEqual([link.screen for link in detail.flow.inbound], ["Home feed"])
        self.assertEqual([link.screen for link in detail.flow.outbound], ["Account settings"])

    async def test_defects_are_bucketed_onto_their_node(self) -> None:
        defect = Defect.from_signal(
            signal=DefectSignal.OVERLAP_CLIPPING,
            source=DefectSource.POST_RUN,
            summary="Button overlaps title",
            evidence=DefectEvidence(screen=_DETAIL),
        )
        by_slug = self.__by_slug(self.__build(defects=[defect]))

        self.assertEqual(len(by_slug["restaurant-detail"].defects), 1)
        self.assertEqual(by_slug["home-feed"].defects, [])

    async def test_narrative_is_the_nodes_own_rich_description(self) -> None:
        by_slug = self.__by_slug(self.__build())

        self.assertEqual(by_slug["home-feed"].narrative, "Home prose")
        self.assertEqual(by_slug["restaurant-detail"].narrative, "Detail prose")
        # HOME_B carries no rich description of its own; there is no activity-level fallback.
        self.assertEqual(by_slug["home-feed-scrolled-further"].narrative, "")


class TestScreenDocumentSlugCollision(unittest.IsolatedAsyncioTestCase):
    """Distinct activities that share a friendly name get de-collided slugs."""

    async def test_colliding_titles_get_numeric_suffixes(self) -> None:
        screens = [
            _screen_row(
                visual_hash=_HOME_A,
                activity="com.app/.MainActivity",
                category="home",
                description="Main menu",
            ),
            _screen_row(
                visual_hash=_DETAIL,
                activity="com.app/.ui.MainActivity",
                category="home",
                description="Main menu",
            ),
        ]
        graph = KnowledgeGraph(provider=_Provider(screens=screens, transitions=[]))
        await graph.load()

        index = ScreenDocumentExporter().build(graph=graph, defects=[], metadata=_metadata())
        slugs = sorted(document.slug for document in index.documents)

        self.assertEqual(slugs, ["main-menu", "main-menu-2"])


class TestScreenDocumentFiltering(unittest.IsolatedAsyncioTestCase):
    """Transient captures with no real content are not documented."""

    async def test_skips_screens_without_description_or_narrative(self) -> None:
        screens = [
            _screen_row(
                visual_hash=_HOME_A,
                activity="com.app/.Home",
                category="home",
                description="Home feed",
                rich="Home prose",
            ),
            # Transient capture: no meaningful description, no narrative.
            _screen_row(visual_hash=_DETAIL, activity="com.app/.Home", category="other"),
            # Failed analysis: the fallback sentinel is not a real description.
            _screen_row(
                visual_hash=_ACCOUNT,
                activity="com.app/.Home",
                category="other",
                description="Fallback state",
            ),
        ]
        graph = KnowledgeGraph(provider=_Provider(screens=screens, transitions=[]))
        await graph.load()

        index = ScreenDocumentExporter().build(graph=graph, defects=[], metadata=_metadata())

        self.assertEqual([document.slug for document in index.documents], ["home-feed"])


class TestScreenDocumentStructuralMerge(unittest.IsolatedAsyncioTestCase):
    """Captures sharing an activity and structure hash collapse into one document."""

    async def test_content_variations_merge_distinct_structures_stay_separate(self) -> None:
        screens = [
            _screen_row(
                visual_hash=_HOME_A,
                activity="com.app/.Login",
                category="auth",
                description="Login screen, empty field",
                rich="Login prose A",
                structure="loginstruct0001",
            ),
            _screen_row(
                visual_hash=_HOME_B,
                activity="com.app/.Login",
                category="auth",
                description="Login screen, number typed",
                rich="Login prose B, longer",
                structure="loginstruct0001",
            ),
            _screen_row(
                visual_hash=_DETAIL,
                activity="com.app/.Login",
                category="auth",
                description="OTP verification screen",
                rich="OTP prose",
                structure="otpstruct000001",
            ),
        ]
        graph = KnowledgeGraph(provider=_Provider(screens=screens, transitions=[]))
        await graph.load()

        index = ScreenDocumentExporter().build(graph=graph, defects=[], metadata=_metadata())

        # The two login captures merge (same structure); OTP stays separate.
        self.assertEqual(len(index.documents), 2)
        login = max(index.documents, key=lambda document: document.fingerprints)
        otp = min(index.documents, key=lambda document: document.fingerprints)

        self.assertEqual(login.fingerprints, 2)
        self.assertEqual(login.visits, 2)
        self.assertEqual(login.narrative, "Login prose B, longer")
        self.assertEqual(otp.fingerprints, 1)
        self.assertEqual(otp.title, "OTP verification screen")


class TestScreenDocumentRenderer(unittest.TestCase):
    """The renderer emits image-free Markdown for screens and an index."""

    def setUp(self) -> None:
        self.__renderer = ScreenDocumentRenderer()

    @staticmethod
    def __document() -> ScreenDocument:
        return ScreenDocument(
            slug="main-detail",
            title="Main (Detail)",
            category=ScreenCategory.DETAIL,
            activity="com.app/.MainActivity",
            purpose="Restaurant detail",
            narrative="Detail prose describing the layout.",
            flow=ScreenFlow(
                inbound=[ScreenLink(action="tap", element="Restaurant card", screen="Main (Home)")],
                outbound=[
                    ScreenLink(action="tap", element="Open settings", screen="Account", count=3)
                ],
            ),
            defects=[
                Defect.from_signal(
                    signal=DefectSignal.OVERLAP_CLIPPING,
                    source=DefectSource.POST_RUN,
                    summary="Button overlaps title",
                    evidence=DefectEvidence(screen="ffffffffffffffff"),
                )
            ],
            visits=2,
            fingerprints=3,
        )

    def test_screen_document_is_image_free(self) -> None:
        markdown = self.__renderer.render_screen(document=self.__document())

        self.assertNotIn("![", markdown)
        self.assertNotIn("<img", markdown)
        for extension in (".png", ".jpg", ".jpeg", ".webp"):
            self.assertNotIn(extension, markdown)

    def test_screen_document_renders_flow_and_defects(self) -> None:
        markdown = self.__renderer.render_screen(document=self.__document())

        self.assertIn("# Main (Detail)", markdown)
        self.assertIn("- Variants: 3", markdown)
        self.assertIn("## Reached From", markdown)
        self.assertIn('tap "Restaurant card" from Main (Home)', markdown)
        self.assertIn("## Leads To", markdown)
        self.assertIn('tap "Open settings" to Account (x3)', markdown)
        self.assertIn("## Defects", markdown)
        self.assertIn("Button overlaps title", markdown)

    def test_index_links_to_each_document(self) -> None:
        index = DocumentIndex(metadata=_metadata(), documents=[self.__document()])

        markdown = self.__renderer.render_index(index=index)

        self.assertIn("# Screen Documentation", markdown)
        self.assertIn("- Screens: 1", markdown)
        self.assertIn("[Main (Detail)](main-detail.md)", markdown)

    def test_empty_index_states_no_screens(self) -> None:
        index = DocumentIndex(metadata=_metadata(), documents=[])

        self.assertIn("No screens were documented.", self.__renderer.render_index(index=index))


if __name__ == "__main__":
    unittest.main()
