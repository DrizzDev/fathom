"""Tests for the StudioExporter — transforms AppGraph to studio file layout."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fathom.app_graph.exporter import StudioExporter, _describe_action
from fathom.app_graph.graph import AppGraph
from fathom.app_graph.models import Element, Locator, Screen, ScreenCategory, Transition


def _el(name: str, **kwargs) -> Element:
    return Element(name=name, **kwargs)


def _populated_graph() -> AppGraph:
    """A small graph with 3 screens and 2 transitions."""
    g = AppGraph(package="com.doordash", label="DoorDash")
    g.begin_exploration("Explore home and search", step_count=5)

    g.add_screen(Screen(
        hash="h1", name="Home", activity="com.doordash/.MainActivity",
        category=ScreenCategory.HOME, purpose="Main landing screen",
        elements=[
            _el("Search bar", locators=[Locator(strategy="resource_id", value="search_input")]),
            _el("Banner carousel"),
            _el("Bottom nav"),
        ],
        actions=["Search for restaurants", "Browse categories"],
        visits=3,
    ))
    g.add_screen(Screen(
        hash="h2", name="Search", activity="com.doordash/.SearchActivity",
        category=ScreenCategory.SEARCH, purpose="Find restaurants and items",
        elements=[_el("Search input"), _el("Recent searches")],
        actions=["Type a query", "Pick from suggestions"],
        visits=2,
    ))
    g.add_screen(Screen(
        hash="h3", name="Search Results", activity="com.doordash/.ResultsActivity",
        category=ScreenCategory.LIST, purpose="Restaurant listing",
        elements=[_el("Restaurant cards"), _el("Filter bar")],
        actions=["Open a restaurant", "Apply filters"],
        visits=1,
    ))

    g.add_transition(Transition(
        source_hash="h1", target_hash="h2",
        action_type="tap", action_target="Search bar",
    ))
    g.add_transition(Transition(
        source_hash="h2", target_hash="h3",
        action_type="type", action_target="Search input", value="biryani",
    ))

    return g


class TestStudioExporterBuild(unittest.TestCase):
    """Building the AppMap data model."""

    def test_build_pages(self) -> None:
        g = _populated_graph()
        exporter = StudioExporter(g)
        app_map = exporter.build_app_map()

        self.assertEqual(app_map.package, "com.doordash")
        self.assertEqual(app_map.label, "DoorDash")
        self.assertEqual(len(app_map.pages), 3)
        self.assertEqual(len(app_map.edges), 2)

    def test_page_node_id_format(self) -> None:
        g = _populated_graph()
        exporter = StudioExporter(g)
        app_map = exporter.build_app_map()

        for page in app_map.pages:
            self.assertTrue(page.node_id.startswith("page:"))
            self.assertEqual(page.node_id, f"page:{page.id}")

    def test_edge_references_node_ids(self) -> None:
        g = _populated_graph()
        exporter = StudioExporter(g)
        app_map = exporter.build_app_map()

        node_ids = {p.node_id for p in app_map.pages}
        for edge in app_map.edges:
            self.assertIn(edge.from_node, node_ids,
                          f"Edge 'from' {edge.from_node} not in pages")
            self.assertIn(edge.to_node, node_ids,
                          f"Edge 'to' {edge.to_node} not in pages")

    def test_edge_labels(self) -> None:
        g = _populated_graph()
        exporter = StudioExporter(g)
        app_map = exporter.build_app_map()

        labels = [e.label for e in app_map.edges]
        self.assertIn('tap "Search bar"', labels)
        self.assertIn('type "biryani"', labels)

    def test_slug_deduplication(self) -> None:
        """Two screens with the same name get unique slugs."""
        g = AppGraph(package="com.test")
        g.add_screen(Screen(hash="a", name="Search", activity="X"))
        g.add_screen(Screen(hash="b", name="Search", activity="Y"))
        exporter = StudioExporter(g)
        app_map = exporter.build_app_map()

        ids = [p.id for p in app_map.pages]
        self.assertEqual(len(set(ids)), 2)
        self.assertIn("search", ids)
        self.assertIn("search-2", ids)


class TestStudioExporterWrite(unittest.TestCase):
    """Writing the full studio file layout."""

    def test_writes_expected_files(self) -> None:
        g = _populated_graph()
        exporter = StudioExporter(g)

        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "app-maps" / g.package
            written = exporter.write(out)

            self.assertTrue((out / "app-map.json").exists())
            self.assertTrue((out / "index.md").exists())
            self.assertTrue((out / "home.md").exists())
            self.assertTrue((out / "search.md").exists())
            self.assertTrue((out / "search-results.md").exists())

    def test_app_map_json_is_valid(self) -> None:
        g = _populated_graph()
        exporter = StudioExporter(g)

        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "maps" / g.package
            exporter.write(out)

            raw = json.loads((out / "app-map.json").read_text())
            self.assertEqual(raw["package"], "com.doordash")
            self.assertEqual(raw["label"], "DoorDash")
            self.assertIn("pages", raw)
            self.assertIn("edges", raw)

            # Verify camelCase keys
            page = raw["pages"][0]
            self.assertIn("nodeId", page)
            self.assertIn("evidenceXml", page)

    def test_page_markdown_content(self) -> None:
        g = _populated_graph()
        exporter = StudioExporter(g)

        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "maps" / g.package
            exporter.write(out)

            md = (out / "home.md").read_text()
            self.assertIn("# Home", md)
            self.assertIn("home", md.lower())
            self.assertIn("## Elements", md)
            self.assertIn("Search bar", md)
            self.assertIn("## Leads To", md)


class TestDescribeAction(unittest.TestCase):
    """Action label formatting."""

    def test_tap(self) -> None:
        self.assertEqual(_describe_action("tap", "Search"), 'tap "Search"')

    def test_type_with_value(self) -> None:
        self.assertEqual(_describe_action("type", "Input", "hello"), 'type "hello"')

    def test_back(self) -> None:
        self.assertEqual(_describe_action("back"), "back")

    def test_scroll(self) -> None:
        self.assertEqual(_describe_action("scroll"), "scroll")


if __name__ == "__main__":
    unittest.main()
