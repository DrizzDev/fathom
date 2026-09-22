"""Tests for App Graph models — internal, studio format, and change intelligence."""

from __future__ import annotations

import json
import unittest

from fathom.app_graph.models import (
    AppMap,
    ChangeType,
    Edge,
    Element,
    Fingerprint,
    GraphDiff,
    Locator,
    LocatorRepair,
    Page,
    Screen,
    ScreenCategory,
    ScreenChange,
    slugify,
)


class TestSlugify(unittest.TestCase):
    def test_simple(self) -> None:
        self.assertEqual(slugify("Home Screen"), "home-screen")

    def test_special_chars(self) -> None:
        self.assertEqual(slugify("Search (Results)"), "search-results")

    def test_whitespace(self) -> None:
        self.assertEqual(slugify("  Cart  "), "cart")

    def test_empty_fallback(self) -> None:
        self.assertEqual(slugify(""), "screen")
        self.assertEqual(slugify("   "), "screen")


class TestLocatorModel(unittest.TestCase):
    def test_defaults(self) -> None:
        loc = Locator(strategy="resource_id", value="btn_search")
        self.assertEqual(loc.confidence, 1.0)
        self.assertIsNone(loc.last_verified)

    def test_serialization(self) -> None:
        loc = Locator(strategy="xpath", value="//button[@text='Go']", confidence=0.8)
        data = loc.model_dump()
        self.assertEqual(data["strategy"], "xpath")
        self.assertEqual(data["confidence"], 0.8)


class TestElementModel(unittest.TestCase):
    def test_with_locators(self) -> None:
        el = Element(
            name="Search bar",
            locators=[
                Locator(strategy="resource_id", value="search_input"),
                Locator(strategy="content_desc", value="Search"),
            ],
            element_type="input",
            interactive=True,
        )
        self.assertEqual(len(el.locators), 2)
        self.assertEqual(el.element_type, "input")

    def test_empty_locators(self) -> None:
        el = Element(name="Banner")
        self.assertEqual(el.locators, [])
        self.assertTrue(el.interactive)


class TestScreenModel(unittest.TestCase):
    def test_defaults(self) -> None:
        s = Screen(hash="abc", name="Home", activity="X")
        self.assertEqual(s.category, ScreenCategory.OTHER)
        self.assertEqual(s.visits, 1)
        self.assertEqual(s.elements, [])

    def test_element_names(self) -> None:
        s = Screen(
            hash="abc", name="Home", activity="X",
            elements=[Element(name="Search"), Element(name="Cart")],
        )
        self.assertEqual(s.element_names(), ["Search", "Cart"])

    def test_find_element(self) -> None:
        s = Screen(
            hash="abc", name="Home", activity="X",
            elements=[Element(name="Search bar")],
        )
        self.assertIsNotNone(s.find_element("search bar"))
        self.assertIsNone(s.find_element("nonexistent"))

    def test_best_locator(self) -> None:
        s = Screen(
            hash="abc", name="Home", activity="X",
            elements=[Element(
                name="Search",
                locators=[
                    Locator(strategy="xpath", value="//x", confidence=0.5),
                    Locator(strategy="resource_id", value="search", confidence=1.0),
                ],
            )],
        )
        best = s.best_locator("Search")
        self.assertIsNotNone(best)
        self.assertEqual(best.strategy, "resource_id")

    def test_serialization(self) -> None:
        s = Screen(
            hash="abc", name="Home", activity="X",
            category=ScreenCategory.HOME, purpose="Main screen",
            elements=[Element(
                name="Btn",
                locators=[Locator(strategy="id", value="btn1")],
            )],
        )
        data = s.model_dump()
        self.assertEqual(data["hash"], "abc")
        self.assertEqual(data["elements"][0]["name"], "Btn")
        self.assertEqual(data["elements"][0]["locators"][0]["value"], "btn1")


class TestGraphDiff(unittest.TestCase):
    def test_properties(self) -> None:
        diff = GraphDiff(
            package="com.example",
            changes=[
                ScreenChange(screen_name="New", change_type=ChangeType.ADDED),
                ScreenChange(screen_name="Old", change_type=ChangeType.REMOVED),
                ScreenChange(screen_name="Changed", change_type=ChangeType.MODIFIED),
            ],
        )
        self.assertEqual(len(diff.added_screens), 1)
        self.assertEqual(len(diff.removed_screens), 1)
        self.assertEqual(len(diff.modified_screens), 1)
        self.assertTrue(diff.has_changes)

    def test_summary(self) -> None:
        diff = GraphDiff(
            package="x",
            changes=[
                ScreenChange(screen_name="A", change_type=ChangeType.ADDED),
                ScreenChange(screen_name="B", change_type=ChangeType.ADDED),
            ],
        )
        self.assertIn("2 new screens", diff.summary())

    def test_no_changes(self) -> None:
        diff = GraphDiff(package="x", changes=[])
        self.assertFalse(diff.has_changes)
        self.assertEqual(diff.summary(), "no changes detected")


class TestLocatorRepair(unittest.TestCase):
    def test_model(self) -> None:
        repair = LocatorRepair(
            screen_name="Home",
            element_name="Search",
            old_locator=Locator(strategy="id", value="old"),
            new_locator=Locator(strategy="id", value="new"),
            test_case="test_search.py",
        )
        self.assertEqual(repair.screen_name, "Home")
        self.assertFalse(repair.persisted)


class TestStudioPageModel(unittest.TestCase):
    def test_camel_case_serialization(self) -> None:
        p = Page(
            id="home", node_id="page:home", name="Home",
            activity="MainActivity", visits=3,
            evidence_xml="evidence/home.xml",
        )
        data = json.loads(p.model_dump_json(by_alias=True))
        self.assertIn("nodeId", data)
        self.assertEqual(data["nodeId"], "page:home")
        self.assertIn("evidenceXml", data)

    def test_node_id_convention(self) -> None:
        p = Page(id="search-results", node_id="page:search-results",
                 name="Search Results", activity="X")
        self.assertEqual(p.node_id, f"page:{p.id}")


class TestStudioEdgeModel(unittest.TestCase):
    def test_alias_serialization(self) -> None:
        e = Edge(from_node="page:home", to_node="page:search", label='tap "Search"')
        data = json.loads(e.model_dump_json(by_alias=True))
        self.assertIn("from", data)
        self.assertIn("to", data)
        self.assertNotIn("from_node", data)
        self.assertNotIn("to_node", data)


class TestStudioAppMapModel(unittest.TestCase):
    def test_camel_case_keys(self) -> None:
        m = AppMap(
            package="com.example", label="Example",
            generated_at="2026-09-21T00:00:00Z",
            exploration_count=2,
        )
        data = json.loads(m.model_dump_json(by_alias=True))
        self.assertIn("generatedAt", data)
        self.assertIn("explorationCount", data)
        self.assertIn("locatorRepairs", data)


if __name__ == "__main__":
    unittest.main()
