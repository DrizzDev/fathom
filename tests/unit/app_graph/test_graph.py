"""Tests for the AppGraph — the cumulative knowledge store."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fathom.app_graph.graph import AppGraph
from fathom.app_graph.models import (
    Block,
    BlockStatus,
    ChangeType,
    Coverage,
    Element,
    Locator,
    LocatorRepair,
    Screen,
    ScreenCategory,
    Transition,
)


def _screen(
    *,
    hash: str = "abc123",
    name: str = "Home",
    activity: str = "com.example/.MainActivity",
    category: ScreenCategory = ScreenCategory.HOME,
    purpose: str = "",
    visits: int = 1,
    structure_hash: str | None = None,
    elements: list[Element] | None = None,
) -> Screen:
    return Screen(
        hash=hash, name=name, activity=activity,
        category=category, purpose=purpose, visits=visits,
        structure_hash=structure_hash,
        elements=elements or [],
    )


def _transition(
    *,
    source: str = "abc123",
    target: str = "def456",
    action_type: str = "tap",
    action_target: str | None = "Search",
    locator: Locator | None = None,
) -> Transition:
    return Transition(
        source_hash=source, target_hash=target,
        action_type=action_type, action_target=action_target,
        locator=locator,
    )


class TestAppGraphAddScreen(unittest.TestCase):
    """Adding screens, with deduplication."""

    def test_add_new_screen(self) -> None:
        g = AppGraph(package="com.example")
        canon = g.add_screen(_screen())
        self.assertEqual(canon, "abc123")
        self.assertEqual(len(g.screens), 1)

    def test_dedup_by_structure_hash(self) -> None:
        g = AppGraph()
        g.add_screen(_screen(hash="a", structure_hash="s1", activity="X"))
        canon = g.add_screen(_screen(hash="b", structure_hash="s1", activity="X"))
        self.assertEqual(canon, "a")
        self.assertEqual(len(g.screens), 1)
        self.assertEqual(g.screens[0].visits, 2)

    def test_dedup_by_name_and_activity(self) -> None:
        g = AppGraph()
        g.add_screen(_screen(hash="a", name="Home", activity="X"))
        canon = g.add_screen(_screen(hash="b", name="Home", activity="X"))
        self.assertEqual(canon, "a")
        self.assertEqual(len(g.screens), 1)

    def test_different_activity_not_deduped(self) -> None:
        g = AppGraph()
        g.add_screen(_screen(hash="a", name="Home", activity="X"))
        g.add_screen(_screen(hash="b", name="Home", activity="Y"))
        self.assertEqual(len(g.screens), 2)

    def test_merge_enriches_existing(self) -> None:
        g = AppGraph()
        g.add_screen(_screen(hash="a", activity="X", purpose=""))
        g.add_screen(_screen(hash="b", activity="X", purpose="The main screen"))
        self.assertEqual(g.screens[0].purpose, "The main screen")

    def test_merge_combines_elements(self) -> None:
        g = AppGraph()
        s1 = _screen(hash="a", elements=[Element(name="Search bar")])
        s2 = _screen(hash="b", elements=[Element(name="Cart icon"), Element(name="Search bar")])
        g.add_screen(s1)
        g.add_screen(s2)
        self.assertEqual(len(g.screens[0].elements), 2)
        names = [e.name for e in g.screens[0].elements]
        self.assertIn("Search bar", names)
        self.assertIn("Cart icon", names)

    def test_merge_combines_locators(self) -> None:
        """When merging elements, locators from both sides are kept."""
        g = AppGraph()
        loc1 = Locator(strategy="resource_id", value="search_bar")
        loc2 = Locator(strategy="xpath", value="//input[@hint='Search']")
        s1 = _screen(hash="a", elements=[Element(name="Search", locators=[loc1])])
        s2 = _screen(hash="b", elements=[Element(name="Search", locators=[loc2])])
        g.add_screen(s1)
        g.add_screen(s2)
        el = g.screens[0].find_element("Search")
        self.assertIsNotNone(el)
        self.assertEqual(len(el.locators), 2)


class TestAppGraphTransitions(unittest.TestCase):
    """Adding transitions, with deduplication."""

    def test_add_transition(self) -> None:
        g = AppGraph()
        g.add_screen(_screen(hash="a"))
        g.add_screen(_screen(hash="b", name="Search", activity="Y"))
        g.add_transition(_transition(source="a", target="b"))
        self.assertEqual(len(g.transitions), 1)

    def test_dedup_increments_count(self) -> None:
        g = AppGraph()
        g.add_screen(_screen(hash="a"))
        g.add_screen(_screen(hash="b", name="Search", activity="Y"))
        g.add_transition(_transition(source="a", target="b"))
        g.add_transition(_transition(source="a", target="b"))
        self.assertEqual(len(g.transitions), 1)
        self.assertEqual(g.transitions[0].count, 2)

    def test_transition_resolves_aliases(self) -> None:
        g = AppGraph()
        g.add_screen(_screen(hash="a", structure_hash="s", activity="X"))
        g.add_screen(_screen(hash="c", name="Search", activity="Y"))
        g.add_screen(_screen(hash="b", structure_hash="s", activity="X"))
        g.add_transition(_transition(source="b", target="c"))
        self.assertEqual(g.transitions[0].source_hash, "a")

    def test_transition_dropped_if_endpoint_missing(self) -> None:
        g = AppGraph()
        g.add_screen(_screen(hash="a"))
        g.add_transition(_transition(source="a", target="missing"))
        self.assertEqual(len(g.transitions), 0)

    def test_transition_preserves_locator(self) -> None:
        g = AppGraph()
        g.add_screen(_screen(hash="a"))
        g.add_screen(_screen(hash="b", name="Search", activity="Y"))
        loc = Locator(strategy="resource_id", value="btn_search")
        g.add_transition(_transition(source="a", target="b", locator=loc))
        self.assertIsNotNone(g.transitions[0].locator)
        self.assertEqual(g.transitions[0].locator.value, "btn_search")


class TestAppGraphFlow(unittest.TestCase):
    """Navigation flow computation."""

    def test_screen_flow(self) -> None:
        g = AppGraph()
        g.add_screen(_screen(hash="a", name="Home", activity="X"))
        g.add_screen(_screen(hash="b", name="Search", activity="Y"))
        g.add_transition(_transition(source="a", target="b"))
        g.add_transition(_transition(source="b", target="a", action_type="back", action_target=None))

        flow = g.screen_flow("a")
        self.assertEqual(len(flow.outbound), 1)
        self.assertEqual(flow.outbound[0].screen, "Search")
        self.assertEqual(len(flow.inbound), 1)
        self.assertEqual(flow.inbound[0].screen, "Search")

    def test_flow_includes_locators(self) -> None:
        g = AppGraph()
        g.add_screen(_screen(hash="a", name="Home", activity="X"))
        g.add_screen(_screen(hash="b", name="Search", activity="Y"))
        loc = Locator(strategy="resource_id", value="search_btn")
        g.add_transition(_transition(source="a", target="b", locator=loc))
        flow = g.screen_flow("a")
        self.assertIsNotNone(flow.outbound[0].locator)
        self.assertEqual(flow.outbound[0].locator.value, "search_btn")


class TestAppGraphPathfinding(unittest.TestCase):
    """Graph traversal."""

    def test_reachable_from(self) -> None:
        g = AppGraph()
        g.add_screen(_screen(hash="a", name="Home", activity="A"))
        g.add_screen(_screen(hash="b", name="Search", activity="B"))
        g.add_screen(_screen(hash="c", name="Results", activity="C"))
        g.add_screen(_screen(hash="d", name="Isolated", activity="D"))
        g.add_transition(_transition(source="a", target="b"))
        g.add_transition(_transition(source="b", target="c"))
        reachable = g.reachable_from("a")
        self.assertIn("a", reachable)
        self.assertIn("b", reachable)
        self.assertIn("c", reachable)
        self.assertNotIn("d", reachable)

    def test_find_path(self) -> None:
        g = AppGraph()
        g.add_screen(_screen(hash="a", name="Home", activity="A"))
        g.add_screen(_screen(hash="b", name="Search", activity="B"))
        g.add_screen(_screen(hash="c", name="Results", activity="C"))
        g.add_transition(_transition(source="a", target="b"))
        g.add_transition(_transition(source="b", target="c"))
        path = g.find_path("a", "c")
        self.assertEqual(path, ["a", "b", "c"])

    def test_find_path_no_route(self) -> None:
        g = AppGraph()
        g.add_screen(_screen(hash="a", name="Home", activity="A"))
        g.add_screen(_screen(hash="b", name="Isolated", activity="B"))
        self.assertEqual(g.find_path("a", "b"), [])


class TestAppGraphBlocks(unittest.TestCase):
    """Block (exploration scope) management."""

    def test_create_block(self) -> None:
        g = AppGraph()
        block = g.create_block("Explore the Home page")
        self.assertEqual(block.intent, "Explore the Home page")
        self.assertEqual(block.status, BlockStatus.PENDING)
        self.assertEqual(len(g.blocks), 1)

    def test_block_lifecycle(self) -> None:
        g = AppGraph()
        block = g.create_block("Explore Cart")
        g.start_block(block.id)
        self.assertEqual(g.blocks[0].status, BlockStatus.RUNNING)
        g.complete_block(block.id, screens_discovered=["h1", "h2"], steps=5)
        self.assertEqual(g.blocks[0].status, BlockStatus.COMPLETED)
        self.assertEqual(g.blocks[0].screens_discovered, ["h1", "h2"])

    def test_block_failure(self) -> None:
        g = AppGraph()
        block = g.create_block("Explore Payment")
        g.start_block(block.id)
        g.complete_block(block.id, error="Device disconnected")
        self.assertEqual(g.blocks[0].status, BlockStatus.FAILED)
        self.assertEqual(g.blocks[0].error, "Device disconnected")


class TestSelfHealing(unittest.TestCase):
    """Locator repair tracking."""

    def test_record_repair(self) -> None:
        g = AppGraph()
        g.add_screen(Screen(
            hash="a", name="Home", activity="X",
            elements=[Element(
                name="Search",
                locators=[Locator(strategy="resource_id", value="old_search")],
            )],
        ))
        repair = LocatorRepair(
            screen_name="Home",
            element_name="Search",
            old_locator=Locator(strategy="resource_id", value="old_search"),
            new_locator=Locator(strategy="resource_id", value="new_search"),
        )
        g.record_repair(repair)

        self.assertEqual(len(g.locator_repairs), 1)
        el = g.screens[0].find_element("Search")
        self.assertEqual(len(el.locators), 2)
        # Old locator confidence should be 0
        old = [l for l in el.locators if l.value == "old_search"][0]
        self.assertEqual(old.confidence, 0.0)
        # New locator should be present
        new = [l for l in el.locators if l.value == "new_search"][0]
        self.assertEqual(new.confidence, 1.0)


class TestChangeIntelligence(unittest.TestCase):
    """Graph diffing for change detection."""

    def test_detect_added_screen(self) -> None:
        old = AppGraph(package="com.example")
        old.add_screen(_screen(hash="a", name="Home", activity="X"))

        new = AppGraph(package="com.example")
        new.add_screen(_screen(hash="a", name="Home", activity="X"))
        new.add_screen(_screen(hash="b", name="Cart", activity="Y"))

        diff = old.diff(new)
        self.assertTrue(diff.has_changes)
        self.assertEqual(len(diff.added_screens), 1)
        self.assertEqual(diff.added_screens[0].screen_name, "Cart")

    def test_detect_removed_screen(self) -> None:
        old = AppGraph(package="com.example")
        old.add_screen(_screen(hash="a", name="Home", activity="X"))
        old.add_screen(_screen(hash="b", name="Legacy", activity="Y"))

        new = AppGraph(package="com.example")
        new.add_screen(_screen(hash="a", name="Home", activity="X"))

        diff = old.diff(new, coverage=Coverage(activities=["X", "Y"]))
        self.assertEqual(len(diff.removed_screens), 1)
        self.assertEqual(diff.removed_screens[0].screen_name, "Legacy")

    def test_detect_modified_screen(self) -> None:
        old = AppGraph(package="com.example")
        old.add_screen(Screen(
            hash="a", name="Home", activity="X",
            elements=[Element(name="Search"), Element(name="Old banner")],
        ))

        new = AppGraph(package="com.example")
        new.add_screen(Screen(
            hash="a2", name="Home", activity="X",
            elements=[Element(name="Search"), Element(name="New promo")],
        ))

        diff = old.diff(new)
        self.assertEqual(len(diff.modified_screens), 1)
        mod = diff.modified_screens[0]
        self.assertIn("New promo", mod.added_elements)
        self.assertIn("Old banner", mod.removed_elements)

    def test_no_changes(self) -> None:
        old = AppGraph(package="com.example")
        old.add_screen(_screen(hash="a", name="Home", activity="X"))

        new = AppGraph(package="com.example")
        new.add_screen(_screen(hash="a2", name="Home", activity="X"))

        diff = old.diff(new)
        self.assertFalse(diff.has_changes)
        self.assertEqual(diff.summary(), "no changes detected")

    def test_diff_summary(self) -> None:
        old = AppGraph(package="com.example")
        old.add_screen(_screen(hash="a", name="Home", activity="X"))

        new = AppGraph(package="com.example")
        new.add_screen(_screen(hash="a", name="Home", activity="X"))
        new.add_screen(_screen(hash="b", name="Cart", activity="Y"))
        new.add_screen(_screen(hash="c", name="Checkout", activity="Z"))

        diff = old.diff(new)
        self.assertIn("2 new screens", diff.summary())


class TestAppGraphPersistence(unittest.TestCase):
    """Save and load round-trip."""

    def test_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.json"

            g = AppGraph(package="com.example", label="Example")
            g.begin_exploration("Explore home", step_count=5)
            g.add_screen(_screen(hash="a", name="Home", activity="X"))
            g.add_screen(_screen(hash="b", name="Search", activity="Y"))
            g.add_transition(_transition(source="a", target="b"))
            g.save(path)

            loaded = AppGraph.load(path)
            self.assertEqual(loaded.package, "com.example")
            self.assertEqual(loaded.label, "Example")
            self.assertEqual(len(loaded.screens), 2)
            self.assertEqual(len(loaded.transitions), 1)
            self.assertEqual(loaded.exploration_count, 1)
            self.assertEqual(loaded.total_steps, 5)
            self.assertEqual(loaded.goals, ["Explore home"])

    def test_round_trip_with_locators(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.json"
            g = AppGraph(package="com.example")
            g.add_screen(Screen(
                hash="a", name="Home", activity="X",
                elements=[Element(
                    name="Search",
                    locators=[Locator(strategy="resource_id", value="search_bar")],
                    element_type="input",
                )],
            ))
            g.save(path)

            loaded = AppGraph.load(path)
            el = loaded.screens[0].find_element("Search")
            self.assertIsNotNone(el)
            self.assertEqual(len(el.locators), 1)
            self.assertEqual(el.locators[0].value, "search_bar")
            self.assertEqual(el.element_type, "input")

    def test_round_trip_blocks_and_repairs(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.json"
            g = AppGraph(package="com.example")
            g.create_block("Explore home")
            g.add_screen(Screen(
                hash="a", name="Home", activity="X",
                elements=[Element(name="Btn", locators=[Locator(strategy="id", value="old")])],
            ))
            g.record_repair(LocatorRepair(
                screen_name="Home", element_name="Btn",
                old_locator=Locator(strategy="id", value="old"),
                new_locator=Locator(strategy="id", value="new"),
            ))
            g.save(path)

            loaded = AppGraph.load(path)
            self.assertEqual(len(loaded.blocks), 1)
            self.assertEqual(len(loaded.locator_repairs), 1)

    def test_load_nonexistent_returns_empty(self) -> None:
        g = AppGraph.load(Path("/tmp/does-not-exist.json"))
        self.assertEqual(len(g.screens), 0)


class TestAppGraphCumulative(unittest.TestCase):
    """Multiple explorations build on the same graph."""

    def test_two_explorations(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.json"

            g = AppGraph(package="com.example", label="Example")
            g.begin_exploration("Explore home", step_count=3)
            g.add_screen(_screen(hash="a", name="Home", activity="X"))
            g.add_screen(_screen(hash="b", name="Search", activity="Y"))
            g.add_transition(_transition(source="a", target="b"))
            g.save(path)

            g2 = AppGraph.load(path)
            g2.begin_exploration("Explore cart", step_count=4)
            g2.add_screen(_screen(hash="b2", name="Search", activity="Y"))
            g2.add_screen(_screen(hash="c", name="Cart", activity="Z"))
            g2.add_transition(_transition(source="b2", target="c", action_target="Cart"))
            g2.save(path)

            final = AppGraph.load(path)
            self.assertEqual(len(final.screens), 3)
            self.assertEqual(len(final.transitions), 2)
            self.assertEqual(final.exploration_count, 2)
            self.assertEqual(final.total_steps, 7)
            self.assertEqual(final.goals, ["Explore home", "Explore cart"])


if __name__ == "__main__":
    unittest.main()

class TestScopedDiff(unittest.TestCase):
    def test_unobserved_screens_are_not_removed(self) -> None:
        old = AppGraph(package="com.example")
        old.add_screen(_screen(hash="home", name="Home", activity="Home"))
        old.add_screen(_screen(hash="cart", name="Cart", activity="Cart"))
        new = AppGraph(package="com.example")
        new.add_screen(_screen(hash="cart", name="Cart", activity="Cart"))
        self.assertEqual(old.diff(new).removed_screens, [])
        self.assertEqual(old.diff(new, coverage=Coverage(activities=["Cart"])).removed_screens, [])
        self.assertEqual(len(old.diff(new, coverage=Coverage(activities=["Home"])).removed_screens), 1)

    def test_other_app_cannot_be_compared(self) -> None:
        with self.assertRaises(ValueError):
            AppGraph(package="one").diff(AppGraph(package="two"))
