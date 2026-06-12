from __future__ import annotations

import json
import unittest

from fathom.constants.exploration import MAX_SCREEN_LABEL_LENGTH, GraphFormat
from fathom.core.exceptions import GraphExportError
from fathom.core.services.exporter.graph import (
    DotGraphFormatter,
    GraphExporter,
    GraphLabeler,
    JsonGraphFormatter,
    MermaidGraphFormatter,
)
from fathom.schemas.exploration import (
    ExplorationSnapshot,
    ExplorationStats,
    ExploredScreen,
    ScreenTransition,
)


def _snapshot() -> ExplorationSnapshot:
    return ExplorationSnapshot(
        screens=[
            ExploredScreen(
                hash="home", activity="com.app/.Home", description="Home feed", visits=2
            ),
            ExploredScreen(
                hash="cart", activity="com.app/.CartActivity", description=None, visits=1
            ),
        ],
        transitions=[
            ScreenTransition(
                source="home", destination="cart", action="tap", target="Cart", count=3
            )
        ],
        stats=ExplorationStats(screens=2, transitions=1, visits=3, unexplored=1),
    )


class TestGraphLabeler(unittest.TestCase):
    """The labeler derives readable names for screens and transitions."""

    def setUp(self) -> None:
        self.__labeler = GraphLabeler()

    def test_friendly_activity_strips_package_suffix_and_spaces_camelcase(self) -> None:
        self.assertEqual(self.__labeler.friendly_activity(activity="com.app/.HomeActivity"), "Home")
        self.assertEqual(
            self.__labeler.friendly_activity(activity="com.app/.SettingsListActivity"),
            "Settings List",
        )

    def test_friendly_activity_falls_back_to_screen(self) -> None:
        self.assertEqual(self.__labeler.friendly_activity(activity="com.app/.Activity"), "Screen")

    def test_screen_prefers_description_then_activity(self) -> None:
        with_description = ExploredScreen(
            hash="h", activity="com.app/.Home", description="Home feed", visits=2
        )
        self.assertEqual(self.__labeler.screen(screen=with_description), "Home feed (visits: 2)")

        without_description = ExploredScreen(hash="h", activity="com.app/.CartActivity", visits=1)
        self.assertEqual(self.__labeler.screen(screen=without_description), "Cart (visits: 1)")

    def test_long_description_is_truncated(self) -> None:
        screen = ExploredScreen(hash="h", activity="a", description="x" * 200, visits=0)
        label = self.__labeler.screen(screen=screen)
        self.assertIn("...", label)
        # Name portion (before the visits suffix) is capped.
        name = label.rsplit(" (visits:", 1)[0]
        self.assertLessEqual(len(name), MAX_SCREEN_LABEL_LENGTH)

    def test_transition_label_includes_target_and_repeat_count(self) -> None:
        transition = ScreenTransition(
            source="a", destination="b", action="tap", target="Cart", count=3
        )
        self.assertEqual(self.__labeler.transition(transition=transition), "tap: Cart (x3)")

        single = ScreenTransition(source="a", destination="b", action="back", target=None, count=1)
        self.assertEqual(self.__labeler.transition(transition=single), "back")


class TestJsonGraphFormatter(unittest.TestCase):
    """JSON output round-trips back into a snapshot."""

    def test_render_is_parseable_and_complete(self) -> None:
        rendered = JsonGraphFormatter().render(snapshot=_snapshot())
        parsed = json.loads(rendered)

        self.assertEqual(len(parsed["screens"]), 2)
        self.assertEqual(parsed["transitions"][0]["target"], "Cart")
        self.assertEqual(parsed["stats"]["screens"], 2)


class TestDotGraphFormatter(unittest.TestCase):
    """DOT output is valid directed-graph source."""

    def test_render_contains_nodes_and_edges(self) -> None:
        rendered = DotGraphFormatter().render(snapshot=_snapshot())

        self.assertIn("digraph exploration {", rendered)
        self.assertIn('"home"', rendered)
        self.assertIn('"home" -> "cart"', rendered)
        self.assertTrue(rendered.rstrip().endswith("}"))

    def test_quotes_in_labels_are_escaped(self) -> None:
        snapshot = ExplorationSnapshot(
            screens=[ExploredScreen(hash="h", activity="a", description='He said "hi"', visits=1)]
        )
        rendered = DotGraphFormatter().render(snapshot=snapshot)
        self.assertIn('\\"hi\\"', rendered)


class TestMermaidGraphFormatter(unittest.TestCase):
    """Mermaid output uses indexed node ids and skips dangling edges."""

    def test_render_emits_indexed_nodes_and_edges(self) -> None:
        rendered = MermaidGraphFormatter().render(snapshot=_snapshot())

        self.assertIn("graph LR", rendered)
        self.assertIn("N0[", rendered)
        self.assertIn("N0 -->", rendered)

    def test_transition_to_unknown_screen_is_skipped(self) -> None:
        snapshot = ExplorationSnapshot(
            screens=[ExploredScreen(hash="home", activity="a", visits=1)],
            transitions=[ScreenTransition(source="home", destination="ghost", action="tap")],
        )
        rendered = MermaidGraphFormatter().render(snapshot=snapshot)
        self.assertNotIn("-->", rendered)


class TestGraphExporter(unittest.TestCase):
    """The exporter dispatches to the formatter for the requested format."""

    def test_render_dispatches_by_format(self) -> None:
        exporter = GraphExporter()
        snapshot = _snapshot()

        self.assertIn("digraph", exporter.render(snapshot=snapshot, graph_format=GraphFormat.DOT))
        self.assertIn(
            "graph LR", exporter.render(snapshot=snapshot, graph_format=GraphFormat.MERMAID)
        )
        self.assertIn(
            '"screens"', exporter.render(snapshot=snapshot, graph_format=GraphFormat.JSON)
        )

    def test_unknown_format_raises(self) -> None:
        exporter = GraphExporter(formatters={GraphFormat.JSON: JsonGraphFormatter()})
        with self.assertRaises(GraphExportError):
            exporter.render(snapshot=_snapshot(), graph_format=GraphFormat.DOT)

    def test_formats_reports_registered_formats(self) -> None:
        self.assertEqual(set(GraphExporter().formats), set(GraphFormat))


if __name__ == "__main__":
    unittest.main()
