"""
Unit tests for GraphExportService HTML export.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from fathom.services.export import GraphExportService, _activity_color_map


class TestGraphExportHtml:
    """
    to_html produces a self-contained, escaped, path-finding-enabled document.
    """

    @staticmethod
    def __graph_data() -> Dict[str, Any]:
        return {
            "nodes": [
                {
                    "visual_hash": "hash_home_" + "0" * 54,
                    "activity": "com.example.app/.HomeActivity",
                    "description": "Home screen with search and nav bar.",
                    "rich_description": "**Purpose**\nMain home.",
                    "visit_count": 3,
                    "first_seen": 1700000000,
                    "last_seen": 1700000500,
                },
                {
                    "visual_hash": "hash_settings_" + "0" * 50,
                    "activity": "com.example.app/.SettingsActivity",
                    "description": "Settings",
                    "visit_count": 1,
                    "first_seen": 1700000100,
                    "last_seen": 1700000100,
                },
            ],
            "edges": [
                {
                    "source_hash": "hash_home_" + "0" * 54,
                    "destination_hash": "hash_settings_" + "0" * 50,
                    "action_type": "tap",
                    "action_target": "Settings icon",
                    "count": 2,
                    "first_seen": 1700000100,
                    "last_seen": 1700000500,
                },
            ],
            "stats": {"unique_screens": 2, "total_transitions": 1, "unique_activities": 2},
        }

    def test_returns_self_contained_document(self) -> None:
        html = GraphExportService.to_html(self.__graph_data())
        assert html.startswith("<!DOCTYPE html>")
        assert html.rstrip().endswith("</html>")
        assert "vis-network" in html
        assert '<script id="fathom-data" type="application/json">' in html

    def test_embeds_every_node_and_edge(self) -> None:
        data = self.__graph_data()
        html = GraphExportService.to_html(data)
        for node in data["nodes"]:
            assert node["visual_hash"] in html
        assert "Settings icon" in html
        assert "tap" in html

    def test_uses_computed_title_when_none_given(self) -> None:
        html = GraphExportService.to_html(self.__graph_data())
        assert "2 screens" in html
        assert "1 transitions" in html

    def test_respects_explicit_title(self) -> None:
        html = GraphExportService.to_html(self.__graph_data(), title="My App Map")
        assert "<title>My App Map</title>" in html
        assert ">My App Map<" in html

    def test_escapes_script_closer_in_descriptions(self) -> None:
        data = {
            "nodes": [
                {
                    "visual_hash": "h" * 64,
                    "activity": "com.example.app/.X",
                    "description": "contains </script> literal",
                    "visit_count": 1,
                }
            ],
            "edges": [],
            "stats": {"unique_screens": 1, "total_transitions": 0, "unique_activities": 1},
        }
        html = GraphExportService.to_html(data)

        payload_start = html.index('<script id="fathom-data"')
        payload_end = html.index("</script>", payload_start)
        embedded = html[payload_start:payload_end]
        assert "</script>" not in embedded

        raw = embedded[embedded.index(">") + 1 :]
        parsed = json.loads(raw.replace("<\\/", "</"))
        assert parsed["graph"]["nodes"][0]["description"] == "contains </script> literal"

    def test_includes_path_finding_ui(self) -> None:
        html = GraphExportService.to_html(self.__graph_data())
        for marker in (
            'id="start"',
            'id="goal"',
            'id="find-paths"',
            'id="clear-paths"',
            'id="allow-back"',
            'id="start-menu"',
            'id="goal-menu"',
            'class="combo-menu"',
            "function attachCombo",
            'data-pane="pane-paths"',
            'id="paths-list"',
            'id="paths-summary"',
            "function findAllPaths",
            "function buildAdjacency",
            "function discoverMaximalJourneys",
            "function renderDiscoveredJourneys",
            'id="paths-count"',
        ):
            assert marker in html

    def test_save_writes_html_by_default(self, tmp_path: Any) -> None:
        data = self.__graph_data()
        written = GraphExportService.save(
            data, output_dir=str(tmp_path), prefix="kg_test", formats={"html": True}
        )
        assert "html" in written
        html_path = tmp_path / "kg_test.html"
        assert html_path.exists()
        content = html_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert data["nodes"][0]["visual_hash"] in content


class TestActivityColorMap:
    """
    Activity-to-color mapping is deterministic and stable across calls.
    """

    def test_is_deterministic_and_stable(self) -> None:
        nodes = [
            {"activity": "com.example.app/.A"},
            {"activity": "com.example.app/.B"},
            {"activity": "com.example.app/.A"},
        ]
        colors = _activity_color_map(nodes)

        assert set(colors.keys()) == {"com.example.app/.A", "com.example.app/.B"}
        assert colors["com.example.app/.A"] != colors["com.example.app/.B"]
        assert _activity_color_map(nodes) == colors
