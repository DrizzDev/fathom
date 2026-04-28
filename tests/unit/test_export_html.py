"""Unit tests for GraphExportService.to_html."""

from __future__ import annotations

import json

import pytest

from fathom.services.export import GraphExportService, _activity_color_map


@pytest.fixture
def graph_data():
    return {
        "nodes": [
            {
                "visual_hash": "hash_home_0000000000000000000000000000000000000000000000000000000000",
                "activity": "com.example.app/.HomeActivity",
                "description": "Home screen with search and nav bar.",
                "rich_description": "**Purpose**\nMain home.",
                "visit_count": 3,
                "first_seen": 1700000000,
                "last_seen": 1700000500,
            },
            {
                "visual_hash": "hash_settings_00000000000000000000000000000000000000000000000000000",
                "activity": "com.example.app/.SettingsActivity",
                "description": "Settings",
                "visit_count": 1,
                "first_seen": 1700000100,
                "last_seen": 1700000100,
            },
        ],
        "edges": [
            {
                "source_hash": "hash_home_0000000000000000000000000000000000000000000000000000000000",
                "destination_hash": "hash_settings_00000000000000000000000000000000000000000000000000000",
                "action_type": "tap",
                "action_target": "Settings icon",
                "count": 2,
                "first_seen": 1700000100,
                "last_seen": 1700000500,
            },
        ],
        "stats": {
            "unique_screens": 2,
            "total_transitions": 1,
            "unique_activities": 2,
        },
    }


def test_to_html_returns_self_contained_document(graph_data):
    html = GraphExportService.to_html(graph_data)

    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    # vis-network must be referenced (CDN script tag)
    assert "vis-network" in html
    # Data payload is embedded, not linked
    assert '<script id="fathom-data" type="application/json">' in html


def test_to_html_embeds_every_node_and_edge(graph_data):
    html = GraphExportService.to_html(graph_data)

    # Every visual_hash should appear in the embedded JSON blob
    for node in graph_data["nodes"]:
        assert node["visual_hash"] in html
    # Edge action/target appear too
    assert "Settings icon" in html
    assert "tap" in html


def test_to_html_uses_computed_title_when_none_given(graph_data):
    html = GraphExportService.to_html(graph_data)

    assert "2 screens" in html
    assert "1 transitions" in html


def test_to_html_respects_explicit_title(graph_data):
    html = GraphExportService.to_html(graph_data, title="My App Map")

    assert "<title>My App Map</title>" in html
    assert ">My App Map<" in html  # also in <h1>


def test_to_html_escapes_script_closer_in_descriptions():
    graph_data = {
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

    html = GraphExportService.to_html(graph_data)

    # The raw `</script>` must not appear inside the JSON payload
    payload_start = html.index('<script id="fathom-data"')
    payload_end = html.index("</script>", payload_start)
    embedded = html[payload_start:payload_end]
    assert "</script>" not in embedded
    # But the literal, as parsed by the browser, should round-trip cleanly.
    # Pull the JSON out and re-parse it.
    json_start = embedded.index(">") + 1
    raw = embedded[json_start:]
    # Un-escape the </ guard so json.loads can parse it
    parsed = json.loads(raw.replace("<\\/", "</"))
    assert parsed["graph"]["nodes"][0]["description"] == "contains </script> literal"


def test_activity_color_map_is_deterministic():
    nodes = [
        {"activity": "com.example.app/.A"},
        {"activity": "com.example.app/.B"},
        {"activity": "com.example.app/.A"},  # duplicate
    ]
    colors = _activity_color_map(nodes)

    assert set(colors.keys()) == {"com.example.app/.A", "com.example.app/.B"}
    # Sorted → A gets palette[0], B gets palette[1]
    assert colors["com.example.app/.A"] != colors["com.example.app/.B"]
    # Stable across calls
    assert _activity_color_map(nodes) == colors


def test_to_html_includes_path_finding_ui(graph_data):
    html = GraphExportService.to_html(graph_data)

    # Header inputs + buttons
    assert 'id="start"' in html
    assert 'id="goal"' in html
    assert 'id="find-paths"' in html
    assert 'id="clear-paths"' in html
    assert 'id="allow-back"' in html
    # Custom combobox menus (one per input)
    assert 'id="start-menu"' in html
    assert 'id="goal-menu"' in html
    assert 'class="combo-menu"' in html
    assert "function attachCombo" in html
    # Sidebar tabs + paths pane
    assert 'data-pane="pane-paths"' in html
    assert 'id="paths-list"' in html
    assert 'id="paths-summary"' in html
    # JS core
    assert "function findAllPaths" in html
    assert "function buildAdjacency" in html
    # Auto-populated discovered journeys
    assert "function discoverMaximalJourneys" in html
    assert "function renderDiscoveredJourneys" in html
    assert 'id="paths-count"' in html


def test_save_writes_html_by_default(graph_data, tmp_path):
    written = GraphExportService.save(
        graph_data,
        output_dir=str(tmp_path),
        prefix="kg_test",
        formats={"html": True},
    )

    assert "html" in written
    html_path = tmp_path / "kg_test.html"
    assert html_path.exists()
    content = html_path.read_text()
    assert "<!DOCTYPE html>" in content
    assert graph_data["nodes"][0]["visual_hash"] in content
