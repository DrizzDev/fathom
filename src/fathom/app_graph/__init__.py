"""App Knowledge Graph — persistent, cumulative graph of explored mobile apps.

Covers the full product spec:

1. **Exploration & Knowledge Creation** — Screens with elements, locators,
   navigation.  Organized into Blocks (user-defined exploration scopes).

2. **App Graph** — Cumulative, persistent graph that merges explorations.
   Deduplicates screens, merges locators, tracks visit counts.

3. **Test Generation Context** — The graph provides screen knowledge, locators,
   and navigation paths as context for test case and Appium script generation.

4. **Self-Healing** — Locator repairs are tracked and written back to the
   graph's element locators so knowledge improves over time.

5. **Change Intelligence** — Diff two graph states to detect added, removed,
   and modified screens after a product update.

6. **Change-Based Test Generation** — The diff output feeds test generation
   to produce new-feature, regression, and negative test cases.

Usage::

    from fathom.app_graph import AppGraph, StudioExporter

    graph = AppGraph.load(path)
    graph.add_screen(...)
    graph.add_transition(...)
    graph.save(path)

    exporter = StudioExporter(graph)
    exporter.write(output_dir)
"""

from __future__ import annotations

from fathom.app_graph.graph import AppGraph
from fathom.app_graph.exporter import StudioExporter
from fathom.app_graph.models import (
    AppMap,
    Block,
    BlockStatus,
    ChangeType,
    Coverage,
    Edge,
    Element,
    Fingerprint,
    GraphDiff,
    LastRun,
    Locator,
    LocatorRepair,
    Page,
    RunAction,
    Screen,
    ScreenCategory,
    ScreenChange,
    ScreenFlow,
    ScreenLink,
    Transition,
)
from fathom.app_graph.renderer import PageRenderer

__all__ = [
    "AppGraph",
    "AppMap",
    "Block",
    "BlockStatus",
    "ChangeType",
    "Coverage",
    "Edge",
    "Element",
    "Fingerprint",
    "GraphDiff",
    "LastRun",
    "Locator",
    "LocatorRepair",
    "Page",
    "PageRenderer",
    "RunAction",
    "Screen",
    "ScreenCategory",
    "ScreenChange",
    "ScreenFlow",
    "ScreenLink",
    "StudioExporter",
    "Transition",
]
