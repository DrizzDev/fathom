"""Export an AppGraph to the Drizz Studio file layout.

The studio runner stores App Maps under ``app-maps/{package}/`` with this
structure::

    app-maps/com.example.app/
    ├── index.json          # AppMap (pages, edges, metadata)
    ├── evidence/
    │   ├── home.png          # screenshot captures
    │   ├── home.xml          # UI hierarchy dumps
    │   ├── search.png
    │   └── search.xml
    ├── home.md               # per-screen page knowledge
    ├── search.md
    └── index.md              # screen index

``StudioExporter.write()`` produces exactly this layout so the studio's
graph viewer, test generator, and self-healer can read the files directly.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fathom.app_graph.graph import AppGraph
from fathom.app_graph.models import (
    AppMap,
    Edge,
    Fingerprint,
    LastRun,
    Page,
    RunAction,
    slugify,
)
from fathom.app_graph.renderer import PageRenderer

logger = logging.getLogger(__name__)


class StudioExporter:
    """Transforms an :class:`AppGraph` into the Drizz Studio file layout."""

    def __init__(self, graph: AppGraph) -> None:
        self._graph = graph
        self._renderer = PageRenderer()

    # ------------------------------------------------------------------
    # Build the studio data model
    # ------------------------------------------------------------------

    def build_app_map(
        self,
        *,
        last_run: Optional[LastRun] = None,
    ) -> AppMap:
        """Build a :class:`AppMap` from the current graph state."""
        slugs = self._assign_slugs()
        pages = self._build_pages(slugs)
        edges = self._build_edges(slugs)

        now = datetime.now(timezone.utc).isoformat()

        return AppMap(
            package=self._graph.package,
            label=self._graph.label,
            goal=self._graph.goals[-1] if self._graph.goals else "",
            goals=self._graph.goals,
            generated_at=now,
            steps=self._graph.total_steps,
            exploration_count=self._graph.exploration_count,
            last_run=last_run,
            pages=pages,
            edges=edges,
            updated_at=now,
        )

    # ------------------------------------------------------------------
    # Write to disk
    # ------------------------------------------------------------------

    def write(
        self,
        output_dir: Path,
        *,
        last_run: Optional[LastRun] = None,
    ) -> list[Path]:
        """Write the full studio file layout and return all paths written."""
        app_map = self.build_app_map(last_run=last_run)
        written: list[Path] = []

        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. index.json
        map_path = output_dir / "index.json"
        map_path.write_text(
            app_map.model_dump_json(indent=2, by_alias=True),
            encoding="utf-8",
        )
        written.append(map_path)

        # 2. Evidence files (screenshots + XML)
        evidence_dir = output_dir / "evidence"
        evidence_dir.mkdir(exist_ok=True)
        for page in app_map.pages:
            screen = self._screen_for_page(page)
            if not screen:
                continue
            if screen.screenshot_path:
                dst = evidence_dir / f"{page.id}.png"
                _copy_if_exists(screen.screenshot_path, dst)
                written.append(dst)
            if screen.xml_path:
                dst = evidence_dir / f"{page.id}.xml"
                _copy_if_exists(screen.xml_path, dst)
                written.append(dst)

        # 3. Page markdown files
        for page in app_map.pages:
            screen = self._screen_for_page(page)
            if not screen:
                continue
            flow = self._graph.screen_flow(screen.hash)
            md = self._renderer.render_page(screen, flow)
            md_path = output_dir / f"{page.id}.md"
            md_path.write_text(md, encoding="utf-8")
            page.file = f"{page.id}.md"
            written.append(md_path)

        # 4. Index
        index_md = self._renderer.render_index(
            self._graph.screens,
            package=self._graph.package,
            generated_at=app_map.generated_at,
        )
        index_path = output_dir / "index.md"
        index_path.write_text(index_md, encoding="utf-8")
        written.append(index_path)

        # Re-write index.json with updated file references
        map_path.write_text(
            app_map.model_dump_json(indent=2, by_alias=True),
            encoding="utf-8",
        )

        logger.info(
            "Wrote studio app map: %d pages, %d edges → %s",
            len(app_map.pages), len(app_map.edges), output_dir,
        )
        return written

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assign_slugs(self) -> dict[str, str]:
        """Map screen hash → unique slug."""
        slugs: dict[str, str] = {}
        used: dict[str, int] = {}

        for screen in self._graph.screens:
            base = slugify(screen.name)
            if base in used:
                used[base] += 1
                slug = f"{base}-{used[base]}"
            else:
                used[base] = 1
                slug = base
            slugs[screen.hash] = slug

        return slugs

    def _build_pages(self, slugs: dict[str, str]) -> list[Page]:
        """Build studio pages from graph screens."""
        pages: list[Page] = []
        for screen in self._graph.screens:
            slug = slugs[screen.hash]
            has_screenshot = bool(screen.screenshot_path and Path(screen.screenshot_path).exists())
            has_xml = bool(screen.xml_path and Path(screen.xml_path).exists())

            pages.append(Page(
                id=slug,
                node_id=f"page:{slug}",
                name=screen.name,
                activity=screen.activity,
                fingerprint=Fingerprint(
                    struct=[],
                    content=[],
                ),
                visits=screen.visits,
                file=f"{slug}.md",
                screenshot=f"evidence/{slug}.png" if has_screenshot else None,
                evidence_xml=f"evidence/{slug}.xml" if has_xml else None,
            ))

        return pages

    def _build_edges(self, slugs: dict[str, str]) -> list[Edge]:
        """Build studio edges from graph transitions."""
        edges: list[Edge] = []
        for t in self._graph.transitions:
            src_slug = slugs.get(t.source_hash)
            tgt_slug = slugs.get(t.target_hash)
            if not src_slug or not tgt_slug:
                continue
            if src_slug == tgt_slug:
                continue  # self-loops are badges in the studio, not edges

            edges.append(Edge(
                from_node=f"page:{src_slug}",
                to_node=f"page:{tgt_slug}",
                label=_describe_action(t.action_type, t.action_target, t.value),
            ))

        return edges

    def _screen_for_page(self, page: Page) -> Optional["Screen"]:
        """Find the internal screen backing a studio page."""
        slug_map = self._assign_slugs()
        for h, slug in slug_map.items():
            if slug == page.id:
                return self._graph.screen_by_hash(h)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _describe_action(
    action_type: str,
    target: Optional[str] = None,
    value: Optional[str] = None,
) -> str:
    """Build a human-readable action label.

    >>> _describe_action("tap", "Search")
    'tap "Search"'
    >>> _describe_action("type", "Search field", "biryani")
    'type "biryani"'
    >>> _describe_action("scroll", None, None)
    'scroll'
    >>> _describe_action("back")
    'back'
    """
    if action_type == "type" and value:
        return f'type "{value}"'
    if target:
        return f'{action_type} "{target}"'
    return action_type


def _copy_if_exists(src: str, dst: Path) -> bool:
    """Copy a file if the source exists.  Returns whether it succeeded."""
    src_path = Path(src)
    if src_path.exists():
        shutil.copy2(src_path, dst)
        return True
    logger.debug("Evidence file not found: %s", src)
    return False
