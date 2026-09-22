"""Render screen knowledge as Markdown page files for Drizz Studio.

The studio's frontend ``markdown()`` function handles standard Markdown:
headings, bold, code, lists, and tables.  The renderer produces one ``.md``
file per logical screen, plus an ``index.md`` that links them all.

Page files include locators for each element so the studio's test generator
and self-healer can reference them directly.
"""

from __future__ import annotations

from fathom.app_graph.models import (
    Element,
    GraphDiff,
    ScreenChange,
    ChangeType,
    Locator,
    Screen,
    ScreenFlow,
    ScreenLink,
    slugify,
)


class PageRenderer:
    """Renders screens into Markdown consumed by Drizz Studio."""

    # ------------------------------------------------------------------
    # Single page
    # ------------------------------------------------------------------

    def render_page(self, screen: Screen, flow: ScreenFlow) -> str:
        """Return a complete Markdown page for one screen."""
        parts: list[str] = []

        # Header
        parts.append(f"# {screen.name}\n")
        parts.append(f"- **Category:** {screen.category}")
        parts.append(f"- **Activity:** `{screen.activity}`")
        parts.append(f"- **Visits:** {screen.visits}")
        parts.append("")

        # Purpose
        if screen.purpose:
            parts.append("## Purpose\n")
            parts.append(screen.purpose)
            parts.append("")

        # Description (full prose, if different from purpose)
        if screen.description and screen.description != screen.purpose:
            parts.append("## Screen\n")
            parts.append(screen.description)
            parts.append("")

        # Elements with locators
        if screen.elements:
            parts.append("## Elements\n")
            for el in screen.elements:
                parts.append(_format_element(el))
            parts.append("")

        # Actions
        if screen.actions:
            parts.append("## What You Can Do\n")
            for action in screen.actions:
                parts.append(f"- {action}")
            parts.append("")

        # Inbound navigation
        if flow.inbound:
            parts.append("## Reached From\n")
            for link in flow.inbound:
                parts.append(_format_link(link, direction="from"))
            parts.append("")

        # Outbound navigation with locators
        if flow.outbound:
            parts.append("## Leads To\n")
            for link in flow.outbound:
                parts.append(_format_link(link, direction="to"))
            parts.append("")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def render_index(
        self,
        screens: list[Screen],
        *,
        package: str = "",
        generated_at: str = "",
    ) -> str:
        """Return a Markdown index linking to per-screen pages."""
        parts: list[str] = []
        parts.append("# Screen Documentation\n")

        if package:
            parts.append(f"- **Package:** `{package}`")
        if generated_at:
            parts.append(f"- **Generated:** {generated_at}")
        parts.append(f"- **Screens:** {len(screens)}")
        total_elements = sum(len(s.elements) for s in screens)
        total_locators = sum(len(el.locators) for s in screens for el in s.elements)
        if total_elements:
            parts.append(f"- **Elements:** {total_elements}")
        if total_locators:
            parts.append(f"- **Locators:** {total_locators}")
        parts.append("")

        if screens:
            parts.append("## Screens\n")
            parts.append("| Screen | Category | Elements | Visits |")
            parts.append("| --- | --- | --- | --- |")
            for s in screens:
                slug = slugify(s.name)
                parts.append(
                    f"| [{s.name}]({slug}.md) | {s.category} "
                    f"| {len(s.elements)} | {s.visits} |"
                )
            parts.append("")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Change Intelligence report
    # ------------------------------------------------------------------

    def render_diff(self, diff: GraphDiff) -> str:
        """Render a GraphDiff as a Markdown change report."""
        parts: list[str] = []
        parts.append("# Change Intelligence Report\n")
        parts.append(f"- **Package:** `{diff.package}`")
        parts.append(f"- **Compared:** {diff.compared_at}")
        parts.append(f"- **Before:** {diff.old_screen_count} screens")
        parts.append(f"- **After:** {diff.new_screen_count} screens")
        parts.append(f"- **Summary:** {diff.summary()}")
        parts.append("")

        if diff.added_screens:
            parts.append("## New Screens\n")
            for c in diff.added_screens:
                parts.append(f"### {c.screen_name}\n")
                parts.append(f"- **Activity:** `{c.activity}`")
                if c.added_elements:
                    parts.append(f"- **Elements:** {', '.join(c.added_elements)}")
                if c.added_navigation:
                    parts.append(f"- **Navigation:**")
                    for nav in c.added_navigation:
                        parts.append(f"  - {nav}")
                parts.append("")

        if diff.removed_screens:
            parts.append("## Removed Screens\n")
            for c in diff.removed_screens:
                parts.append(f"- ~~{c.screen_name}~~ (`{c.activity}`)")
            parts.append("")

        if diff.modified_screens:
            parts.append("## Modified Screens\n")
            for c in diff.modified_screens:
                parts.append(f"### {c.screen_name}\n")
                if c.added_elements:
                    parts.append(f"- **New elements:** {', '.join(c.added_elements)}")
                if c.removed_elements:
                    parts.append(f"- **Removed elements:** {', '.join(c.removed_elements)}")
                if c.added_navigation:
                    parts.append(f"- **New navigation:** {', '.join(c.added_navigation)}")
                if c.removed_navigation:
                    parts.append(f"- **Removed navigation:** {', '.join(c.removed_navigation)}")
                if c.purpose_changed:
                    parts.append(f"- **Purpose changed:** {c.old_purpose} → {c.new_purpose}")
                parts.append("")

        if not diff.has_changes:
            parts.append("No changes detected between the two graph states.\n")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_element(el: Element) -> str:
    """Format an element as a Markdown bullet with locator details."""
    type_tag = f" `{el.element_type}`" if el.element_type else ""
    interactive_tag = "" if el.interactive else " *(display only)*"
    line = f"- **{el.name}**{type_tag}{interactive_tag}"

    if el.locators:
        active = [loc for loc in el.locators if loc.confidence > 0]
        if active:
            best = max(active, key=lambda l: l.confidence)
            line += f"\n  - Locator: `{best.strategy}` = `{best.value}`"
            if len(active) > 1:
                line += f" (+{len(active) - 1} fallback{'s' if len(active) > 2 else ''})"

    return line


def _format_link(link: ScreenLink, *, direction: str) -> str:
    """Format a navigation link as a Markdown bullet.

    Examples::

        - tap "Search" from **Home** (×2)
        - type "biryani" to **Search Results** (×1)
          - Locator: `resource_id` = `search_input`
    """
    element_part = f' "{link.element}"' if link.element else ""
    value_part = f' "{link.value}"' if link.value else ""
    count_part = f" (×{link.count})" if link.count > 1 else ""
    line = f"- {link.action}{element_part}{value_part} {direction} **{link.screen}**{count_part}"

    if link.locator and link.locator.confidence > 0:
        line += f"\n  - Locator: `{link.locator.strategy}` = `{link.locator.value}`"

    return line
