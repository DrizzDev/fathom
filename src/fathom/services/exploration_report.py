"""
Knowledge graph analysis and exploration report generation.

Generates comprehensive exploration reports with graph metrics, cycle detection,
reachability analysis, and navigation insights.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional

from fathom.infrastructure.memory.knowledge_graph import (
    GraphNode,
    KnowledgeGraph,
)

logger = getLogger(__name__)


class ExplorationReportGenerator:
    """Generates comprehensive exploration reports with graph analysis."""

    def __init__(self, knowledge_graph: KnowledgeGraph) -> None:
        """Initialize report generator with knowledge graph."""
        self.kg = knowledge_graph

    def generate_full_report(
        self,
        workflow_id: str,
        duration_seconds: float,
        target_package: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive exploration report with all graph metrics.

        Returns
        -------
        Dict[str, Any]
            Complete report with graph analysis, metrics, and insights.
        """
        start_time = time.time()

        stats = self.kg.get_stats()
        cycles = self.kg.detect_cycles()

        # Get top screens
        top_screens = sorted(
            self.kg.nodes.values(),
            key=lambda n: (n.visit_count, n.last_seen or 0),
            reverse=True,
        )[:10]

        # Analyze reachability from each major screen
        reachability_analysis = self._analyze_reachability()

        # Identify critical screens
        critical_screens = self._identify_critical_screens()

        # Find key paths
        key_paths = self._find_key_paths()

        # Find all paths to target/exit screens
        paths_to_targets = self._find_all_paths_to_targets()

        # Collect screen translations keyed by normalized activity (one per unique activity)
        from fathom.infrastructure.memory.knowledge_graph import normalize_activity

        screen_translations: Dict[str, Dict[str, Any]] = {}
        for node in self.kg.nodes.values():
            norm = normalize_activity(node.activity)
            if node.rich_description and norm not in screen_translations:
                screen_translations[norm] = {
                    "activity": norm,
                    "description": node.description,
                    "rich_description": node.rich_description,
                }

        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "workflow_id": workflow_id,
                "target_package": target_package or "auto-detected",
                "exploration_duration_seconds": duration_seconds,
                "report_generation_time_seconds": time.time() - start_time,
            },
            "summary": {
                "unique_screens": stats["unique_screens"],
                "total_transitions": stats["total_transitions"],
                "total_visits": stats["total_visits"],
                "unique_activities": stats["unique_activities"],
                "activities": stats["activities"],
                "unexplored_screens": stats["unexplored"],
            },
            "graph_analysis": {
                "diameter": self.kg.get_graph_diameter(),
                "cycle_count": len(cycles),
                "cycles": self._format_cycles(cycles[:20]),  # Top 20 cycles
                "connected_components": self._analyze_connected_components(),
            },
            "screen_rankings": {
                "most_visited": self._format_screen_list(top_screens),
                "critical_screens": critical_screens,
            },
            "reachability_analysis": reachability_analysis,
            "navigation_paths": key_paths,
            "paths_to_targets": paths_to_targets,
            "activity_breakdown": self._analyze_activities(),
            "screen_translations": screen_translations,
            "recommendations": self._generate_recommendations(stats, cycles),
        }

        return report

    def _format_screen_list(self, screens: List[Any]) -> List[Dict[str, Any]]:
        """Format screen list for report."""
        result = []
        for screen in screens:
            context = self.kg.get_visualization_context(screen.visual_hash)
            result.append(
                {
                    "hash": screen.visual_hash[:16],
                    "description": screen.description or "no description",
                    "activity": screen.activity,
                    "visits": screen.visit_count,
                    "outgoing_edges": len(context["outgoing_edges"]),
                    "inbound_edges": len(context["inbound_edges"]),
                    "in_cycle": context["in_cycle"],
                }
            )
        return result

    def _format_cycles(self, cycles: List[List[str]]) -> List[Dict[str, Any]]:
        """Format cycles for report."""
        result = []
        for cycle in cycles:
            screens = []
            for node_hash in cycle[:-1]:  # Exclude last (which repeats first)
                screen = self.kg.get_screen(node_hash)
                if screen:
                    screens.append(screen.description or node_hash[:16])
            result.append(
                {
                    "length": len(cycle) - 1,
                    "screens": screens,
                }
            )
        return result

    def _analyze_reachability(self) -> Dict[str, Any]:
        """Analyze reachability from major screens."""
        major_screens = sorted(
            self.kg.nodes.values(),
            key=lambda n: n.visit_count,
            reverse=True,
        )[:5]

        reachability = {}
        for screen in major_screens:
            forward = self.kg.get_connected_component(screen.visual_hash)
            backward = self.kg.get_reverse_connected_component(screen.visual_hash)
            total = self.kg.node_count

            reachability[screen.description or screen.visual_hash[:16]] = {
                "forward_reach": len(forward),
                "forward_coverage": f"{(len(forward) / total * 100):.1f}%",
                "backward_reach": len(backward),
                "is_isolated": len(forward) == 1 and len(backward) == 1,
            }

        return reachability

    def _identify_critical_screens(self) -> List[Dict[str, Any]]:
        """Identify critical screens (bottlenecks, high connectivity)."""
        critical = []

        for screen in self.kg.nodes.values():
            context = self.kg.get_visualization_context(screen.visual_hash)

            edge_count = len(context["outgoing_edges"]) + len(context["inbound_edges"])
            is_hub = edge_count >= 5
            is_bottleneck = context["backward_reachable"] > self.kg.node_count * 0.5

            if is_hub or is_bottleneck:
                critical.append(
                    {
                        "name": screen.description or screen.visual_hash[:16],
                        "activity": screen.activity,
                        "type": "hub" if is_hub else "bottleneck",
                        "connectivity": edge_count,
                        "backward_reach": context["backward_reachable"],
                        "forward_reach": context["forward_reachable"],
                    }
                )

        return sorted(critical, key=lambda x: x["connectivity"], reverse=True)

    def _find_key_paths(self) -> List[Dict[str, Any]]:
        """Find key user journeys (entry to exit, major flows, and longest paths)."""
        paths: List[Dict[str, Any]] = []

        # First, find entry/exit screens
        entry_screens = [
            s
            for s in self.kg.nodes.values()
            if len(self.kg.get_reverse_connected_component(s.visual_hash)) <= 2
        ]
        exit_screens = [
            s
            for s in self.kg.nodes.values()
            if len(self.kg.get_connected_component(s.visual_hash)) <= 2
        ]

        # Find paths from entry to exit
        for entry in entry_screens[:3]:
            for exit_screen in exit_screens[:3]:
                if entry.visual_hash != exit_screen.visual_hash:
                    path = self.kg.find_path(entry.visual_hash, exit_screen.visual_hash)
                    if path and len(path) > 2:
                        screens: List[str] = []
                        for h, _ in path:
                            screen = self.kg.get_screen(h)
                            if screen:
                                screens.append(screen.description or h[:16])
                            else:
                                screens.append(h[:16])
                        paths.append(
                            {
                                "from": entry.description or entry.visual_hash[:16],
                                "to": exit_screen.description or exit_screen.visual_hash[:16],
                                "steps": len(path) - 1,
                                "path": screens,  # Complete path
                                "type": "entry_exit_journey",
                            }
                        )

        # Also find longest paths (diameter paths)
        diameter = self.kg.get_graph_diameter()
        if diameter and diameter > 3:
            max_paths_found: List[Dict[str, Any]] = []
            nodes = list(self.kg.nodes.keys())
            for start in nodes[: min(10, len(nodes))]:
                for end in nodes:
                    if start != end:
                        path = self.kg.find_path(start, end)
                        if path and len(path) - 1 == diameter:
                            screens = []
                            for h, _ in path:
                                screen = self.kg.get_screen(h)
                                if screen:
                                    screens.append(screen.description or h[:16])
                                else:
                                    screens.append(h[:16])
                            start_screen = self.kg.get_screen(start)
                            end_screen = self.kg.get_screen(end)
                            max_paths_found.append(
                                {
                                    "from": start_screen.description
                                    if start_screen
                                    else start[:16],
                                    "to": end_screen.description if end_screen else end[:16],
                                    "steps": len(path) - 1,
                                    "path": screens,
                                    "type": "longest_path",
                                }
                            )
                if max_paths_found:
                    break
            paths.extend(max_paths_found[:2])

        return paths[:5]  # Top 5 key paths

    def _find_all_paths_to_targets(self) -> Dict[str, Any]:
        """Find all paths leading to exit/target screens."""
        result: Dict[str, Any] = {}

        # Identify target screens (screens with low outgoing edges, i.e., exit points)
        target_candidates: List[tuple[Any, int]] = []
        for screen in self.kg.nodes.values():
            context = self.kg.get_visualization_context(screen.visual_hash)
            outgoing = len(context.get("outgoing_edges", []))
            # Exit screens typically have 0-2 outgoing edges
            if outgoing <= 2:
                target_candidates.append((screen, outgoing))

        # Sort by visit count (more visited = more important target)
        target_candidates.sort(key=lambda x: x[0].visit_count, reverse=True)

        # For top 3 target screens, find all paths to them
        for target_screen, _ in target_candidates[:3]:
            target_hash = target_screen.visual_hash
            all_paths_to_target: List[Dict[str, Any]] = []

            # Find paths from all other screens to this target
            for source_screen in self.kg.nodes.values():
                if source_screen.visual_hash != target_hash:
                    paths = self.kg.find_all_paths(
                        source_screen.visual_hash, target_hash, max_depth=8
                    )

                    for path in paths[:3]:  # Top 3 paths per source
                        screens: List[str] = []
                        for h, _ in path:
                            screen_obj: Optional[GraphNode] = self.kg.get_screen(h)
                            if screen_obj:
                                screens.append(screen_obj.description or h[:16])
                            else:
                                screens.append(h[:16])

                        all_paths_to_target.append(
                            {
                                "from": source_screen.description or source_screen.visual_hash[:16],
                                "to": target_screen.description or target_hash[:16],
                                "steps": len(path) - 1,
                                "path": screens,  # Complete path
                            }
                        )

            # Store top paths to this target
            target_name = target_screen.description or target_hash[:16]
            result[target_name] = {
                "target": target_name,
                "total_paths_found": len(all_paths_to_target),
                "sample_paths": all_paths_to_target[:10],  # Top 10 paths
            }

        return result

    def _analyze_activities(self) -> Dict[str, Any]:
        """Analyze activity distribution."""
        from fathom.infrastructure.memory.knowledge_graph import normalize_activity

        activity_map: Dict[str, List[str]] = {}

        for screen in self.kg.nodes.values():
            norm = normalize_activity(screen.activity)
            if norm not in activity_map:
                activity_map[norm] = []
            activity_map[norm].append(screen.description or screen.visual_hash[:16])

        return {
            activity: {
                "screen_count": len(screens),
                "samples": screens[:3],
            }
            for activity, screens in activity_map.items()
        }

    def _analyze_connected_components(self) -> Dict[str, Any]:
        """Analyze graph connectivity."""
        visited_set = set()
        components = []

        for node_hash in self.kg.nodes:
            if node_hash not in visited_set:
                component = self.kg.get_connected_component(node_hash)
                visited_set.update(component)
                samples = []
                for h in list(component)[:3]:
                    screen = self.kg.get_screen(h)
                    if screen:
                        samples.append(screen.description or h[:16])
                components.append(
                    {
                        "size": len(component),
                        "samples": samples,
                    }
                )

        # Extract sizes with proper typing
        sizes = [int(c.get("size", 0)) for c in components]  # type: ignore[call-overload]
        largest = max(sizes, default=0) if sizes else 0
        sorted_components = sorted(
            components,
            key=lambda x: int(x.get("size", 0)),  # type: ignore[call-overload]
            reverse=True,
        )[:5]

        return {
            "total_components": len(components),
            "largest_component": largest,
            "components": sorted_components,
        }

    def _generate_recommendations(
        self,
        stats: Dict[str, Any],
        cycles: List[List[str]],
    ) -> List[str]:
        """Generate recommendations based on graph analysis."""
        recommendations = []

        # Check coverage
        if stats["unexplored"] > stats["unique_screens"] * 0.3:
            recommendations.append("⚠️  Low coverage: More than 30% of screens are under-explored")

        # Check cycles
        if len(cycles) > stats["unique_screens"]:
            recommendations.append(
                "⚠️  High cycle count: Graph has many loops. Watch for infinite navigation."
            )

        # Check isolation
        isolated = [
            s
            for s in self.kg.nodes.values()
            if len(self.kg.get_connected_component(s.visual_hash)) == 1
        ]
        if isolated:
            recommendations.append(
                f"⚠️  Isolated screens: {len(isolated)} screens are unreachable from others"
            )

        # Check connectivity
        avg_edges = sum(len(e) for e in self.kg._KnowledgeGraph__edges.values()) / max(  # type: ignore[attr-defined, misc]
            stats["unique_screens"],
            1,
        )
        if avg_edges < 1.5:
            recommendations.append("→ Low connectivity: Consider exploring longer navigation paths")

        if not recommendations:
            recommendations.append("✓ Graph appears well-explored and connected!")

        return recommendations

    async def save_report(
        self,
        report: Dict[str, Any],
        output_dir: Optional[str] = None,
    ) -> Path:
        """
        Save exploration report to file (both JSON and markdown formats).

        Parameters
        ----------
        report:
            Report dictionary to save
        output_dir:
            Output directory (default: assets/reports/)

        Returns
        -------
        Path
            Path to saved JSON report file
        """
        if output_dir is None:
            output_dir = "assets/reports"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workflow_id = report["metadata"]["workflow_id"]

        # Save JSON
        json_filename = f"exploration_report_{workflow_id}_{timestamp}.json"
        json_filepath = output_path / json_filename
        with json_filepath.open("w") as f:
            json.dump(report, f, indent=2)

        # Save markdown
        md_filename = f"exploration_report_{workflow_id}_{timestamp}.md"
        md_filepath = output_path / md_filename
        markdown_content = self.export_report_markdown(report)
        with md_filepath.open("w") as f:
            f.write(markdown_content)

        # Save screen translations file to per-app memory directory
        translations_path = self.save_translations_file(
            report=report,
            workflow_id=workflow_id,
            timestamp=timestamp,
        )

        logger.info(
            "Exploration report saved to %s, %s, and %s",
            json_filepath,
            md_filepath,
            translations_path or "no translations",
        )
        return json_filepath

    def save_translations_file(
        self,
        report: Dict[str, Any],
        workflow_id: str,
        timestamp: str,
    ) -> Optional[Path]:
        """
        Save all screen translations to a single combined markdown file
        in the per-app memory directory (assets/memory/{package}/).

        Returns the file path, or None if there are no translations.
        """

        translations = report.get("screen_translations", {})
        if not translations:
            return None

        package: str = report["metadata"].get("target_package", "unknown")
        app_dir = Path("assets/memory") / package
        app_dir.mkdir(parents=True, exist_ok=True)

        filename = f"screen_translations_{workflow_id}_{timestamp}.md"
        filepath = app_dir / filename

        lines = [f"# Screen Descriptions: {workflow_id}\n"]
        meta = report.get("metadata", {})
        lines.append(f"**Generated:** {meta.get('generated_at', 'unknown')}")
        lines.append(f"**Package:** `{meta.get('target_package', 'unknown')}`")
        lines.append(f"**Activities described:** {len(translations)}\n")
        lines.append("---\n")

        for activity, entry in translations.items():
            desc = entry.get("description") or "Unnamed screen"
            rich = entry.get("rich_description", "")

            # Activity section heading
            lines.append(f"## {activity}\n")
            lines.append(f"*{desc}*\n")

            # Split the rich description into the initial observation and
            # additional observations (separated by "---" + "### Additional Observation").
            parts = rich.split("\n\n---\n\n### Additional Observation\n")

            # First observation
            lines.append("### Initial Observation\n")
            lines.append(parts[0].strip())
            lines.append("")

            # Additional observations
            for i, part in enumerate(parts[1:], start=2):
                lines.append(f"### Observation {i}\n")
                lines.append(part.strip())
                lines.append("")

            lines.append("---\n")

        with filepath.open("w") as f:
            f.write("\n".join(lines))

        logger.info("Screen translations saved to %s", filepath)

        # Also render a PDF alongside the markdown
        self.save_translations_pdf(
            report=report,
            workflow_id=workflow_id,
            timestamp=timestamp,
        )

        return filepath

    def save_translations_pdf(
        self,
        report: Dict[str, Any],
        workflow_id: str,
        timestamp: str,
    ) -> Optional[Path]:
        """
        Render screen translations as a styled PDF using reportlab.

        Returns the file path, or None if rendering fails.
        """

        translations = report.get("screen_translations", {})
        if not translations:
            return None

        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import mm
            from reportlab.platypus import (
                Paragraph,
                SimpleDocTemplate,
                Spacer,
            )
        except ImportError:
            logger.warning("reportlab not installed — skipping PDF generation")
            return None

        package: str = report["metadata"].get("target_package", "unknown")
        app_dir = Path("assets/memory") / package
        app_dir.mkdir(parents=True, exist_ok=True)

        filename = f"screen_translations_{workflow_id}_{timestamp}.pdf"
        filepath = app_dir / filename

        try:
            doc = SimpleDocTemplate(
                str(filepath),
                pagesize=A4,
                leftMargin=20 * mm,
                rightMargin=20 * mm,
                topMargin=15 * mm,
                bottomMargin=15 * mm,
            )

            styles = getSampleStyleSheet()

            # Custom styles
            title_style = ParagraphStyle(
                "ScreenTitle",
                parent=styles["Title"],
                fontSize=18,
                spaceAfter=6,
            )
            meta_style = ParagraphStyle(
                "Meta",
                parent=styles["Normal"],
                fontSize=9,
                textColor="#666666",
                spaceAfter=2,
            )
            activity_style = ParagraphStyle(
                "ActivityHeading",
                parent=styles["Heading2"],
                fontSize=13,
                spaceBefore=14,
                spaceAfter=4,
                textColor="#1a1a1a",
            )
            obs_style = ParagraphStyle(
                "ObservationHeading",
                parent=styles["Heading3"],
                fontSize=11,
                spaceBefore=8,
                spaceAfter=3,
                textColor="#444444",
            )
            body_style = ParagraphStyle(
                "Body",
                parent=styles["Normal"],
                fontSize=9,
                leading=13,
                spaceAfter=4,
            )
            section_style = ParagraphStyle(
                "SectionLabel",
                parent=styles["Normal"],
                fontSize=9,
                leading=13,
                spaceAfter=2,
                textColor="#0066cc",
            )

            story: List[Any] = []

            # Title
            meta = report.get("metadata", {})
            story.append(Paragraph(f"Screen Descriptions: {workflow_id}", title_style))
            story.append(
                Paragraph(
                    f"Package: {meta.get('target_package', 'unknown')} &nbsp;|&nbsp; "
                    f"Generated: {meta.get('generated_at', 'unknown')} &nbsp;|&nbsp; "
                    f"Activities: {len(translations)}",
                    meta_style,
                )
            )
            story.append(Spacer(1, 8 * mm))

            for activity, entry in translations.items():
                desc = entry.get("description") or "Unnamed screen"
                rich = entry.get("rich_description", "")

                # Activity heading
                story.append(Paragraph(activity, activity_style))
                story.append(Paragraph(f"<i>{desc}</i>", body_style))

                # Split observations
                parts = rich.split("\n\n---\n\n### Additional Observation\n")

                for idx, part in enumerate(parts):
                    label = "Initial Observation" if idx == 0 else f"Observation {idx + 1}"
                    story.append(Paragraph(label, obs_style))

                    # Render each line, highlighting section headers
                    for line in part.strip().splitlines():
                        stripped = line.strip()
                        if not stripped:
                            continue
                        if stripped.startswith("## "):
                            story.append(Paragraph(stripped[3:], section_style))
                        elif stripped.startswith("**Activity:**"):
                            continue  # Skip redundant activity line
                        else:
                            # Escape XML-sensitive chars for reportlab
                            safe = (
                                stripped.replace("&", "&amp;")
                                .replace("<", "&lt;")
                                .replace(">", "&gt;")
                            )
                            story.append(Paragraph(safe, body_style))

                story.append(Spacer(1, 6 * mm))

            doc.build(story)
            logger.info("Screen translations PDF saved to %s", filepath)
            return filepath

        except Exception:
            logger.warning("PDF generation failed", exc_info=True)
            return None

    def export_report_markdown(self, report: Dict[str, Any]) -> str:
        """Export report as comprehensive human-readable markdown."""
        lines = []

        meta = report["metadata"]
        lines.append(f"# Exploration Report: {meta['workflow_id']}")
        lines.append(f"\n**Generated:** {meta['generated_at']}")
        lines.append(f"**Package:** `{meta['target_package']}`")
        lines.append(f"**Duration:** {meta['exploration_duration_seconds']:.1f}s")
        lines.append(f"**Report Generation Time:** {meta['report_generation_time_seconds']:.4f}s")

        # Summary
        summary = report["summary"]
        lines.append("\n## 📊 Summary Statistics")
        lines.append(f"- **Unique Screens:** {summary['unique_screens']}")
        lines.append(f"- **Total Transitions:** {summary['total_transitions']}")
        lines.append(f"- **Total Visits:** {summary['total_visits']}")
        lines.append(f"- **Activities:** {summary['unique_activities']}")
        lines.append(f"- **Unexplored Screens:** {summary['unexplored_screens']}")

        # Graph Analysis
        graph = report["graph_analysis"]
        lines.append("\n## 🕸️ Graph Analysis")
        lines.append(f"- **Graph Diameter:** {graph['diameter'] or 'N/A'} steps")
        lines.append(f"- **Cycles Detected:** {graph['cycle_count']}")
        lines.append(
            f"- **Connected Components:** {graph['connected_components']['total_components']}"
        )
        lines.append(f"- **Total Edges:** {len(report['screen_rankings'].get('most_visited', []))}")

        # Top Screens
        rankings = report["screen_rankings"]
        lines.append("\n## 🔝 Most Visited Screens")
        lines.append("| Rank | Screen | Visits | Edges |")
        lines.append("|------|--------|--------|-------|")
        for i, screen in enumerate(rankings["most_visited"][:10], 1):
            lines.append(
                f"| {i} | {screen['description'][:35]} | {screen['visits']} | {screen['outgoing_edges']} |"
            )

        # Critical Screens
        if rankings.get("critical_screens"):
            lines.append("\n## ⚠️ Critical Screens (Hubs & Bottlenecks)")
            for screen in rankings["critical_screens"][:8]:
                lines.append(
                    f"- **{screen['name']}** ({screen['type']}) - {screen.get('connections', 'N/A')} connections"
                )

        # Navigation Paths
        if report.get("navigation_paths"):
            lines.append("\n## 🛣️ Key Navigation Paths")
            longest_paths = [
                p for p in report["navigation_paths"] if p.get("type") == "longest_path"
            ]
            entry_paths = [
                p for p in report["navigation_paths"] if p.get("type") == "entry_exit_journey"
            ]

            if longest_paths:
                lines.append("\n### 📏 Longest Paths (Diameter Paths)")
                for i, path in enumerate(longest_paths, 1):
                    path_str = " → ".join(path["path"])
                    lines.append(
                        f"{i}. **{path['from']}** to **{path['to']}** ({path['steps']} steps)"
                    )
                    lines.append(f"   - Route: `{path_str}`")

            if entry_paths:
                lines.append("\n### 🚪 Entry-to-Exit Journeys")
                for i, path in enumerate(entry_paths[:5], 1):
                    path_str = " → ".join(path["path"])
                    lines.append(
                        f"{i}. **{path['from']}** to **{path['to']}** ({path['steps']} steps)"
                    )
                    lines.append(f"   - Route: `{path_str}`")

        # All Paths to Target Screens
        if report.get("paths_to_targets"):
            lines.append("\n## 🎯 All Paths to Target/Exit Screens")
            for target_name, target_info in report["paths_to_targets"].items():
                lines.append(
                    f"\n### Paths to **{target_name}** ({target_info['total_paths_found']} unique paths found)"
                )
                for i, path in enumerate(target_info["sample_paths"][:5], 1):
                    path_str = " → ".join(path["path"])
                    lines.append(f"{i}. From **{path['from']}** ({path['steps']} steps)")
                    lines.append(f"   - `{path_str}`")

        # Cycles
        if graph.get("cycles"):
            lines.append("\n## 🔄 Detected Cycles")
            for i, cycle in enumerate(graph["cycles"][:10], 1):
                cycle_str = " → ".join(cycle["screens"])
                lines.append(f"{i}. {cycle_str} (length: {cycle['length']})")

        # Reachability
        reachability = report.get("reachability_analysis", {})
        if reachability:
            lines.append("\n## 📡 Reachability Analysis")
            lines.append("| Screen | Forward Reach | Backward Reach |")
            lines.append("|--------|---------------|----------------|")
            for screen in reachability.get("by_screen", [])[:10]:
                forward = (
                    f"{screen['forward_reach_percentage']:.1f}%"
                    if screen.get("forward_reach_percentage")
                    else "0%"
                )
                backward = screen.get("backward_reach_count", 0)
                lines.append(f"| {screen['screen'][:20]} | {forward} | {backward} |")

        # Activity Breakdown
        if report.get("activity_breakdown"):
            lines.append("\n## 🏷️ Activity Breakdown")
            for activity, screens in report["activity_breakdown"].items():
                lines.append(f"- **{activity}:** {len(screens)} screens")

        # Recommendations
        if report.get("recommendations"):
            lines.append("\n## 💡 Recommendations")
            for i, rec in enumerate(report["recommendations"], 1):
                lines.append(f"{i}. {rec}")

        lines.append(f"\n---\n*Report generated on {meta['generated_at']}*")

        return "\n".join(lines)

    def export_report_summary(self, report: Dict[str, Any]) -> str:
        """Export report as human-readable text summary."""
        lines = []
        lines.append("=" * 70)
        lines.append("EXPLORATION REPORT SUMMARY")
        lines.append("=" * 70)

        meta = report["metadata"]
        lines.append(f"\nGenerated: {meta['generated_at']}")
        lines.append(f"Workflow: {meta['workflow_id']}")
        lines.append(f"Package: {meta['target_package']}")
        lines.append(f"Duration: {meta['exploration_duration_seconds']:.1f}s")

        summary = report["summary"]
        lines.append(f"\n{'SUMMARY':=^70}")
        lines.append(f"  Unique Screens: {summary['unique_screens']}")
        lines.append(f"  Total Transitions: {summary['total_transitions']}")
        lines.append(f"  Total Visits: {summary['total_visits']}")
        lines.append(f"  Activities: {summary['unique_activities']}")

        graph_analysis = report["graph_analysis"]
        lines.append(f"\n{'GRAPH ANALYSIS':=^70}")
        lines.append(f"  Diameter: {graph_analysis['diameter'] or 'N/A'}")
        lines.append(f"  Cycles: {graph_analysis['cycle_count']}")
        lines.append(
            f"  Connected Components: {graph_analysis['connected_components']['total_components']}"
        )

        rankings = report["screen_rankings"]
        lines.append(f"\n{'TOP SCREENS':=^70}")
        for i, screen in enumerate(rankings["most_visited"][:5], 1):
            lines.append(
                f"  {i}. {screen['description'][:40]:40} "
                f"(visits: {screen['visits']}, edges: {screen['outgoing_edges']})"
            )

        if rankings["critical_screens"]:
            lines.append(f"\n{'CRITICAL SCREENS':=^70}")
            for screen in rankings["critical_screens"][:3]:
                lines.append(f"  • {screen['name'][:40]:40} ({screen['type']})")

        recommendations = report["recommendations"]
        lines.append(f"\n{'RECOMMENDATIONS':=^70}")
        for rec in recommendations:
            lines.append(f"  {rec}")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)
