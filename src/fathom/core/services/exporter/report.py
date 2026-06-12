"""
Generates and renders structured exploration-analysis reports.

The generator reads the live knowledge graph and produces a typed
:class:`ExplorationReport`; the renderer turns that report into Markdown. The
report timestamp is injected, never read from the clock, so generation is
deterministic and testable.
"""

from __future__ import annotations

from typing import Dict, List, Set

from fathom.constants.exploration import (
    BOTTLENECK_REACH_RATIO,
    HUB_CONNECTIVITY_THRESHOLD,
    LOW_COVERAGE_RATIO,
    MAX_REPORTED_CYCLES,
    MIN_AVERAGE_CONNECTIVITY,
    TOP_SCREEN_LIMIT,
    CriticalScreenKind,
    RecommendationLevel,
)
from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph
from fathom.schemas.report import (
    ActivityCoverage,
    ComponentSummary,
    CoverageSummary,
    CriticalScreen,
    ExplorationReport,
    NavigationCycle,
    Recommendation,
    ReportMetadata,
    ScreenInsight,
)

_SHORT_HASH_LENGTH = 16


class ExplorationReportGenerator:
    """
    Builds a structured analysis report from a knowledge graph.
    """

    def __init__(self, *, graph: KnowledgeGraph) -> None:
        self.__graph = graph

    def generate(
        self, *, workflow: str, package: str, generated_at: str, duration: float = 0.0
    ) -> ExplorationReport:
        """
        Produces the full typed report for a completed exploration run.
        """

        cycles = self.__graph.detect_cycles()
        inbound = self.__inbound_counts()
        cycle_members = {node_hash for cycle in cycles for node_hash in cycle}

        return ExplorationReport(
            metadata=ReportMetadata(
                workflow=workflow, package=package, generated_at=generated_at, duration=duration
            ),
            coverage=self.__coverage(cycle_count=len(cycles)),
            most_visited=self.__most_visited(inbound=inbound, cycle_members=cycle_members),
            critical_screens=self.__critical_screens(inbound=inbound),
            activities=self.__activities(),
            cycles=self.__cycles(cycles=cycles),
            components=self.__components(),
            recommendations=self.__recommendations(cycle_count=len(cycles)),
        )

    def __inbound_counts(self) -> Dict[str, int]:
        """
        Counts the transitions leading into each screen.
        """

        counts: Dict[str, int] = {}
        for edges in self.__graph.edges.values():
            for edge in edges:
                counts[edge.destination_hash] = counts.get(edge.destination_hash, 0) + 1
        return counts

    def __coverage(self, *, cycle_count: int) -> CoverageSummary:
        """
        Summarises headline coverage and connectivity metrics.
        """

        stats = self.__graph.get_stats()
        screens = int(stats.get("unique_screens", 0))
        unexplored = int(stats.get("unexplored", 0))
        coverage = ((screens - unexplored) / screens * 100.0) if screens else 0.0

        return CoverageSummary(
            screens=screens,
            transitions=int(stats.get("total_transitions", 0)),
            visits=int(stats.get("total_visits", 0)),
            activities=int(stats.get("unique_activities", 0)),
            unexplored=unexplored,
            coverage=round(coverage, 1),
            diameter=self.__graph.get_graph_diameter(),
            cycles=cycle_count,
        )

    def __most_visited(
        self, *, inbound: Dict[str, int], cycle_members: Set[str]
    ) -> List[ScreenInsight]:
        """
        Ranks the most-visited screens with their connectivity profile.
        """

        ranked = sorted(
            self.__graph.nodes.values(),
            key=lambda node: (node.visit_count, node.last_seen or 0),
            reverse=True,
        )[:TOP_SCREEN_LIMIT]

        return [
            ScreenInsight(
                hash=node.visual_hash,
                activity=node.activity,
                description=node.description,
                visits=node.visit_count,
                outgoing=len(self.__graph.get_neighbors(visual_hash=node.visual_hash)),
                inbound=inbound.get(node.visual_hash, 0),
                in_cycle=node.visual_hash in cycle_members,
            )
            for node in ranked
        ]

    def __critical_screens(self, *, inbound: Dict[str, int]) -> List[CriticalScreen]:
        """
        Identifies hubs and bottlenecks in the navigation graph.

        Reachability is computed per screen, so cost scales with graph size;
        exploration graphs are small enough for this to stay inexpensive.
        """

        graph = self.__graph
        bottleneck_threshold = graph.node_count * BOTTLENECK_REACH_RATIO
        critical: List[CriticalScreen] = []

        for node in graph.nodes.values():
            outgoing = len(graph.get_neighbors(visual_hash=node.visual_hash))
            connectivity = outgoing + inbound.get(node.visual_hash, 0)
            backward = len(graph.get_reverse_connected_component(end_hash=node.visual_hash))

            is_hub = connectivity >= HUB_CONNECTIVITY_THRESHOLD
            is_bottleneck = backward > bottleneck_threshold
            if not (is_hub or is_bottleneck):
                continue

            critical.append(
                CriticalScreen(
                    name=node.description or node.visual_hash[:_SHORT_HASH_LENGTH],
                    activity=node.activity,
                    kind=CriticalScreenKind.HUB if is_hub else CriticalScreenKind.BOTTLENECK,
                    connectivity=connectivity,
                    forward_reach=len(graph.get_connected_component(start_hash=node.visual_hash)),
                    backward_reach=backward,
                )
            )

        return sorted(critical, key=lambda screen: screen.connectivity, reverse=True)

    def __activities(self) -> List[ActivityCoverage]:
        """
        Counts the screens discovered per normalised activity.
        """

        counts: Dict[str, int] = {}
        for node in self.__graph.nodes.values():
            activity = KnowledgeGraph.normalize_activity(node.activity)
            counts[activity] = counts.get(activity, 0) + 1

        return [
            ActivityCoverage(activity=activity, screens=count)
            for activity, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
        ]

    def __cycles(self, *, cycles: List[List[str]]) -> List[NavigationCycle]:
        """
        Formats the detected navigation cycles for reporting.
        """

        result: List[NavigationCycle] = []
        for cycle in cycles[:MAX_REPORTED_CYCLES]:
            screens = []
            for node_hash in cycle[:-1]:  # The final hash repeats the first.
                node = self.__graph.get_screen(visual_hash=node_hash)
                screens.append(
                    node.description
                    if node and node.description
                    else node_hash[:_SHORT_HASH_LENGTH]
                )
            result.append(NavigationCycle(length=max(len(cycle) - 1, 1), screens=screens))
        return result

    def __components(self) -> ComponentSummary:
        """
        Summarises the forward-reachability clusters of the screen graph.
        """

        graph = self.__graph
        seen: Set[str] = set()
        sizes: List[int] = []

        for node_hash in graph.nodes:
            if node_hash in seen:
                continue
            component = graph.get_connected_component(start_hash=node_hash)
            seen.update(component)
            sizes.append(len(component))

        return ComponentSummary(count=len(sizes), largest=max(sizes, default=0))

    def __recommendations(self, *, cycle_count: int) -> List[Recommendation]:
        """
        Derives actionable coverage recommendations from the graph.
        """

        graph = self.__graph
        stats = graph.get_stats()
        screens = int(stats.get("unique_screens", 0))
        unexplored = int(stats.get("unexplored", 0))

        recommendations: List[Recommendation] = []

        if screens and unexplored > screens * LOW_COVERAGE_RATIO:
            recommendations.append(
                Recommendation(
                    level=RecommendationLevel.WARNING,
                    message="More than 30% of screens remain under-explored.",
                )
            )

        if cycle_count > screens:
            recommendations.append(
                Recommendation(
                    level=RecommendationLevel.WARNING,
                    message="High cycle count - watch for repeating navigation loops.",
                )
            )

        dead_ends = sum(
            1
            for node in graph.nodes.values()
            if len(graph.get_connected_component(start_hash=node.visual_hash)) == 1
        )
        if dead_ends:
            recommendations.append(
                Recommendation(
                    level=RecommendationLevel.WARNING,
                    message=f"{dead_ends} screens are dead ends with no outgoing transitions.",
                )
            )

        average_connectivity = graph.edge_count / screens if screens else 0.0
        if screens and average_connectivity < MIN_AVERAGE_CONNECTIVITY:
            recommendations.append(
                Recommendation(
                    level=RecommendationLevel.NOTE,
                    message="Low connectivity - consider exploring longer navigation paths.",
                )
            )

        if not recommendations:
            recommendations.append(
                Recommendation(
                    level=RecommendationLevel.OK,
                    message="Graph appears well-explored and connected.",
                )
            )

        return recommendations


class MarkdownReportRenderer:
    """
    Renders a structured exploration report as Markdown.
    """

    __LEVEL_MARKERS: Dict[RecommendationLevel, str] = {
        RecommendationLevel.WARNING: "WARNING",
        RecommendationLevel.NOTE: "NOTE",
        RecommendationLevel.OK: "OK",
    }

    def render(self, *, report: ExplorationReport) -> str:
        """
        Returns the full report as a Markdown document.
        """

        sections = [
            self.__header(metadata=report.metadata),
            self.__coverage(coverage=report.coverage),
            self.__activities(activities=report.activities),
            self.__most_visited(screens=report.most_visited),
            self.__critical(screens=report.critical_screens),
            self.__cycles(cycles=report.cycles),
            self.__components(components=report.components),
            self.__recommendations(recommendations=report.recommendations),
        ]
        return "\n\n".join(section for section in sections if section)

    @staticmethod
    def __header(*, metadata: ReportMetadata) -> str:
        """
        Renders the title and run metadata.
        """

        return "\n".join(
            [
                "# Exploration Report",
                "",
                f"- Workflow: {metadata.workflow}",
                f"- Package: {metadata.package}",
                f"- Generated: {metadata.generated_at}",
                f"- Duration: {metadata.duration:.1f}s",
            ]
        )

    @staticmethod
    def __coverage(*, coverage: CoverageSummary) -> str:
        """
        Renders the headline coverage metrics.
        """

        diameter = str(coverage.diameter) if coverage.diameter is not None else "n/a"
        return "\n".join(
            [
                "## Coverage",
                "",
                f"- Screens: {coverage.screens}",
                f"- Transitions: {coverage.transitions}",
                f"- Visits: {coverage.visits}",
                f"- Activities: {coverage.activities}",
                f"- Coverage: {coverage.coverage:.1f}%",
                f"- Unexplored: {coverage.unexplored}",
                f"- Cycles: {coverage.cycles}",
                f"- Diameter: {diameter}",
            ]
        )

    @staticmethod
    def __activities(*, activities: List[ActivityCoverage]) -> str:
        """
        Renders the per-activity screen breakdown.
        """

        if not activities:
            return ""
        rows = [f"| {item.activity} | {item.screens} |" for item in activities]
        return "\n".join(["## Activities", "", "| Activity | Screens |", "| --- | --- |", *rows])

    @staticmethod
    def __most_visited(*, screens: List[ScreenInsight]) -> str:
        """
        Renders the most-visited screen ranking.
        """

        if not screens:
            return ""
        rows = [
            f"| {screen.description or screen.hash[:_SHORT_HASH_LENGTH]} | {screen.visits} | "
            f"{screen.outgoing} | {screen.inbound} | {'yes' if screen.in_cycle else 'no'} |"
            for screen in screens
        ]
        return "\n".join(
            [
                "## Most Visited Screens",
                "",
                "| Screen | Visits | Out | In | In cycle |",
                "| --- | --- | --- | --- | --- |",
                *rows,
            ]
        )

    @staticmethod
    def __critical(*, screens: List[CriticalScreen]) -> str:
        """
        Renders the hubs and bottlenecks.
        """

        if not screens:
            return ""
        rows = [
            f"| {screen.name} | {screen.kind.value} | {screen.connectivity} | "
            f"{screen.forward_reach} | {screen.backward_reach} |"
            for screen in screens
        ]
        return "\n".join(
            [
                "## Critical Screens",
                "",
                "| Screen | Kind | Connectivity | Forward | Backward |",
                "| --- | --- | --- | --- | --- |",
                *rows,
            ]
        )

    @staticmethod
    def __cycles(*, cycles: List[NavigationCycle]) -> str:
        """
        Renders the detected navigation cycles.
        """

        if not cycles:
            return ""
        rows = [f"- ({cycle.length}) {' -> '.join(cycle.screens)}" for cycle in cycles]
        return "\n".join(["## Navigation Cycles", "", *rows])

    @staticmethod
    def __components(*, components: ComponentSummary) -> str:
        """
        Renders the connectivity-cluster summary.
        """

        return "\n".join(
            [
                "## Connectivity",
                "",
                f"- Clusters: {components.count}",
                f"- Largest cluster: {components.largest}",
            ]
        )

    def __recommendations(self, *, recommendations: List[Recommendation]) -> str:
        """
        Renders the actionable recommendations.
        """

        if not recommendations:
            return ""
        rows = [
            f"- [{self.__LEVEL_MARKERS[item.level]}] {item.message}" for item in recommendations
        ]
        return "\n".join(["## Recommendations", "", *rows])
