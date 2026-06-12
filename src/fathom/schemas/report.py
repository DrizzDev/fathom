"""
Typed model of a structured exploration-analysis report.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from fathom.constants.exploration import CriticalScreenKind, RecommendationLevel


class ReportMetadata(BaseModel):
    """
    Identifying context for an exploration report.
    """

    workflow: str = Field(description="Identifier of the exploration run")
    package: str = Field(description="Android package the run targeted")
    generated_at: str = Field(description="ISO-8601 timestamp the report was produced")
    duration: float = Field(default=0.0, ge=0.0, description="Exploration run length in seconds")


class CoverageSummary(BaseModel):
    """
    Headline coverage and connectivity metrics for an exploration run.
    """

    screens: int = Field(default=0, ge=0, description="Unique screens discovered")
    transitions: int = Field(default=0, ge=0, description="Total recorded transitions")
    visits: int = Field(default=0, ge=0, description="Total screen visits across the run")
    activities: int = Field(default=0, ge=0, description="Distinct activities discovered")
    unexplored: int = Field(default=0, ge=0, description="Screens not yet fully explored")
    coverage: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Percentage of screens fully explored"
    )
    diameter: Optional[int] = Field(
        default=None, description="Longest shortest-path between any two screens"
    )
    cycles: int = Field(default=0, ge=0, description="Navigation cycles detected")


class ScreenInsight(BaseModel):
    """
    Connectivity and visit profile for a single screen.
    """

    hash: str = Field(description="Canonical visual hash identifying the screen")
    activity: str = Field(description="Android activity the screen belongs to")
    description: Optional[str] = Field(default=None, description="Human-readable screen summary")
    visits: int = Field(default=0, ge=0, description="Times the screen was visited")
    outgoing: int = Field(default=0, ge=0, description="Outgoing transitions from the screen")
    inbound: int = Field(default=0, ge=0, description="Transitions leading into the screen")
    in_cycle: bool = Field(default=False, description="Whether the screen lies on a cycle")


class CriticalScreen(BaseModel):
    """
    A hub or bottleneck screen in the navigation graph.
    """

    name: str = Field(description="Description or short hash of the screen")
    activity: str = Field(description="Android activity the screen belongs to")
    kind: CriticalScreenKind = Field(description="Why the screen is structurally significant")
    connectivity: int = Field(default=0, ge=0, description="Combined inbound and outbound edges")
    forward_reach: int = Field(default=0, ge=0, description="Screens reachable from this one")
    backward_reach: int = Field(default=0, ge=0, description="Screens that can reach this one")


class ActivityCoverage(BaseModel):
    """
    Screen count for a single activity.
    """

    activity: str = Field(description="Normalised Android activity name")
    screens: int = Field(default=0, ge=0, description="Screens belonging to the activity")


class NavigationCycle(BaseModel):
    """
    A detected navigation loop.
    """

    length: int = Field(ge=1, description="Number of screens in the loop")
    screens: List[str] = Field(default_factory=list, description="Screens on the loop, in order")


class ComponentSummary(BaseModel):
    """
    Connected-component structure of the screen graph.
    """

    count: int = Field(default=0, ge=0, description="Number of disconnected components")
    largest: int = Field(default=0, ge=0, description="Screen count of the largest component")


class Recommendation(BaseModel):
    """
    A single actionable insight about the exploration coverage.
    """

    level: RecommendationLevel = Field(description="Severity of the recommendation")
    message: str = Field(description="Human-readable guidance")


class ExplorationReport(BaseModel):
    """
    Structured analysis of a completed exploration run.
    """

    metadata: ReportMetadata = Field(description="Identifying context for the run")
    coverage: CoverageSummary = Field(description="Headline coverage metrics")
    most_visited: List[ScreenInsight] = Field(
        default_factory=list, description="Most-visited screens, ranked"
    )
    critical_screens: List[CriticalScreen] = Field(
        default_factory=list, description="Hubs and bottlenecks"
    )
    activities: List[ActivityCoverage] = Field(
        default_factory=list, description="Per-activity screen counts"
    )
    cycles: List[NavigationCycle] = Field(
        default_factory=list, description="Detected navigation loops"
    )
    components: ComponentSummary = Field(
        default_factory=ComponentSummary, description="Graph connectivity structure"
    )
    recommendations: List[Recommendation] = Field(
        default_factory=list, description="Actionable coverage insights"
    )
