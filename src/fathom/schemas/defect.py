"""
Typed models for defects detected during exploration.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, computed_field

from fathom.constants.defect import (
    DefectKind,
    DefectSeverity,
    DefectSignal,
    DefectSource,
    DefectVerification,
)
from fathom.constants.screen import HitOutcome
from fathom.schemas.actions import Bounds, CoordinateSource
from fathom.schemas.report import ReportMetadata


class DefectEvidence(BaseModel):
    """
    Where a defect was observed and what supports it.
    """

    screen: str = Field(description="Canonical visual hash of the screen the defect was found on")
    activity: Optional[str] = Field(
        default=None, description="Android activity the screen belongs to"
    )
    bounds: Optional[Bounds] = Field(
        default=None, description="Region of the screen the defect occupies, when localizable"
    )
    excerpt: Optional[str] = Field(
        default=None, description="Offending text or short detail backing the defect"
    )
    screenshot: Optional[str] = Field(
        default=None,
        description="Screenshot URI for UI review; never required to read the report",
    )


class Defect(BaseModel):
    """
    A single problem found in the application under exploration.
    """

    signal: DefectSignal = Field(description="Specific observation evidencing the defect")
    severity: DefectSeverity = Field(description="How badly the defect degrades the experience")
    source: DefectSource = Field(description="Run stage that produced the defect")
    summary: str = Field(description="One-line human-readable description")
    evidence: DefectEvidence = Field(description="Where the defect was observed")
    occurrence: int = Field(
        default=1, ge=1, description="Times the defect was observed across the run"
    )
    verification: DefectVerification = Field(
        default=DefectVerification.CONFIRMED,
        description="Whether the defect is corroborated enough to lead the report",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def kind(self) -> DefectKind:
        """
        Broad category, derived from the signal so it cannot drift.
        """

        return self.signal.kind

    @classmethod
    def from_signal(
        cls,
        *,
        signal: DefectSignal,
        source: DefectSource,
        summary: str,
        evidence: DefectEvidence,
        severity: Optional[DefectSeverity] = None,
        verification: DefectVerification = DefectVerification.CONFIRMED,
    ) -> "Defect":
        """
        Builds a defect, defaulting severity from the signal when not supplied.
        """

        return cls(
            signal=signal,
            severity=severity or signal.default_severity,
            source=source,
            summary=summary,
            evidence=evidence,
            verification=verification,
        )

    @property
    def signature(self) -> str:
        """
        Stable dedup key: the same signal on the same screen region is one defect.
        """

        if self.evidence.bounds is not None:
            anchor = self.evidence.bounds.coord_bucket()
        else:
            anchor = self.evidence.excerpt or ""
        return f"{self.signal.value}|{self.evidence.screen}|{anchor}"


class ScreenDefects(BaseModel):
    """
    All defects attributed to one screen.
    """

    screen: str = Field(description="Canonical visual hash of the screen")
    defects: List[Defect] = Field(default_factory=list, description="Defects found on the screen")


class StepSignals(BaseModel):
    """
    Per-step runtime signals the inline detector inspects.
    """

    screen: str = Field(description="Canonical visual hash of the pre-action screen")
    activity: Optional[str] = Field(
        default=None, description="Android activity of the pre-action screen"
    )
    action_target: Optional[str] = Field(
        default=None, description="Human-readable target of the action that was executed"
    )
    expects_transition: bool = Field(
        description="Whether the action predicted a visible screen change"
    )
    screen_changed: bool = Field(description="Whether the screen actually changed after the action")
    left_package: bool = Field(
        default=False,
        description="Whether the action left the target package without recovery",
    )
    usable_capture: bool = Field(
        default=True, description="Whether the post-action capture yielded a usable screen"
    )
    post_activity: Optional[str] = Field(
        default=None, description="Android activity after the action settled"
    )
    grounding: Optional[CoordinateSource] = Field(
        default=None, description="Evidence source that produced the tapped coordinate"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Planner confidence in the tapped action"
    )
    target_hit: HitOutcome = Field(
        default=HitOutcome.UNKNOWN,
        description="Whether the tap landed inside an interactive element",
    )
    screenshot: Optional[str] = Field(
        default=None, description="Screenshot URI of the screen the step acted on"
    )


class ScreenSnapshot(BaseModel):
    """
    Read-only view of one screen handed to a screen-level defect detector.
    """

    screen: str = Field(description="Canonical visual hash of the screen")
    activity: Optional[str] = Field(
        default=None, description="Android activity the screen belongs to"
    )
    texts: List[str] = Field(
        default_factory=list, description="Visible text fragments from OCR and the hierarchy"
    )
    screenshot: Optional[bytes] = Field(
        default=None, description="Raw screenshot bytes for the vision detector, when available"
    )
    screenshot_uri: Optional[str] = Field(
        default=None, description="Screenshot URI stamped onto any defect found on the screen"
    )


class BugReport(BaseModel):
    """
    Aggregated defects for a completed exploration run.
    """

    metadata: ReportMetadata = Field(description="Identifying context for the run")
    defects: List[Defect] = Field(
        default_factory=list, description="All defects, most severe first"
    )
    by_kind: Dict[DefectKind, int] = Field(
        default_factory=dict, description="Defect count per kind"
    )
    by_severity: Dict[DefectSeverity, int] = Field(
        default_factory=dict, description="Defect count per severity"
    )
    needs_review: List[Defect] = Field(
        default_factory=list,
        description="Uncorroborated defects held back from the headline for manual triage",
    )
