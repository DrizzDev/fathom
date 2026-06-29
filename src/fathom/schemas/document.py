"""
Typed models for the per-screen exploration documentation.

A document describes one logical screen (one activity + one category) so a reader
understands it without ever seeing a screenshot. The many fingerprints the crawl
captures for a screen collapse into a single document, with the extra captures
counted as variants.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from fathom.constants.document import SCREEN_DOCUMENT_SCHEMA_VERSION
from fathom.constants.screen import ScreenCategory
from fathom.schemas.actions import Action
from fathom.schemas.defect import Defect
from fathom.schemas.report import ReportMetadata


class LinkSemantics(BaseModel):
    """
    Recorded intent and classification of a transition, sourced from the crawl.
    """

    outcome: Optional[str] = Field(
        default=None, description="Predicted screen effect of the transition"
    )
    category: Optional[str] = Field(
        default=None, description="UI role of the element the action addressed"
    )
    region: Optional[str] = Field(
        default=None, description="Screen region the addressed element sits in"
    )
    overlay: bool = Field(
        default=False, description="Whether an overlay was present when the action ran"
    )
    rationale: Optional[str] = Field(
        default=None, description="Why the action was taken during exploration"
    )

    @classmethod
    def of(cls, *, action: Action) -> "LinkSemantics":
        """
        Builds link semantics from the action that drove a transition.
        """

        return cls(
            outcome=action.expected_outcome.value if action.expected_outcome else None,
            category=action.element_category,
            region=action.region,
            overlay=action.overlay_detected,
            rationale=action.rationale.strip() or None,
        )


class ScreenLink(BaseModel):
    """
    One navigational edge between two logical screens.
    """

    action: str = Field(description="Action type that drives the transition")
    element: Optional[str] = Field(
        default=None, description="Human-readable element the action addressed"
    )
    screen: str = Field(description="Title of the logical screen on the other end")
    count: int = Field(default=1, ge=1, description="Times the transition was observed")
    value: Optional[str] = Field(
        default=None, description="Text entered on the transition, for type actions"
    )
    semantics: Optional[LinkSemantics] = Field(
        default=None, description="Recorded intent and classification of the transition"
    )


class ScreenFlow(BaseModel):
    """
    How a logical screen connects to the rest of the application.
    """

    inbound: List[ScreenLink] = Field(
        default_factory=list, description="Transitions that arrive at this screen"
    )
    outbound: List[ScreenLink] = Field(
        default_factory=list, description="Transitions that leave this screen"
    )


class ScreenDocument(BaseModel):
    """
    A self-contained, image-free description of one logical screen.
    """

    slug: str = Field(description="Filename-safe identifier, unique within the run")
    title: str = Field(description="Human-readable screen title")
    category: ScreenCategory = Field(description="Functional kind of the screen")
    activity: str = Field(description="Android activity the screen belongs to")
    purpose: str = Field(default="", description="One-line summary of what the screen is for")
    narrative: str = Field(
        default="", description="Full prose description of the screen's elements and actions"
    )
    elements: List[str] = Field(
        default_factory=list,
        description="Notable interactive or informative elements on the screen, one per entry",
    )
    actions: List[str] = Field(
        default_factory=list,
        description="Concrete things a user can accomplish on the screen, one per entry",
    )
    interactions: List[ScreenLink] = Field(
        default_factory=list,
        description="In-place actions performed on the screen that stay on it, e.g. typing",
    )
    flow: ScreenFlow = Field(
        default_factory=ScreenFlow, description="Inbound and outbound navigation"
    )
    defects: List[Defect] = Field(
        default_factory=list, description="Defects attributed to this screen, most severe first"
    )
    visits: int = Field(default=0, ge=0, description="Total visits across all fingerprints")
    fingerprints: int = Field(
        default=1, ge=1, description="Distinct captures that collapsed into this logical screen"
    )


class DocumentIndex(BaseModel):
    """
    The full set of per-screen documents for a completed exploration run.
    """

    schema_version: str = Field(
        default=SCREEN_DOCUMENT_SCHEMA_VERSION,
        description="Version of the screen-documentation contract this artifact follows",
    )
    metadata: ReportMetadata = Field(description="Identifying context for the run")
    documents: List[ScreenDocument] = Field(
        default_factory=list, description="One document per logical screen"
    )
