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

from fathom.constants.screen import ScreenCategory
from fathom.schemas.defect import Defect
from fathom.schemas.report import ReportMetadata


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

    metadata: ReportMetadata = Field(description="Identifying context for the run")
    documents: List[ScreenDocument] = Field(
        default_factory=list, description="One document per logical screen"
    )
