"""
Structured functional content of a screen for per-screen documentation.

The ``describe_screen`` tool already returns a screen's purpose, its notable
elements, and what a user can achieve on it as discrete fields. This value object
keeps that structure first-class through persistence and export, rather than
flattening it into one prose blob that downstream consumers must re-parse.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class ScreenContent(BaseModel):
    """
    A screen's purpose, notable elements, and achievable actions, kept structured.
    """

    purpose: str = Field(
        default="", description="What the screen is for and the primary tasks available on it"
    )
    elements: List[str] = Field(
        default_factory=list,
        description="Notable interactive or informative elements, one per entry",
    )
    actions: List[str] = Field(
        default_factory=list,
        description="Concrete things a user can accomplish on the screen, one per entry",
    )
