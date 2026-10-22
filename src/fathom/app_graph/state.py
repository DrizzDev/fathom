"""Typed persistent application knowledge and observed discovery events."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from fathom.app_graph.models import Block, LocatorRepair, Screen, Transition


class Identity(BaseModel):
    """Organization and application boundary for shared knowledge."""

    tenant: str = Field(min_length=1, description="Trusted organization identity.")
    package: str = Field(min_length=1, description="Application package or bundle identifier.")


class Snapshot(BaseModel):
    """Complete cumulative graph state suitable for a repository."""

    package: str = Field(default="", description="Application identifier.")
    label: str = Field(default="", description="Display label.")
    goals: list[str] = Field(default_factory=list, description="Exploration goals.")
    exploration_count: int = Field(default=0, description="Distinct exploration count.")
    total_steps: int = Field(default=0, description="Observed discovery count.")
    screens: list[Screen] = Field(default_factory=list, description="Logical pages.")
    transitions: list[Transition] = Field(default_factory=list, description="Observed navigation.")
    aliases: dict[str, str] = Field(default_factory=dict, description="Canonical screen aliases.")
    blocks: list[Block] = Field(default_factory=list, description="Exploration blocks.")
    locator_repairs: list[LocatorRepair] = Field(
        default_factory=list, description="Verified repair history."
    )


class Observation(BaseModel):
    """One evidenced screen observation in an exploration block."""

    identity: Identity = Field(description="Application knowledge boundary.")
    execution: str = Field(min_length=1, description="Stable run/block identifier.")
    sequence: int = Field(ge=0, description="Capture sequence within the run.")
    goal: str = Field(min_length=1, description="Requested exploration scope.")
    screen: Screen = Field(description="Observed page and native locators.")
    image: bytes = Field(repr=False, description="Observed PNG evidence.")
    xml: Optional[str] = Field(default=None, repr=False, description="Native hierarchy evidence.")
    transition: Optional[Transition] = Field(
        default=None, description="Successful preceding action and observed destination."
    )


class Completion(BaseModel):
    """Terminal result of an exploration block."""

    identity: Identity = Field(description="Application boundary.")
    execution: str = Field(description="Stable block identifier.")
    error: Optional[str] = Field(default=None, description="Failure or cancellation reason.")


class Start(BaseModel):
    """Persistent Block registration before the first capture."""

    identity: Identity = Field(description="Application boundary.")
    execution: str = Field(min_length=1, description="Stable block identifier.")
    goal: str = Field(min_length=1, description="Requested exploration scope.")


class Description(BaseModel):
    """Model interpretation grounded in a previously captured page."""

    identity: Identity = Field(description="Application boundary.")
    screen: str = Field(description="Observed page hash.")
    text: str = Field(description="Screen description from the exploration model.")
