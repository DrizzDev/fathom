"""
Named markers for the per-screen Markdown documentation contract.

These headings and relation words are the structural contract the test-authoring
consumer parses, so they live as named constants rather than inline literals.
"""

from __future__ import annotations

from enum import StrEnum


class SectionHeading(StrEnum):
    """Second-level headings the screen-document renderer emits, in render order."""

    PURPOSE = "Purpose"
    SCREEN = "Screen"
    ELEMENTS = "Elements"
    ACTIONS = "What You Can Do"
    REACHED_FROM = "Reached From"
    LEADS_TO = "Leads To"
    DEFECTS = "Defects"


class Relation(StrEnum):
    """Connective word linking a navigation bullet to the screen on its other end."""

    INBOUND = "from"
    OUTBOUND = "to"
