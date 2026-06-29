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


# Generic element-type descriptors the vision model appends to a target's visible
# text (for example "Continue button", "Email input field"). They are stripped from
# a link's element so the rendered target matches the element's exact on-screen text,
# which is what the test-authoring consumer grounds against. Ordered longest-first so
# multi-word descriptors are matched before their single-word tails.
GENERIC_ELEMENT_SUFFIXES: tuple[str, ...] = (
    "input field",
    "text field",
    "search field",
    "search bar",
    "button",
    "checkbox",
    "dropdown",
    "toggle",
    "switch",
    "icon",
    "field",
    "option",
    "tab",
)

# Shortest remaining text, after stripping a descriptor, that is still treated as a
# usable visible-text target; below this the original target is kept unchanged.
MINIMUM_VISIBLE_TARGET_LENGTH: int = 2

# Version of the published screen-documentation JSON artifact. The consumer reads
# this to detect contract changes; bump the minor for additive fields and the major
# for breaking changes to the structured screen schema.
SCREEN_DOCUMENT_SCHEMA_VERSION: str = "1.0"

# Filename of the published machine-readable screen-documentation artifact, written
# beside the per-screen Markdown so the consumer can prefer it over re-parsing prose.
SCREEN_DOCUMENT_ARTIFACT_FILENAME: str = "index.json"
