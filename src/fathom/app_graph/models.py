"""Data models for the App Knowledge Graph.

Three families live here:

1. **Internal models** — what Fathom stores and reasons about (``Screen``,
   ``Transition``, ``Element``, ``Locator``, ``Block``, ``LocatorRepair``).

2. **Studio models** — the exact JSON shapes Drizz Studio reads from disk
   (``Page``, ``Edge``, ``AppMap``).

3. **Change Intelligence models** — diffing two graph states to detect what
   changed in the application (``GraphDiff``, ``ScreenChange``).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ScreenCategory(StrEnum):
    HOME = "home"
    LIST = "list"
    DETAIL = "detail"
    FORM = "form"
    AUTH = "auth"
    PAYMENT = "payment"
    SETTINGS = "settings"
    ONBOARDING = "onboarding"
    SEARCH = "search"
    OTHER = "other"


class BlockStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ChangeType(StrEnum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


# ---------------------------------------------------------------------------
# Locators — the foundation for self-healing
# ---------------------------------------------------------------------------

class Locator(BaseModel):
    """A single way to find a UI element on screen.

    Multiple locator strategies are stored per element so the system can
    fall back when one breaks.  When self-healing fixes a locator, the
    old one is kept with ``confidence=0`` and the new one is added.
    """

    strategy: str  # resource_id, xpath, content_desc, text, bounds, class_name, uiautomator
    value: str  # the actual selector value
    confidence: float = 1.0  # 0.0–1.0, lowered when healed away
    last_verified: Optional[str] = None  # ISO-8601 timestamp of last successful use


class Element(BaseModel):
    """An interactive or informative UI element on a screen.

    Each element carries one or more locators that Appium can use to find it.
    """

    name: str  # human-readable, e.g. "Search bar", "Add to cart button"
    locators: list[Locator] = Field(default_factory=list)
    interactive: bool = True  # can the user tap/type into this?
    element_type: str = ""  # button, input, text, image, toggle, …
    bounds: Optional[str] = None  # "x,y,w,h" from UI hierarchy


# ---------------------------------------------------------------------------
# Blocks — user-defined exploration scopes
# ---------------------------------------------------------------------------

class Block(BaseModel):
    """A user-defined exploration scope (a 'Block' in the product spec).

    Examples:
        - 'Explore the Home page'
        - 'Explore Search → Restaurant → Cart'
        - 'Explore the Cart and Checkout flow'
    """

    id: str
    intent: str  # what the user asked to explore
    status: BlockStatus = BlockStatus.PENDING
    screens_discovered: list[str] = Field(default_factory=list)  # screen hashes
    transitions_discovered: int = 0
    steps: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Self-healing — locator repair records
# ---------------------------------------------------------------------------

class LocatorRepair(BaseModel):
    """Record of a locator that was healed during execution.

    When Appium fails to find an element, the AI inspects the screen,
    identifies the correct element, and records the repair here.  The
    repair is written back to both the screen's element locators and
    the test script.
    """

    screen_name: str
    element_name: str
    old_locator: Locator
    new_locator: Locator
    repaired_at: str = ""  # ISO-8601
    test_case: str = ""  # which test triggered the repair
    persisted: bool = False  # written back to graph + script?


# ---------------------------------------------------------------------------
# Internal models — what the graph stores
# ---------------------------------------------------------------------------

class ScreenLink(BaseModel):
    """One navigational edge between two screens."""

    model_config = ConfigDict(frozen=True)

    action: str
    element: Optional[str] = None
    screen: str  # target screen name
    count: int = 1
    value: Optional[str] = None  # text entered (for type actions)
    locator: Optional[Locator] = None  # locator used for this navigation


class ScreenFlow(BaseModel):
    """Inbound and outbound navigation for a single screen."""

    inbound: list[ScreenLink] = Field(default_factory=list)
    outbound: list[ScreenLink] = Field(default_factory=list)


class Screen(BaseModel):
    """A logical screen in the App Knowledge Graph.

    A logical screen may be backed by several visual captures that share the
    same Android activity and structural fingerprint.  The graph deduplicates
    them into one ``Screen`` and increments ``visits``.
    """

    hash: str  # canonical visual hash (primary key in the graph)
    name: str
    activity: str
    category: ScreenCategory = ScreenCategory.OTHER
    purpose: str = ""
    elements: list[Element] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    visits: int = 1
    structure_hash: Optional[str] = None
    screenshot_path: Optional[str] = None  # absolute path to captured PNG
    xml_path: Optional[str] = None  # absolute path to captured XML
    description: Optional[str] = None  # short prose summary

    def element_names(self) -> list[str]:
        """Return just the element names as a flat list."""
        return [e.name for e in self.elements]

    def find_element(self, name: str) -> Optional[Element]:
        """Find an element by name (case-insensitive)."""
        lower = name.lower()
        for el in self.elements:
            if el.name.lower() == lower:
                return el
        return None

    def best_locator(self, element_name: str) -> Optional[Locator]:
        """Return the highest-confidence locator for an element."""
        el = self.find_element(element_name)
        if not el or not el.locators:
            return None
        return max(el.locators, key=lambda loc: loc.confidence)


class Transition(BaseModel):
    """A directed edge between two screens."""

    model_config = ConfigDict(frozen=True)

    source_hash: str
    target_hash: str
    action_type: str  # tap, type, scroll, back, …
    action_target: Optional[str] = None  # element description
    value: Optional[str] = None  # text entered (for type actions)
    count: int = 1
    locator: Optional[Locator] = None  # locator used for the action


# ---------------------------------------------------------------------------
# Change Intelligence models
# ---------------------------------------------------------------------------

class ScreenChange(BaseModel):
    """How a single screen changed between two graph states."""

    screen_name: str
    change_type: ChangeType
    activity: str = ""
    # What specifically changed (only for MODIFIED)
    added_elements: list[str] = Field(default_factory=list)
    removed_elements: list[str] = Field(default_factory=list)
    added_navigation: list[str] = Field(default_factory=list)  # "tap X → ScreenY"
    removed_navigation: list[str] = Field(default_factory=list)
    purpose_changed: bool = False
    old_purpose: str = ""
    new_purpose: str = ""


class GraphDiff(BaseModel):
    """Diff between an old graph state and a new exploration.

    Used by Change Intelligence to detect what changed in the application
    after a product update, and to generate change-specific tests.
    """

    package: str
    compared_at: str = ""  # ISO-8601
    old_screen_count: int = 0
    new_screen_count: int = 0
    changes: list[ScreenChange] = Field(default_factory=list)

    @property
    def added_screens(self) -> list[ScreenChange]:
        return [c for c in self.changes if c.change_type == ChangeType.ADDED]

    @property
    def removed_screens(self) -> list[ScreenChange]:
        return [c for c in self.changes if c.change_type == ChangeType.REMOVED]

    @property
    def modified_screens(self) -> list[ScreenChange]:
        return [c for c in self.changes if c.change_type == ChangeType.MODIFIED]

    @property
    def has_changes(self) -> bool:
        return any(c.change_type != ChangeType.UNCHANGED for c in self.changes)

    def summary(self) -> str:
        added = len(self.added_screens)
        removed = len(self.removed_screens)
        modified = len(self.modified_screens)
        parts = []
        if added:
            parts.append(f"{added} new screen{'s' if added != 1 else ''}")
        if removed:
            parts.append(f"{removed} removed")
        if modified:
            parts.append(f"{modified} modified")
        return ", ".join(parts) if parts else "no changes detected"


# ---------------------------------------------------------------------------
# Studio models — exact JSON shapes Drizz Studio reads
# ---------------------------------------------------------------------------

def _to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(w.capitalize() for w in parts[1:])


class Fingerprint(BaseModel):
    """Structural fingerprint attached to every studio page."""

    model_config = ConfigDict(populate_by_name=True, alias_generator=_to_camel)

    pkg: str = ""
    struct: list[str] = Field(default_factory=list)
    content: list[str] = Field(default_factory=list)
    state: list[str] = Field(default_factory=list)


class Page(BaseModel):
    """A page in the Drizz Studio App Map."""

    model_config = ConfigDict(populate_by_name=True, alias_generator=_to_camel)

    id: str  # slug, e.g. "home", "search-results"
    node_id: str  # always "page:" + id
    name: str
    activity: str
    fingerprint: Fingerprint = Field(default_factory=Fingerprint)
    visits: int = 1
    file: Optional[str] = None  # e.g. "home.md"
    screenshot: Optional[str] = None  # e.g. "evidence/home.png"
    evidence_xml: Optional[str] = None  # e.g. "evidence/home.xml"


class Edge(BaseModel):
    """A navigation edge in the Drizz Studio App Map."""

    model_config = ConfigDict(populate_by_name=True)

    from_node: str = Field(alias="from")  # nodeId of source, e.g. "page:home"
    to_node: str = Field(alias="to")  # nodeId of target
    label: str  # e.g. 'tap "Search"'


class RunAction(BaseModel):
    """One step recorded during the most recent exploration run."""

    model_config = ConfigDict(populate_by_name=True, alias_generator=_to_camel)

    step: int
    screen: str
    activity: str
    action: str
    reason: str = ""
    result: str = "screen changed"


class LastRun(BaseModel):
    """Summary of the most recent exploration run."""

    model_config = ConfigDict(populate_by_name=True, alias_generator=_to_camel)

    started_at: str  # ISO-8601
    ending: str = ""
    actions: list[RunAction] = Field(default_factory=list)


class AppMap(BaseModel):
    """Top-level Drizz Studio App Map — written as ``app-map.json``."""

    model_config = ConfigDict(populate_by_name=True, alias_generator=_to_camel)

    package: str
    label: str
    goal: str = ""
    goals: list[str] = Field(default_factory=list)
    generated_at: str = ""  # ISO-8601
    steps: int = 0
    exploration_count: int = 1
    last_run: Optional[LastRun] = None
    locator_repairs: list[dict] = Field(default_factory=list)
    repair_log: Optional[str] = None
    pages: list[Page] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    updated_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Convert a screen name to a filename-safe slug.

    >>> slugify("Home Screen")
    'home-screen'
    >>> slugify("  Search (Results) ")
    'search-results'
    """
    s = _SLUG_STRIP.sub("-", name.lower().strip()).strip("-")
    return s or "screen"
