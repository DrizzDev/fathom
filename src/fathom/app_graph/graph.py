"""App Knowledge Graph — the cumulative screen-and-navigation store.

One ``AppGraph`` per application package.  Screens are deduplicated by
``(activity, structure_hash)`` when a structure hash is available, and by
``(activity, name)`` as a fallback.  Transitions are deduplicated by
``(source, target, action_type, action_target)``.

Persistence is plain JSON so the graph can live alongside the studio's own
file layout on the runner's disk.  Call :meth:`save` / :meth:`load` to
round-trip.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fathom.app_graph.models import (
    Block,
    BlockStatus,
    ChangeType,
    Element,
    GraphDiff,
    Locator,
    LocatorRepair,
    Screen,
    ScreenChange,
    ScreenFlow,
    ScreenLink,
    Transition,
    slugify,
)

logger = logging.getLogger(__name__)


class AppGraph:
    """Persistent, cumulative knowledge graph of a mobile application.

    Covers the full product spec:
    - Exploration & knowledge creation (screens, elements, locators)
    - Cumulative graph building across blocks/explorations
    - Self-healing locator repair tracking
    - Change intelligence (graph diffing)
    """

    def __init__(self, *, package: str = "", label: str = "") -> None:
        self.package = package
        self.label = label
        self._screens: dict[str, Screen] = {}
        self._transitions: list[Transition] = []
        self._aliases: dict[str, str] = {}
        self._goals: list[str] = []
        self._exploration_count: int = 0
        self._total_steps: int = 0
        self._blocks: list[Block] = []
        self._locator_repairs: list[LocatorRepair] = []

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    @property
    def screens(self) -> list[Screen]:
        return list(self._screens.values())

    @property
    def transitions(self) -> list[Transition]:
        return list(self._transitions)

    @property
    def goals(self) -> list[str]:
        return list(self._goals)

    @property
    def exploration_count(self) -> int:
        return self._exploration_count

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def blocks(self) -> list[Block]:
        return list(self._blocks)

    @property
    def locator_repairs(self) -> list[LocatorRepair]:
        return list(self._locator_repairs)

    def screen_by_hash(self, h: str) -> Optional[Screen]:
        return self._screens.get(self.resolve(h))

    def screen_by_name(self, name: str) -> Optional[Screen]:
        """Find a screen by its display name (case-insensitive)."""
        lower = name.lower()
        for s in self._screens.values():
            if s.name.lower() == lower:
                return s
        return None

    def resolve(self, h: str) -> str:
        """Resolve a raw hash to its canonical equivalent."""
        return self._aliases.get(h, h)

    def outgoing(self, screen_hash: str) -> list[Transition]:
        canon = self.resolve(screen_hash)
        return [t for t in self._transitions if t.source_hash == canon]

    def incoming(self, screen_hash: str) -> list[Transition]:
        canon = self.resolve(screen_hash)
        return [t for t in self._transitions if t.target_hash == canon]

    def screen_flow(self, screen_hash: str) -> ScreenFlow:
        """Build the full navigation context for one screen."""
        canon = self.resolve(screen_hash)
        screen = self._screens.get(canon)
        if not screen:
            return ScreenFlow()

        inbound: list[ScreenLink] = []
        for t in self.incoming(canon):
            src = self._screens.get(t.source_hash)
            if not src or src.hash == canon:
                continue
            inbound.append(ScreenLink(
                action=t.action_type,
                element=t.action_target,
                screen=src.name,
                count=t.count,
                value=t.value,
                locator=t.locator,
            ))

        outbound: list[ScreenLink] = []
        for t in self.outgoing(canon):
            tgt = self._screens.get(t.target_hash)
            if not tgt or tgt.hash == canon:
                continue
            outbound.append(ScreenLink(
                action=t.action_type,
                element=t.action_target,
                screen=tgt.name,
                count=t.count,
                value=t.value,
                locator=t.locator,
            ))

        return ScreenFlow(inbound=inbound, outbound=outbound)

    def reachable_from(self, screen_hash: str, max_depth: int = 10) -> set[str]:
        """All screen hashes reachable from the given screen (BFS)."""
        canon = self.resolve(screen_hash)
        visited: set[str] = set()
        queue = [canon]
        depth = 0
        while queue and depth < max_depth:
            next_queue: list[str] = []
            for h in queue:
                if h in visited:
                    continue
                visited.add(h)
                for t in self.outgoing(h):
                    if t.target_hash not in visited:
                        next_queue.append(t.target_hash)
            queue = next_queue
            depth += 1
        return visited

    def find_path(self, start: str, end: str, max_depth: int = 10) -> list[str]:
        """BFS shortest path from start to end. Returns list of screen hashes."""
        start, end = self.resolve(start), self.resolve(end)
        if start == end:
            return [start]
        visited: set[str] = {start}
        queue: list[list[str]] = [[start]]
        while queue:
            path = queue.pop(0)
            if len(path) > max_depth:
                break
            current = path[-1]
            for t in self.outgoing(current):
                if t.target_hash == end:
                    return path + [end]
                if t.target_hash not in visited:
                    visited.add(t.target_hash)
                    queue.append(path + [t.target_hash])
        return []

    # ------------------------------------------------------------------
    # Blocks — exploration scopes
    # ------------------------------------------------------------------

    def create_block(self, intent: str) -> Block:
        """Create a new exploration block."""
        block = Block(
            id=str(uuid.uuid4())[:8],
            intent=intent,
            status=BlockStatus.PENDING,
        )
        self._blocks.append(block)
        return block

    def start_block(self, block_id: str) -> None:
        """Mark a block as running."""
        for b in self._blocks:
            if b.id == block_id:
                b.status = BlockStatus.RUNNING
                b.started_at = datetime.now(timezone.utc).isoformat()
                return

    def complete_block(
        self,
        block_id: str,
        *,
        screens_discovered: list[str] | None = None,
        transitions_discovered: int = 0,
        steps: int = 0,
        error: str | None = None,
    ) -> None:
        """Mark a block as completed (or failed)."""
        for b in self._blocks:
            if b.id == block_id:
                b.status = BlockStatus.FAILED if error else BlockStatus.COMPLETED
                b.completed_at = datetime.now(timezone.utc).isoformat()
                b.screens_discovered = screens_discovered or []
                b.transitions_discovered = transitions_discovered
                b.steps = steps
                b.error = error
                return

    # ------------------------------------------------------------------
    # Mutation API — feed exploration results
    # ------------------------------------------------------------------

    def begin_exploration(self, goal: str, step_count: int = 0) -> None:
        """Mark the start of a new exploration run."""
        if goal and goal not in self._goals:
            self._goals.append(goal)
        self._exploration_count += 1
        self._total_steps += step_count

    def add_screen(self, screen: Screen) -> str:
        """Add or merge a screen.  Returns the canonical hash."""
        canonical = self._find_canonical(screen)
        if canonical:
            self._merge_screen(canonical, screen)
            if screen.hash != canonical:
                self._aliases[screen.hash] = canonical
            return canonical

        self._screens[screen.hash] = screen
        return screen.hash

    def add_transition(self, transition: Transition) -> None:
        """Add or increment a transition."""
        src = self.resolve(transition.source_hash)
        tgt = self.resolve(transition.target_hash)
        if src not in self._screens or tgt not in self._screens:
            logger.debug(
                "Dropping transition %s→%s: endpoint not in graph",
                src[:8], tgt[:8],
            )
            return

        resolved = Transition(
            source_hash=src,
            target_hash=tgt,
            action_type=transition.action_type,
            action_target=transition.action_target,
            value=transition.value,
            count=transition.count,
            locator=transition.locator,
        )

        for i, existing in enumerate(self._transitions):
            if (
                existing.source_hash == resolved.source_hash
                and existing.target_hash == resolved.target_hash
                and existing.action_type == resolved.action_type
                and existing.action_target == resolved.action_target
            ):
                self._transitions[i] = Transition(
                    source_hash=existing.source_hash,
                    target_hash=existing.target_hash,
                    action_type=existing.action_type,
                    action_target=existing.action_target,
                    value=resolved.value or existing.value,
                    count=existing.count + resolved.count,
                    locator=resolved.locator or existing.locator,
                )
                return

        self._transitions.append(resolved)

    # ------------------------------------------------------------------
    # Self-healing — locator repairs
    # ------------------------------------------------------------------

    def record_repair(self, repair: LocatorRepair) -> None:
        """Record a locator repair from a self-healing event.

        Also updates the element's locators on the screen: lowers the old
        locator's confidence and adds the new one.
        """
        self._locator_repairs.append(repair)

        # Update the screen's element locators
        screen = self.screen_by_name(repair.screen_name)
        if not screen:
            return
        element = screen.find_element(repair.element_name)
        if not element:
            return

        # Lower confidence of old locator
        for loc in element.locators:
            if loc.strategy == repair.old_locator.strategy and loc.value == repair.old_locator.value:
                loc.confidence = 0.0

        # Add new locator if not already present
        already = any(
            loc.strategy == repair.new_locator.strategy and loc.value == repair.new_locator.value
            for loc in element.locators
        )
        if not already:
            element.locators.append(repair.new_locator)

    # ------------------------------------------------------------------
    # Change Intelligence — diff two graph states
    # ------------------------------------------------------------------

    def diff(self, new_graph: AppGraph) -> GraphDiff:
        """Compare this graph (old/baseline) against a new exploration.

        Returns a GraphDiff describing what screens were added, removed,
        or modified.  Used by Change Intelligence to understand the impact
        of a product update.
        """
        old_by_key = {_screen_key(s): s for s in self._screens.values()}
        new_by_key = {_screen_key(s): s for s in new_graph._screens.values()}

        changes: list[ScreenChange] = []

        # Screens in new but not old → ADDED
        for key, screen in new_by_key.items():
            if key not in old_by_key:
                changes.append(ScreenChange(
                    screen_name=screen.name,
                    change_type=ChangeType.ADDED,
                    activity=screen.activity,
                    added_elements=screen.element_names(),
                    added_navigation=[
                        f"{t.action_type} → {new_graph._screens.get(t.target_hash, Screen(hash='', name='?', activity='')).name}"
                        for t in new_graph.outgoing(screen.hash)
                    ],
                ))

        # Screens in old but not new → REMOVED
        for key, screen in old_by_key.items():
            if key not in new_by_key:
                changes.append(ScreenChange(
                    screen_name=screen.name,
                    change_type=ChangeType.REMOVED,
                    activity=screen.activity,
                    removed_elements=screen.element_names(),
                    removed_navigation=[
                        f"{t.action_type} → {self._screens.get(t.target_hash, Screen(hash='', name='?', activity='')).name}"
                        for t in self.outgoing(screen.hash)
                    ],
                ))

        # Screens in both → check for MODIFIED
        for key in old_by_key:
            if key not in new_by_key:
                continue
            old_screen = old_by_key[key]
            new_screen = new_by_key[key]
            change = _compare_screens(old_screen, new_screen, self, new_graph)
            if change:
                changes.append(change)

        return GraphDiff(
            package=self.package or new_graph.package,
            compared_at=datetime.now(timezone.utc).isoformat(),
            old_screen_count=len(self._screens),
            new_screen_count=len(new_graph._screens),
            changes=changes,
        )

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _find_canonical(self, screen: Screen) -> Optional[str]:
        """Find an existing screen that is the same logical screen."""
        for h, existing in self._screens.items():
            if existing.activity != screen.activity:
                continue
            if (
                screen.structure_hash
                and existing.structure_hash
                and screen.structure_hash == existing.structure_hash
            ):
                return h
            if screen.name and existing.name and screen.name == existing.name:
                return h
        return None

    def _merge_screen(self, canonical: str, incoming: Screen) -> None:
        """Merge new observation into existing screen."""
        existing = self._screens[canonical]
        merged = Screen(
            hash=canonical,
            name=existing.name or incoming.name,
            activity=existing.activity,
            category=incoming.category if existing.category == "other" else existing.category,
            purpose=existing.purpose or incoming.purpose,
            elements=_merge_elements(existing.elements, incoming.elements),
            actions=_merge_str_lists(existing.actions, incoming.actions),
            visits=existing.visits + incoming.visits,
            structure_hash=existing.structure_hash or incoming.structure_hash,
            screenshot_path=incoming.screenshot_path or existing.screenshot_path,
            xml_path=incoming.xml_path or existing.xml_path,
            description=existing.description or incoming.description,
        )
        self._screens[canonical] = merged

    # ------------------------------------------------------------------
    # Persistence — plain JSON
    # ------------------------------------------------------------------

    def save(self, path: Path) -> Path:
        """Write the graph to a JSON file.  Returns the path written."""
        data = {
            "package": self.package,
            "label": self.label,
            "goals": self._goals,
            "exploration_count": self._exploration_count,
            "total_steps": self._total_steps,
            "screens": [s.model_dump() for s in self._screens.values()],
            "transitions": [t.model_dump() for t in self._transitions],
            "aliases": self._aliases,
            "blocks": [b.model_dump() for b in self._blocks],
            "locator_repairs": [r.model_dump() for r in self._locator_repairs],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info("Saved app graph: %d screens, %d transitions → %s",
                     len(self._screens), len(self._transitions), path)
        return path

    @classmethod
    def load(cls, path: Path) -> AppGraph:
        """Load a previously saved graph.  Returns a fresh graph if the
        file does not exist.
        """
        graph = cls()
        if not path.exists():
            return graph

        data = json.loads(path.read_text(encoding="utf-8"))
        graph.package = data.get("package", "")
        graph.label = data.get("label", "")
        graph._goals = data.get("goals", [])
        graph._exploration_count = data.get("exploration_count", 0)
        graph._total_steps = data.get("total_steps", 0)
        graph._aliases = data.get("aliases", {})

        for raw in data.get("screens", []):
            screen = Screen.model_validate(raw)
            graph._screens[screen.hash] = screen

        for raw in data.get("transitions", []):
            graph._transitions.append(Transition.model_validate(raw))

        for raw in data.get("blocks", []):
            graph._blocks.append(Block.model_validate(raw))

        for raw in data.get("locator_repairs", []):
            graph._locator_repairs.append(LocatorRepair.model_validate(raw))

        logger.info("Loaded app graph: %d screens, %d transitions ← %s",
                     len(graph._screens), len(graph._transitions), path)
        return graph

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        activities = {s.activity for s in self._screens.values()}
        total_elements = sum(len(s.elements) for s in self._screens.values())
        total_locators = sum(
            len(loc)
            for s in self._screens.values()
            for el in s.elements
            for loc in [el.locators]
        )
        return {
            "screens": len(self._screens),
            "transitions": len(self._transitions),
            "visits": sum(s.visits for s in self._screens.values()),
            "activities": sorted(activities),
            "explorations": self._exploration_count,
            "elements": total_elements,
            "locators": total_locators,
            "blocks": len(self._blocks),
            "repairs": len(self._locator_repairs),
        }

    def __repr__(self) -> str:
        return (
            f"AppGraph(package={self.package!r}, "
            f"screens={len(self._screens)}, "
            f"transitions={len(self._transitions)})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _screen_key(screen: Screen) -> str:
    """Stable identity key for comparing screens across graph versions."""
    return f"{screen.activity}|{screen.name}"


def _merge_str_lists(a: list[str], b: list[str]) -> list[str]:
    """Merge two string lists preserving order and deduplicating."""
    seen: set[str] = set()
    result: list[str] = []
    for item in [*a, *b]:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _merge_elements(a: list[Element], b: list[Element]) -> list[Element]:
    """Merge element lists, combining locators for elements with the same name."""
    by_name: dict[str, Element] = {}
    for el in a:
        by_name[el.name.lower()] = el
    for el in b:
        key = el.name.lower()
        if key in by_name:
            existing = by_name[key]
            # Merge locators from both
            seen_locs = {(l.strategy, l.value) for l in existing.locators}
            merged_locators = list(existing.locators)
            for loc in el.locators:
                if (loc.strategy, loc.value) not in seen_locs:
                    merged_locators.append(loc)
            by_name[key] = Element(
                name=existing.name,
                locators=merged_locators,
                interactive=existing.interactive or el.interactive,
                element_type=existing.element_type or el.element_type,
                bounds=el.bounds or existing.bounds,
            )
        else:
            by_name[key] = el
    return list(by_name.values())


def _compare_screens(
    old: Screen,
    new: Screen,
    old_graph: AppGraph,
    new_graph: AppGraph,
) -> Optional[ScreenChange]:
    """Compare two versions of the same screen.  Returns None if unchanged."""
    old_el_names = set(old.element_names())
    new_el_names = set(new.element_names())
    added_el = sorted(new_el_names - old_el_names)
    removed_el = sorted(old_el_names - new_el_names)

    old_nav = {f"{t.action_type} → {old_graph._screens.get(t.target_hash, Screen(hash='', name='?', activity='')).name}"
               for t in old_graph.outgoing(old.hash)}
    new_nav = {f"{t.action_type} → {new_graph._screens.get(t.target_hash, Screen(hash='', name='?', activity='')).name}"
               for t in new_graph.outgoing(new.hash)}
    added_nav = sorted(new_nav - old_nav)
    removed_nav = sorted(old_nav - new_nav)

    purpose_changed = bool(old.purpose and new.purpose and old.purpose != new.purpose)

    if not added_el and not removed_el and not added_nav and not removed_nav and not purpose_changed:
        return None

    return ScreenChange(
        screen_name=new.name,
        change_type=ChangeType.MODIFIED,
        activity=new.activity,
        added_elements=added_el,
        removed_elements=removed_el,
        added_navigation=added_nav,
        removed_navigation=removed_nav,
        purpose_changed=purpose_changed,
        old_purpose=old.purpose if purpose_changed else "",
        new_purpose=new.purpose if purpose_changed else "",
    )
