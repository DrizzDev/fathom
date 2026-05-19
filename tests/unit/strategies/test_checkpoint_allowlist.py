"""
Verify CHECKPOINT_ALLOWED_*_MODULES tracks every Pydantic / Enum class
transitively reachable through :class:`IntentGraphState`.

Why this exists:
    LangGraph's checkpoint serializer (``JsonPlusSerializer``) refuses
    to deserialize unregistered Pydantic types, falling back to a
    ``dict`` and emitting a warning. In the intent graph the missing
    class then arrives at downstream nodes as a plain ``dict``, the
    ``isinstance(...)`` guards fail, and the run unwinds through the
    silent SUPERVISE → EXECUTE → OBSERVE → RECORD cascade — the bug
    that surfaced on staging as "Record node received no valid step
    result".

What this enforces:
    1. Every Pydantic ``BaseModel`` and ``enum.Enum`` reachable through
       :class:`IntentGraphState` annotations (transitively, including
       ``Optional``, ``List``, ``Dict``, ``Union`` arms, and nested
       Pydantic field types) must appear in
       :data:`CHECKPOINT_ALLOWED_JSON_MODULES`.
    2. The msgpack variant must equal the json variant.
    3. Every listed entry must resolve to a real importable class —
       catches typos / renames / stale entries.

Adding a new typed field to :class:`IntentGraphState` (or a new field
to any reachable schema) breaks (1) and forces a maintainer to extend
the allow-list explicitly. That is the entire point.
"""

from __future__ import annotations

import enum
import importlib
import typing
import unittest
from typing import Set, Tuple, get_args, get_origin

from pydantic import BaseModel

from fathom.strategies.graph.state import IntentGraphState
from fathom.strategies.intent import (
    CHECKPOINT_ALLOWED_JSON_MODULES,
    CHECKPOINT_ALLOWED_MSGPACK_MODULES,
)


def _walk_transitive_classes(root: type) -> Set[Tuple[str, str]]:
    """
    Return the transitive closure of Pydantic / Enum classes reachable
    from ``root`` via type annotations.
    """

    discovered: Set[Tuple[str, str]] = set()
    visited: Set[type] = set()

    def visit(annotation: object) -> None:
        if annotation is None or annotation is type(None):
            return

        origin = get_origin(annotation)
        if origin is not None:
            for argument in get_args(annotation):
                visit(argument)
            return

        if not isinstance(annotation, type):
            return

        if annotation in visited:
            return
        visited.add(annotation)

        if issubclass(annotation, BaseModel):
            discovered.add((annotation.__module__, annotation.__name__))
            for field_info in annotation.model_fields.values():
                visit(field_info.annotation)
            return

        if issubclass(annotation, enum.Enum):
            discovered.add((annotation.__module__, annotation.__name__))

    hints = typing.get_type_hints(root)
    for annotation in hints.values():
        visit(annotation)

    return discovered


class CheckpointAllowlistTest(unittest.TestCase):
    """
    Guard the LangGraph deserialization allow-list against schema drift.
    """

    def test_allowlist_covers_intent_graph_state_closure(self) -> None:
        """
        Every Pydantic / Enum class reachable from IntentGraphState
        must appear in CHECKPOINT_ALLOWED_JSON_MODULES.
        """

        required = _walk_transitive_classes(IntentGraphState)
        listed = set(CHECKPOINT_ALLOWED_JSON_MODULES)

        missing = required - listed
        self.assertFalse(
            missing,
            msg=(
                "CHECKPOINT_ALLOWED_JSON_MODULES is missing classes reachable "
                "through IntentGraphState. Add the entries below to "
                "fathom/strategies/intent.py — checkpoint deserialization will "
                "otherwise silently drop these to dict and break downstream "
                "isinstance() guards.\n\nMissing entries:\n"
                + "\n".join(f"  ({module!r}, {name!r})," for module, name in sorted(missing))
            ),
        )

    def test_msgpack_allowlist_matches_json_allowlist(self) -> None:
        """
        The msgpack and json allow-lists must stay in sync. The
        serializer applies one or the other depending on the encoded
        payload format, so a divergence corrupts replay for the
        diverging types.
        """

        self.assertEqual(
            CHECKPOINT_ALLOWED_MSGPACK_MODULES,
            CHECKPOINT_ALLOWED_JSON_MODULES,
            msg=(
                "CHECKPOINT_ALLOWED_MSGPACK_MODULES drifted from "
                "CHECKPOINT_ALLOWED_JSON_MODULES. They must stay aliased."
            ),
        )

    def test_every_allowlist_entry_resolves(self) -> None:
        """
        Every (module, name) entry must import to a real class —
        guards against typos and stale entries after refactors.
        """

        unresolved: list[Tuple[str, str, str]] = []
        for module_name, class_name in CHECKPOINT_ALLOWED_JSON_MODULES:
            try:
                module = importlib.import_module(module_name)
            except ImportError as exception:
                unresolved.append((module_name, class_name, f"import failed: {exception}"))
                continue

            if not hasattr(module, class_name):
                unresolved.append((module_name, class_name, "attribute missing"))
                continue

            target = getattr(module, class_name)
            if not isinstance(target, type):
                unresolved.append((module_name, class_name, f"not a class: {type(target)!r}"))

        self.assertFalse(
            unresolved,
            msg=(
                "CHECKPOINT_ALLOWED_JSON_MODULES contains stale or typo'd "
                "entries — each was added in the past but no longer "
                "resolves to a real class. Remove or correct:\n"
                + "\n".join(
                    f"  ({module!r}, {name!r}): {reason}" for module, name, reason in unresolved
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
