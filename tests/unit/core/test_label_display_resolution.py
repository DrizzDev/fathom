"""Tests for manifest-backed display-name resolution in ReferenceResolutionService.

When an Action carries a ``label_id`` but no human-readable target name
(either empty, generic ``"element"``, or the namespaced ``"label:{id}"``
placeholder that ``parsing.py`` stamps for label-only emissions), the
resolution service must look up the element in the manifest and copy a
display name from its XML attributes onto the Action at snap time.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from fathom.constants import ActionType
from fathom.core.services.resolution import ReferenceResolutionService
from fathom.schemas.actions import Action


class _NullMemory:
    """Minimal MemoryPort stub for the resolution service."""

    async def get(self, key: str) -> Optional[str]:
        return None

    async def set(self, key: str, value: str) -> None:
        return None

    async def delete(self, key: str) -> None:
        return None

    async def keys(self) -> list[str]:
        return []


@pytest.fixture
def service() -> ReferenceResolutionService:
    return ReferenceResolutionService(ledger=_NullMemory())  # type: ignore[arg-type]


def _element(**attrs: Any) -> Dict[str, Any]:
    """Build a label_map entry with the given attributes + a stub bounds."""

    return {"bounds": "[100,200][300,280]", **attrs}


class TestLabelDisplayResolution:
    @pytest.mark.asyncio
    async def test_label_placeholder_replaced_with_manifest_text(
        self, service: ReferenceResolutionService
    ) -> None:
        action = Action(
            action_type=ActionType.TAP,
            rationale="r",
            target="label:42",
            natural_language_target="label:42",
            label_id="42",
        )
        resolved = await service.resolve(action=action, elements={"42": _element(text="Submit")})

        assert resolved.target == "Submit"
        assert resolved.natural_language_target == "Submit"
        assert resolved.export_target == "Submit"

    @pytest.mark.asyncio
    async def test_content_desc_used_when_text_missing(
        self, service: ReferenceResolutionService
    ) -> None:
        action = Action(
            action_type=ActionType.TAP,
            rationale="r",
            target="label:7",
            natural_language_target="label:7",
            label_id="7",
        )
        resolved = await service.resolve(
            action=action, elements={"7": _element(**{"content-desc": "Close dialog"})}
        )
        assert resolved.target == "Close dialog"

    @pytest.mark.asyncio
    async def test_name_used_when_text_and_content_desc_missing(
        self, service: ReferenceResolutionService
    ) -> None:
        action = Action(
            action_type=ActionType.TAP,
            rationale="r",
            target="label:9",
            natural_language_target="label:9",
            label_id="9",
        )
        resolved = await service.resolve(
            action=action, elements={"9": _element(name="Primary button")}
        )
        assert resolved.target == "Primary button"

    @pytest.mark.asyncio
    async def test_resource_id_used_as_last_resort(
        self, service: ReferenceResolutionService
    ) -> None:
        action = Action(
            action_type=ActionType.TAP,
            rationale="r",
            target="label:3",
            natural_language_target="label:3",
            label_id="3",
        )
        resolved = await service.resolve(
            action=action, elements={"3": _element(**{"resource-id": "com.app:id/go"})}
        )
        assert resolved.target == "com.app:id/go"

    @pytest.mark.asyncio
    async def test_generic_element_target_replaced(
        self, service: ReferenceResolutionService
    ) -> None:
        action = Action(
            action_type=ActionType.TAP,
            rationale="r",
            target="element",
            natural_language_target="element",
            label_id="11",
        )
        resolved = await service.resolve(action=action, elements={"11": _element(text="Pay now")})
        assert resolved.target == "Pay now"
        assert resolved.natural_language_target == "Pay now"

    @pytest.mark.asyncio
    async def test_explicit_target_not_overwritten(
        self, service: ReferenceResolutionService
    ) -> None:
        """When the LLM provided a concrete target_name, it wins over the manifest."""

        action = Action(
            action_type=ActionType.TAP,
            rationale="r",
            target="Manually named target",
            natural_language_target="Manually named target",
            label_id="9",
        )
        resolved = await service.resolve(
            action=action, elements={"9": _element(text="Different from LLM")}
        )
        assert resolved.target == "Manually named target"
        assert resolved.natural_language_target == "Manually named target"

    @pytest.mark.asyncio
    async def test_existing_export_target_not_overwritten(
        self, service: ReferenceResolutionService
    ) -> None:
        """When export_target is already populated, the manifest name does not
        clobber it — useful for positional targets where script_target wins."""

        action = Action(
            action_type=ActionType.TAP,
            rationale="r",
            target="label:5",
            natural_language_target="label:5",
            label_id="5",
            export_target="the first search result",
        )
        resolved = await service.resolve(
            action=action, elements={"5": _element(text="R for Rabbit Pant Diaper")}
        )
        assert resolved.export_target == "the first search result"

    @pytest.mark.asyncio
    async def test_no_display_attributes_keeps_placeholder(
        self, service: ReferenceResolutionService
    ) -> None:
        """Elements with no display text fall through — bounds still snap."""

        action = Action(
            action_type=ActionType.TAP,
            rationale="r",
            target="label:99",
            natural_language_target="label:99",
            label_id="99",
        )
        resolved = await service.resolve(action=action, elements={"99": {"bounds": "[0,0][10,10]"}})
        # Placeholder stays; no display name available.
        assert resolved.target == "label:99"
        assert resolved.bounds is not None
        assert resolved.bounds.x == 0
