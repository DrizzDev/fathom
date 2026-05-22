from __future__ import annotations

import unittest

from fathom.constants.command import CommandScopeKind
from fathom.constants.scroll import ScrollEvidenceSource
from fathom.core.execution.scroll import ScrollScopeResolver
from fathom.schemas.actions import Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.command import CommandAnchor, CommandScope
from fathom.schemas.observation import (
    KeyboardObservation,
    ScreenObservation,
    ScrollRegion,
)
from fathom.schemas.screens import ScreenHashBundle
from fathom.utils.coordinates import CoordinateConverter


class ScrollScopeResolverTest(unittest.IsolatedAsyncioTestCase):
    """
    Covers container-first resolution for scroll commands.
    """

    async def test_prefers_large_vertical_page_scope_for_main_scroll_target(self) -> None:
        """
        Main-page scroll intent should resolve to the page scope, not a nested carousel.
        """

        observation = ScreenObservation(
            activity="app",
            hashes=ScreenHashBundle(visual_hash="a", xml_hash="b", interaction_hash="c"),
            elements=(),
            overlays=(),
            keyboard=KeyboardObservation(visible=False, bounds=None, dismiss=()),
            scroll=(
                ScrollRegion(
                    identifier="carousel",
                    label_id="3",
                    bounds=Bounds(
                        x=0,
                        y=700,
                        width=1080,
                        height=180,
                        coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                        source=CoordinateSource.XML,
                    ),
                    direction="horizontal",
                    axis="horizontal",
                    kind=CommandScopeKind.CAROUSEL,
                    confidence=0.95,
                    source=ScrollEvidenceSource.SURFACE,
                ),
                ScrollRegion(
                    identifier="page",
                    label_id=None,
                    bounds=Bounds(
                        x=0,
                        y=393,
                        width=1080,
                        height=1861,
                        coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                        source=CoordinateSource.VIEWPORT,
                    ),
                    direction="vertical",
                    axis="vertical",
                    kind=CommandScopeKind.VIEWPORT,
                    confidence=0.72,
                    source=ScrollEvidenceSource.SURFACE,
                ),
            ),
            calls_to_action=(),
            focused=None,
        )
        converter = CoordinateConverter(logical_width=1080, logical_height=2340)
        fallback = CommandScope(
            identifier="fallback",
            kind=CommandScopeKind.VIEWPORT,
            bounds=Bounds(
                x=0,
                y=393,
                width=1080,
                height=1861,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                source=CoordinateSource.VIEWPORT,
            ),
            region=converter.region_from_bounds(
                bounds=Bounds(
                    x=0,
                    y=393,
                    width=1080,
                    height=1861,
                    coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                    source=CoordinateSource.VIEWPORT,
                ),
                source=CoordinateSource.VIEWPORT,
            ),
            axis="vertical",
            confidence=0.2,
        )

        scope = await ScrollScopeResolver().resolve(
            anchor=CommandAnchor(target="main scrollable area"),
            observation=observation,
            fallback=fallback,
            converter=converter,
        )

        self.assertEqual(scope.identifier, "page")
        self.assertEqual(scope.kind, CommandScopeKind.VIEWPORT)

    async def test_exact_observation_region_anchor_resolves_matching_scope(self) -> None:
        """
        Observation-only region hints must resolve by observation_region_id, not masquerade as manifest labels.
        """

        observation = ScreenObservation(
            activity="app",
            hashes=ScreenHashBundle(visual_hash="a", xml_hash="b", interaction_hash="c"),
            elements=(),
            overlays=(),
            keyboard=KeyboardObservation(visible=False, bounds=None, dismiss=()),
            scroll=(
                ScrollRegion(
                    identifier="page_scroll_region",
                    observation_region_id="page_scroll_region",
                    bounds=Bounds(
                        x=0,
                        y=393,
                        width=1080,
                        height=1861,
                        coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                        source=CoordinateSource.VIEWPORT,
                    ),
                    direction="vertical",
                    axis="vertical",
                    kind=CommandScopeKind.VIEWPORT,
                    confidence=0.72,
                    source=ScrollEvidenceSource.SURFACE,
                ),
            ),
            calls_to_action=(),
            focused=None,
        )
        converter = CoordinateConverter(logical_width=1080, logical_height=2340)
        fallback = CommandScope(
            identifier="fallback",
            kind=CommandScopeKind.VIEWPORT,
            bounds=Bounds(
                x=0,
                y=393,
                width=1080,
                height=1861,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                source=CoordinateSource.VIEWPORT,
            ),
            region=converter.region_from_bounds(
                bounds=Bounds(
                    x=0,
                    y=393,
                    width=1080,
                    height=1861,
                    coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                    source=CoordinateSource.VIEWPORT,
                ),
                source=CoordinateSource.VIEWPORT,
            ),
            axis="vertical",
            confidence=0.2,
        )

        scope = await ScrollScopeResolver().resolve(
            anchor=CommandAnchor(
                target="feed",
                observation_region_id="page_scroll_region",
            ),
            observation=observation,
            fallback=fallback,
            converter=converter,
        )

        self.assertEqual(scope.observation_region_id, "page_scroll_region")
        self.assertIsNone(scope.manifest_label_id)
