"""Tests for fathom.utils.coordinates.CoordinateConverter."""

from __future__ import annotations

import logging

import pytest

from fathom.schemas.actions import Bounds
from fathom.schemas.configuration import (
    DeviceRuntimeConfiguration,
    InteractionPolicyConfiguration,
    InteractionRuntimeConfiguration,
    ScrollInteractionPolicy,
    SwipeInteractionPolicy,
)
from fathom.utils.coordinates import CoordinateConverter

SCREEN_W = 1080
SCREEN_H = 1920


def _make_converter(
    *,
    width: int = SCREEN_W,
    height: int = SCREEN_H,
    swipe: SwipeInteractionPolicy | None = None,
    scroll: ScrollInteractionPolicy | None = None,
) -> CoordinateConverter:
    config = DeviceRuntimeConfiguration(
        interaction=InteractionRuntimeConfiguration(
            policy=InteractionPolicyConfiguration(
                swipe=swipe or SwipeInteractionPolicy(),
                scroll=scroll or ScrollInteractionPolicy(),
            )
        )
    )
    return CoordinateConverter(screen_width=width, screen_height=height, configuration=config)


def _pixel_bounds(x: int, y: int, w: int, h: int) -> Bounds:
    return Bounds(x=x, y=y, width=w, height=h, coord_system="pixel")


class TestSwipeCoordinatesMiddle:
    """Bounds well inside the screen → endpoints stay in-screen, axis-correct."""

    @pytest.mark.parametrize("direction", ["up", "down", "left", "right"])
    def test_endpoints_within_screen(self, direction: str) -> None:
        converter = _make_converter()
        bounds = _pixel_bounds(x=300, y=600, w=480, h=720)

        x1, y1, x2, y2 = converter.swipe_coordinates(bounds=bounds, direction=direction)

        assert 0 <= x1 < SCREEN_W
        assert 0 <= x2 < SCREEN_W
        assert 0 <= y1 < SCREEN_H
        assert 0 <= y2 < SCREEN_H

    def test_up_moves_finger_upward(self) -> None:
        converter = _make_converter()
        bounds = _pixel_bounds(x=300, y=600, w=480, h=720)

        x1, y1, x2, y2 = converter.swipe_coordinates(bounds=bounds, direction="up")

        assert x1 == x2  # vertical swipe — x is fixed
        assert y1 > y2  # start lower, end higher (finger moves up)

    def test_down_moves_finger_downward(self) -> None:
        converter = _make_converter()
        bounds = _pixel_bounds(x=300, y=600, w=480, h=720)

        x1, y1, x2, y2 = converter.swipe_coordinates(bounds=bounds, direction="down")

        assert x1 == x2
        assert y1 < y2

    def test_left_moves_finger_leftward(self) -> None:
        converter = _make_converter()
        bounds = _pixel_bounds(x=300, y=600, w=480, h=720)

        x1, y1, x2, y2 = converter.swipe_coordinates(bounds=bounds, direction="left")

        assert y1 == y2
        assert x1 > x2

    def test_right_moves_finger_rightward(self) -> None:
        converter = _make_converter()
        bounds = _pixel_bounds(x=300, y=600, w=480, h=720)

        x1, y1, x2, y2 = converter.swipe_coordinates(bounds=bounds, direction="right")

        assert y1 == y2
        assert x1 < x2


class TestSwipeCoordinatesEdgeClamping:
    """Bounds whose center sits near a screen edge must produce in-screen endpoints."""

    def test_top_edge_up_swipe_does_not_go_negative(self) -> None:
        converter = _make_converter()
        bounds = _pixel_bounds(x=400, y=10, w=200, h=80)

        x1, y1, x2, y2 = converter.swipe_coordinates(bounds=bounds, direction="up")

        margin = int(SCREEN_H * 0.05)
        assert y1 >= margin
        assert y2 >= margin
        assert y1 <= SCREEN_H - 1 - margin
        assert y2 <= SCREEN_H - 1 - margin

    def test_bottom_edge_down_swipe_stays_in_screen(self) -> None:
        converter = _make_converter()
        bounds = _pixel_bounds(x=400, y=SCREEN_H - 50, w=200, h=40)

        x1, y1, x2, y2 = converter.swipe_coordinates(bounds=bounds, direction="down")

        margin = int(SCREEN_H * 0.05)
        assert y1 >= margin
        assert y2 >= margin
        assert y1 <= SCREEN_H - 1 - margin
        assert y2 <= SCREEN_H - 1 - margin

    def test_left_edge_left_swipe_does_not_go_negative(self) -> None:
        converter = _make_converter()
        bounds = _pixel_bounds(x=5, y=900, w=80, h=200)

        x1, y1, x2, y2 = converter.swipe_coordinates(bounds=bounds, direction="left")

        margin = int(SCREEN_W * 0.05)
        assert x1 >= margin
        assert x2 >= margin
        assert x1 <= SCREEN_W - 1 - margin
        assert x2 <= SCREEN_W - 1 - margin

    def test_right_edge_right_swipe_stays_in_screen(self) -> None:
        converter = _make_converter()
        bounds = _pixel_bounds(x=SCREEN_W - 60, y=900, w=40, h=200)

        x1, y1, x2, y2 = converter.swipe_coordinates(bounds=bounds, direction="right")

        margin = int(SCREEN_W * 0.05)
        assert x1 >= margin
        assert x2 >= margin
        assert x1 <= SCREEN_W - 1 - margin
        assert x2 <= SCREEN_W - 1 - margin


class TestSwipeCoordinatesMinDistanceRecovery:
    """When edge clamping shrinks the gesture below min_distance, recenter on midline."""

    def test_top_edge_recovers_to_midline(self) -> None:
        converter = _make_converter()
        bounds = _pixel_bounds(x=400, y=10, w=200, h=40)

        x1, y1, x2, y2 = converter.swipe_coordinates(bounds=bounds, direction="up")

        # Recovery threshold is min(min_distance_px=200, safe_span). Recovered
        # gesture should be at least min_distance_px in magnitude.
        assert abs(y1 - y2) >= 200

    def test_left_edge_recovers_to_midline(self) -> None:
        converter = _make_converter()
        bounds = _pixel_bounds(x=5, y=900, w=40, h=200)

        x1, y1, x2, y2 = converter.swipe_coordinates(bounds=bounds, direction="left")

        assert abs(x1 - x2) >= 200


class TestSwipeCoordinatesUnknownDirection:
    """Malformed direction strings used to return a zero-length no-op; now they
    log a warning and execute a midline scroll-up gesture."""

    def test_logs_warning_and_returns_real_gesture(self, caplog: pytest.LogCaptureFixture) -> None:
        converter = _make_converter()
        bounds = _pixel_bounds(x=300, y=600, w=480, h=720)

        with caplog.at_level(logging.WARNING, logger="fathom.utils.coordinates"):
            x1, y1, x2, y2 = converter.swipe_coordinates(bounds=bounds, direction="diagonal")

        assert any("Unknown swipe direction" in record.message for record in caplog.records)
        assert (x1, y1) != (x2, y2)
        assert x1 == x2  # midline fallback is a vertical gesture
        assert y1 > y2  # scroll-up


class TestSwipeCoordinatesNormalizedVsPixel:
    """Equivalent normalized and pixel bounds must produce the same gesture."""

    def test_equivalent_bounds_match(self) -> None:
        converter = _make_converter()
        normalized = Bounds(x=400, y=500, width=200, height=300)
        pixel = _pixel_bounds(
            x=int(400 * SCREEN_W / 1000),
            y=int(500 * SCREEN_H / 1000),
            w=int(200 * SCREEN_W / 1000),
            h=int(300 * SCREEN_H / 1000),
        )

        from_normalized = converter.swipe_coordinates(bounds=normalized, direction="up")
        from_pixel = converter.swipe_coordinates(bounds=pixel, direction="up")

        assert from_normalized == from_pixel


class TestSwipeCoordinatesConfigKnobs:
    """edge_margin_ratio and min_distance_px on the policies must take effect."""

    def test_larger_edge_margin_pushes_endpoints_inward(self) -> None:
        wide_margin = _make_converter(
            scroll=ScrollInteractionPolicy(edge_margin_ratio=0.2),
        )
        bounds = _pixel_bounds(x=400, y=10, w=200, h=40)

        _, y1, _, y2 = wide_margin.swipe_coordinates(bounds=bounds, direction="up")

        margin = int(SCREEN_H * 0.2)
        assert y1 >= margin
        assert y2 >= margin
        assert y1 <= SCREEN_H - 1 - margin
        assert y2 <= SCREEN_H - 1 - margin

    def test_larger_min_distance_forces_recovery(self) -> None:
        big_min = _make_converter(
            scroll=ScrollInteractionPolicy(min_distance_px=600),
        )
        bounds = _pixel_bounds(x=400, y=200, w=200, h=80)

        _, y1, _, y2 = big_min.swipe_coordinates(bounds=bounds, direction="up")

        # With min_distance_px=600, even an off-center bounds should produce
        # a gesture at least 600px tall (or the safe span, whichever is smaller).
        margin = int(SCREEN_H * 0.05)
        safe_span = (SCREEN_H - 1 - margin) - margin
        assert abs(y1 - y2) >= min(600, safe_span)


class TestSwipeCoordinatesAcrossResolutions:
    """Same logical bounds on different resolutions should all produce in-screen
    endpoints. Regression for resolution-specific bugs."""

    @pytest.mark.parametrize(
        "width,height",
        [(1080, 1920), (1170, 2532), (720, 1280), (1440, 3120)],
    )
    @pytest.mark.parametrize("direction", ["up", "down", "left", "right"])
    def test_endpoints_in_screen_across_resolutions(
        self, width: int, height: int, direction: str
    ) -> None:
        converter = _make_converter(width=width, height=height)
        bounds = Bounds(x=400, y=500, width=200, height=300)  # normalized

        x1, y1, x2, y2 = converter.swipe_coordinates(bounds=bounds, direction=direction)

        assert 0 <= x1 < width
        assert 0 <= x2 < width
        assert 0 <= y1 < height
        assert 0 <= y2 < height
