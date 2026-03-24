"""
Shared action execution helper.

Converts an ``Action`` with normalized coordinates (0-1000 scale) into
concrete device method calls with pixel coordinates.  Used by both the
intent graph (``nodes.py``) and the BFS exploration strategy so that
coordinate resolution is consistent everywhere.
"""

from __future__ import annotations

import asyncio
from logging import getLogger
from typing import Tuple

from fathom.constants import ActionType
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.results import ActionResult
from fathom.tools.device import DeviceTool
from fathom.utils.coordinates import CoordinateConverter

logger = getLogger(__name__)


async def execute_device_action(device: DeviceTool, action: Action) -> ActionResult:
    """
    Execute an ``Action`` on a device with proper coordinate conversion.

    Converts normalized (0-1000) bounds to absolute pixel coordinates
    using the device's screen size, then dispatches to the appropriate
    device method (``tap``, ``type_text``, ``swipe``, ``back``, etc.).

    Falls back to ``device.execute(action.model_dump())`` for action
    types that are not explicitly handled.
    """

    size = await device.get_screen_size()
    converter = CoordinateConverter(screen_width=size[0], screen_height=size[1])

    if action.action_type == ActionType.TAP:
        coords = (
            converter.center_to_pixels(bounds=action.bounds)
            if action.bounds
            else (size[0] // 2, size[1] // 2)
        )
        return await device.tap(x=coords[0], y=coords[1])

    if action.action_type == ActionType.TYPE:
        if not action.bounds:
            return ActionResult(
                success=False,
                duration=0,
                error="Type action requires bounds for focus tap guard",
            )
        x, y = converter.center_to_pixels(bounds=action.bounds)
        focus = await device.tap(x=x, y=y)
        if not focus.success:
            return ActionResult(
                success=False,
                duration=0,
                error=f"Focus tap failed: {focus.error or 'unknown'}",
            )
        return await device.type_text(text=action.text or "", replace=True)

    if action.action_type == ActionType.LONG_PRESS:
        coords = (
            converter.center_to_pixels(bounds=action.bounds)
            if action.bounds
            else (size[0] // 2, size[1] // 2)
        )
        return await device.long_press(x=coords[0], y=coords[1])

    if action.action_type in (
        ActionType.SWIPE_LEFT,
        ActionType.SWIPE_RIGHT,
        ActionType.SWIPE_UP,
        ActionType.SWIPE_DOWN,
    ):
        direction = action.action_type.value.split("_")[1]
        swipe_coords = converter.swipe_coordinates(
            bounds=action.bounds or Bounds(x=200, y=200, width=600, height=600),
            direction=direction,
        )
        return await device.swipe(
            x1=swipe_coords[0],
            y1=swipe_coords[1],
            x2=swipe_coords[2],
            y2=swipe_coords[3],
        )

    if action.action_type in (ActionType.SCROLL, ActionType.SWIPE):
        # Default scroll = swipe up in center area
        swipe_coords = converter.swipe_coordinates(
            bounds=action.bounds or Bounds(x=200, y=200, width=600, height=600),
            direction="up",
        )
        return await device.swipe(
            x1=swipe_coords[0],
            y1=swipe_coords[1],
            x2=swipe_coords[2],
            y2=swipe_coords[3],
        )

    if action.action_type == ActionType.WAIT:
        duration = action.wait_duration or 1000
        await asyncio.sleep(delay=duration / 1000.0)
        return ActionResult(success=True, duration=duration)

    if action.action_type == ActionType.BACK:
        return await device.back()

    if action.action_type == ActionType.HOME:
        return await device.home()

    if action.action_type in (
        ActionType.VALIDATE,
        ActionType.COMPLETE,
        ActionType.SAVE_MEMORY,
        ActionType.RETRIEVE_MEMORY,
    ):
        return ActionResult(success=True, duration=0)

    logger.warning("Unrecognized action type: %s", action.action_type)
    return ActionResult(
        success=False, duration=0, error=f"Unrecognized action type: {action.action_type}"
    )


async def get_action_coordinates(device: DeviceTool, action: Action) -> Tuple[int, ...]:
    """
    Compute the pixel coordinates for an action.

    Returns a tuple of (x, y) for taps or (x1, y1, x2, y2) for swipes.
    Returns an empty tuple for actions that don't have coordinates.
    """

    size = await device.get_screen_size()
    converter = CoordinateConverter(screen_width=size[0], screen_height=size[1])

    if action.action_type in (ActionType.TAP, ActionType.TYPE, ActionType.LONG_PRESS):
        if action.bounds:
            return converter.center_to_pixels(bounds=action.bounds)
        return (size[0] // 2, size[1] // 2)

    if action.action_type in (
        ActionType.SWIPE,
        ActionType.SCROLL,
        ActionType.SWIPE_UP,
        ActionType.SWIPE_DOWN,
        ActionType.SWIPE_LEFT,
        ActionType.SWIPE_RIGHT,
    ):
        if action.bounds:
            direction = "up"
            if "_" in action.action_type.value:
                direction = action.action_type.value.split("_")[1]
            return converter.swipe_coordinates(bounds=action.bounds, direction=direction)
        return (size[0] // 2, size[1] * 3 // 4, size[0] // 2, size[1] // 4)

    return ()


async def ensure_target_package(
    device: DeviceTool,
    target_package: str,
    *,
    max_back_attempts: int = 3,
) -> bool:
    """
    Verify the device is still inside the target application.

    If the foreground package doesn't match ``target_package``, attempt to
    return using BACK presses. If that fails, re-launch the app directly.

    Returns ``True`` if the device is (or was recovered to) the target
    package, ``False`` if recovery failed entirely.
    """

    current = await device.get_current_package()
    if current == target_package:
        return True

    logger.warning(
        "Package drift detected: expected %s, got %s — recovering",
        target_package,
        current,
    )

    # Try BACK presses first (cheaper, preserves navigation stack)
    for attempt in range(1, max_back_attempts + 1):
        await device.back()
        await asyncio.sleep(0.5)
        current = await device.get_current_package()
        if current == target_package:
            logger.info(
                "Recovered to target package via BACK (attempt %d/%d)",
                attempt,
                max_back_attempts,
            )
            return True

    # BACK didn't work — force-launch the app
    logger.warning(
        "BACK recovery failed after %d attempts, re-launching %s",
        max_back_attempts,
        target_package,
    )
    launch_result = await device.launch_app(package_name=target_package)
    if not launch_result.success:
        logger.error("Failed to re-launch %s: %s", target_package, launch_result.error)
        return False

    await asyncio.sleep(2.0)  # Give app time to fully launch
    current = await device.get_current_package()
    recovered = current == target_package

    if recovered:
        logger.info("Recovered to target package via re-launch")
    else:
        logger.error("Re-launch did not restore target package (got %s)", current)

    return recovered
