from __future__ import annotations

from logging import getLogger

from fathom.interfaces.device import DevicePort
from fathom.interfaces.swipe import SwipeAttemptDispatcher
from fathom.interfaces.vision import VisualHasher
from fathom.schemas.actions import GesturePath
from fathom.schemas.swipe import DeviceOutcome, SwipeAttempt, VisualOutcome

logger = getLogger(__name__)


class DeviceSwipeDispatcher(SwipeAttemptDispatcher):
    """
    Concrete dispatcher: dispatches one swipe via DevicePort, captures one screenshot, hashes via VisualHasher.
    """

    def __init__(
        self,
        *,
        device: DevicePort,
        hasher: VisualHasher,
    ) -> None:
        """
        Bind to the device and the visual hasher used for cheap pre/post comparison.
        """

        self.__device = device
        self.__hasher = hasher

    async def attempt(
        self,
        *,
        path: GesturePath,
        index: int,
        original_before: str,
    ) -> SwipeAttempt:
        """
        Dispatch one swipe attempt and return the typed outcome relative to the original pre-action hash.
        """

        device_outcome = await self.__dispatch(path=path)
        if not device_outcome.succeeded:
            return SwipeAttempt(
                index=index,
                path=path,
                device=device_outcome,
                visual=VisualOutcome(changed=False, before=original_before, after=None),
            )

        visual_outcome = await self.__capture_visual(original_before=original_before)
        return SwipeAttempt(
            index=index,
            path=path,
            device=device_outcome,
            visual=visual_outcome,
        )

    async def __dispatch(self, *, path: GesturePath) -> DeviceOutcome:
        """
        Issue the device swipe primitive and translate exceptions into a typed DeviceOutcome.
        """

        try:
            result = await self.__device.swipe(
                x1=path.start_x,
                y1=path.start_y,
                x2=path.end_x,
                y2=path.end_y,
                duration=path.duration,
            )
            return DeviceOutcome(succeeded=bool(result.success), error=result.error)
        except Exception as exception:
            logger.warning(f"DeviceSwipeDispatcher: swipe failed: {exception}")
            return DeviceOutcome(succeeded=False, error=str(exception))

    async def __capture_visual(self, *, original_before: str) -> VisualOutcome:
        """
        Capture the post-attempt screenshot and compare its hash against the original pre-action hash.
        """

        try:
            image = await self.__device.capture_screen()
            after_hash = self.__hasher.hash(image=image)
        except Exception as exception:
            logger.warning(f"DeviceSwipeDispatcher: capture failed: {exception}")
            return VisualOutcome(changed=False, before=original_before, after=None)

        return VisualOutcome(
            changed=after_hash != original_before,
            before=original_before,
            after=after_hash,
        )
