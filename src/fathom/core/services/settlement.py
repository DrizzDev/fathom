from __future__ import annotations

import asyncio
import time
from logging import getLogger
from typing import TYPE_CHECKING

from fathom.constants import ActionExecutionKind
from fathom.constants.execution import (
    CAPTURE_OVERHEAD_MS,
    MAX_STABILITY_WAIT_MS,
    POST_ACTION_OBSERVATION_TIMEOUT_SECONDS,
)
from fathom.constants.platform import DeviceConnectionType, DevicePlatform
from fathom.interfaces.device import DevicePort
from fathom.interfaces.settlement import ScreenComparisonPort, ScreenStatePort
from fathom.schemas.effect import ActionEffect, ActionEffectStatus
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.settlement import ScreenSettlement, ScreenSettlementEvidence
from fathom.schemas.vision import ActionKind, ActionKindResolver

if TYPE_CHECKING:
    from fathom.schemas.configuration import FathomConfiguration

logger = getLogger(__name__)


class ScreenSettlementService:
    """
    Applies post-action wait and one bounded recapture for transition actions.
    """

    def __init__(
        self,
        *,
        device: DevicePort,
        state: ScreenStatePort,
        comparison: ScreenComparisonPort,
        configuration: FathomConfiguration,
    ) -> None:
        """
        Bind device, screen-state, comparison, and settlement configuration.
        """

        self.__state = state
        self.__device = device
        self.__comparison = comparison
        self.__configuration = configuration

    async def pause(self) -> None:
        """
        Apply the configured post-action settlement wait.
        """

        await self.pause_for(configuration=self.__configuration)

    @staticmethod
    async def pause_for(*, configuration: FathomConfiguration) -> None:
        """
        Apply a capped post-action settlement wait.
        """

        requested = float(configuration.engine.stability_wait)
        capped = min(requested * 1000.0, MAX_STABILITY_WAIT_MS)
        applied = max(0.0, capped - CAPTURE_OVERHEAD_MS) / 1000.0

        logger.info(
            "Screen settlement wait applied",
            extra={
                "event": "screen.settlement.wait",
                "wait.applied.seconds": applied,
                "wait.requested.seconds": requested,
                "wait.capture_overhead.ms": CAPTURE_OVERHEAD_MS,
            },
        )
        await asyncio.sleep(delay=applied)

    async def compare(self, *, evidence: ScreenSettlementEvidence) -> ScreenSettlement:
        """
        Recapture once when a transition-capable action first appears unchanged.
        """

        initial = ActionEffect.from_screen_diff(diff=evidence.after.diff)

        if not self.__should_recapture(evidence=evidence, effect=initial):
            return ScreenSettlement(
                diff=evidence.after.diff,
                hashes=evidence.after.hashes,
                capture=evidence.after.capture,
            )

        await asyncio.sleep(delay=self.__configuration.engine.transition_grace_period)

        try:
            capture = await asyncio.wait_for(
                self.__capture(reference=evidence.after.capture),
                timeout=self.__capture_timeout(),
            )
        except Exception as exception:  # noqa: BLE001 - observation must not fail the run
            logger.warning(
                "Screen settlement recapture failed",
                extra={
                    "event": "screen.settlement.failed",
                    "workflow.id": evidence.workflow_id,
                    "error.kind": type(exception).__name__,
                    "action.type": evidence.execution.step.action.action_type.value,
                },
            )
            return ScreenSettlement(
                diff=evidence.after.diff,
                hashes=evidence.after.hashes,
                capture=evidence.after.capture,
            )

        hashes = self.__state.resolve_capture_hashes(capture=capture, elements=[])
        after_state = self.__state.build_screen_state(
            capture=capture,
            xml_hash=hashes.xml_hash,
            visual_hash=hashes.visual_hash,
            interaction_hash=hashes.interaction_hash,
        )
        diff = await asyncio.to_thread(
            self.__comparison.compare,
            after=capture,
            after_state=after_state,
            before=evidence.before.capture,
            before_state=evidence.before.state,
        )
        settled = ActionEffect.from_screen_diff(diff=diff)

        logger.info(
            "Screen settlement recaptured",
            extra={
                "event": "screen.settlement.recaptured",
                "workflow.id": evidence.workflow_id,
                "effect.initial": initial.status.value,
                "effect.settled": settled.status.value,
                "hash.settled": hashes.visual_hash[:8],
                "hash.initial": evidence.after.hashes.visual_hash[:8],
                "action.type": evidence.execution.step.action.action_type.value,
            },
        )

        return ScreenSettlement(capture=capture, hashes=hashes, diff=diff)

    def __capture_timeout(self) -> float:
        """
        Return the wall-clock timeout for one settlement recapture.
        """

        device = getattr(self.__configuration, "device", None)

        if device is None:
            runtime = self.__device.configuration
            timeout = runtime.command_timeout if runtime is not None else None
            return float(timeout or POST_ACTION_OBSERVATION_TIMEOUT_SECONDS)

        if device.type == DeviceConnectionType.REMOTE:
            return float(device.remote.request_timeout)

        if device.platform == DevicePlatform.IOS:
            return float(device.ios.command_timeout)

        return float(device.android.snapshot_timeout or POST_ACTION_OBSERVATION_TIMEOUT_SECONDS)

    async def __capture(self, *, reference: ScreenCapture) -> ScreenCapture:
        """
        Capture only the current screenshot for delayed settlement comparison.
        """

        image = await self.__device.capture_screen()

        if not image:
            raise RuntimeError("Screen settlement returned an empty screenshot.")

        try:
            package_name = await self.__device.get_current_package()
        except Exception:
            package_name = reference.activity

        return ScreenCapture(
            image=image,
            width=reference.width,
            height=reference.height,
            timestamp=int(time.time() * 1000),
            activity=package_name or reference.activity,
        )

    @staticmethod
    def __should_recapture(*, evidence: ScreenSettlementEvidence, effect: ActionEffect) -> bool:
        """
        Return whether the executed action deserves one light settlement recapture.
        """

        execution = evidence.execution

        if execution.execution_result is None or not execution.execution_result.success:
            return False

        if execution.step.action.execution_kind is not ActionExecutionKind.DEVICE:
            return False

        if effect.status is not ActionEffectStatus.NO_PROGRESS:
            return False

        kind = ActionKindResolver.resolve(action_type=execution.step.action.action_type)
        return kind in (ActionKind.NAVIGATION, ActionKind.INPUT)
