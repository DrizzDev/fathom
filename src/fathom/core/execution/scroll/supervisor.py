from __future__ import annotations

from typing import Optional

from fathom.core.execution.command import CommandSupervisor
from fathom.core.execution.scroll.planner import ScrollPlanner
from fathom.core.execution.scroll.resolver import ScrollScopeResolver
from fathom.core.execution.scroll.runtime.policy import ScrollRuntimePolicy
from fathom.interfaces.device import DevicePort
from fathom.interfaces.scroll import (
    ScrollDetectPort,
    ScrollSurfacePort,
    ScrollVerifyPort,
    TraceRecorder,
)
from fathom.schemas.actions import GesturePath
from fathom.schemas.configuration import ScrollInteractionPolicy
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.results import ActionResult, ActionTraceEvent
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.scroll import ScrollContext, ScrollOutcome
from fathom.utils.coordinates import CoordinateConverter


class ScrollCommandSupervisor:
    """
    Scope-resolved supervisor for scroll-like commands.
    """

    def __init__(
        self,
        *,
        device: DevicePort,
        detector: ScrollDetectPort,
        surface: ScrollSurfacePort,
        verifier: Optional[ScrollVerifyPort] = None,
        resolver: Optional[ScrollScopeResolver] = None,
        planner: Optional[ScrollPlanner] = None,
        runtime_policy: Optional[ScrollRuntimePolicy] = None,
    ) -> None:
        """
        Bind the collaborators required for supervised scroll execution.
        """

        self.__supervisor = CommandSupervisor(
            device=device,
            resolver=resolver or ScrollScopeResolver(),
            planner=planner or ScrollPlanner(),
            detector=detector,
            runtime_policy=runtime_policy or ScrollRuntimePolicy(),
            surface=surface,
            verifier=verifier,
        )

    async def execute(
        self,
        *,
        before: ScreenCapture,
        observation: ScreenObservation,
        context: ScrollContext,
        current: GesturePath,
        converter: CoordinateConverter,
        policy: ScrollInteractionPolicy.AdaptivePolicy,
        trace_recorder: TraceRecorder | None = None,
    ) -> tuple[ActionResult, ScrollOutcome, tuple[ActionTraceEvent, ...]]:
        """
        Execute one scroll command inside one resolved scope.
        """

        execution = await self.__supervisor.execute(
            before=before,
            observation=observation,
            context=context,
            current=current,
            converter=converter,
            policy=policy,
            trace_recorder=trace_recorder,
        )
        return execution.action, execution.outcome, execution.trace_events


class AdaptiveScrollSupervisor(ScrollCommandSupervisor):
    """
    Backward-compatible alias for the scoped scroll supervisor.
    """
