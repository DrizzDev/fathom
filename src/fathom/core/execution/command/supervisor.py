from __future__ import annotations

import time
from io import BytesIO
from logging import getLogger
from typing import Optional, Tuple, cast

from PIL import Image

from fathom.constants.interaction import SwipeSpeed
from fathom.interfaces.command import CommandScopeResolvePort
from fathom.interfaces.device import DevicePort
from fathom.interfaces.scroll import (
    ScrollDetectPort,
    ScrollPlanPort,
    ScrollRuntimePolicyPort,
    ScrollSurfacePort,
    ScrollVerifyPort,
    TraceRecorder,
)
from fathom.schemas.actions import GesturePath
from fathom.schemas.command import CommandPolicy
from fathom.schemas.configuration import ScrollInteractionPolicy
from fathom.schemas.execution import ScopedCommandExecution
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.results import ActionResult, ActionTraceAttempt, ActionTraceEvent
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.scroll import (
    ScrollAttempt,
    ScrollContext,
    ScrollOutcome,
    ScrollScope,
    ScrollVerdict,
)
from fathom.utils.coordinates import CoordinateConverter

logger = getLogger(__name__)


class CommandSupervisor:
    """
    Coordinates scope resolution, planning, device execution, and observation for scoped commands.
    """

    def __init__(
        self,
        *,
        device: DevicePort,
        detector: ScrollDetectPort,
        surface: ScrollSurfacePort,
        planner: ScrollPlanPort,
        resolver: CommandScopeResolvePort,
        runtime_policy: ScrollRuntimePolicyPort,
        verifier: Optional[ScrollVerifyPort] = None,
    ) -> None:
        """
        Bind the collaborators required for one supervised scoped-command execution.
        """

        self.__device = device
        self.__detector = detector
        self.__surface = surface
        self.__planner = planner
        self.__resolver = resolver
        self.__runtime_policy = runtime_policy
        self.__verifier = verifier

    async def execute(
        self,
        *,
        current: GesturePath,
        before: ScreenCapture,
        context: ScrollContext,
        observation: ScreenObservation,
        converter: CoordinateConverter,
        policy: ScrollInteractionPolicy.AdaptivePolicy,
        trace_recorder: TraceRecorder | None = None,
    ) -> ScopedCommandExecution:
        """
        Execute one scoped command using bounded adaptive attempts.
        """

        capture_width, capture_height = self.__capture_size(capture=before)
        scope = await self.__resolve_scope(
            context=context,
            observation=observation,
            converter=converter,
        )
        attempts = await self.__plan_attempts(
            capture_height=capture_height,
            capture_width=capture_width,
            context=context,
            current=current,
            observation=observation,
            scope=scope,
            converter=converter,
            policy=policy,
        )
        return await self.__run_attempts(
            attempts=attempts,
            before=before,
            context=context,
            policy=policy,
            scope=scope,
            trace_recorder=trace_recorder,
        )

    async def __resolve_scope(
        self,
        *,
        context: ScrollContext,
        observation: ScreenObservation,
        converter: CoordinateConverter,
    ) -> ScrollScope:
        """
        Resolve the execution scope for the current command.
        """

        requested_scope = self.__runtime_policy.requested_scope(
            context=context,
            converter=converter,
        )
        return cast(
            "ScrollScope",
            await self.__resolver.resolve(
                fallback=requested_scope,
                converter=converter,
                anchor=context.anchor,
                observation=observation,
            ),
        )

    async def __plan_attempts(
        self,
        *,
        capture_height: int,
        capture_width: int,
        context: ScrollContext,
        current: GesturePath,
        observation: ScreenObservation,
        scope: ScrollScope,
        converter: CoordinateConverter,
        policy: ScrollInteractionPolicy.AdaptivePolicy,
    ) -> tuple[ScrollAttempt, ...]:
        """
        Build bounded attempts for the resolved scope.
        """

        surfaces = await self.__surface.inspect(
            path=current,
            observation=observation,
            capture_width=capture_width,
            capture_height=capture_height,
        )
        return self.__planner.plan(
            scope=scope,
            policy=policy,
            current=current,
            context=context,
            surfaces=surfaces,
            converter=converter,
            capture_height=capture_height,
        )

    async def __run_attempts(
        self,
        *,
        attempts: tuple[ScrollAttempt, ...],
        before: ScreenCapture,
        context: ScrollContext,
        policy: ScrollInteractionPolicy.AdaptivePolicy,
        scope: ScrollScope,
        trace_recorder: TraceRecorder | None,
    ) -> ScopedCommandExecution:
        """
        Dispatch attempts and classify the final outcome.
        """

        requested_policy = CommandPolicy(
            budget=policy.budget,
            attempts=policy.maximum_attempts,
        )
        command_policy = CommandPolicy(
            budget=requested_policy.budget,
            attempts=self.__runtime_policy.maximum_internal_attempts(policy=requested_policy),
        )
        deadline = time.time() + (command_policy.budget / 1000.0)
        completed: list[ScrollAttempt] = []
        trace_events: list[ActionTraceEvent] = []
        current_before = before
        final_action = ActionResult(success=False, duration=0, error="scroll_not_executed")
        final_verdict = self.__runtime_policy.initial_verdict()

        for attempt_index, attempt in enumerate(attempts[: command_policy.attempts]):
            if time.time() >= deadline:
                final_verdict = self.__runtime_policy.budget_verdict()
                break

            action, verdict, after = await self.__dispatch_attempt(
                attempt=attempt,
                attempt_index=attempt_index,
                before=current_before,
                context=context,
                policy=policy,
                trace_events=trace_events,
                trace_recorder=trace_recorder,
            )
            final_action = action
            final_verdict = verdict
            completed.append(attempt.model_copy(update={"verdict": verdict}))

            if after is None:
                break

            current_before = after
            if self.__runtime_policy.is_success(verdict=verdict):
                break
            if not self.__runtime_policy.should_continue(
                verdict=verdict,
                deadline=deadline,
                policy=command_policy,
                completed_count=len(completed),
            ):
                break

        return ScopedCommandExecution(
            action=final_action,
            outcome=ScrollOutcome(
                scope=scope,
                final=final_verdict,
                attempts=tuple(completed),
                success=self.__runtime_policy.is_success(verdict=final_verdict),
            ),
            trace_events=tuple(trace_events),
        )

    async def __dispatch_attempt(
        self,
        *,
        attempt: ScrollAttempt,
        attempt_index: int,
        before: ScreenCapture,
        context: ScrollContext,
        policy: ScrollInteractionPolicy.AdaptivePolicy,
        trace_events: list[ActionTraceEvent],
        trace_recorder: TraceRecorder | None,
    ) -> tuple[ActionResult, ScrollVerdict, Optional[ScreenCapture]]:
        """
        Dispatch one attempt and classify the result.
        """

        started = time.time()
        self.__planner.log_attempt(attempt=attempt, attempt_index=attempt_index)
        action = await self.__device.swipe(
            x1=attempt.path.start_x,
            y1=attempt.path.start_y,
            x2=attempt.path.end_x,
            y2=attempt.path.end_y,
            speed=SwipeSpeed.SLOW,
            duration=attempt.path.duration,
        )
        trace_events.append(
            ActionTraceEvent(
                capture=before,
                coords=attempt.path.to_coordinates(),
                attempt=ActionTraceAttempt(index=attempt_index),
            )
        )
        if trace_recorder is not None:
            await trace_recorder(trace_events[-1])

        if not action.success:
            verdict = self.__runtime_policy.device_failure_verdict(error=action.error)
            return action, verdict, None

        after = await self.__capture_after(before=before)
        verdict = await self.__evaluate_attempt(
            after=after,
            before=before,
            policy=policy,
            context=context,
            attempt=attempt,
        )
        result = ActionResult(
            duration=int((time.time() - started) * 1000),
            success=self.__runtime_policy.is_success(verdict=verdict),
            error=None if self.__runtime_policy.is_success(verdict=verdict) else verdict.detail,
        )
        return result, verdict, after

    async def __evaluate_attempt(
        self,
        *,
        after: ScreenCapture,
        attempt: ScrollAttempt,
        before: ScreenCapture,
        context: ScrollContext,
        policy: ScrollInteractionPolicy.AdaptivePolicy,
    ) -> ScrollVerdict:
        """
        Evaluate one completed attempt.
        """

        verdict = await self.__detector.evaluate(
            after=after,
            before=before,
            direction=context.direction,
            region=attempt.capture_region,
        )
        if (
            verdict.kind is self.__planner.ambiguous_verdict_kind()
            and policy.verify
            and self.__verifier is not None
        ):
            return await self.__verifier.verify(
                after=after,
                before=before,
                direction=context.direction,
                region=attempt.capture_region,
            )
        return verdict

    async def __capture_after(self, *, before: ScreenCapture) -> ScreenCapture:
        """
        Capture the screen after one dispatched attempt.
        """

        image, xml_content = await self.__device.get_snapshot()
        try:
            activity = await self.__device.get_current_package()
        except Exception:
            activity = before.activity
        return ScreenCapture(
            image=image,
            xml_content=xml_content,
            width=before.width,
            height=before.height,
            activity=activity,
            metadata=dict(before.metadata),
            timestamp=int(time.time() * 1000),
        )

    @staticmethod
    def __capture_size(*, capture: ScreenCapture) -> Tuple[int, int]:
        """
        Resolve actual capture image size when possible.
        """

        try:
            with Image.open(BytesIO(capture.image)) as screenshot:
                size = screenshot.size
                return int(size[0]), int(size[1])
        except Exception:
            return capture.width, capture.height
