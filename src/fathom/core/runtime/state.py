from __future__ import annotations

from fathom.constants.runtime import (
    DEFAULT_LOOP_THRESHOLD,
    DEFAULT_LOOP_WINDOW,
    DEFAULT_REALIGNMENT_BUDGET,
)
from fathom.core.runtime.checkpoint import CheckpointCodec
from fathom.core.runtime.effects import EffectHistory
from fathom.core.runtime.failures import FailureMemory
from fathom.core.runtime.healing import HealingUsage
from fathom.core.runtime.realignment import RealignmentTracker
from fathom.core.runtime.recovery import RecoveryRuntimeState
from fathom.core.runtime.screen import ScreenRuntimeState
from fathom.core.runtime.tasks import TaskRuntimeState


class RuntimeState:
    """
    Aggregate root for runtime state components.
    """

    def __init__(
        self,
        *,
        tasks: TaskRuntimeState,
        screen: ScreenRuntimeState,
        effects: EffectHistory,
        failures: FailureMemory,
        recovery: RecoveryRuntimeState,
        checkpoint: CheckpointCodec,
        healing: HealingUsage,
        realignment: RealignmentTracker,
    ) -> None:
        """
        Initialize the runtime state aggregate.
        """

        self.__tasks = tasks
        self.__screen = screen
        self.__effects = effects
        self.__healing = healing
        self.__failures = failures
        self.__recovery = recovery
        self.__checkpoint = checkpoint
        self.__realignment = realignment

    @classmethod
    def create(
        cls,
        *,
        loop_threshold: int = DEFAULT_LOOP_THRESHOLD,
        loop_window: int = DEFAULT_LOOP_WINDOW,
        realignment_budget: int = DEFAULT_REALIGNMENT_BUDGET,
    ) -> "RuntimeState":
        """
        Create a runtime state aggregate with default components.
        """

        return cls(
            healing=HealingUsage(),
            effects=EffectHistory(),
            failures=FailureMemory(),
            tasks=TaskRuntimeState(),
            screen=ScreenRuntimeState(
                loop_threshold=loop_threshold,
                loop_window=loop_window,
            ),
            checkpoint=CheckpointCodec(),
            recovery=RecoveryRuntimeState(),
            realignment=RealignmentTracker(budget=realignment_budget),
        )

    @property
    def tasks(self) -> TaskRuntimeState:
        """
        Return task runtime state.
        """

        return self.__tasks

    @property
    def screen(self) -> ScreenRuntimeState:
        """
        Return screen runtime state.
        """

        return self.__screen

    @property
    def effects(self) -> EffectHistory:
        """
        Return effect history.
        """

        return self.__effects

    @property
    def failures(self) -> FailureMemory:
        """
        Return failure memory.
        """

        return self.__failures

    @property
    def recovery(self) -> RecoveryRuntimeState:
        """
        Return recovery runtime state.
        """

        return self.__recovery

    @property
    def checkpoint(self) -> CheckpointCodec:
        """
        Return checkpoint codec.
        """

        return self.__checkpoint

    @property
    def healing(self) -> HealingUsage:
        """
        Return healing-usage state.
        """

        return self.__healing

    @property
    def realignment(self) -> RealignmentTracker:
        """
        Return the HITL realignment tracker.
        """

        return self.__realignment
