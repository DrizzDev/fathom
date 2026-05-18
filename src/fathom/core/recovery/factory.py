from __future__ import annotations

from logging import getLogger
from typing import Callable, Dict, List

from pydantic import BaseModel, ConfigDict, Field

from fathom.core.exceptions import ConfigurationError
from fathom.core.recovery.strategies.alternative import AlternativeTargetRecovery
from fathom.core.recovery.strategies.escalation import HumanEscalationRecovery
from fathom.core.recovery.strategies.failure import BoundedFailureRecovery
from fathom.core.recovery.strategies.keyboard import KeyboardRecovery
from fathom.core.recovery.strategies.overlay import OverlayRecovery
from fathom.core.recovery.strategies.replan import ReplanRecovery
from fathom.core.recovery.strategies.scroll import ScrollBoundaryRecovery
from fathom.core.recovery.strategy import RecoveryStrategy
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort

logger = getLogger(__name__)


class RecoveryContext(BaseModel):
    """
    Typed parameter bag handed to strategy builders. New ports become
    optional fields when they land without breaking existing builders.
    """

    llm: LLMPort = Field(description="Language-model port shared with strategies")
    memory: MemoryPort = Field(
        description="Memory port for strategies that need prior-attempt context"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


StrategyBuilder = Callable[[RecoveryContext], RecoveryStrategy]


class RecoveryStrategyFactory:
    """
    Name-keyed registry of recovery strategy builders.
    Mirrors the ``PromptFactory`` pattern: built-in builders live in an immutable map;
    custom registrations live in a separate mutable map so test isolation via :meth:`reset` does not touch defaults.
    """

    __BUILTINS: Dict[str, StrategyBuilder] = {
        "replan": ReplanRecovery.build,
        "overlay": OverlayRecovery.build,
        "keyboard": KeyboardRecovery.build,
        "task_replan": ReplanRecovery.build,
        "scroll": ScrollBoundaryRecovery.build,
        "failure": BoundedFailureRecovery.build,
        "escalation": HumanEscalationRecovery.build,
        "alternative": AlternativeTargetRecovery.build,
    }

    __custom: Dict[str, StrategyBuilder] = {}

    @classmethod
    def register(cls, name: str, builder: StrategyBuilder) -> None:
        """
        Register a custom strategy builder under ``name``.
        Last-write-wins among customs; customs shadow builtins on name collision.
        """

        cls.__custom[name] = builder

    @classmethod
    def reset(cls) -> None:
        """
        Drop all custom registrations and restore the factory to the
        builtin-only state. Intended for test teardown.
        """

        cls.__custom.clear()

    @classmethod
    def build(cls, *, names: List[str], context: RecoveryContext) -> List[RecoveryStrategy]:
        """
        Construct the strategy list in priority order. Raises
        :class:`ConfigurationError` for any unknown name so a typo in configuration fails fast at the call site.
        """

        strategies: List[RecoveryStrategy] = []

        for name in names:
            builder = cls.__custom.get(name) or cls.__BUILTINS.get(name)
            if builder is None:
                raise ConfigurationError(
                    f"Unknown recovery strategy {name!r}; available: {cls.available()}"
                )

            strategies.append(builder(context))

        return strategies

    @classmethod
    def available(cls) -> List[str]:
        """
        Return all registered strategy names (builtins + customs), sorted.
        """

        return sorted(set(cls.__BUILTINS) | set(cls.__custom))
