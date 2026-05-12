from __future__ import annotations

from logging import getLogger
from typing import Callable, Dict, List

from pydantic import BaseModel, ConfigDict, Field

from fathom.core.recovery.strategies.replan import ReplanRecovery
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
    """

    __builders: Dict[str, StrategyBuilder] = {
        "replan": lambda context: ReplanRecovery(llm=context.llm),
    }

    @classmethod
    def register(cls, name: str, builder: StrategyBuilder) -> None:
        """
        Register a strategy builder under ``name`` (last-write-wins).
        """

        cls.__builders[name] = builder

    @classmethod
    def build(cls, *, names: List[str], context: RecoveryContext) -> List[RecoveryStrategy]:
        """
        Construct the strategy list in priority order; unknown names skipped.
        """

        strategies: List[RecoveryStrategy] = []

        for name in names:
            if (builder := cls.__builders.get(name)) is None:
                logger.warning("[RecoveryStrategyFactory] Unknown strategy %r — skipping", name)
                continue

            strategies.append(builder(context))

        return strategies

    @classmethod
    def available(cls) -> List[str]:
        """
        Return registered strategy names.
        """

        return sorted(cls.__builders)
