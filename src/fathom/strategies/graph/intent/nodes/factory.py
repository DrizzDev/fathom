from __future__ import annotations

from typing import Any, Callable, Dict

from fathom.constants.graph import NodeName
from fathom.core.recovery import (
    RecoveryContext,
    RecoveryCoordinator,
    RecoveryStrategyFactory,
)
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.intent.nodes.analyze import AnalyzeNode
from fathom.strategies.graph.intent.nodes.execute import ExecuteNode
from fathom.strategies.graph.intent.nodes.ground import GroundNode
from fathom.strategies.graph.intent.nodes.observe import ObserveNode
from fathom.strategies.graph.intent.nodes.provider import IntentNodeProvider
from fathom.strategies.graph.intent.nodes.record import RecordNode
from fathom.strategies.graph.intent.nodes.supervise import SuperviseNode
from fathom.strategies.graph.intent.nodes.verify import VerifyNode


class IntentGraphFactory:
    """
    Builds the intent-graph node callables for the supplied graph context.
    """

    @staticmethod
    def build(*, context: GraphContext) -> Dict[str, Callable[..., Any]]:
        """
        Return the keyed callable map consumed by the LangGraph builder.
        """

        recovery_context = RecoveryContext(
            llm=context.llm,
            memory=context.memory,
        )
        recovery_strategies = RecoveryStrategyFactory.build(
            context=recovery_context,
            names=list(context.recovery.strategies),
        )
        recovery_coordinator = RecoveryCoordinator(
            policy=context.recovery,
            strategies=recovery_strategies,
        )

        provider = IntentNodeProvider(
            context=context,
            recovery=recovery_coordinator,
            screen_comparator=context.comparator,
        )

        return {
            NodeName.GROUND: GroundNode(provider=provider),
            NodeName.ANALYZE: AnalyzeNode(provider=provider),
            NodeName.SUPERVISE: SuperviseNode(provider=provider),
            NodeName.EXECUTE: ExecuteNode(provider=provider),
            NodeName.OBSERVE: ObserveNode(provider=provider),
            NodeName.RECORD: RecordNode(provider=provider),
            NodeName.VERIFY: VerifyNode(provider=provider),
        }
