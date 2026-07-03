from __future__ import annotations

from logging import getLogger
from typing import Tuple

from fathom.constants.tools import StateNamespace
from fathom.core.exceptions import InvariantViolation
from fathom.interfaces.memory import MemoryPort
from fathom.schemas.tools import StateUpdate, ToolArtifact, ToolData, ToolDiagnostic

logger = getLogger(__name__)


class ToolUpdateRouter:
    """
    Routes non-command model-tool response parts at the graph boundary.
    """

    def __init__(self, *, memory: MemoryPort) -> None:
        """
        Bind application services used by routed response parts.
        """

        self.__memory = memory

    async def route(
        self,
        *,
        updates: Tuple[StateUpdate, ...],
        data: Tuple[ToolData, ...],
        artifacts: Tuple[ToolArtifact, ...],
        diagnostics: Tuple[ToolDiagnostic, ...],
        workflow_id: str,
    ) -> None:
        """
        Apply non-command response parts without creating executable history.
        """

        for update in updates:
            await self.__apply_update(update=update)

        if updates or data or artifacts or diagnostics:
            logger.info(
                "Tool response routed",
                extra={
                    "component": "graph.intent.tool_update",
                    "event": "tool.update.routed",
                    "workflow.id": workflow_id,
                    "update.count": len(updates),
                    "data.count": len(data),
                    "artifact.count": len(artifacts),
                    "diagnostic.count": len(diagnostics),
                    "update.namespaces": sorted({update.namespace.value for update in updates}),
                    "update.keys": sorted(update.key for update in updates),
                },
            )

    async def __apply_update(self, *, update: StateUpdate) -> None:
        """
        Apply one runtime state update.
        """

        if update.namespace is StateNamespace.MEMORY:
            await self.__memory.set(key=update.key, value=update.value)
            return

        raise InvariantViolation(f"Unsupported state namespace: {update.namespace.value}")
