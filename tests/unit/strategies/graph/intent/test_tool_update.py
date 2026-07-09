from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

from fathom.constants.tools import StateNamespace
from fathom.interfaces.memory import MemoryPort
from fathom.schemas.tools import StateUpdate
from fathom.strategies.graph.intent.tool_update import ToolUpdateRouter


class ToolUpdateRouterTest(unittest.IsolatedAsyncioTestCase):
    """
    Covers routing of non-command model-tool response parts.
    """

    async def test_routes_memory_update_without_logging_value(self) -> None:
        """
        MEMORY updates are applied to memory and values are not logged.
        """

        memory_set = AsyncMock()
        memory = cast("MemoryPort", SimpleNamespace(set=memory_set))
        router = ToolUpdateRouter(memory=memory)

        with self.assertLogs(
            "fathom.strategies.graph.intent.tool_update",
            level="INFO",
        ) as logs:
            await router.route(
                updates=(
                    StateUpdate(
                        namespace=StateNamespace.MEMORY,
                        key="item_price",
                        value="₹94",
                    ),
                ),
                data=(),
                artifacts=(),
                diagnostics=(),
                workflow_id="wf-tool-response",
            )

        memory_set.assert_awaited_once_with(key="item_price", value="₹94")
        rendered = "\n".join(logs.output)
        self.assertIn("Tool response routed", rendered)
        self.assertNotIn("₹94", rendered)
