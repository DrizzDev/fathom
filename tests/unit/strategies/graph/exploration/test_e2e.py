from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, Mock, patch

from fathom.constants.exploration import DISABLED_LOOP_THRESHOLD
from fathom.constants.state import CommonStateKey as CKey
from fathom.constants.state import CompletionReason
from fathom.core.agent.state import AgentState
from fathom.core.services.exporter.artifacts import ExplorationArtifactWriter
from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph
from fathom.schemas.actions import Action
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.metrics import ExecutionMetrics
from fathom.schemas.results import ExecutionResult, GenerateResult
from fathom.schemas.screens import ScreenState
from fathom.strategies.graph.exploration.builder import ExplorationGraphBuilder

_MAX_STEPS = 5
_RECURSION_LIMIT = 200


class _NullProvider:
    """IMemoryProvider stand-in whose writes are no-ops."""

    async def store_observation(self, screen: ScreenState, description: Optional[str]) -> None:
        return None

    async def store_transition(
        self, source_hash: str, action: Action, destination_hash: str
    ) -> None:
        return None


class _Call:
    """Minimal tool-call stand-in exposing name and args."""

    def __init__(self, name: str, args: Dict[str, Any]) -> None:
        self.name = name
        self.args = args


class _ScanningModel:
    """A fake LLM that taps a fresh element on every scan, never exhausting."""

    def __init__(self) -> None:
        self.__calls = 0

    async def generate(self, **_: Any) -> GenerateResult:
        """Returns an explore_ui call targeting a unique element each time."""

        self.__calls += 1
        return GenerateResult(
            tool_calls=[
                _Call(
                    "explore_ui",
                    {
                        "action": {
                            "action_type": "tap",
                            "rationale": "explore",
                            "target_name": f"Element {self.__calls}",
                        },
                        "assistant_message": "tap",
                        "screen_description": "Home",
                        "content_exhausted": False,
                    },
                )
            ]
        )


class TestExplorationWorkflowEndToEnd(unittest.IsolatedAsyncioTestCase):
    """A full graph run loops to completion and yields writable artifacts."""

    @staticmethod
    def __context(
        *, agent_state: AgentState, graph: KnowledgeGraph, image: bytes = b"image"
    ) -> Mock:
        capture = Mock(
            image=image,
            xml_content=None,
            activity="com.app/.Home",
            width=1080,
            height=1920,
            screenshot_uri=None,
        )
        capture.model_copy = Mock(return_value=capture)
        screen_state = ScreenState(
            activity="com.app/.Home",
            timestamp=0,
            activity_hash="home",
            visual_hash="0000000000000000",
        )
        return Mock(
            intent="Explore application",
            workflow_id="e2e",
            package_name="com.app",
            max_steps=_MAX_STEPS,
            is_cancelled=False,
            llm=_ScanningModel(),
            configuration=Mock(llm=Mock(use_cache=False)),
            exploration_graph=graph,
            agent_state=agent_state,
            metrics=ExecutionMetrics(),
            perception=Mock(
                perceive=AsyncMock(return_value=capture),
                build_state=Mock(return_value=screen_state),
            ),
            action_executor=Mock(
                act=AsyncMock(return_value=ExecutionResult(success=True, duration=1))
            ),
            history=Mock(enqueue_save_step=Mock()),
            memory=Mock(store_experience=AsyncMock()),
            device=Mock(get_current_package=AsyncMock(return_value="com.app")),
            hitl=Mock(
                check_signal=AsyncMock(return_value=None),
                is_pause_requested=AsyncMock(return_value=False),
            ),
            telemetry=Mock(info=AsyncMock()),
        )

    async def test_run_completes_and_writes_artifacts(self) -> None:
        graph = KnowledgeGraph(provider=_NullProvider())
        agent_state = AgentState(
            intent="Explore application",
            max_steps=_MAX_STEPS,
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
            loop_threshold=DISABLED_LOOP_THRESHOLD,
        )
        context = self.__context(agent_state=agent_state, graph=graph)
        compiled = ExplorationGraphBuilder(context=context).build()

        with patch("fathom.strategies.graph.exploration.nodes.stability_wait", new=AsyncMock()):
            await compiled.ainvoke({}, config={"recursion_limit": _RECURSION_LIMIT})

        # The run consumed its full step budget rather than stalling early: with
        # loop detection disabled, repeated visits to the same screen do not abort.
        self.assertEqual(agent_state.step_count, _MAX_STEPS)
        self.assertGreaterEqual(graph.node_count, 1)

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "reports"
            written = ExplorationArtifactWriter().write(
                graph=graph,
                directory=directory,
                workflow="e2e",
                package="com.app",
                generated_at="2026-06-12T00:00:00",
                duration=1.0,
            )
            self.assertEqual(
                {path.name for path in written if path.parent == directory},
                {"graph.json", "graph.dot", "graph.mermaid", "report.md"},
            )
            # Per-screen documentation lands under a screens/ subdirectory.
            self.assertIn(directory / "screens" / "index.md", written)
            self.assertTrue(all(path.exists() for path in written))

    async def test_unusable_capture_ends_clean_not_recursion(self) -> None:
        # Perception yields an empty screenshot: grounding must fail fast with a
        # clear reason. A tight recursion budget would raise GraphRecursionError
        # if the old silent wedge persisted, so reaching END proves the fix.
        graph = KnowledgeGraph(provider=_NullProvider())
        agent_state = AgentState(
            intent="Explore application",
            max_steps=_MAX_STEPS,
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
            loop_threshold=DISABLED_LOOP_THRESHOLD,
        )
        context = self.__context(agent_state=agent_state, graph=graph, image=b"")
        compiled = ExplorationGraphBuilder(context=context).build()

        final_state = await compiled.ainvoke({}, config={"recursion_limit": 5})

        self.assertTrue(final_state[CKey.IS_COMPLETE])
        self.assertEqual(final_state[CKey.COMPLETION_REASON], CompletionReason.PERCEPTION_FAILED)
        self.assertEqual(agent_state.step_count, 0)


if __name__ == "__main__":
    unittest.main()
