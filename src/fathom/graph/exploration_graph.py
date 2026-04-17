"""
LangGraph StateGraph for DFS exploration workflows.

Topology::

    ┌────────┐
    │ ground │──capture_failed?──→ END
    └───┬────┘
        │
    ┌───▼──────────┐
    │  dfs_route   │──COMPLETE──→ END
    └───┬──────────┘
        │
        ├── SCAN ──────→ scan ──→ execute ──→ record ──→ ground (loop)
        │                 └─exhausted──→ dfs_route
        │
        ├── BACKTRACK ─→ navigate ──→ record ──→ ground (loop)
        │
        └── ADVANCE ──→ navigate ──→ record ──→ ground (recovery)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import asyncio

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph.state import CompiledStateGraph

from logging import getLogger

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from fathom.graph.exploration_nodes import (
    ExplorationNodeContext,
    build_exploration_nodes,
    make_route_after_bfs_route,
    make_route_after_ground,
    make_route_after_record,
    make_route_after_scan,
)
from fathom.graph.exploration_state import ExplorationGraphState
from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph
from fathom.interfaces import IMemoryProvider
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool
from fathom.tools.vision import VisionTool

logger = getLogger(__name__)


def build_exploration_graph(
    device: DeviceTool,
    capture: CaptureTool,
    vision: VisionTool,
    knowledge_graph: KnowledgeGraph,
    memory: IMemoryProvider,
    *,
    max_steps: int = 100,
    workflow_id: str = "default",
    checkpointer: Optional[MemorySaver] = None,
    cancel_event: Optional[asyncio.Event] = None,
    pause_event: Optional[asyncio.Event] = None,
    target_package: Optional[str] = None,
    focus: Optional[str] = None,
) -> tuple["CompiledStateGraph[Any, Any, Any]", ExplorationNodeContext]:
    """
    Build and compile a LangGraph :class:`StateGraph` for DFS exploration.

    Parameters
    ----------
    device:
        ADB device tool for physical interactions.
    capture:
        Screen capture tool.
    vision:
        VLM orchestrator (Gemini) for screen analysis.
    knowledge_graph:
        Persistent knowledge graph (per-app, SQLite-backed).
    memory:
        Persistent memory provider.
    max_steps:
        Safety-limit on loop iterations.
    workflow_id:
        Unique run identifier (used for checkpoints).
    checkpointer:
        Optional LangGraph checkpointer. Defaults to in-memory.
    cancel_event:
        Optional ``asyncio.Event`` signalled on workflow cancellation.
    target_package:
        When provided, the exploration is scoped to this Android package.
        After every action the ``record`` node verifies the foreground
        package and recovers if the device has drifted.

    Returns
    -------
    tuple[CompiledStateGraph, ExplorationNodeContext]
        The compiled graph and the shared node context.
    """

    # ── 1. Shared context ──────────────────────────────────────────
    ctx = ExplorationNodeContext(
        device=device,
        capture=capture,
        vision=vision,
        knowledge_graph=knowledge_graph,
        memory=memory,
        max_steps=max_steps,
        workflow_id=workflow_id,
        cancel_event=cancel_event,
        pause_event=pause_event,
        target_package=target_package,
        focus=focus,
    )

    # ── 2. Build node functions ────────────────────────────────────
    nodes = build_exploration_nodes(ctx)

    # ── 3. Assemble the graph ──────────────────────────────────────
    graph = StateGraph(ExplorationGraphState)

    graph.add_node("ground", nodes["ground"])
    graph.add_node("bfs_route", nodes["bfs_route"])
    graph.add_node("scan", nodes["scan"])
    graph.add_node("execute", nodes["execute"])
    graph.add_node("navigate", nodes["navigate"])
    graph.add_node("record", nodes["record"])

    # ── 4. Wire edges ─────────────────────────────────────────────
    graph.set_entry_point("ground")

    graph.add_conditional_edges(
        "ground",
        make_route_after_ground(ctx),
        {"bfs_route": "bfs_route", "done": END},
    )

    graph.add_conditional_edges(
        "bfs_route",
        make_route_after_bfs_route(ctx),
        {"scan": "scan", "navigate": "navigate", "done": END},
    )

    graph.add_conditional_edges(
        "scan",
        make_route_after_scan(ctx),
        {"execute": "execute", "bfs_route": "bfs_route", "done": END},
    )

    graph.add_edge("execute", "record")
    graph.add_edge("navigate", "record")

    graph.add_conditional_edges(
        "record",
        make_route_after_record(ctx),
        {"ground": "ground", "done": END},
    )

    # ── 5. Compile ─────────────────────────────────────────────────
    saver = checkpointer or MemorySaver()
    compiled = graph.compile(checkpointer=saver)

    logger.info(
        "Exploration graph compiled  (nodes=%d, max_steps=%d)",
        len(nodes),
        max_steps,
    )

    return compiled, ctx
