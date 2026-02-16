"""SQLite knowledge adapter - implements knowledge graph using SQLite and rustworkx."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite
import rustworkx as rx

from fathom.interfaces.knowledge import KnowledgePort
from fathom.schemas.actions import Action


class SQLiteKnowledge(KnowledgePort):
    """
    SQLite adapter for application knowledge graph.

    Uses SQLite for persistence and rustworkx for graph operations.
    """

    def __init__(self, *, database_path: str = "assets/memory/knowledge_graph.db") -> None:
        """Initialize SQLite knowledge adapter."""
        self.__initialized = False
        self.__path = Path(database_path)
        self.__path.parent.mkdir(parents=True, exist_ok=True)
        self.__graph: Optional[rx.PyDiGraph] = None
        self.__screen_to_node: Dict[str, int] = {}

    async def __initialize(self) -> None:
        """Initialize the database schema and load graph."""
        if self.__initialized:
            return

        async with aiosqlite.connect(self.__path) as db:
            # Create tables
            await db.execute(
                "CREATE TABLE IF NOT EXISTS screens (screen_id TEXT PRIMARY KEY, metadata TEXT)"
            )
            await db.execute(
                "CREATE TABLE IF NOT EXISTS transitions "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "from_screen TEXT, to_screen TEXT, action_json TEXT)"
            )
            await db.commit()

        # Load graph from database
        await self.__load_graph()
        self.__initialized = True

    async def __load_graph(self) -> None:
        """Load graph structure from database."""
        self.__graph = rx.PyDiGraph()
        self.__screen_to_node = {}

        async with aiosqlite.connect(self.__path) as db:
            # Load screens as nodes
            async with db.execute("SELECT screen_id, metadata FROM screens") as cursor:
                async for row in cursor:
                    screen_id = row[0]
                    metadata = json.loads(row[1]) if row[1] else {}
                    if self.__graph is not None:
                        node_idx = self.__graph.add_node({"screen_id": screen_id, **metadata})
                        self.__screen_to_node[screen_id] = node_idx

            # Load transitions as edges
            async with db.execute(
                "SELECT from_screen, to_screen, action_json FROM transitions"
            ) as cursor:
                async for row in cursor:
                    from_screen, to_screen, action_json = row
                    if from_screen in self.__screen_to_node and to_screen in self.__screen_to_node:
                        from_idx = self.__screen_to_node[from_screen]
                        to_idx = self.__screen_to_node[to_screen]
                        action_data = json.loads(action_json) if action_json else {}
                        if self.__graph is not None:
                            self.__graph.add_edge(from_idx, to_idx, action_data)

    async def add_screen(self, *, screen_id: str, metadata: Dict[str, Any]) -> None:
        """Add screen node to graph."""
        await self.__initialize()

        # Add to database
        async with aiosqlite.connect(self.__path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO screens (screen_id, metadata) VALUES (?, ?)",
                (screen_id, json.dumps(metadata)),
            )
            await db.commit()

        # Add to in-memory graph
        if screen_id not in self.__screen_to_node:
            if self.__graph is None:
                raise RuntimeError("Knowledge graph not initialized")
            node_idx = self.__graph.add_node({"screen_id": screen_id, **metadata})
            self.__screen_to_node[screen_id] = node_idx

    async def add_transition(self, *, from_screen: str, to_screen: str, action: Action) -> None:
        """Add transition edge between screens."""
        await self.__initialize()

        # Ensure both screens exist
        if from_screen not in self.__screen_to_node:
            await self.add_screen(screen_id=from_screen, metadata={})
        if to_screen not in self.__screen_to_node:
            await self.add_screen(screen_id=to_screen, metadata={})

        # Add to database
        async with aiosqlite.connect(self.__path) as db:
            await db.execute(
                "INSERT INTO transitions (from_screen, to_screen, action_json) VALUES (?, ?, ?)",
                (from_screen, to_screen, action.model_dump_json()),
            )
            await db.commit()

        # Add to in-memory graph
        from_idx = self.__screen_to_node[from_screen]
        to_idx = self.__screen_to_node[to_screen]
        if self.__graph is None:
            raise RuntimeError("Knowledge graph not initialized")
        self.__graph.add_edge(from_idx, to_idx, action.model_dump())

    async def find_path(self, *, from_screen: str, to_screen: str) -> Optional[List[Action]]:
        """Find action sequence to reach target screen."""
        await self.__initialize()

        if from_screen not in self.__screen_to_node or to_screen not in self.__screen_to_node:
            return None

        from_idx = self.__screen_to_node[from_screen]
        to_idx = self.__screen_to_node[to_screen]

        try:
            # Use Dijkstra's algorithm to find shortest path
            if self.__graph is None:
                raise RuntimeError("Knowledge graph not initialized")
            path_indices = rx.dijkstra_shortest_paths(
                self.__graph, from_idx, target=to_idx, weight_fn=lambda _: 1
            )

            if to_idx not in path_indices:
                return None

            # Extract actions from path
            path = path_indices[to_idx]
            actions = []
            for i in range(len(path) - 1):
                edge_data = self.__graph.get_edge_data(path[i], path[i + 1])
                if edge_data:
                    actions.append(Action(**edge_data))

            return actions if actions else None
        except Exception:
            return None

    async def get_neighbors(self, *, screen_id: str) -> List[str]:
        """Get screens reachable from given screen."""
        await self.__initialize()

        if screen_id not in self.__screen_to_node:
            return []

        node_idx = self.__screen_to_node[screen_id]
        if self.__graph is None:
            raise RuntimeError("Knowledge graph not initialized")
        neighbor_indices = self.__graph.successor_indices(node_idx)

        neighbors = []
        for neighbor_idx in neighbor_indices:
            node_data = self.__graph.get_node_data(neighbor_idx)
            if node_data and "screen_id" in node_data:
                neighbors.append(node_data["screen_id"])

        return neighbors
