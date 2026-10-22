from __future__ import annotations

from fathom.app_graph.graph import AppGraph
from fathom.app_graph.models import Block, BlockStatus
from fathom.app_graph.state import Completion, Description, Observation, Start


class Ledger:
    """Applies evidenced discovery and block lifecycle policy to a graph."""

    def start(self, *, graph: AppGraph, start: Start) -> None:
        """Register each run once and preserve existing discovery knowledge."""
        snapshot = graph.snapshot()
        if any(block.id == start.execution for block in snapshot.blocks):
            return
        snapshot.blocks.append(Block(id=start.execution, intent=start.goal))
        graph.restore(snapshot=snapshot)
        graph.start_block(block_id=start.execution)
        graph.begin_exploration(goal=start.goal)

    def observe(self, *, graph: AppGraph, observation: Observation) -> None:
        """Merge a new observation and its verified navigation."""
        self.start(
            graph=graph,
            start=Start(
                identity=observation.identity,
                execution=observation.execution,
                goal=observation.goal,
            ),
        )
        canonical = graph.add_screen(screen=observation.screen)
        if observation.transition is not None:
            graph.add_transition(transition=observation.transition)
        snapshot = graph.snapshot()
        block = next(item for item in snapshot.blocks if item.id == observation.execution)
        if canonical not in block.screens_discovered:
            block.screens_discovered.append(canonical)
        previous_steps = block.steps
        block.steps = max(block.steps, observation.sequence)
        block.transitions_discovered += int(observation.transition is not None)
        snapshot.total_steps += block.steps - previous_steps
        graph.restore(snapshot=snapshot)

    def complete(self, *, graph: AppGraph, completion: Completion) -> None:
        """Finalize an existing block without discarding its discoveries."""
        block = next((item for item in graph.blocks if item.id == completion.execution), None)
        if block is None or block.status in (BlockStatus.COMPLETED, BlockStatus.FAILED):
            return
        graph.complete_block(
            block_id=block.id,
            screens_discovered=block.screens_discovered,
            transitions_discovered=block.transitions_discovered,
            steps=block.steps,
            error=completion.error,
        )

    def describe(self, *, graph: AppGraph, description: Description) -> None:
        """Describe an existing canonical page without inventing navigation."""
        snapshot = graph.snapshot()
        canonical = graph.resolve(description.screen)
        for screen in snapshot.screens:
            if screen.hash == canonical:
                screen.description = description.text
                screen.purpose = description.text
        graph.restore(snapshot=snapshot)
