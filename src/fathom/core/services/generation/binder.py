from __future__ import annotations

from typing import List, Tuple

from fathom.schemas.flow import (
    Evidence,
    Flow,
    FlowNode,
    LaunchNode,
    StepLaunch,
)


class LaunchBinder:
    """
    Stamps a flow's launch nodes with deterministic launch provenance from the normalized evidence markers.
    """

    def bind(self, *, flow: Flow, evidence: Evidence) -> Flow:
        """
        Rewrite each launch node's package, provenance, and grounding steps from the matching evidence marker.
        """

        markers: Tuple[StepLaunch, ...] = tuple(
            step.launch for step in evidence.steps if step.launch is not None
        )
        if not markers:
            return flow

        order = 0
        nodes: List[FlowNode] = []

        for node in flow.nodes:
            if isinstance(node, LaunchNode):
                if order < len(markers):
                    node = self.__bound(node=node, marker=markers[order])

                order += 1

            nodes.append(node)

        return flow.model_copy(update={"nodes": tuple(nodes)})

    @staticmethod
    def __bound(*, node: LaunchNode, marker: StepLaunch) -> LaunchNode:
        """
        Return the launch node with package, provenance, and source steps copied from the marker.
        """

        return node.model_copy(
            update={
                "package": marker.package,
                "provenance": marker.provenance,
                "source_steps": marker.source_steps,
            }
        )
