from __future__ import annotations

from typing import List, Tuple

from fathom.interfaces.dialect import Dialect
from fathom.schemas.flow import Flow, FlowNode
from fathom.schemas.generation import ScriptCommand


class ScriptCommandBuilder:
    """
    Builds rendered command metadata from flow nodes without relying on line positions.
    """

    def __init__(self, *, dialect: Dialect) -> None:
        """
        Bind the dialect renderer used to render each top-level command node.
        """

        self.__dialect = dialect

    def build(self, *, flow: Flow) -> Tuple[ScriptCommand, ...]:
        """
        Return rendered top-level commands paired with their evidence source steps.
        """

        commands: List[ScriptCommand] = []

        for node in flow.nodes:
            commands.append(
                ScriptCommand(
                    source_steps=node.source_steps,
                    text=self.__render(flow=flow, node=node),
                    verified_by=("execution",) if node.source_steps else (),
                )
            )

        return tuple(commands)

    def __render(self, *, flow: Flow, node: FlowNode) -> str:
        """
        Render a single top-level node in the context of its parent flow.
        """

        command = Flow(
            nodes=(node,),
            intent=flow.intent,
            package=flow.package,
            partial=flow.partial,
        )
        return self.__dialect.renderer.render(flow=command).strip()
