from __future__ import annotations

from typing import List, Tuple

from fathom.constants.generation import ScriptCommandRole
from fathom.interfaces.dialect import Dialect
from fathom.schemas.flow import BranchNode, CheckNode, Flow, FlowNode, LaunchNode
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
            source_steps = self.__source_steps(node=node)
            commands.append(
                ScriptCommand(
                    role=self.__role(node=node),
                    source_steps=source_steps,
                    text=self.__render(flow=flow, node=node),
                    verified_by=("execution",) if source_steps else (),
                )
            )

        return tuple(commands)

    @staticmethod
    def __role(*, node: FlowNode) -> ScriptCommandRole:
        """
        Return the semantic role of one flow node.
        """

        if isinstance(node, LaunchNode):
            return ScriptCommandRole.LAUNCH

        if isinstance(node, BranchNode):
            return ScriptCommandRole.BRANCH

        if isinstance(node, CheckNode):
            return ScriptCommandRole.CHECK

        return ScriptCommandRole.ACTION

    def __source_steps(self, *, node: FlowNode) -> Tuple[int, ...]:
        """
        Return all evidence steps represented by one rendered top-level command.
        """

        if not isinstance(node, BranchNode):
            return node.source_steps

        sources: List[int] = []

        for step in node.source_steps:
            if step not in sources:
                sources.append(step)

        for leaf in node.body:
            for step in leaf.source_steps:
                if step not in sources:
                    sources.append(step)

        return tuple(sources)

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
