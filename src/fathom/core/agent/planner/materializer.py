from __future__ import annotations

from fathom.core.agent.action import ActionBuilder
from fathom.core.agent.command import CommandGate
from fathom.schemas.results import AnalysisResult


class CommandMaterializer:
    """
    Builds an executable action from a parsed execute_ui command after catalog validation.
    """

    def __init__(self, *, command_gate: CommandGate, action_builder: ActionBuilder) -> None:
        """
        Bind the command gate and action builder the materialization delegates to.
        """

        self.__command_gate = command_gate
        self.__action_builder = action_builder

    def materialize(self, *, analysis: AnalysisResult) -> AnalysisResult:
        """
        Return the analysis with its command materialized into an executable action, or unchanged.
        """

        if analysis.tool_response is None or analysis.tool_response.command is None:
            return analysis

        accepted = self.__command_gate.validate(command=analysis.tool_response.command)
        action = self.__action_builder.build(command=accepted)
        return analysis.model_copy(update={"action": action})
