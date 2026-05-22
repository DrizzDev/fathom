from __future__ import annotations

from fathom.core.prompts.tools import ToolRegistry


class TestToolRegistry:
    """
    Pins the execute_ui tool contract surfaced to the model.
    """

    def test_execute_ui_action_requires_confidence(self) -> None:
        """
        The function-declaration schema must require action.confidence so
        the model cannot legally omit it on the first tool call.
        """

        definitions = ToolRegistry.get_all_definitions()["function_declarations"]
        execute_ui = next(
            definition for definition in definitions if definition["name"] == "execute_ui"
        )
        required = execute_ui["parameters"]["properties"]["action"]["required"]

        assert "confidence" in required
