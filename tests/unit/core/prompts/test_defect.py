from __future__ import annotations

import unittest

from fathom.constants.defect import DETECT_DEFECTS_TOOL, VISION_DEFECT_SIGNALS
from fathom.core.prompts.defect import DefectPromptBuilder
from fathom.core.prompts.tools import ToolRegistry


class DefectPromptBuilderTest(unittest.TestCase):
    """
    Verifies the defect-inspection system instruction.
    """

    def test_prompt_directs_the_tool_and_restraint(self) -> None:
        """
        The prompt names the tool and tells the model to return nothing on a clean screen.
        """

        prompt = DefectPromptBuilder().build_system_prompt()

        self.assertIn(DETECT_DEFECTS_TOOL, prompt)
        self.assertIn("empty", prompt.lower())


class DefectToolDefinitionTest(unittest.TestCase):
    """
    Verifies the detect_defects tool schema.
    """

    def test_tool_exposes_defects_array_constrained_to_vision_signals(self) -> None:
        """
        The tool requires signal + summary per defect and constrains signals to the vision set.
        """

        declarations = ToolRegistry.get_defect_definitions()["function_declarations"]
        self.assertEqual(len(declarations), 1)

        tool = declarations[0]
        self.assertEqual(tool["name"], DETECT_DEFECTS_TOOL)

        item = tool["parameters"]["properties"]["defects"]["items"]
        self.assertEqual(item["required"], ["signal", "summary"])
        self.assertEqual(
            set(item["properties"]["signal"]["enum"]),
            {signal.value for signal in VISION_DEFECT_SIGNALS},
        )


if __name__ == "__main__":
    unittest.main()
