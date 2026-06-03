from __future__ import annotations

import unittest

from fathom.constants.tools import ToolName
from fathom.core.prompts.tools import ToolRegistry


class ToolRegistryTest(unittest.TestCase):
    """
    Pins the tool catalog filtering and ordering contracts.
    """

    def test_execute_ui_action_requires_confidence(self) -> None:
        """
        execute_ui's schema must require action.confidence on every tool call.
        """

        definitions = ToolRegistry.get_all_definitions()["function_declarations"]
        execute_ui = next(
            definition for definition in definitions if definition["name"] == "execute_ui"
        )
        required = execute_ui["parameters"]["properties"]["action"]["required"]

        self.assertIn("confidence", required)

    def test_condition_field_announces_is_conditional_dependency(self) -> None:
        """
        The schema's condition description must teach the planner the is_conditional contract.
        """

        definitions = ToolRegistry.get_all_definitions()["function_declarations"]
        execute_ui = next(
            definition for definition in definitions if definition["name"] == "execute_ui"
        )
        condition = execute_ui["parameters"]["properties"]["action"]["properties"]["condition"]
        is_conditional = execute_ui["parameters"]["properties"]["action"]["properties"][
            "is_conditional"
        ]

        self.assertIn("conditional wait", condition["description"])
        self.assertIn("'condition' field is REQUIRED", is_conditional["description"])
        self.assertIn("MANDATORY whenever is_conditional=true", condition["description"])

    def test_definitions_filters_to_requested_names(self) -> None:
        """
        definitions(names=…) returns only the requested tool declarations.
        """

        result = ToolRegistry.definitions(
            names=frozenset({ToolName.EXECUTE_UI, ToolName.STORE_MEMORY}),
        )
        names = [declaration["name"] for declaration in result["function_declarations"]]

        self.assertEqual(set(names), {"execute_ui", "store_memory"})

    def test_definitions_preserves_catalog_order(self) -> None:
        """
        Output order follows the catalog regardless of input set iteration order.
        """

        result = ToolRegistry.definitions(
            names=frozenset({ToolName.VALIDATE_STATE, ToolName.ASK_USER, ToolName.EXECUTE_UI}),
        )
        names = [declaration["name"] for declaration in result["function_declarations"]]

        self.assertEqual(names, ["ask_user", "execute_ui", "validate_state"])

    def test_definitions_returns_empty_for_empty_set(self) -> None:
        """
        An empty allowed set yields an empty declarations payload.
        """

        result = ToolRegistry.definitions(names=frozenset())

        self.assertEqual(result["function_declarations"], [])

    def test_get_all_definitions_returns_full_catalog(self) -> None:
        """
        The backwards-compatible getter still returns the full catalog.
        """

        names = {
            declaration["name"]
            for declaration in ToolRegistry.get_all_definitions()["function_declarations"]
        }

        self.assertEqual(
            names,
            {
                "ask_user",
                "execute_ui",
                "verify_goal",
                "store_memory",
                "recall_memory",
                "validate_state",
            },
        )

    def test_bbox_description_demands_tight_visible_extent(self) -> None:
        """
        The bbox schema description must instruct the planner to hug visible glyph extent only.
        """

        definitions = ToolRegistry.get_all_definitions()["function_declarations"]
        execute_ui = next(
            definition for definition in definitions if definition["name"] == "execute_ui"
        )
        bbox = execute_ui["parameters"]["properties"]["action"]["properties"]["bbox"]

        self.assertIn("hug the visible glyph", bbox["description"])
        self.assertIn("exclude surrounding card", bbox["description"].lower())

    def test_target_name_rejects_interaction_kind_suffix(self) -> None:
        """
        The target_name description must tell the planner to drop interaction-kind suffixes.
        """

        definitions = ToolRegistry.get_all_definitions()["function_declarations"]
        execute_ui = next(
            definition for definition in definitions if definition["name"] == "execute_ui"
        )
        target_name = execute_ui["parameters"]["properties"]["action"]["properties"]["target_name"]

        self.assertIn("EXACT visible text", target_name["description"])
        self.assertIn("Do NOT append interaction-kind suffixes", target_name["description"])
