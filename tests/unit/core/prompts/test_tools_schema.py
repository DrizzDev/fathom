from __future__ import annotations

import unittest
from typing import Any, Dict, List

from fathom.constants.screen import ScreenCategory
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

    def test_action_type_enum_omits_legacy_enter(self) -> None:
        """The execute_ui action_type enum must not advertise the deprecated 'enter' action."""

        definitions = ToolRegistry.get_all_definitions()["function_declarations"]
        execute_ui = next(
            definition for definition in definitions if definition["name"] == "execute_ui"
        )
        action_type = execute_ui["parameters"]["properties"]["action"]["properties"]["action_type"]

        self.assertNotIn("enter", action_type["enum"])

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

    def test_verify_goal_schema_exposes_completion_reason_fields(self) -> None:
        """
        The verify_goal schema must surface goal_completion_reason, subgoal_completion_reason,
        and completion_criteria_met so the LLM can populate them when claiming completion.
        """

        definitions = ToolRegistry.get_all_definitions()["function_declarations"]
        verify_goal = next(
            definition for definition in definitions if definition["name"] == "verify_goal"
        )
        properties = verify_goal["parameters"]["properties"]

        self.assertIn("goal_completion_reason", properties)
        self.assertIn("subgoal_completion_reason", properties)
        self.assertIn("completion_criteria_met", properties)

    def test_verify_goal_completion_reason_descriptions_signal_required_when_complete(
        self,
    ) -> None:
        """
        Reason field descriptions must communicate the implicit "required when complete" contract.
        """

        definitions = ToolRegistry.get_all_definitions()["function_declarations"]
        verify_goal = next(
            definition for definition in definitions if definition["name"] == "verify_goal"
        )
        properties = verify_goal["parameters"]["properties"]

        self.assertIn(
            "Required when goal_completed=true",
            properties["goal_completion_reason"]["description"],
        )
        self.assertIn(
            "Required when sub_goal_completed=true",
            properties["subgoal_completion_reason"]["description"],
        )


class ExplorationToolDefinitionsTest(unittest.TestCase):
    """
    Pins the explore_ui and describe_screen schemas used by the exploration scan.
    """

    @staticmethod
    def __names(definitions: Dict[str, Any]) -> List[str]:
        """
        Extracts the declared tool names from a function-declarations payload.
        """

        return [declaration["name"] for declaration in definitions["function_declarations"]]

    def test_exploration_definitions_expose_both_tools(self) -> None:
        """
        The exploration payload exposes explore_ui then describe_screen, in order.
        """

        self.assertEqual(
            self.__names(ToolRegistry.get_exploration_definitions()),
            ["explore_ui", "describe_screen"],
        )

    def test_translation_definitions_expose_describe_screen(self) -> None:
        """
        The standalone translation payload exposes only describe_screen.
        """

        self.assertEqual(
            self.__names(ToolRegistry.get_translation_definitions()),
            ["describe_screen"],
        )

    def test_explore_ui_requires_action_and_carries_exploration_grounding(self) -> None:
        """
        explore_ui requires the action and assistant_message and grounds region/category.
        """

        explore = ToolRegistry.get_exploration_definitions()["function_declarations"][0]

        self.assertIn("action", explore["parameters"]["required"])
        self.assertIn("assistant_message", explore["parameters"]["required"])

        action_properties = explore["parameters"]["properties"]["action"]["properties"]
        self.assertIn("element_category", action_properties)
        self.assertIn("region", action_properties)

    def test_describe_screen_requires_functional_fields(self) -> None:
        """
        describe_screen requires the activity, purpose, elements, and actions fields.
        """

        describe = ToolRegistry.get_translation_definitions()["function_declarations"][0]

        for field in ("activity_name", "screen_purpose", "elements", "achievable_actions"):
            self.assertIn(field, describe["parameters"]["required"])

    def test_describe_screen_constrains_category_to_the_enum(self) -> None:
        """
        describe_screen requires screen_category and limits it to ScreenCategory values.
        """

        describe = ToolRegistry.get_translation_definitions()["function_declarations"][0]
        properties = describe["parameters"]["properties"]

        self.assertIn("screen_category", describe["parameters"]["required"])
        self.assertEqual(
            set(properties["screen_category"]["enum"]),
            {category.value for category in ScreenCategory},
        )
