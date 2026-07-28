from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.constants.capability import CompletionMode
from fathom.constants.subgoal import TaskProof
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.core.services.directive import DirectivePolicy
from fathom.schemas.subgoal import SubGoal, SubGoalKind


class DirectivePolicyTest(unittest.TestCase):
    """
    Golden master for the directive -> SubGoalKind projection: VALIDATE validates; every other
    directive and the legacy None directive map to ACTION.
    """

    def __policy(self) -> DirectivePolicy:
        """
        Build a directive policy over the full command catalog.
        """

        return DirectivePolicy(catalog=CommandCatalogProvider().build())

    def test_only_validate_directive_projects_to_validation(self) -> None:
        """
        Only the validate directive projects to a VALIDATION sub-goal kind; the rest project to ACTION.
        """

        policy = self.__policy()

        for action_type in ActionType:
            with self.subTest(action_type=action_type):
                expected = (
                    SubGoalKind.VALIDATION
                    if action_type is ActionType.VALIDATE
                    else SubGoalKind.ACTION
                )
                self.assertEqual(policy.kind(directive=action_type), expected)

    def test_missing_directive_defaults_to_action(self) -> None:
        """
        A legacy decomposition with no directive defaults to the ACTION kind.
        """

        self.assertEqual(self.__policy().kind(directive=None), SubGoalKind.ACTION)

    def test_projects_store_sub_goal_to_capture_verified_task(self) -> None:
        """
        A STORE-directed sub-goal projects to a task whose proof requirement is CAPTURE_VERIFIED.
        """

        task = self.__policy().project(
            sub_goal=SubGoal(
                description="Store the visible order id",
                index=2,
                directive=ActionType.STORE,
            )
        )

        self.assertEqual(task.index, 2)
        self.assertEqual(task.completion, CompletionMode.CAPTURE_VERIFIED)

    def test_projects_directiveless_sub_goal_to_screen_verified_task(self) -> None:
        """
        A legacy sub-goal without a directive projects to the strict screen-verified requirement.
        """

        task = self.__policy().project(sub_goal=SubGoal(description="Open the notes list", index=0))

        self.assertEqual(task.completion, CompletionMode.SCREEN_VERIFIED)
        self.assertEqual(task.kind, SubGoalKind.ACTION)

    def test_durable_proof_projects_to_outcome_verified_task(self) -> None:
        """
        The decomposer's durable bit demands observed outcome proof for an ambiguous TAP task.
        """

        task = self.__policy().project(
            sub_goal=SubGoal(
                description="Add Diet Coke to the cart",
                index=1,
                directive=ActionType.TAP,
                proof=TaskProof.DURABLE,
            )
        )

        self.assertEqual(task.completion, CompletionMode.OUTCOME_VERIFIED)

    def test_transient_proof_keeps_the_screen_verified_path(self) -> None:
        """
        A declared transient TAP keeps the cheap screen-verified requirement.
        """

        task = self.__policy().project(
            sub_goal=SubGoal(
                description="Open the cart",
                index=0,
                directive=ActionType.TAP,
                proof=TaskProof.TRANSIENT,
            )
        )

        self.assertEqual(task.completion, CompletionMode.SCREEN_VERIFIED)

    def test_durable_proof_never_overrides_the_capture_family(self) -> None:
        """
        STORE keeps its capture-verified requirement regardless of the declared bit.
        """

        task = self.__policy().project(
            sub_goal=SubGoal(
                description="Store the visible order id",
                index=2,
                directive=ActionType.STORE,
                proof=TaskProof.DURABLE,
            )
        )

        self.assertEqual(task.completion, CompletionMode.CAPTURE_VERIFIED)
