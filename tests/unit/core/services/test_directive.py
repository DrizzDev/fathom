from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.core.services.directive import DirectivePolicy
from fathom.schemas.subgoal import SubGoalKind


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
