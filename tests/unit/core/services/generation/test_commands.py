from __future__ import annotations

import unittest

from fathom.adapters.dialect.drizz.factory import DrizzDialectFactory
from fathom.constants.generation import ScriptCommandRole
from fathom.core.services.generation.commands import ScriptCommandBuilder
from fathom.schemas.flow import BranchNode, Flow, Guard, Selector, TapNode


class ScriptCommandBuilderTest(unittest.TestCase):
    """
    Covers rendered command metadata used by script fallback composition.
    """

    def test_branch_command_carries_guard_and_body_source_steps(self) -> None:
        """
        A rendered IF command represents both its guard and conditional body steps.
        """

        flow = Flow(
            intent="login",
            package="com.example",
            nodes=(
                BranchNode(
                    source_steps=(1,),
                    guard=Guard(
                        condition="save password prompt is visible",
                        source_step=1,
                    ),
                    body=(
                        TapNode(
                            source_steps=(2,),
                            selector=Selector(text="Not now button"),
                        ),
                    ),
                ),
            ),
        )

        commands = ScriptCommandBuilder(dialect=DrizzDialectFactory().create()).build(flow=flow)

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0].source_steps, (1, 2))
        self.assertEqual(commands[0].role, ScriptCommandRole.BRANCH)
