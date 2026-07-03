from __future__ import annotations

import unittest

from fathom.adapters.dialect.drizz.factory import DrizzDialectFactory
from fathom.constants.dialect import DialectName
from fathom.constants.flow import CheckKind
from fathom.core.dialect.dialect import DialectRegistry
from fathom.schemas.flow import Check, CheckNode, Flow, LaunchNode


class DrizzDialectFactoryTest(unittest.TestCase):
    """
    Cover assembling the Drizz dialect and resolving it through the registry.
    """

    def test_factory_builds_resolvable_dialect_that_round_trips(self) -> None:
        """
        The factory yields a registry-resolvable dialect whose render output passes its checker.
        """

        registry = DialectRegistry()
        registry.register(dialect=DrizzDialectFactory().create())
        dialect = registry.resolve(name=DialectName.DRIZZ)

        self.assertEqual(dialect.name, DialectName.DRIZZ)

        flow = Flow(
            intent="open and verify",
            package="com.example",
            nodes=(
                LaunchNode(package="com.example", source_steps=(0,)),
                CheckNode(
                    checks=(Check(kind=CheckKind.VISIBLE, subject="home"),), source_steps=(1,)
                ),
            ),
        )
        text = dialect.renderer.render(flow=flow)
        self.assertTrue(dialect.checker.check(text=text).ok)
