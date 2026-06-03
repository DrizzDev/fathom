from __future__ import annotations

import unittest

from fathom.constants.qualification import QualificationLabel, RationaleCategory
from fathom.core.services.qualifier.gate import QualificationGatePolicy
from fathom.core.services.qualifier.permissive import PermissiveIntentQualifier
from fathom.schemas.configuration import QualifierConfiguration


class PermissiveIntentQualifierTest(unittest.IsolatedAsyncioTestCase):
    """
    Permissive qualifier must never produce a verdict the gate would block.
    """

    async def asyncSetUp(self) -> None:
        """
        Provide a fresh permissive qualifier and the default gate policy per test.
        """

        self.__qualifier = PermissiveIntentQualifier()
        self.__policy = QualificationGatePolicy(configuration=QualifierConfiguration())

    async def test_returns_executable_for_normal_intent(self) -> None:
        """
        A well-formed intent must be accepted with PERMISSIVE rationale.
        """

        verdict = await self.__qualifier.qualify(intent="Search for McPuff")
        self.assertEqual(verdict.label, QualificationLabel.EXECUTABLE)
        self.assertEqual(verdict.rationale.category, RationaleCategory.PERMISSIVE)
        self.assertFalse(self.__policy.should_block(verdict=verdict))

    async def test_returns_executable_for_gibberish(self) -> None:
        """
        Gibberish must still pass the permissive service without inspection.
        """

        verdict = await self.__qualifier.qualify(intent="qwerty asdkjfh")
        self.assertEqual(verdict.label, QualificationLabel.EXECUTABLE)
        self.assertFalse(self.__policy.should_block(verdict=verdict))

    async def test_returns_executable_for_question(self) -> None:
        """
        A question must pass through the permissive service unchanged.
        """

        verdict = await self.__qualifier.qualify(intent="what is 2 + 2?")
        self.assertEqual(verdict.label, QualificationLabel.EXECUTABLE)
        self.assertFalse(self.__policy.should_block(verdict=verdict))


if __name__ == "__main__":
    unittest.main()
