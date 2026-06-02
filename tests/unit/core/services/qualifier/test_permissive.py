from __future__ import annotations

import unittest

from fathom.constants.qualification import QualificationLabel, RationaleCategory
from fathom.core.services.qualifier.permissive import PermissiveIntentQualifier
from fathom.schemas.configuration import QualifierConfiguration


class PermissiveIntentQualifierTest(unittest.IsolatedAsyncioTestCase):
    """
    Permissive qualifier must never block, regardless of input shape.
    """

    async def asyncSetUp(self) -> None:
        """
        Provide a fresh PermissiveIntentQualifier and the default floor per test.
        """

        self.__qualifier = PermissiveIntentQualifier()
        self.__floor = QualifierConfiguration().confidence

    async def test_returns_executable_for_normal_intent(self) -> None:
        """
        A well-formed intent must be accepted with PERMISSIVE rationale.
        """

        verdict = await self.__qualifier.qualify(intent="Search for McPuff")
        self.assertEqual(verdict.label, QualificationLabel.EXECUTABLE)
        self.assertEqual(verdict.rationale.category, RationaleCategory.PERMISSIVE)
        self.assertFalse(verdict.should_block(floor=self.__floor))

    async def test_returns_executable_for_gibberish(self) -> None:
        """
        Gibberish must still pass the permissive service without inspection.
        """

        verdict = await self.__qualifier.qualify(intent="qwerty asdkjfh")
        self.assertEqual(verdict.label, QualificationLabel.EXECUTABLE)
        self.assertFalse(verdict.should_block(floor=self.__floor))

    async def test_returns_executable_for_question(self) -> None:
        """
        A question must pass through the permissive service unchanged.
        """

        verdict = await self.__qualifier.qualify(intent="what is 2 + 2?")
        self.assertEqual(verdict.label, QualificationLabel.EXECUTABLE)
        self.assertFalse(verdict.should_block(floor=self.__floor))


if __name__ == "__main__":
    unittest.main()
