from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock

from fathom.core.services.qualifier import (
    IntentQualifierFactory,
    LLMIntentQualifier,
    PermissiveIntentQualifier,
)
from fathom.interfaces.llm import LLMPort
from fathom.schemas.configuration import QualifierConfiguration
from fathom.schemas.results import GenerateResult


class _NoOpLLM(LLMPort):
    """
    LLMPort double that should never be touched by the permissive branch.
    """

    @property
    def model_name(self) -> str:
        """
        Return a static identifier so logs stay informative.
        """

        return "no-op-test-llm"

    async def generate(self, **_: Any) -> GenerateResult:
        """
        Fail loudly if the factory ever wires this LLM into a permissive qualifier.
        """

        raise AssertionError("LLM.generate must not be called on the permissive branch")

    async def cleanup(self) -> None:
        """
        Honour the lifecycle contract with no-op semantics.
        """

        return None


class IntentQualifierFactoryTest(unittest.TestCase):
    """
    Factory must select LLM-backed when enabled and permissive when disabled,
    and must NOT touch the LLM on the disabled branch.
    """

    def test_enabled_returns_llm_backed_wrapping_supplied_llm(self) -> None:
        """
        With enabled=True the factory returns an LLMIntentQualifier bound to the supplied LLM.
        """

        llm = MagicMock(spec=LLMPort)
        qualifier = IntentQualifierFactory.create(
            llm=llm, configuration=QualifierConfiguration(enabled=True)
        )
        self.assertIsInstance(qualifier, LLMIntentQualifier)

    def test_disabled_returns_permissive_and_never_touches_llm(self) -> None:
        """
        With enabled=False the factory returns a permissive qualifier and ignores the LLM.
        """

        qualifier = IntentQualifierFactory.create(
            llm=_NoOpLLM(), configuration=QualifierConfiguration(enabled=False)
        )
        self.assertIsInstance(qualifier, PermissiveIntentQualifier)

    def test_disabled_default_configuration_returns_llm_backed(self) -> None:
        """
        Default QualifierConfiguration has enabled=True so the factory builds the LLM-backed impl.
        """

        llm = MagicMock(spec=LLMPort)
        qualifier = IntentQualifierFactory.create(
            llm=llm, configuration=QualifierConfiguration()
        )
        self.assertIsInstance(qualifier, LLMIntentQualifier)


if __name__ == "__main__":
    unittest.main()
