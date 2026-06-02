from __future__ import annotations

import unittest
from typing import List
from unittest.mock import MagicMock

from fathom.core.services.qualifier import (
    LLMIntentQualifier,
    PermissiveIntentQualifier,
    QualifierComposer,
)
from fathom.interfaces.factory import LLMFactoryPort
from fathom.interfaces.llm import LLMPort
from fathom.runtime.assembly import RunAssemblyBuilder
from fathom.schemas.configuration import LLMConfiguration, QualifierConfiguration
from fathom.settings.env import FathomSettings


class _SpyLLMFactory(LLMFactoryPort):
    """
    LLM factory double that captures configurations and returns mock LLM ports.
    """

    def __init__(self) -> None:
        """
        Initialize the spy with an empty capture list.
        """

        self.captured_configurations: List[LLMConfiguration] = []

    def create(self, *, configuration: LLMConfiguration) -> LLMPort:
        """
        Record the configuration and return a fresh mock LLM port.
        """

        self.captured_configurations.append(configuration)
        return MagicMock(spec=LLMPort)


class _ForbiddenLLMFactory(LLMFactoryPort):
    """
    LLM factory double that fails the test if `create` is ever called.
    """

    def create(self, *, configuration: LLMConfiguration) -> LLMPort:
        """
        Fail loudly if the composer ever asks for an LLM when it should not.
        """

        _ = configuration
        raise AssertionError(
            "LLMFactory.create must not be called when the qualifier is disabled"
        )


class QualifierComposerTest(unittest.TestCase):
    """
    Composer must derive the qualifier LLM from the bound assembly when enabled,
    and skip LLM construction entirely when disabled. This is the regression check
    for the staging bug where a builder-internal FathomSettings saw no credentials.
    """

    def __assembly(self, *, api_key: str = "bound-key") -> RunAssemblyBuilder:
        """
        Build an assembly with fully-credentialed bound settings.
        """

        bound_settings = FathomSettings(
            gemini_api_key=api_key,
            vertex_location="bound-location",
            vertex_project_id="bound-project",
            google_application_credentials="/fake/credentials.json",
        )
        return RunAssemblyBuilder(settings=bound_settings)

    def test_enabled_constructs_dedicated_llm_from_bound_assembly(self) -> None:
        """
        Enabled qualifier configuration must call LLMFactory with a configuration that
        carries the credentials from the assembly's bound settings.
        """

        factory = _SpyLLMFactory()
        composer = QualifierComposer(assembly=self.__assembly(), llm_factory=factory)

        qualifier = composer.compose(
            planner_llm=MagicMock(spec=LLMPort),
            configuration=QualifierConfiguration(),
        )

        self.assertIsInstance(qualifier, LLMIntentQualifier)
        self.assertEqual(len(factory.captured_configurations), 1)

        captured = factory.captured_configurations[0]

        self.assertEqual(captured.temperature, 0.0)
        self.assertEqual(captured.api_key, "bound-key")
        self.assertEqual(captured.project_id, "bound-project")
        self.assertEqual(captured.location, "bound-location")
        self.assertEqual(captured.credentials, "/fake/credentials.json")
        

    def test_disabled_returns_permissive_without_constructing_llm(self) -> None:
        """
        Disabled qualifier configuration must skip LLMFactory and install permissive.
        """

        composer = QualifierComposer(
            assembly=self.__assembly(), llm_factory=_ForbiddenLLMFactory()
        )

        qualifier = composer.compose(
            planner_llm=MagicMock(spec=LLMPort),
            configuration=QualifierConfiguration(enabled=False),
        )

        self.assertIsInstance(qualifier, PermissiveIntentQualifier)

    def test_qualifier_knobs_reach_the_underlying_llm_configuration(self) -> None:
        """
        Temperature, use_cache and thinking_level flow through to the LLM configuration.
        """

        factory = _SpyLLMFactory()
        composer = QualifierComposer(assembly=self.__assembly(), llm_factory=factory)

        composer.compose(
            planner_llm=MagicMock(spec=LLMPort),
            configuration=QualifierConfiguration(
                temperature=0.0,
                use_cache=False,
                thinking_level="minimal",
            ),
        )

        captured = factory.captured_configurations[0]

        self.assertFalse(captured.use_cache)
        self.assertEqual(captured.temperature, 0.0)
        self.assertEqual(captured.thinking_level, "minimal")


if __name__ == "__main__":
    unittest.main()
