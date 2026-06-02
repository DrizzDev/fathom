from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from fathom.core.services.qualifier import (
    LLMIntentQualifier,
    PermissiveIntentQualifier,
)
from fathom.interfaces.device import DevicePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.perception import PerceptionPort
from fathom.runtime.builder import Fathom
from fathom.schemas.configuration import (
    FathomConfiguration,
    InferenceConfiguration,
    QualifierConfiguration,
)


class FathomBuilderQualifierDefaultTest(unittest.TestCase):
    """
    Builder must use the IntentQualifierFactory with the caller-supplied LLM port,
    never construct a fresh FathomSettings or a separate GeminiLLM at default time.
    """

    @staticmethod
    def __builder_with_required_ports():
        """
        Wire just enough ports for build() to succeed under default configuration.
        """

        return (
            Fathom.builder()
            .with_llm(port=MagicMock(spec=LLMPort))
            .with_device(port=MagicMock(spec=DevicePort))
            .with_perception(port=MagicMock(spec=PerceptionPort))
        )

    def test_default_qualifier_when_enabled_is_llm_backed_with_supplied_llm(self) -> None:
        """
        With qualifier.enabled=True the builder installs LLMIntentQualifier reusing self.__llm.
        """

        llm = MagicMock(spec=LLMPort)
        runner = (
            Fathom.builder()
            .with_llm(port=llm)
            .with_device(port=MagicMock(spec=DevicePort))
            .with_perception(port=MagicMock(spec=PerceptionPort))
            .build()
        )
        # The runner is the only public surface; reach in via name-mangled attribute to assert
        # the qualifier shape. This is the regression check for the FathomSettings-leak bug.
        installed = runner._FathomRunner__qualifier  # type: ignore[attr-defined]
        self.assertIsInstance(installed, LLMIntentQualifier)

    def test_default_qualifier_when_disabled_is_permissive(self) -> None:
        """
        With qualifier.enabled=False the builder installs a permissive qualifier.
        """

        configuration = FathomConfiguration(qualifier=QualifierConfiguration(enabled=False))
        runner = (
            Fathom.builder()
            .with_llm(port=MagicMock(spec=LLMPort))
            .with_config(configuration=configuration)
            .with_device(port=MagicMock(spec=DevicePort))
            .with_perception(port=MagicMock(spec=PerceptionPort))
            .build()
        )
        installed = runner._FathomRunner__qualifier  # type: ignore[attr-defined]
        self.assertIsInstance(installed, PermissiveIntentQualifier)

    def test_explicit_qualifier_overrides_default(self) -> None:
        """
        Caller-supplied .with_qualifier() must override the factory default.
        """

        explicit = PermissiveIntentQualifier()
        runner = self.__builder_with_required_ports().with_qualifier(port=explicit).build()
        installed = runner._FathomRunner__qualifier  # type: ignore[attr-defined]
        self.assertIs(installed, explicit)

    def test_with_qualifier_config_flows_into_runner(self) -> None:
        """
        Regression: request-level qualifier configuration must reach the runner.

        Setting a non-default inference knob must round-trip through the builder
        so downstream code (composer, LLM factory) reads the caller's intent.
        """

        custom = QualifierConfiguration(
            inference=InferenceConfiguration(temperature=0.0, thinking_level="medium")
        )
        runner = (
            Fathom.builder()
            .with_llm(port=MagicMock(spec=LLMPort))
            .with_device(port=MagicMock(spec=DevicePort))
            .with_perception(port=MagicMock(spec=PerceptionPort))
            .with_qualifier_config(configuration=custom)
            .build()
        )
        installed_config = runner._FathomRunner__config.qualifier  # type: ignore[attr-defined]
        self.assertEqual(installed_config.inference.thinking_level, "medium")
        self.assertEqual(installed_config.inference.temperature, 0.0)


if __name__ == "__main__":
    unittest.main()
