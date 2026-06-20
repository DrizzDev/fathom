from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock

from fathom.adapters.interaction.noop import NoopInteraction
from fathom.constants.storage import StorageBackend
from fathom.core.services.qualifier import (
    LLMIntentQualifier,
    PermissiveIntentQualifier,
)
from fathom.interfaces.device import DevicePort
from fathom.interfaces.factory import LLMFactoryPort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.perception import PerceptionPort
from fathom.interfaces.storage import StoragePort
from fathom.runtime.assembly import RunAssemblyBuilder
from fathom.runtime.builder import Fathom
from fathom.schemas.configuration import (
    FathomConfiguration,
    LLMConfiguration,
    QualifierConfiguration,
    StorageConfiguration,
)
from fathom.settings.env import FathomSettings


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
            .with_interaction(port=NoopInteraction())
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
            .with_interaction(port=NoopInteraction())
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
            .with_interaction(port=NoopInteraction())
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

        custom = QualifierConfiguration.evolve(thinking_level="medium")
        runner = (
            Fathom.builder()
            .with_llm(port=MagicMock(spec=LLMPort))
            .with_device(port=MagicMock(spec=DevicePort))
            .with_perception(port=MagicMock(spec=PerceptionPort))
            .with_qualifier_config(configuration=custom)
            .with_interaction(port=NoopInteraction())
            .build()
        )
        installed_config = runner._FathomRunner__config.qualifier  # type: ignore[attr-defined]
        self.assertEqual(installed_config.inference.thinking_level, "medium")
        self.assertEqual(installed_config.inference.temperature, 0.0)


class _SpyLLMFactory(LLMFactoryPort):
    """
    LLM factory double that captures every configuration it receives and
    returns a fresh mock LLM. Used to assert with_assembly() routes the
    inference knobs through assembly.build_qualifier_model_configuration.
    """

    def __init__(self) -> None:
        """
        Start with an empty capture log.
        """

        self.captured: list[LLMConfiguration] = []

    def create(self, *, configuration: LLMConfiguration) -> LLMPort:
        """
        Record the configuration and return a fresh mock LLM port.
        """

        self.captured.append(configuration)
        return MagicMock(spec=LLMPort)


class FathomBuilderWithAssemblyTest(unittest.TestCase):
    """
    .with_assembly() is the SDK-level escape hatch that makes
    QualifierConfiguration.inference.{model, timeout, max_retries, ...} actually
    take effect. Without it, the builder falls back to running the qualifier on
    the planner LLM and ignores the inference block — the documented prior
    behavior that Enricher silently inherited.
    """

    @staticmethod
    def __assembly() -> RunAssemblyBuilder:
        """
        Build a credentialed assembly for the qualifier-model builder to read.
        """

        return RunAssemblyBuilder(
            settings=FathomSettings(
                gemini_api_key="test-key",
                vertex_location="bound-location",
                vertex_project_id="bound-project",
            )
        )

    def test_with_assembly_builds_dedicated_qualifier_llm_via_factory(self) -> None:
        """
        With assembly supplied, the builder must build a NEW LLM via the factory
        using the qualifier model configuration — not reuse the planner LLM.
        """

        planner = MagicMock(spec=LLMPort)
        factory = _SpyLLMFactory()

        runner = (
            Fathom.builder()
            .with_llm(port=planner)
            .with_device(port=MagicMock(spec=DevicePort))
            .with_perception(port=MagicMock(spec=PerceptionPort))
            .with_assembly(assembly=self.__assembly(), llm_factory=factory)
            .with_interaction(port=NoopInteraction())
            .build()
        )

        # Factory called exactly once with the qualifier's model defaults.
        self.assertEqual(len(factory.captured), 1)
        captured = factory.captured[0]
        self.assertEqual(captured.model, QualifierConfiguration().inference.model)
        self.assertEqual(captured.timeout, QualifierConfiguration().inference.timeout)
        self.assertEqual(captured.max_retries, QualifierConfiguration().inference.max_retries)

        # The runner's qualifier LLM is NOT the planner; it's the dedicated one.
        qualifier = runner._FathomRunner__qualifier  # type: ignore[attr-defined]
        self.assertIsInstance(qualifier, LLMIntentQualifier)

        # Runner takes ownership: the dedicated LLM is in owned_resources so
        # runner.cleanup() will close it without the SDK caller tracking it.
        owned = runner._FathomRunner__owned_resources  # type: ignore[attr-defined]
        self.assertEqual(len(owned), 1)

    def test_without_assembly_qualifier_runs_on_planner_llm(self) -> None:
        """
        Backward compat: with no assembly, the qualifier reuses the planner LLM
        and inference.* settings DO NOT take effect. Owned resources is empty —
        the planner LLM lifecycle stays with the caller-supplied port.
        """

        planner = MagicMock(spec=LLMPort)
        runner = (
            Fathom.builder()
            .with_llm(port=planner)
            .with_device(port=MagicMock(spec=DevicePort))
            .with_perception(port=MagicMock(spec=PerceptionPort))
            .with_interaction(port=NoopInteraction())
            .build()
        )

        owned = runner._FathomRunner__owned_resources  # type: ignore[attr-defined]
        self.assertEqual(owned, [])

    def test_with_assembly_respects_explicit_qualifier_override(self) -> None:
        """
        .with_qualifier(port=...) takes precedence over .with_assembly() — the
        factory must not be invoked at all if the caller injected their own
        qualifier port.
        """

        explicit = PermissiveIntentQualifier()
        factory = _SpyLLMFactory()

        runner = (
            Fathom.builder()
            .with_llm(port=MagicMock(spec=LLMPort))
            .with_device(port=MagicMock(spec=DevicePort))
            .with_perception(port=MagicMock(spec=PerceptionPort))
            .with_assembly(assembly=self.__assembly(), llm_factory=factory)
            .with_qualifier(port=explicit)
            .with_interaction(port=NoopInteraction())
            .build()
        )

        self.assertEqual(len(factory.captured), 0)
        self.assertIs(runner._FathomRunner__qualifier, explicit)  # type: ignore[attr-defined]
        self.assertEqual(runner._FathomRunner__owned_resources, [])  # type: ignore[attr-defined]

    def test_with_assembly_and_disabled_qualifier_skips_factory(self) -> None:
        """
        When qualifier is disabled the builder must install the permissive
        qualifier and skip dedicated LLM construction entirely — no factory
        call, no owned resource.
        """

        factory = _SpyLLMFactory()
        runner = (
            Fathom.builder()
            .with_llm(port=MagicMock(spec=LLMPort))
            .with_config(
                configuration=FathomConfiguration(
                    qualifier=QualifierConfiguration(enabled=False),
                )
            )
            .with_device(port=MagicMock(spec=DevicePort))
            .with_perception(port=MagicMock(spec=PerceptionPort))
            .with_assembly(assembly=self.__assembly(), llm_factory=factory)
            .with_interaction(port=NoopInteraction())
            .build()
        )

        self.assertEqual(len(factory.captured), 0)
        self.assertIsInstance(
            runner._FathomRunner__qualifier,  # type: ignore[attr-defined]
            PermissiveIntentQualifier,
        )

    def test_with_assembly_passes_evolved_inference_to_factory(self) -> None:
        """
        Custom inference knobs via QualifierConfiguration.evolve must reach the
        dedicated LLM configuration the factory receives.
        """

        custom = QualifierConfiguration.evolve(model="gemini-2.5-flash", timeout=8.0)
        factory = _SpyLLMFactory()

        Fathom.builder().with_llm(port=MagicMock(spec=LLMPort)).with_device(
            port=MagicMock(spec=DevicePort)
        ).with_perception(port=MagicMock(spec=PerceptionPort)).with_qualifier_config(
            configuration=custom
        ).with_assembly(
            assembly=self.__assembly(), llm_factory=factory
        ).with_interaction(
            port=NoopInteraction()
        ).build()

        captured = factory.captured[0]
        self.assertEqual(captured.model, "gemini-2.5-flash")
        self.assertEqual(captured.timeout, 8.0)


class FathomBuilderStorageUnificationTest(unittest.TestCase):
    """
    Builder must keep its two storage state slots in sync.

    Before the unification ``with_storage`` set only ``self.__storage`` (the
    port). The artifact pipeline reads ``self.__config.storage`` instead, so a
    deployment that wired a multi-backend storage port still got an artifact
    pipeline backed by the un-updated ``FathomConfiguration.storage`` default
    — and ``EfsSink`` was selected on every stag run, silently dropping every
    screenshot/annotated/trace upload while history uploads (which use the
    port directly) continued to land in GCS.
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
            .with_interaction(port=NoopInteraction())
        )

    def test_with_storage_propagates_configuration_into_runner_config(self) -> None:
        """
        ``with_storage`` must update both the port AND ``__config.storage``
        so the artifact-pipeline sink selector reads the operator's chosen
        backends — not the StorageConfiguration field default.
        """

        configuration = StorageConfiguration(
            backends={StorageBackend.LOCAL, StorageBackend.CLOUD},
            storage_bucket="example-bucket",
        )

        runner = (
            self.__builder_with_required_ports()
            .with_storage(port=MagicMock(spec=StoragePort), configuration=configuration)
            .build()
        )

        installed_storage_config = runner._FathomRunner__config.storage  # type: ignore[attr-defined]
        self.assertEqual(
            installed_storage_config.backends,
            {StorageBackend.LOCAL, StorageBackend.CLOUD},
        )
        self.assertEqual(installed_storage_config.storage_bucket, "example-bucket")

    def test_with_storage_port_and_config_reference_the_same_intent(self) -> None:
        """
        The port and the configuration must reflect a single operator decision.
        Passing them together at the same call site is what makes them
        impossible to drift apart by accident.
        """

        port = MagicMock(spec=StoragePort)
        configuration = StorageConfiguration(
            backends={StorageBackend.CLOUD},
            storage_bucket="cloud-only-bucket",
        )

        runner = (
            self.__builder_with_required_ports()
            .with_storage(port=port, configuration=configuration)
            .build()
        )

        self.assertIs(runner._FathomRunner__storage, port)  # type: ignore[attr-defined]
        self.assertEqual(
            runner._FathomRunner__config.storage.backends,  # type: ignore[attr-defined]
            {StorageBackend.CLOUD},
        )
        self.assertEqual(
            runner._FathomRunner__config.storage.storage_bucket,  # type: ignore[attr-defined]
            "cloud-only-bucket",
        )


_ = Any  # exported for downstream extensions


if __name__ == "__main__":
    unittest.main()
