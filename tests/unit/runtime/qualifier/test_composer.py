from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

from fathom.constants.run import ExecutionMode
from fathom.core.services.qualifier import (
    LLMIntentQualifier,
    PermissiveIntentQualifier,
)
from fathom.interfaces.factory import LLMFactoryPort
from fathom.interfaces.llm import LLMPort
from fathom.runtime.assembly import RunAssemblyBuilder
from fathom.runtime.qualifier import QualifierComposer
from fathom.schemas.composition import QualifierComposition
from fathom.schemas.configuration import (
    LLMConfiguration,
    QualifierConfiguration,
)
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
        raise AssertionError("LLMFactory.create must not be called when the qualifier is disabled")


class _RaisingLLMFactory(LLMFactoryPort):
    """
    LLM factory double that simulates a construction failure such as Vertex auth refusal.
    """

    def create(self, *, configuration: LLMConfiguration) -> LLMPort:
        """
        Raise a vendor-shaped exception to exercise the composer's failure-log path.
        """

        _ = configuration
        raise RuntimeError("simulated vertex auth failure")


class QualifierComposerTest(unittest.IsolatedAsyncioTestCase):
    """
    Composer must return a QualifierComposition with the qualifier port and any
    infrastructure resources it created. The composition root drains the
    resources after the run; the qualifier itself never owns lifecycle.
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

    async def test_enabled_constructs_dedicated_llm_and_returns_it_as_resource(self) -> None:
        """
        Enabled qualifier configuration must build a dedicated LLM, install it
        on the qualifier, AND register it in composition.resources so the
        composition root can close it after the run.
        """

        factory = _SpyLLMFactory()
        composer = QualifierComposer(assembly=self.__assembly(), llm_factory=factory)

        composition = await composer.compose(
            planner_llm=MagicMock(spec=LLMPort),
            configuration=QualifierConfiguration(),
        )

        self.assertIsInstance(composition, QualifierComposition)
        self.assertIsInstance(composition.qualifier, LLMIntentQualifier)
        self.assertIsInstance(composition.resources, tuple)
        self.assertEqual(len(composition.resources), 1)
        self.assertEqual(len(factory.captured_configurations), 1)

        captured = factory.captured_configurations[0]
        self.assertEqual(captured.temperature, 0.0)
        self.assertEqual(captured.api_key, "bound-key")
        self.assertEqual(captured.project_id, "bound-project")
        self.assertEqual(captured.location, "bound-location")
        self.assertEqual(captured.credentials, "/fake/credentials.json")

    async def test_disabled_returns_permissive_with_no_resources(self) -> None:
        """
        Disabled qualifier configuration must skip LLMFactory entirely and
        return a composition with zero owned resources.
        """

        composer = QualifierComposer(assembly=self.__assembly(), llm_factory=_ForbiddenLLMFactory())

        composition = await composer.compose(
            planner_llm=MagicMock(spec=LLMPort),
            configuration=QualifierConfiguration(enabled=False),
        )

        self.assertIsInstance(composition, QualifierComposition)
        self.assertIsInstance(composition.qualifier, PermissiveIntentQualifier)
        self.assertEqual(composition.resources, ())

    async def test_llm_construction_failure_logs_qualifier_context_and_propagates(self) -> None:
        """
        LLM construction failures must be logged with qualifier-specific context
        and re-raised so callers fail fast instead of receiving an opaque error.
        """

        composer = QualifierComposer(assembly=self.__assembly(), llm_factory=_RaisingLLMFactory())

        with (
            self.assertLogs("fathom.runtime.qualifier.composer", level="WARNING") as captured,
            self.assertRaises(RuntimeError),
        ):
            await composer.compose(
                planner_llm=MagicMock(spec=LLMPort),
                configuration=QualifierConfiguration(),
            )

        self.assertTrue(
            any(
                "Dedicated qualifier LLM construction failed" in record
                for record in captured.output
            ),
            msg=f"Expected construction-failed log; got {captured.output}",
        )

    async def test_compose_emits_composed_log_with_implementation_details(self) -> None:
        """
        Every compose() call must emit a composed INFO log so we can verify in
        production which qualifier wiring is actually in use.
        """

        composer = QualifierComposer(assembly=self.__assembly(), llm_factory=_SpyLLMFactory())

        with self.assertLogs("fathom.runtime.qualifier.composer", level="INFO") as captured:
            await composer.compose(
                planner_llm=MagicMock(spec=LLMPort),
                configuration=QualifierConfiguration(),
            )

        self.assertTrue(
            any("Composed qualifier port" in record for record in captured.output),
            msg=f"Expected composed-qualifier log; got {captured.output}",
        )

    async def test_qualifier_knobs_reach_the_underlying_llm_configuration(self) -> None:
        """
        Temperature, use_cache and thinking_level flow through to the LLM configuration.
        """

        factory = _SpyLLMFactory()
        composer = QualifierComposer(assembly=self.__assembly(), llm_factory=factory)

        await composer.compose(
            planner_llm=MagicMock(spec=LLMPort),
            configuration=QualifierConfiguration.evolve(thinking_level="minimal"),
        )

        captured = factory.captured_configurations[0]
        self.assertEqual(captured.thinking_level, "minimal")

    async def test_post_construction_compose_failure_cleans_dedicated_llm(self) -> None:
        """
        If compose() fails after creating a dedicated qualifier LLM, ownership
        never reaches the caller. The composer must close that LLM itself.
        """

        dedicated_llm = MagicMock(spec=LLMPort)
        dedicated_llm.cleanup = AsyncMock()

        factory = _SpyLLMFactory()
        factory.create = MagicMock(return_value=dedicated_llm)
        composer = QualifierComposer(assembly=self.__assembly(), llm_factory=factory)

        with (
            patch(
                "fathom.runtime.qualifier.composer.IntentQualifierFactory.create",
                side_effect=RuntimeError("qualifier construction failed"),
            ),
            self.assertRaises(RuntimeError),
        ):
            await composer.compose(
                planner_llm=MagicMock(spec=LLMPort),
                configuration=QualifierConfiguration(),
            )

        dedicated_llm.cleanup.assert_awaited_once_with()

    async def test_post_construction_compose_cancellation_cleans_dedicated_llm(self) -> None:
        """Regression: CancelledError is BaseException-derived in Python 3.8+, so catching only Exception would leak the dedicated qualifier LLM if a future refactor introduces an await inside compose()'s try block and the activity is cancelled at that point."""

        dedicated_llm = MagicMock(spec=LLMPort)
        dedicated_llm.cleanup = AsyncMock()

        factory = _SpyLLMFactory()
        factory.create = MagicMock(return_value=dedicated_llm)
        composer = QualifierComposer(assembly=self.__assembly(), llm_factory=factory)

        with (
            patch(
                "fathom.runtime.qualifier.composer.IntentQualifierFactory.create",
                side_effect=asyncio.CancelledError(),
            ),
            self.assertRaises(asyncio.CancelledError),
        ):
            await composer.compose(
                planner_llm=MagicMock(spec=LLMPort),
                configuration=QualifierConfiguration(),
            )

        dedicated_llm.cleanup.assert_awaited_once_with()


class QualifierComposerShouldComposeTest(unittest.TestCase):
    """
    should_compose is the single, request-driven decision both Temporal and CLI
    use to decide whether to build a dedicated qualifier. Mismatch here is
    exactly the CLI-exploration parity bug — keep this test as the contract.
    """

    @staticmethod
    def __request(*, mode: ExecutionMode, enabled: bool) -> SimpleNamespace:
        """
        Build a minimal RunRequest-shaped object for the decision.
        """

        return SimpleNamespace(
            objective=SimpleNamespace(mode=mode),
            interaction=SimpleNamespace(
                qualifier_configuration=QualifierConfiguration(enabled=enabled)
            ),
        )

    def test_intent_with_enabled_qualifier_returns_true(self) -> None:
        """
        Intent runs whose qualifier configuration is enabled must opt into composition.
        """

        request = self.__request(mode=ExecutionMode.INTENT, enabled=True)
        self.assertTrue(QualifierComposer.should_compose(request=request))

    def test_intent_with_disabled_qualifier_returns_false(self) -> None:
        """
        Intent runs that explicitly disable qualification fall through to the
        builder's permissive default; the composer must not build a dedicated LLM.
        """

        request = self.__request(mode=ExecutionMode.INTENT, enabled=False)
        self.assertFalse(QualifierComposer.should_compose(request=request))

    def test_exploration_never_composes(self) -> None:
        """
        Exploration runs never qualify — neither enabled=True nor enabled=False
        on an exploration request must trigger composition. This is the
        regression check for the CLI parity bug.
        """

        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                request = self.__request(mode=ExecutionMode.EXPLORATION, enabled=enabled)
                self.assertFalse(QualifierComposer.should_compose(request=request))


if __name__ == "__main__":
    unittest.main()
