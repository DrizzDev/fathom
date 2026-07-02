from __future__ import annotations

import unittest
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from fathom.interfaces.factory import LLMFactoryPort
from fathom.interfaces.llm import LLMPort
from fathom.runtime.command.executor import CommandExecutor
from fathom.settings.env import FathomSettings


class _RecordingLLMFactory(LLMFactoryPort):
    """
    LLM factory double that records every create() call so the test can verify
    whether the executor asked for any LLM at all on a given build path.
    """

    def __init__(self) -> None:
        """
        Initialize the recording factory with an empty call log.
        """

        self.calls: list[Any] = []

    def create(self, *, configuration: Any) -> LLMPort:
        """
        Record the configuration and return a fresh mock LLM port.
        """

        self.calls.append(configuration)
        return MagicMock(spec=LLMPort)


class _InteractionLikeResource:
    """
    Resource shaped like InteractionPort for partial-build teardown tests.
    """

    def __init__(self) -> None:
        """
        Initialize teardown tracking mocks.
        """

        self.aclose = AsyncMock()

    async def cleanup(self, *, request: object) -> None:
        """
        Domain cleanup operation; must not be used for teardown.
        """

        _ = request
        raise AssertionError("Domain cleanup must not be used as teardown")


class CommandExecutorQualifierParityTest(unittest.IsolatedAsyncioTestCase):
    """
    Regression: CLI exploration must NOT instantiate a dedicated qualifier
    LLM. Previously the executor unconditionally called QualifierComposer
    even for exploration runs, paying for / failing on infrastructure with
    no intent to gate.
    """

    @staticmethod
    def __executor(*, llm_factory: LLMFactoryPort) -> CommandExecutor:
        """
        Build a CommandExecutor with the recording LLM factory and minimal
        other factories — the test only exercises the build decision, so
        downstream factories return mocks that never get used.
        """

        return CommandExecutor(
            settings=FathomSettings(),
            llm_factory=llm_factory,
            device_factory=MagicMock(),
            signal_factory=MagicMock(),
            telemetry_factory=MagicMock(),
            perception_factory=MagicMock(),
            device_defaults_resolver=MagicMock(),
        )

    @staticmethod
    def __exploration_request() -> Any:
        """
        Build a minimal exploration request shape.
        """

        from fathom.schemas.run import ExplorationRunRequest

        return ExplorationRunRequest.model_validate(
            {
                "principal": {
                    "tenant": "tenant-test",
                    "agent": "agent:fathom",
                    "operator": "operator-test",
                    "conversation": "conversation-test",
                },
                "resources": {"targets": [{}]},
                "runtime": {"interactive": False},
                "objective": {"mode": "EXPLORATION"},
            }
        )

    async def test_exploration_request_does_not_invoke_qualifier_composer(self) -> None:
        """
        With an exploration request, the executor must not call
        QualifierComposer.compose() at all. Only the planner LLM is
        constructed; no dedicated qualifier LLM is built.
        """

        llm_factory = _RecordingLLMFactory()
        executor = self.__executor(llm_factory=llm_factory)

        with (
            patch("fathom.runtime.command.executor.QualifierComposer") as composer_class,
            patch("fathom.runtime.command.executor.RunAssemblyBuilder") as assembly_class,
        ):
            assembly = MagicMock()
            assembly.build_device_configuration = MagicMock(return_value=MagicMock())
            assembly.build_telemetry_configuration = MagicMock(return_value=MagicMock())
            assembly.build_planner_model_configuration = MagicMock(return_value=MagicMock())
            assembly.build_interaction_storage_configuration = MagicMock(return_value=MagicMock())

            assembly_class.return_value = assembly

            composer_instance = MagicMock()
            composer_class.return_value = composer_instance
            composer_class.should_compose = MagicMock(return_value=False)
            composer_instance.compose = MagicMock(
                side_effect=AssertionError(
                    "QualifierComposer.compose must not be called on exploration runs"
                )
            )

            # The factory will satisfy planner LLM creation; we don't run the
            # full build flow because device/perception/etc. are mocked.
            interaction_adapter = MagicMock()
            interaction_adapter.initialize = AsyncMock()

            with patch("fathom.runtime.command.executor.InteractionFactory") as interaction_factory:
                interaction_factory.return_value.create.return_value = interaction_adapter

                build_exception: Optional[BaseException] = None
                try:
                    await executor._CommandExecutor__create_runner(  # type: ignore[attr-defined]
                        signal_adapter=MagicMock(),
                        request=self.__exploration_request(),
                    )
                except Exception as exception:
                    # Some downstream constructors may fail under mocks; the
                    # assertion under test is the composer not being called.
                    build_exception = exception

            if interaction_adapter.initialize.await_count != 1:
                self.fail(
                    "Command runner build did not reach interaction initialization: "
                    f"{build_exception!r}"
                )

            composer_instance.compose.assert_not_called()

    async def test_partial_build_prefers_aclose_over_domain_cleanup(self) -> None:
        """
        CLI partial-build teardown must call aclose() before cleanup(request=...).
        """

        adapter = _InteractionLikeResource()

        await CommandExecutor._CommandExecutor__drain_partial_resources(  # type: ignore[attr-defined]
            resources=[adapter]
        )

        adapter.aclose.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
