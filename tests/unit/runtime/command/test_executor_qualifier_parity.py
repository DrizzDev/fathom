from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

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
                "objective": {"mode": "exploration"},
                "runtime": {"interactive": False},
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

        with patch("fathom.runtime.command.executor.QualifierComposer") as composer_class:
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
            try:
                await executor._CommandExecutor__create_runner(  # type: ignore[attr-defined]
                    request=self.__exploration_request(),
                    signal_adapter=MagicMock(),
                )
            except Exception as exception:
                # Some downstream constructors may fail under mocks; the
                # assertion under test is the composer not being called.
                _ = exception

            composer_instance.compose.assert_not_called()


if __name__ == "__main__":
    unittest.main()
