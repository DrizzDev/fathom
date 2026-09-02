"""
Reasoning effort: a provider-neutral inference control.

Some frontier models accept function tools only when reasoning is explicitly switched
off. Fathom sent no reasoning parameter at all, so every tool-calling request to those
models was rejected outright and the harness could not drive them. The alternative to
this control was a per-model conditional inside the adapter, which encodes a vendor's
release schedule in Fathom's control flow and needs editing on every launch.

Two distinctions carry the design and are pinned here: *unset* is not *none* - one
omits the parameter and takes whatever the provider defaults to, the other pins a level
- and the value that reaches provenance is the one that applied, not the one requested.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase, mock

from pydantic import ValidationError

from fathom.adapters.llm.openai import OpenAILLM
from fathom.constants.llm import ReasoningEffort
from fathom.schemas.configuration import LLMConfiguration
from fathom.settings.env import FathomSettings


class _FakeCompletions:
    """
    Captures the composed request instead of sending it.
    """

    def __init__(self, response: object) -> None:
        self.response = response
        self.request: dict = {}

    async def create(self, **request: object) -> object:
        self.request = dict(request)
        return self.response


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)

    async def close(self) -> None:
        return None


def _response() -> object:
    """
    One tool call, which is the shape every planner turn expects back.
    """

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(name="tap", arguments='{"x":1,"y":2}')
                        )
                    ],
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


_TOOLS = {
    "tools": [
        {"name": "tap", "description": "Tap.", "parameters": {"type": "object", "properties": {}}}
    ]
}


class ReasoningEffortIsSentTest(IsolatedAsyncioTestCase):
    """
    What actually reaches the provider.
    """

    @staticmethod
    def __adapter(*, effort: ReasoningEffort | None, model: str = "gpt-5.6-sol"):
        adapter = OpenAILLM(
            configuration=LLMConfiguration(
                provider="openai",
                api_key="test-key",
                model=model,
                max_retries=0,
                reasoning_effort=effort,
            )
        )
        completions = _FakeCompletions(response=_response())
        adapter._OpenAILLM__client = _FakeClient(completions)  # type: ignore[attr-defined]
        return adapter, completions

    async def test_a_pinned_level_is_sent(self) -> None:
        """
        The regression: without this the request omitted the parameter and the provider
        refused every tool-calling call.
        """

        adapter, completions = self.__adapter(effort=ReasoningEffort.NONE)
        await adapter.generate(use_cache=False, prompt=["go"], tools=_TOOLS)

        self.assertEqual(completions.request["reasoning_effort"], "none")

    async def test_the_parameter_is_omitted_when_unset(self) -> None:
        """
        Unset must stay distinguishable from "none". Defaulting it here would silently
        change every existing run's request shape.
        """

        adapter, completions = self.__adapter(effort=None)
        await adapter.generate(use_cache=False, prompt=["go"], tools=_TOOLS)

        self.assertNotIn("reasoning_effort", completions.request)

    async def test_every_level_travels_verbatim(self) -> None:
        """
        The adapter maps nothing and interprets nothing; which levels a given model
        accepts is the provider's business, and guessing here would mean maintaining a
        second copy of the provider's matrix.
        """

        for effort in ReasoningEffort:
            with self.subTest(effort=effort):
                adapter, completions = self.__adapter(effort=effort)
                await adapter.generate(use_cache=False, prompt=["go"], tools=_TOOLS)
                self.assertEqual(completions.request["reasoning_effort"], effort.value)

    async def test_it_is_independent_of_the_model_name(self) -> None:
        """
        No per-model conditional. A control keyed on model strings needs editing every
        time a provider ships one.
        """

        adapter, completions = self.__adapter(effort=ReasoningEffort.NONE, model="gemini-3.5-flash")
        await adapter.generate(use_cache=False, prompt=["go"], tools=_TOOLS)

        self.assertEqual(completions.request["reasoning_effort"], "none")

    async def test_temperature_handling_is_unchanged(self) -> None:
        """
        The two controls are independent, and gpt-5 models still reject temperature.
        """

        adapter, completions = self.__adapter(effort=ReasoningEffort.NONE)
        await adapter.generate(use_cache=False, prompt=["go"], tools=_TOOLS)

        self.assertNotIn("temperature", completions.request)


class ReasoningEffortReachesProvenanceTest(TestCase):
    """
    A run that pinned a level and one that took the default are different runs.
    """

    @staticmethod
    def __identity(*, effort: ReasoningEffort | None):
        return OpenAILLM(
            configuration=LLMConfiguration(
                provider="openai",
                api_key="test-key",
                model="gpt-5.6-sol",
                reasoning_effort=effort,
            )
        ).identity

    def test_a_pinned_level_is_reported(self) -> None:
        self.assertEqual(self.__identity(effort=ReasoningEffort.NONE).settings.thinking, "none")

    def test_an_unset_control_reports_none(self) -> None:
        """
        None means the provider's default applied. It is recorded as absent rather than
        as a guess at what that default was.
        """

        self.assertIsNone(self.__identity(effort=None).settings.thinking)

    def test_the_reported_value_is_the_effective_one(self) -> None:
        for effort in ReasoningEffort:
            with self.subTest(effort=effort):
                self.assertEqual(self.__identity(effort=effort).settings.thinking, effort.value)


class ReasoningEffortConfigurationTest(TestCase):
    """
    Validation, so an unusable value fails at startup rather than mid-run.
    """

    def test_the_configured_value_is_validated(self) -> None:
        with self.assertRaises(ValidationError):
            LLMConfiguration(provider="openai", model="m", reasoning_effort="exhaustive")

    def test_it_defaults_to_unset(self) -> None:
        self.assertIsNone(LLMConfiguration(provider="openai", model="m").reasoning_effort)

    def test_none_is_a_level_not_an_absence(self) -> None:
        """
        The distinction the whole control rests on: a model that accepts tools only with
        reasoning switched off needs "off" to be expressible.
        """

        configured = LLMConfiguration(
            provider="openai", model="m", reasoning_effort=ReasoningEffort.NONE
        )

        self.assertIs(configured.reasoning_effort, ReasoningEffort.NONE)
        self.assertIsNotNone(configured.reasoning_effort)


class ReasoningEffortSettingsTest(TestCase):
    """
    The environment variable a benchmark host pins the control through.
    """

    @staticmethod
    def __settings(**environ: str) -> FathomSettings:
        with mock.patch.dict(os.environ, environ, clear=False):
            return FathomSettings()

    def test_the_variable_is_read(self) -> None:
        settings = self.__settings(FATHOM_INFERENCE_REASONING_EFFORT="none")

        self.assertIs(settings.inference_reasoning_effort, ReasoningEffort.NONE)

    def test_an_unset_variable_leaves_the_control_unset(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(FathomSettings().inference_reasoning_effort)

    def test_an_unusable_value_is_refused(self) -> None:
        """
        At startup, where it costs one message, rather than on the first inference call
        of a paid benchmark run.
        """

        with self.assertRaises(ValidationError):
            self.__settings(FATHOM_INFERENCE_REASONING_EFFORT="exhaustive")

    def test_the_drizz_prefixed_alias_is_accepted(self) -> None:
        """
        Matching the other inference controls, which all carry both spellings.
        """

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.dict(
                os.environ, {"DRIZZ_FATHOM_INFERENCE_REASONING_EFFORT": "low"}, clear=False
            ),
        ):
            self.assertIs(FathomSettings().inference_reasoning_effort, ReasoningEffort.LOW)
