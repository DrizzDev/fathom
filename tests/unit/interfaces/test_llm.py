from __future__ import annotations

import unittest
from typing import Any, Dict, Optional, Sequence

from fathom.constants.llm import ModelProvider
from fathom.interfaces.llm import LLMPort, ModelBound, PromptPart
from fathom.schemas.configuration import LLMConfiguration
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.llm import ModelIdentity, StructuredOutput
from fathom.schemas.results import GenerateResult


class _LegacyLLM(LLMPort):
    """
    LLM implementation written before the identity capability existed.

    Implements only the three members the port required then, so it stands in for every
    third-party adapter already in the wild.
    """

    @property
    def model_name(self) -> str:
        """
        Model identifier this adapter reports.
        """

        return "legacy-1"

    async def generate(
        self,
        *,
        use_cache: bool,
        prompt: Sequence[PromptPart],
        tools: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        conversation_history: Optional[Sequence[ConversationTurn]] = None,
        structured_output: Optional[StructuredOutput] = None,
    ) -> GenerateResult:
        """
        Unused by these tests.
        """

        raise NotImplementedError

    async def cleanup(self) -> None:
        """
        Unused by these tests.
        """


class LLMPortIdentityDefaultTest(unittest.TestCase):
    """
    The identity capability is additive: adapters written against the previous port contract
    still construct, and report an undeclared provider rather than a guessed one.
    """

    def test_adapter_predating_the_capability_still_instantiates(self) -> None:
        """
        No new abstract member was added, so an existing implementation is still concrete.
        """

        self.assertEqual(_LegacyLLM().model_name, "legacy-1")

    def test_default_identity_declares_unknown_and_keeps_the_model_name(self) -> None:
        """
        The default reports UNKNOWN with the adapter's own model, never a class-derived name.
        """

        identity = _LegacyLLM().identity

        self.assertEqual(identity.provider, ModelProvider.UNKNOWN)
        self.assertEqual(identity.model, "legacy-1")
        self.assertNotIn("legacy", identity.provider.value)

    def test_default_identity_declares_no_inference_settings(self) -> None:
        """
        An adapter that states no settings reports them absent rather than as defaults.
        """

        settings = _LegacyLLM().identity.settings

        self.assertIsNone(settings.cache)
        self.assertIsNone(settings.thinking)
        self.assertIsNone(settings.temperature)

    def test_default_derive_requires_provider_implementation(self) -> None:
        """
        A provider that cannot derive a role must fail explicitly instead of silently sharing
        planner settings with the architect.
        """

        adapter = _LegacyLLM()

        with self.assertRaises(NotImplementedError):
            adapter.derive(overrides=LLMConfiguration(temperature=0.0))


class ModelBoundProtocolTest(unittest.TestCase):
    """
    The ModelBound capability is structural, so a port satisfies it by exposing its adapter.
    """

    def test_port_without_an_llm_attribute_does_not_satisfy_the_capability(self) -> None:
        """
        An LLM adapter is not itself a model-bound role port; it has no adapter to expose.
        """

        self.assertFalse(isinstance(_LegacyLLM(), ModelBound))

    def test_port_exposing_its_adapter_satisfies_the_capability(self) -> None:
        """
        Any object exposing an ``llm`` property is recognized without inheriting anything.
        """

        adapter = _LegacyLLM()

        class _Role:
            """
            Minimal role port exposing the adapter it was composed with.
            """

            @property
            def llm(self) -> LLMPort:
                """
                The adapter this role was composed with.
                """

                return adapter

        self.assertIsInstance(_Role(), ModelBound)
        self.assertIs(_Role().llm, adapter)


class ModelIdentityValueTest(unittest.TestCase):
    """
    Identity is a frozen value so a recorded manifest cannot be mutated after resolution.
    """

    def test_identity_is_immutable(self) -> None:
        """
        Assigning to a resolved identity raises rather than rewriting a recorded run.
        """

        identity = ModelIdentity(provider=ModelProvider.GEMINI, model="gemini-3.5-flash")

        with self.assertRaises(ValueError):
            identity.model = "swapped"  # type: ignore[misc]
