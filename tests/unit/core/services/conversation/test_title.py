from __future__ import annotations

import unittest
from typing import Any, Dict, Optional, Sequence

from fathom.constants.conversation import THREAD_TITLE_MAX_LENGTH
from fathom.core.prompts.title import TitlePromptBuilder
from fathom.core.services.conversation.title import TitleComposer
from fathom.interfaces.llm import LLMPort, PromptPart
from fathom.schemas.conversation import ConversationTurn, TitleContext
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.results import GenerateResult


class FakeTitleLlm(LLMPort):
    """
    LLM test double for conversation title composition.
    """

    def __init__(self, *, content: str = '"HealthTap Login"') -> None:
        """
        Store the generated content returned by the fake model.
        """

        self.__content = content

    @property
    def model_name(self) -> str:
        """
        Return the fake title model name.
        """

        return "gemini-2.5-flash"

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
        Return a compact generated title.
        """

        _ = (use_cache, prompt, tools, system_instruction, conversation_history, structured_output)
        return GenerateResult(content=self.__content)

    async def cleanup(self) -> None:
        """
        Release no resources.
        """

        return None


class TitleComposerTest(unittest.IsolatedAsyncioTestCase):
    """
    Covers generated and fallback conversation title composition.
    """

    async def test_compose_uses_llm_title_when_available(self) -> None:
        """
        LLM-backed title composition returns the model-authored title.
        """

        title = await TitleComposer(llm=FakeTitleLlm(content='"Test login flow"')).compose(
            context=TitleContext(
                intent=(
                    "Open HealthTap and login with email siddhant.sisodiya@drizz.dev "
                    "and password Testin@23"
                ),
                package="com.healthtap.userhtexpress",
            )
        )

        self.assertEqual(title, "Test login flow")
        self.assertNotIn("siddhant.sisodiya", title)
        self.assertNotIn("Testin@23", title)

    async def test_compose_rejects_rogue_intent_echo(self) -> None:
        """
        Model output that looks like copied raw intent fails closed to the fallback title.
        """

        title = await TitleComposer(
            llm=FakeTitleLlm(
                content=(
                    "Open HealthTap and login with email siddhant.sisodiya@drizz.dev "
                    "and password Testin@23"
                )
            )
        ).compose(context=TitleContext(intent=self.__long_intent(), package="com.healthtap"))

        self.assertEqual(title, "Authoring com.healthtap")

    async def test_compose_without_llm_uses_package_fallback(self) -> None:
        """
        Missing title model uses stable runtime context instead of raw intent text.
        """

        title = await TitleComposer().compose(
            context=TitleContext(
                intent=(
                    "Open HealthTap and login with email siddhant.sisodiya@drizz.dev "
                    "and password Testin@23"
                ),
                package="com.healthtap.userhtexpress",
            )
        )

        self.assertEqual(title, "Authoring com.healthtap.userhtexpress")
        self.assertNotIn("HealthTap", title)
        self.assertNotIn("siddhant.sisodiya", title)

    async def test_initial_uses_generic_authoring_fallback_without_package(self) -> None:
        """
        Missing package context still gives the UI a non-empty authoring title.
        """

        title = TitleComposer().initial(
            context=TitleContext(intent="Open HealthTap and login", package=None)
        )

        self.assertEqual(title, "Authoring session")

    async def test_compose_uses_fallback_for_empty_model_output(self) -> None:
        """
        Empty model output falls back to stable runtime context.
        """

        title = await TitleComposer(llm=FakeTitleLlm(content=" \n\t ")).compose(
            context=TitleContext(intent=self.__long_intent(), package="com.example")
        )

        self.assertEqual(title, "Authoring com.example")

    async def test_compose_rejects_overlong_model_output(self) -> None:
        """
        Overlong model output falls back instead of shipping copied prose.
        """

        title = await TitleComposer(
            llm=FakeTitleLlm(content=" ".join("checkout" for _ in range(30)))
        ).compose(context=TitleContext(intent=self.__long_intent(), package="com.example"))

        self.assertEqual(title, "Authoring com.example")

    async def test_compose_rejects_identifier_shaped_model_output(self) -> None:
        """
        Long token-shaped model output falls back instead of shipping identifiers.
        """

        title = await TitleComposer(
            llm=FakeTitleLlm(content="booking abcdefghijklmnopqrstuvwxyz0123456789abcd")
        ).compose(context=TitleContext(intent=self.__long_intent(), package="com.example"))

        self.assertEqual(title, "Authoring com.example")

    async def test_fit_still_respects_ledger_limit(self) -> None:
        """
        Explicit title fitting remains bounded by the ledger title limit.
        """

        title = TitleComposer().normalize(value=self.__long_intent())

        self.assertLessEqual(len(title), THREAD_TITLE_MAX_LENGTH)

    async def test_prompt_requests_semantic_title_not_intent_copy(self) -> None:
        """
        Title prompt asks for a semantic task title instead of intent shortening.
        """

        prompt = TitlePromptBuilder()

        system = prompt.build_system_instruction()
        user = prompt.build_prompt(intent="Open HealthTap and login with email and password")

        self.assertIn("short title for an authoring run", system)
        self.assertIn("Create a fresh 2-6 word action phrase", system)
        self.assertIn("Do not copy, shorten, paraphrase", system)
        self.assertIn("Test login flow", user[0])
        self.assertIn("Booking Uber", user[0])
        self.assertIn("Checking wishlist", user[0])
        self.assertIn("Cleaning up notifications", user[0])
        self.assertIn("Transform the intent into a task label", user[0])

    @staticmethod
    def __long_intent() -> str:
        """
        Return a repeated intent like staging supplied for the HealthTap run.
        """

        return "\n".join(
            "HealthTap app is opened, now login with email and password" for _ in range(10)
        )
