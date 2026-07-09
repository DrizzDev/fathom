from __future__ import annotations

from typing import Optional

from fathom.core.prompts.title import TitlePromptBuilder
from fathom.interfaces.llm import LLMPort
from fathom.schemas.conversation import TitleContext, TitlePolicy


class TitleComposer:
    """
    Composes bounded titles from runtime context.
    """

    def __init__(
        self,
        *,
        use_cache: bool = False,
        llm: Optional[LLMPort] = None,
        policy: Optional[TitlePolicy] = None,
        prompt: Optional[TitlePromptBuilder] = None,
    ) -> None:
        """
        Bind the optional model, prompt builder, and title policy.
        """

        self.__llm = llm
        self.__use_cache = use_cache

        self.__policy = policy or TitlePolicy()
        self.__prompt = prompt or TitlePromptBuilder()

    def initial(self, *, context: TitleContext) -> str:
        """
        Return an immediate deterministic title for client display.
        """

        return self.__fit(value=self.__fallback(context=context))

    async def compose(self, *, context: TitleContext) -> str:
        """
        Return a generated title from runtime context.
        """

        if self.__llm is None:
            return self.initial(context=context)

        response = await self.__llm.generate(
            use_cache=self.__use_cache,
            prompt=self.__prompt.build_prompt(intent=context.intent),
            system_instruction=self.__prompt.build_system_instruction(),
        )
        title = self.__title(value=response.content)
        fitted = self.__fit(value=title)

        return fitted if self.__valid(value=fitted) else self.initial(context=context)

    def normalize(self, *, value: str) -> str:
        """
        Return a bounded explicit title.
        """

        return self.__fit(value=value)

    def __fit(self, *, value: str) -> str:
        """
        Return a whitespace-normalized title that fits the stored title boundary.
        """

        title = " ".join(value.split())
        if len(title) <= self.__policy.limit:
            return title

        return title[: self.__policy.limit].rstrip()

    def __valid(self, *, value: str) -> bool:
        """
        Return whether a generated title looks like a short action phrase.
        """

        if not value or len(value) > self.__policy.phrase or "@" in value:
            return False

        return all(len(token) <= self.__policy.token for token in value.split())

    def __fallback(self, *, context: TitleContext) -> str:
        """
        Return a deterministic authoring title from stable runtime context.
        """

        package = (context.package or "").strip()
        if not package:
            return self.__policy.fallback

        return f"{self.__policy.prefix} {package}"

    @staticmethod
    def __title(*, value: str) -> str:
        """
        Return the first non-empty generated title line without surrounding quote wrappers.
        """

        for line in value.splitlines():
            if title := line.strip().strip("\"'`").strip():
                return title

        return ""
