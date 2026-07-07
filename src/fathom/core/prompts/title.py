from __future__ import annotations

from typing import Tuple

from fathom.interfaces.llm import PromptPart


class TitlePromptBuilder:
    """
    Builds prompts for concise title generation.
    """

    def build_system_instruction(self) -> str:
        """
        Return the title-generation system instruction.
        """

        return (
            "Write a short title for an authoring run. The title names the workflow "
            "purpose, not the user's wording. Create a fresh 2-6 word action phrase. "
            "Do not copy, shorten, paraphrase, or preserve the request text. Never include "
            "literal values such as emails, passwords, tokens, URLs, phone numbers, "
            "addresses, names, or credentials. Return only the title text, with no "
            "quotes, markdown, wrappers, or explanation."
        )

    def build_prompt(self, *, intent: str) -> Tuple[PromptPart, ...]:
        """
        Return the title-generation user prompt.
        """

        return (
            "Create a title for the authoring run. Transform the intent into a task label.\n"
            "Examples:\n"
            "Intent: Login with email X and password Y\n"
            "Title: Test login flow\n"
            "Intent: Book an Uber from home to airport\n"
            "Title: Booking Uber\n"
            "Intent: Open wishlist and check saved items\n"
            "Title: Checking wishlist\n"
            "Intent: Clear all pending notifications\n"
            "Title: Cleaning up notifications\n"
            f"Intent: {intent}\n"
            "Title:",
        )
