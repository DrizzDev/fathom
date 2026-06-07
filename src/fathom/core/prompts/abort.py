from __future__ import annotations

from abc import ABC, abstractmethod


class AbortPromptBuilder(ABC):
    """
    Builds prompts for the operator-abort classifier.
    """

    @abstractmethod
    def build_system_instruction(self) -> str:
        """
        Build the system instruction shared across all classification calls.
        """

        raise NotImplementedError

    @abstractmethod
    def build_user_prompt(self, *, response: str) -> str:
        """
        Build the per-call user prompt wrapping the operator response.
        """

        raise NotImplementedError


class GeminiAbortPromptBuilder(AbortPromptBuilder):
    """
    Gemini-specific prompt builder for the operator-abort classifier.
    """

    __SYSTEM_INSTRUCTION = (
        "You are a strict binary classifier. Given a single human message, "
        "decide whether the speaker is commanding an automation agent to STOP / "
        "CANCEL / ABORT the entire workflow.\n\n"
        "Rules:\n"
        '- Output ONLY a JSON object with two keys: "aborted" (true|false) and '
        '"confidence" (float in [0,1]).\n'
        "- Treat as ABORT: 'cancel', 'stop the workflow', 'kill the run', 'close "
        "the execution', 'mark as done', 'we are done here', etc.\n"
        "- DO NOT classify as ABORT if the message commands a UI action (e.g. "
        "'tap on stop', 'click cancel', 'press the close button', 'tap the X'), "
        "describes UI state ('the stop button is at the top right'), or is "
        "unrelated guidance.\n"
        "- Multilingual input is allowed (Hindi, Spanish, Hinglish, etc.).\n"
        "- When ambiguous, prefer aborted=false."
    )

    def build_system_instruction(self) -> str:
        """
        Return the static system instruction.
        """

        return self.__SYSTEM_INSTRUCTION

    def build_user_prompt(self, *, response: str) -> str:
        """
        Return the user prompt wrapping the operator response.
        """

        return response
