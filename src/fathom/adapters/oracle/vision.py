from __future__ import annotations

from typing import Optional

from fathom.core.prompts.oracle import OraclePromptBuilder
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.oracle import OraclePort
from fathom.schemas.criterion import Verdict
from fathom.schemas.llm import StructuredOutput


class VisionOracle(OraclePort):
    """
    Reads a criterion against the settled screenshot through the LLM vision channel.
    """

    def __init__(self, *, llm: LLMPort, prompts: Optional[OraclePromptBuilder] = None) -> None:
        """
        Bind the LLM port and prompt builder used for the vision reading.
        """

        self.__llm = llm
        self.__output = StructuredOutput(payload=Verdict)
        self.__prompts = prompts if prompts is not None else OraclePromptBuilder()

    async def read(self, *, criterion: str, image: bytes) -> Verdict:
        """
        Return the structured verdict for one criterion against one settled capture.
        """

        response = await self.__llm.generate(
            use_cache=False,
            structured_output=self.__output,
            system_instruction=self.__prompts.build_system_instruction(),
            prompt=[self.__prompts.build_user_prompt(criterion=criterion), image],
        )

        return Verdict.model_validate_json(response.content)
