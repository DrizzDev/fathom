from __future__ import annotations

from typing import Tuple

from fathom.authoring.agent.packet import AuthoringPacketBuilder
from fathom.authoring.agent.prompts import AuthoringPromptFactory
from fathom.authoring.agent.reference import AuthoringReferenceProvider
from fathom.interfaces.authoring import AuthoringArtifactProvider
from fathom.interfaces.llm import PromptPart
from fathom.schemas.authoring import AuthoringTask


class AuthoringRequest:
    """
    Model-ready request assembled for one authoring task.
    """

    def __init__(
        self,
        *,
        instruction: str,
        parts: Tuple[PromptPart, ...],
    ) -> None:
        """
        Bind system instruction and ordered prompt parts.
        """

        self.__parts = parts
        self.__instruction = instruction

    @property
    def instruction(self) -> str:
        """
        Return the system instruction for the authoring request.
        """

        return self.__instruction

    @property
    def parts(self) -> Tuple[PromptPart, ...]:
        """
        Return ordered prompt parts for the authoring request.
        """

        return self.__parts


class AuthoringRequestBuilder:
    """
    Builds model-ready requests from authoring tasks.
    """

    def __init__(self, *, artifacts: AuthoringArtifactProvider) -> None:
        """
        Bind artifact resolution and pure prompt collaborators.
        """

        self.__artifacts = artifacts
        self.__packets = AuthoringPacketBuilder()
        self.__prompts = AuthoringPromptFactory()
        self.__references = AuthoringReferenceProvider()

    def build(self, *, task: AuthoringTask) -> AuthoringRequest:
        """
        Build the complete model request for an authoring task.
        """

        prompt = self.__prompts.prompt(kind=task.kind)
        reference = self.__references.reference(dialect=task.dialect)
        packet = self.__packets.build(task=task, dialect=reference)
        user_prompt = prompt.user_prompt(packet=packet)

        return AuthoringRequest(
            instruction=prompt.system_instruction(),
            parts=(user_prompt, *self.__artifacts.build(task=task)),
        )
