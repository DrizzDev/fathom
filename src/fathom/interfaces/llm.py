from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from fathom.constants.llm import ModelProvider
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.llm import ModelIdentity, StructuredOutput
from fathom.schemas.results import GenerateResult

if TYPE_CHECKING:
    from fathom.schemas.configuration import LLMConfiguration

# Type alias for LLM prompt parts (text, images, or structured content)
PromptPart = Union[str, bytes, Dict[str, str]]


@runtime_checkable
class ModelBound(Protocol):
    """
    Capability for a port that runs on a language model and exposes the adapter it was given.

    Lets the composition root tell whether two roles share one adapter or hold distinct ones,
    which a comparison of their reported identities cannot decide.
    """

    @property
    def llm(self) -> "LLMPort":
        """
        The adapter this port was composed with.
        """

        ...


class LLMPort(ABC):
    """
    Provider-neutral port for language-model generation.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Name of the model being used.
        """

        raise NotImplementedError

    @property
    def identity(self) -> ModelIdentity:
        """
        Provider, model, and settings this adapter declares for itself.

        Default declares UNKNOWN: an adapter that does not state its provider is reported as
        undeclared rather than guessed from its class name.
        """

        return ModelIdentity(provider=ModelProvider.UNKNOWN, model=self.model_name)

    @abstractmethod
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
        Generate one response, optionally constrained by a structured-output contract.

        conversation_history supplies provider-neutral prior turns for multi-turn feedback
        loops; when present, prompt is appended as the final user turn. structured_output is
        a vendor-neutral constrained-decoding contract that adapters map to their provider's API.
        """

        raise NotImplementedError

    async def prewarm(
        self,
        *,
        system_instruction: Optional[str],
        tools: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Prewarm provider-side prompt cache when supported.

        Default is a no-op. Providers that support cached content (e.g. Gemini)
        override this to pre-create the cache entry before the first generate call.
        """

        return  # noqa: PLE0101

    def derive(self, *, overrides: "LLMConfiguration") -> "LLMPort":
        """
        Spawn a sibling adapter sharing credentials and model, with only the explicitly-set
        fields of ``overrides`` applied on top.

        Every provider adapter must implement this operation. Returning the planner itself would
        silently discard role-specific settings such as deterministic temperature and caching.
        """

        _ = overrides
        raise NotImplementedError(
            f"{type(self).__name__} does not implement provider adapter derivation"
        )

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Release resources.
        """

        raise NotImplementedError
