"""Intent classification service.

Wraps a single ``LLMPort.generate`` call with the ``classify_intent``
tool definition and extracts the structured boolean from the resulting
``tool_calls[0].args``. Used by ``IntentStrategy.execute`` to decide
whether the incoming intent should flow through the existing
``IntentDecomposer`` multi-step path or be wrapped as a single
sub-goal and executed directly.

Fails safe: on any exception, missing tool call, or malformed args,
returns ``True`` (decompose) so the agent falls back to the
well-tested multi-step path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fathom.core.prompts.classification import (
    CLASSIFICATION_SYSTEM,
    CLASSIFICATION_TOOL_DEFINITION,
    build_classification_user_prompt,
)
from fathom.interfaces.llm import LLMPort

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Decides whether an intent needs decomposition via LLM tool call."""

    def __init__(self, llm: LLMPort) -> None:
        self.__llm = llm

    async def should_decompose(self, intent: str) -> bool:
        """Return True if ``intent`` should be decomposed, False otherwise.

        The boolean comes from the LLM via a forced tool call on
        ``CLASSIFICATION_TOOL_DEFINITION``. On any failure mode
        (exception, empty tool_calls, missing ``should_decompose`` key)
        the method logs a warning and returns True so the caller falls
        back to the existing multi-step decomposition path.
        """

        prompt = build_classification_user_prompt(intent=intent)

        try:
            result = await self.__llm.generate(
                use_cache=False,
                prompt=prompt,
                tools=CLASSIFICATION_TOOL_DEFINITION,
                system_instruction=CLASSIFICATION_SYSTEM,
            )
        except Exception as exception:
            logger.warning(
                "[IntentClassifier] LLM call failed (%s); defaulting to decompose",
                exception,
            )
            return True

        if not result.tool_calls:
            logger.warning("[IntentClassifier] No tool call in response; defaulting to decompose")
            return True

        args = self.__extract_tool_call_args(result.tool_calls[0])
        if "should_decompose" not in args:
            logger.warning(
                "[IntentClassifier] Tool call missing should_decompose arg; defaulting to decompose"
            )
            return True

        should_decompose = bool(args.get("should_decompose", True))
        reason = str(args.get("reason", ""))
        logger.info(
            "[IntentClassifier] intent=%r should_decompose=%s (%s)",
            intent[:80],
            should_decompose,
            reason,
        )
        return should_decompose

    @staticmethod
    def __extract_tool_call_args(tool_call: Any) -> Dict[str, Any]:
        """Pull the ``args`` dict out of a tool-call object or dict.

        Handles both the dict shape (``{"args": {...}}``) and the
        object shape (``tool_call.args``) that different LLM adapters
        can produce \u2014 same pattern as
        ``fathom.adapters.summarization.llm.LLMSummarizer``.
        """

        if isinstance(tool_call, dict):
            raw = tool_call.get("args", {})
        else:
            raw = getattr(tool_call, "args", {})
        if not isinstance(raw, dict):
            return {}
        return raw
