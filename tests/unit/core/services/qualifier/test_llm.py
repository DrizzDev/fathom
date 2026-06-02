from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Optional, Sequence

from fathom.constants.qualification import (
    DEFAULT_REJECTION_MESSAGE,
    QualificationLabel,
    RationaleCategory,
)
from fathom.core.services.qualifier.llm import LLMIntentQualifier
from fathom.interfaces.llm import LLMPort
from fathom.schemas.configuration import QualifierConfiguration
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.results import GenerateResult

CONFIDENCE_FLOOR: float = QualifierConfiguration().confidence_floor


class ScriptedLLM(LLMPort):
    """
    Minimal scripted LLM that returns a predefined sequence of contents.
    """

    def __init__(self, *, contents: List[str]) -> None:
        """
        Initialize with a queue of scripted contents to return on each call.
        """

        self.__contents = list(contents)
        self.calls = 0

    @property
    def model_name(self) -> str:
        """
        Identifier used by the qualifier and observability layers.
        """

        return "scripted-test"

    async def generate(
        self,
        *,
        use_cache: bool,
        prompt: Sequence[Any],
        tools: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        conversation_history: Optional[Sequence[ConversationTurn]] = None,
    ) -> GenerateResult:
        """
        Pop and return the next scripted content as a generate result.
        """

        self.calls += 1
        content = self.__contents.pop(0)
        return GenerateResult(content=content)

    async def cleanup(self) -> None:
        """
        Match the LLMPort lifecycle contract; nothing to release for the scripted LLM.
        """

        return None


class ExplodingLLM(LLMPort):
    """
    Scripted LLM that always raises on generate.
    """

    @property
    def model_name(self) -> str:
        """
        Identifier returned to observability layers when the LLM is the exploding stub.
        """

        return "exploding"

    async def generate(self, **_: Any) -> GenerateResult:
        """
        Always raise to exercise the qualifier's fail-open path.
        """

        raise RuntimeError("boom")

    async def cleanup(self) -> None:
        """
        Match the LLMPort lifecycle contract; nothing to release for the exploding stub.
        """

        return None


class VerdictPayload:
    """
    Builder of canonical verdict JSON payloads matching the production prompt contract.
    """

    @staticmethod
    def json(*, label: str, confidence: float) -> str:
        """
        Serialize a verdict-shaped payload that the LLM adapter knows how to parse.
        """

        category = (
            "ui_task" if "EXECUTABLE" in label and label != "NOT_EXECUTABLE" else "informational"
        )
        return json.dumps(
            {
                "label": label,
                "confidence": confidence,
                "rationale": {"category": category, "reasoning": "test"},
            }
        )


REAL_EXECUTABLE_INTENTS = (
    "Search for McPuff",
    "open the contact app add new contact to it",
    "Search for a product, add it to cart, and complete checkout.",
    "Scroll vertically until you find Dominoes on the screen",
    "Scroll until you find McDonal's on the screen",
    "Scroll the food catalog section until you find Salad on the screen",
    "Scroll down until you find Jars & containers on the screen",
    "Open Meesho and then scroll down until you find Jars & containers on the screen",
    (
        "Open Swiggy app. Tap on Location dropdown. Select 'HSR Layout'. Tap on search bar. "
        "Enter 'Biryani' in search bar. Scroll up 40% auto suggest page. Tap on Show results."
    ),
    "Find my invoices",
    "Open billing page",
    "Check my settings",
    "See open pull requests",
    "Look for failed deployments",
)


CLEARLY_NON_EXECUTABLE_INTENTS = (
    "what is 2 + 2?",
    "tell me a joke",
    "explain recursion",
    "who founded google?",
    "asdkfjhqwoeiruzxcv",
    "why is the sky blue?",
    "compare React and Vue",
    "write me a poem about the moon",
)


class LLMIntentQualifierTest(unittest.IsolatedAsyncioTestCase):
    """
    Adapter must parse structured verdicts, fail open on errors, and apply the threshold.
    """

    async def test_empty_intent_blocks_with_default_rejection_message(self) -> None:
        """
        Whitespace-only intent must be blocked deterministically with the rejection message.
        """

        qualifier = LLMIntentQualifier(llm=ScriptedLLM(contents=[]))
        verdict = await qualifier.qualify(intent="   ")
        self.assertEqual(verdict.label, QualificationLabel.NOT_EXECUTABLE)
        self.assertEqual(verdict.rationale.category, RationaleCategory.EMPTY)
        self.assertTrue(verdict.should_block(floor=CONFIDENCE_FLOOR))
        self.assertEqual(verdict.message, DEFAULT_REJECTION_MESSAGE)

    async def test_executable_intent_passes_through(self) -> None:
        """
        EXECUTABLE verdict must pass through with no user message attached.
        """

        llm = ScriptedLLM(contents=[VerdictPayload.json(label="EXECUTABLE", confidence=0.95)])
        qualifier = LLMIntentQualifier(llm=llm)
        verdict = await qualifier.qualify(intent="Search for McPuff")
        self.assertEqual(verdict.label, QualificationLabel.EXECUTABLE)
        self.assertEqual(verdict.rationale.category, RationaleCategory.UI_TASK)
        self.assertFalse(verdict.should_block(floor=CONFIDENCE_FLOOR))
        self.assertIsNone(verdict.message)

    async def test_not_executable_high_confidence_attaches_user_message(self) -> None:
        """
        Confident NOT_EXECUTABLE must block and surface the default rejection message.
        """

        llm = ScriptedLLM(contents=[VerdictPayload.json(label="NOT_EXECUTABLE", confidence=0.97)])
        qualifier = LLMIntentQualifier(llm=llm)
        verdict = await qualifier.qualify(intent="what is 2 + 2?")
        self.assertTrue(verdict.should_block(floor=CONFIDENCE_FLOOR))
        self.assertEqual(verdict.message, DEFAULT_REJECTION_MESSAGE)

    async def test_not_executable_low_confidence_does_not_block(self) -> None:
        """
        NOT_EXECUTABLE below the floor must pass through (bias toward allow).
        """

        llm = ScriptedLLM(contents=[VerdictPayload.json(label="NOT_EXECUTABLE", confidence=0.6)])
        qualifier = LLMIntentQualifier(llm=llm)
        verdict = await qualifier.qualify(intent="maybe do something")
        self.assertFalse(verdict.should_block(floor=CONFIDENCE_FLOOR))
        self.assertIsNone(verdict.message)

    async def test_non_json_response_fails_open(self) -> None:
        """
        Non-JSON LLM responses must fail open as PROBABLY_EXECUTABLE.
        """

        qualifier = LLMIntentQualifier(llm=ScriptedLLM(contents=["definitely not json"]))
        verdict = await qualifier.qualify(intent="Search for McPuff")
        self.assertEqual(verdict.label, QualificationLabel.PROBABLY_EXECUTABLE)
        self.assertEqual(verdict.rationale.category, RationaleCategory.QUALIFIER_ERROR)
        self.assertFalse(verdict.should_block(floor=CONFIDENCE_FLOOR))

    async def test_invalid_label_fails_open(self) -> None:
        """
        Unknown label values must fail open instead of crashing the gate.
        """

        llm = ScriptedLLM(
            contents=[
                json.dumps(
                    {
                        "label": "TOTALLY_INVENTED",
                        "confidence": 0.99,
                        "rationale": {"category": "ui_task", "reasoning": "x"},
                    }
                )
            ]
        )
        qualifier = LLMIntentQualifier(llm=llm)
        verdict = await qualifier.qualify(intent="Search for McPuff")
        self.assertEqual(verdict.label, QualificationLabel.PROBABLY_EXECUTABLE)
        self.assertFalse(verdict.should_block(floor=CONFIDENCE_FLOOR))

    async def test_invalid_category_fails_open(self) -> None:
        """
        Unknown rationale category values must fail open instead of crashing the gate.
        """

        llm = ScriptedLLM(
            contents=[
                json.dumps(
                    {
                        "label": "EXECUTABLE",
                        "confidence": 0.9,
                        "rationale": {"category": "nonsense_value", "reasoning": "x"},
                    }
                )
            ]
        )
        qualifier = LLMIntentQualifier(llm=llm)
        verdict = await qualifier.qualify(intent="Search for McPuff")
        self.assertEqual(verdict.rationale.category, RationaleCategory.QUALIFIER_ERROR)

    async def test_markdown_fence_is_stripped(self) -> None:
        """
        Markdown code fences around the JSON response must be removed before parsing.
        """

        wrapped = "```json\n" + VerdictPayload.json(label="EXECUTABLE", confidence=0.9) + "\n```"
        qualifier = LLMIntentQualifier(llm=ScriptedLLM(contents=[wrapped]))
        verdict = await qualifier.qualify(intent="Search for McPuff")
        self.assertEqual(verdict.label, QualificationLabel.EXECUTABLE)

    async def test_llm_exception_fails_open(self) -> None:
        """
        Exceptions raised by the LLM must fail open and never block execution.
        """

        qualifier = LLMIntentQualifier(llm=ExplodingLLM())
        verdict = await qualifier.qualify(intent="Search for McPuff")
        self.assertEqual(verdict.label, QualificationLabel.PROBABLY_EXECUTABLE)
        self.assertEqual(verdict.rationale.category, RationaleCategory.QUALIFIER_ERROR)
        self.assertFalse(verdict.should_block(floor=CONFIDENCE_FLOOR))


class CorpusRegressionTest(unittest.IsolatedAsyncioTestCase):
    """
    Every real observed intent from logs/assets must pass the threshold gate
    even when the model returns a borderline verdict.
    """

    async def test_real_corpus_borderline_verdicts_never_block(self) -> None:
        """
        Every observed real-log intent must pass the gate even at borderline labels.
        """

        for intent in REAL_EXECUTABLE_INTENTS:
            for label in (
                QualificationLabel.EXECUTABLE,
                QualificationLabel.PROBABLY_EXECUTABLE,
                QualificationLabel.PROBABLY_NOT_EXECUTABLE,
            ):
                with self.subTest(intent=intent, label=label):
                    llm = ScriptedLLM(
                        contents=[VerdictPayload.json(label=label.value, confidence=0.99)]
                    )
                    qualifier = LLMIntentQualifier(llm=llm)
                    verdict = await qualifier.qualify(intent=intent)
                    self.assertFalse(
                        verdict.should_block(floor=CONFIDENCE_FLOOR), f"{intent} blocked at {label}"
                    )

    async def test_clearly_non_executable_blocks_when_model_is_confident(self) -> None:
        """
        Clearly non-executable intents must be blocked when the model is confident.
        """

        for intent in CLEARLY_NON_EXECUTABLE_INTENTS:
            with self.subTest(intent=intent):
                llm = ScriptedLLM(
                    contents=[VerdictPayload.json(label="NOT_EXECUTABLE", confidence=0.95)]
                )
                qualifier = LLMIntentQualifier(llm=llm)
                verdict = await qualifier.qualify(intent=intent)
                self.assertTrue(verdict.should_block(floor=CONFIDENCE_FLOOR))
                self.assertEqual(verdict.message, DEFAULT_REJECTION_MESSAGE)


if __name__ == "__main__":
    unittest.main()
