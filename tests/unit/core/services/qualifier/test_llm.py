from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Optional, Sequence

from fathom.constants.qualification import (
    DEFAULT_REJECTION_MESSAGE,
    QualificationLabel,
    RationaleCategory,
)
from fathom.core.services.qualifier.gate import QualificationGatePolicy
from fathom.core.services.qualifier.llm import LLMIntentQualifier
from fathom.interfaces.llm import LLMPort
from fathom.schemas.configuration import QualifierConfiguration
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.results import GenerateResult

GATE_POLICY: QualificationGatePolicy = QualificationGatePolicy(
    configuration=QualifierConfiguration()
)


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
    def json(
        *,
        label: str,
        confidence: float,
        category: str = "ui_task",
    ) -> str:
        """
        Serialize a verdict-shaped payload the LLM adapter knows how to parse.
        """

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
)


UNDER_SPECIFIED_UI_INTENTS = (
    "Open browser",
    "Open photos",
    "Check my settings",
    "Submit a form",
    "Find my invoices",
    "Add item to cart",
    "Open billing page",
    "Upload a PDF",
    "Login",
)


POLITE_QUESTION_EXECUTABLE_INTENTS = (
    "Can you open Swiggy and search for Biryani?",
    "Can you tap the checkout button?",
    "Could you scroll down and find the price?",
    "Can you go back to the previous screen?",
    "Can you open Chrome and search for weather today?",
    "Could you add this item to cart?",
    "Can you tap Continue?",
    "Can you open Settings and turn on Bluetooth?",
)


CONVERSATIONAL_MUST_BLOCK_INTENTS = (
    "Why is this not done yet?",
    "Why did this fail?",
    "Why is checkout not working?",
    "How does this app work?",
    "What should I do next?",
    "Can you explain why this failed?",
    "Is my order confirmed?",
    "Should I click continue?",
    "Tell me what to do here.",
    "Explain this screen to me.",
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
    Adapter must parse structured verdicts, fail open on errors, and apply the binary gate.
    """

    async def test_empty_intent_blocks_with_default_rejection_message(self) -> None:
        """
        Whitespace-only intent must be blocked deterministically with the rejection message.
        """

        qualifier = LLMIntentQualifier(llm=ScriptedLLM(contents=[]))
        verdict = await qualifier.qualify(intent="   ")
        self.assertEqual(verdict.label, QualificationLabel.NOT_EXECUTABLE)
        self.assertEqual(verdict.rationale.category, RationaleCategory.EMPTY)
        self.assertTrue(GATE_POLICY.should_block(verdict=verdict))
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
        self.assertFalse(GATE_POLICY.should_block(verdict=verdict))
        self.assertIsNone(verdict.message)

    async def test_not_executable_blocks_regardless_of_confidence(self) -> None:
        """
        NOT_EXECUTABLE must block at every confidence level. The binary gate has
        no threshold; the model committing to "not a UI request" is enough.
        """

        for confidence in (0.1, 0.5, 0.95):
            with self.subTest(confidence=confidence):
                llm = ScriptedLLM(
                    contents=[
                        VerdictPayload.json(
                            label="NOT_EXECUTABLE",
                            confidence=confidence,
                            category="informational",
                        )
                    ]
                )
                qualifier = LLMIntentQualifier(llm=llm)
                verdict = await qualifier.qualify(intent="what is 2 + 2?")
                self.assertTrue(GATE_POLICY.should_block(verdict=verdict))
                self.assertIsNone(verdict.message)

    async def test_non_json_response_fails_open(self) -> None:
        """
        Non-JSON LLM responses must fail open as EXECUTABLE with QUALIFIER_ERROR.
        """

        qualifier = LLMIntentQualifier(llm=ScriptedLLM(contents=["definitely not json"]))
        verdict = await qualifier.qualify(intent="Search for McPuff")
        self.assertEqual(verdict.label, QualificationLabel.EXECUTABLE)
        self.assertEqual(verdict.rationale.category, RationaleCategory.QUALIFIER_ERROR)
        self.assertFalse(GATE_POLICY.should_block(verdict=verdict))

    async def test_valid_json_but_not_object_fails_open(self) -> None:
        """
        Valid JSON whose root is not an object (array, string, number, null) must
        fail open instead of raising AttributeError when payload.get() is called.
        """

        for body in ("[]", '"foo"', "123", "null", "true"):
            with self.subTest(body=body):
                qualifier = LLMIntentQualifier(llm=ScriptedLLM(contents=[body]))
                verdict = await qualifier.qualify(intent="Search for McPuff")
                self.assertEqual(verdict.label, QualificationLabel.EXECUTABLE)
                self.assertEqual(verdict.rationale.category, RationaleCategory.QUALIFIER_ERROR)
                self.assertFalse(GATE_POLICY.should_block(verdict=verdict))

    async def test_rationale_is_not_object_fails_open(self) -> None:
        """
        A payload where 'rationale' is not an object (e.g. a string) must fail open.
        """

        body = json.dumps({"label": "EXECUTABLE", "confidence": 0.9, "rationale": "oops"})
        qualifier = LLMIntentQualifier(llm=ScriptedLLM(contents=[body]))
        verdict = await qualifier.qualify(intent="Search for McPuff")
        self.assertEqual(verdict.label, QualificationLabel.EXECUTABLE)
        self.assertEqual(verdict.rationale.category, RationaleCategory.QUALIFIER_ERROR)

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
        self.assertEqual(verdict.label, QualificationLabel.EXECUTABLE)
        self.assertEqual(verdict.rationale.category, RationaleCategory.QUALIFIER_ERROR)
        self.assertFalse(GATE_POLICY.should_block(verdict=verdict))

    async def test_unknown_probably_executable_string_fails_open(self) -> None:
        """
        QualificationLabel is binary. Any non-binary label string (e.g. a stale
        prompt response still emitting "PROBABLY_EXECUTABLE") must fail open as
        EXECUTABLE with QUALIFIER_ERROR — the gate must never block on a parse
        failure.
        """

        llm = ScriptedLLM(
            contents=[
                json.dumps(
                    {
                        "confidence": 0.8,
                        "label": "PROBABLY_EXECUTABLE",
                        "rationale": {"category": "ui_task", "reasoning": "x"},
                    }
                )
            ]
        )
        qualifier = LLMIntentQualifier(llm=llm)
        verdict = await qualifier.qualify(intent="Open browser")

        self.assertFalse(GATE_POLICY.should_block(verdict=verdict))
        self.assertEqual(verdict.label, QualificationLabel.EXECUTABLE)
        self.assertEqual(verdict.rationale.category, RationaleCategory.QUALIFIER_ERROR)

    async def test_unknown_probably_not_executable_string_fails_open(self) -> None:
        """
        Mirror of the executable-leaning case: a stale "PROBABLY_NOT_EXECUTABLE"
        label string must fail open rather than crash or block.
        """

        llm = ScriptedLLM(
            contents=[
                json.dumps(
                    {
                        "confidence": 0.9,
                        "label": "PROBABLY_NOT_EXECUTABLE",
                        "rationale": {"category": "ambiguous", "reasoning": "x"},
                    }
                )
            ]
        )
        qualifier = LLMIntentQualifier(llm=llm)
        verdict = await qualifier.qualify(intent="Submit a form")

        self.assertFalse(GATE_POLICY.should_block(verdict=verdict))
        self.assertEqual(verdict.label, QualificationLabel.EXECUTABLE)
        self.assertEqual(verdict.rationale.category, RationaleCategory.QUALIFIER_ERROR)

    async def test_invalid_category_fails_open(self) -> None:
        """
        Unknown rationale category values must fail open instead of crashing the gate.
        """

        llm = ScriptedLLM(
            contents=[
                json.dumps(
                    {
                        "confidence": 0.9,
                        "label": "EXECUTABLE",
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

        self.assertFalse(GATE_POLICY.should_block(verdict=verdict))
        self.assertEqual(verdict.label, QualificationLabel.EXECUTABLE)
        self.assertEqual(verdict.rationale.category, RationaleCategory.QUALIFIER_ERROR)


class CorpusRegressionTest(unittest.IsolatedAsyncioTestCase):
    """
    Pin gate behavior against every bucket in the test corpus when the LLM
    returns the expected label. These are scripted-LLM tests — they prove the
    gate routes labels correctly. Prompt accuracy is covered by the live test.
    """

    async def test_real_executable_intents_pass(self) -> None:
        """
        Every real-log executable intent must pass when the model returns EXECUTABLE.
        """

        for intent in REAL_EXECUTABLE_INTENTS:
            with self.subTest(intent=intent):
                llm = ScriptedLLM(
                    contents=[VerdictPayload.json(label="EXECUTABLE", confidence=0.95)]
                )
                qualifier = LLMIntentQualifier(llm=llm)
                verdict = await qualifier.qualify(intent=intent)

                self.assertFalse(GATE_POLICY.should_block(verdict=verdict))

    async def test_under_specified_ui_intents_pass(self) -> None:
        """
        Under-specified UI actions ('Open browser', 'Submit a form', ...) must
        pass the binary gate. Specificity is the strategy's problem.
        """

        for intent in UNDER_SPECIFIED_UI_INTENTS:
            with self.subTest(intent=intent):
                llm = ScriptedLLM(
                    contents=[VerdictPayload.json(label="EXECUTABLE", confidence=0.85)]
                )
                qualifier = LLMIntentQualifier(llm=llm)
                verdict = await qualifier.qualify(intent=intent)

                self.assertFalse(GATE_POLICY.should_block(verdict=verdict))

    async def test_polite_question_executable_intents_pass(self) -> None:
        """
        Polite-question-form UI commands ('Can you open Swiggy?') must pass the
        gate. Question mark is not a block signal.
        """

        for intent in POLITE_QUESTION_EXECUTABLE_INTENTS:
            with self.subTest(intent=intent):
                llm = ScriptedLLM(
                    contents=[VerdictPayload.json(label="EXECUTABLE", confidence=0.95)]
                )
                qualifier = LLMIntentQualifier(llm=llm)
                verdict = await qualifier.qualify(intent=intent)

                self.assertFalse(GATE_POLICY.should_block(verdict=verdict))

    async def test_conversational_intents_block(self) -> None:
        """
        Answer-seeking and meta-conversation must block when the model commits
        to NOT_EXECUTABLE — this is the new corpus protecting against polite
        questions being treated as UI actions.
        """

        for intent in CONVERSATIONAL_MUST_BLOCK_INTENTS:
            with self.subTest(intent=intent):
                llm = ScriptedLLM(
                    contents=[
                        VerdictPayload.json(
                            confidence=0.95,
                            label="NOT_EXECUTABLE",
                            category="conversational",
                        )
                    ]
                )
                qualifier = LLMIntentQualifier(llm=llm)
                verdict = await qualifier.qualify(intent=intent)

                self.assertTrue(GATE_POLICY.should_block(verdict=verdict))

    async def test_clearly_non_executable_intents_block(self) -> None:
        """
        Clearly non-executable intents must block when the model is confident.
        """

        for intent in CLEARLY_NON_EXECUTABLE_INTENTS:
            with self.subTest(intent=intent):
                llm = ScriptedLLM(
                    contents=[
                        VerdictPayload.json(
                            confidence=0.95,
                            label="NOT_EXECUTABLE",
                            category="informational",
                        )
                    ]
                )
                qualifier = LLMIntentQualifier(llm=llm)
                verdict = await qualifier.qualify(intent=intent)

                self.assertTrue(GATE_POLICY.should_block(verdict=verdict))


if __name__ == "__main__":
    unittest.main()
