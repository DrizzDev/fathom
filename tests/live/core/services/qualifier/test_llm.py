from __future__ import annotations

import asyncio
import logging
import time
import unittest
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.qualification import QualificationLabel
from fathom.core.services.qualifier import QualifierComposer
from fathom.interfaces.qualifier import IntentQualifierPort
from fathom.runtime.assembly import RunAssemblyBuilder
from fathom.runtime.factories import LLMFactory
from fathom.schemas.configuration import QualifierConfiguration
from fathom.settings.env import FathomSettings

logging.getLogger("google.genai").setLevel(logging.WARNING)
logging.getLogger("fathom.core.services.qualifier.llm").setLevel(logging.WARNING)


class IntentCase(BaseModel):
    """
    Single live-test case: an intent plus whether the threshold gate must block it.
    """

    model_config = ConfigDict(frozen=True)

    must_block: bool = Field(description="Expected gate decision for this case.")
    intent: str = Field(description="The user-facing intent string the qualifier will classify.")


class LiveResult(BaseModel):
    """
    Outcome of running a single intent through the live qualifier.
    """

    case: IntentCase = Field(description="The case that produced this outcome.")
    label: QualificationLabel = Field(description="Label the qualifier returned.")

    confidence: float = Field(description="Confidence the qualifier returned.")
    category: str = Field(description="Rationale category the qualifier returned.")
    blocked: bool = Field(description="Whether the threshold gate blocked the run.")

    latency: float = Field(description="Seconds taken by the qualify call.")
    correct: bool = Field(description="Whether the outcome matched must_block.")
    error: Optional[str] = Field(default=None, description="Error message if the qualifier raised.")


class IntentCorpus:
    """
    Static corpus of intent test cases assembled from real logs and the requirement spec.
    """

    OBSERVED_EXECUTABLE: List[str] = [
        "Open Meesho and then scroll down until you find Jars & containers on the screen",
        (
            "Open Swiggy app. Tap on Location dropdown. Select 'HSR Layout'. "
            "Tap on search bar. Enter 'Biryani' in search bar. Scroll up 40% auto suggest "
            "page. Tap on Show results. Validate srp page is loaded. "
            "Tap on heart icon in restaurant list. Validate login half card is displayed"
        ),
        "Scroll down until you find Jars & containers on the screen",
        "Scroll vertically until you find Asha Tiffin on the screen",
        "Search for McPuff",
        "Search for a product, add it to cart, and complete checkout.",
        "open the contact app add new contact to it",
    ]
    BORDERLINE_EXECUTABLE: List[str] = [
        "Upload a PDF",
        "Submit a form",
        "Add item to cart",
        "Find my invoices",
        "Check my settings",
        "Open billing page",
        "Create a Jira ticket",
    ]
    CLEARLY_NON_EXECUTABLE: List[str] = [
        "yuiol",
        "Æsdfghjk",
        "sdfghjk, yuiol",
        "what is abc?",
        "tell me 2 + 2",
        "write me a poem",
        "asdkfjhqwoeiruzxcv",
        "who founded google?",
        "can you explain kubernetes?",
    ]

    @classmethod
    def cases(cls) -> List[IntentCase]:
        """
        Return every intent case to run through the live qualifier.
        """

        cases: List[IntentCase] = []

        for intent in cls.OBSERVED_EXECUTABLE + cls.BORDERLINE_EXECUTABLE:
            cases.append(IntentCase(intent=intent, must_block=False))

        for intent in cls.CLEARLY_NON_EXECUTABLE:
            cases.append(IntentCase(intent=intent, must_block=True))

        return cases


class LiveReport:
    """
    Pretty-printer for a batch of live qualification results.
    """

    HEADER_INTENT_WIDTH: int = 70

    @classmethod
    def render(cls, *, results: List[LiveResult]) -> str:
        """
        Render a markdown-ish table summarizing every case alongside an aggregate.
        """

        lines: List[str] = ["", "Live Intent Qualifier Results", "=" * 110]

        header = (
            f"{'#':<3} {'OK':<3} {'Intent':<{cls.HEADER_INTENT_WIDTH}} "
            f"{'Label':<24} {'Conf':>5} {'Block':<5} {'Lat(s)':>7}"
        )
        lines.append(header)
        lines.append("-" * 110)

        for index, result in enumerate(results, start=1):
            mark = "✓" if result.correct else "✗"
            label = result.label.value if result.label else "ERROR"

            intent = result.case.intent
            if len(intent) > cls.HEADER_INTENT_WIDTH:
                intent = intent[: cls.HEADER_INTENT_WIDTH - 1] + "…"

            blocked = "yes" if result.blocked else "no"
            lines.append(
                f"{index:<3} {mark:<3} {intent:<{cls.HEADER_INTENT_WIDTH}} "
                f"{label:<24} {result.confidence:>5.2f} {blocked:<5} {result.latency:>7.2f}"
            )
            if result.error:
                lines.append(f"    ! error: {result.error}")

        lines.append("-" * 110)

        correct_count = sum(1 for result in results if result.correct)
        executable_total = sum(1 for result in results if not result.case.must_block)
        executable_correct = sum(
            1 for result in results if not result.case.must_block and result.correct
        )
        non_executable_total = sum(1 for result in results if result.case.must_block)
        non_executable_correct = sum(
            1 for result in results if result.case.must_block and result.correct
        )

        lines.append(f"Summary: {correct_count}/{len(results)} correct")
        lines.append(
            f"  Executable (must pass through):    {executable_correct}/{executable_total}"
        )
        lines.append(
            f"  Non-executable (must be blocked):  {non_executable_correct}/{non_executable_total}"
        )

        latencies = [result.latency for result in results if result.error is None]
        if latencies:
            lines.append(
                f"  Latency seconds: avg={sum(latencies) / len(latencies):.2f} "
                f"min={min(latencies):.2f} max={max(latencies):.2f}"
            )

        lines.append("=" * 110)
        return "\n".join(lines)


class ProductionQualifierWiring:
    """
    Reproduce the exact qualifier wiring the Temporal activity uses in production.
    """

    @staticmethod
    def build(*, configuration: QualifierConfiguration) -> IntentQualifierPort:
        """
        Construct the qualifier port through QualifierComposer with real settings + factory.
        """

        llm_factory = LLMFactory()
        assembly = RunAssemblyBuilder(settings=FathomSettings())

        planner_llm = llm_factory.create(
            configuration=assembly.build_qualifier_model_configuration(configuration=configuration)
        )
        return QualifierComposer(assembly=assembly, llm_factory=llm_factory).compose(
            planner_llm=planner_llm, configuration=configuration
        )


class LLMIntentQualifierLiveTest(unittest.IsolatedAsyncioTestCase):
    """
    End-to-end check that the real Gemini qualifier classifies every observed and
    requirement-mandated intent correctly under the threshold gate.
    """

    async def __classify(
        self, *, qualifier: IntentQualifierPort, case: IntentCase, floor: float
    ) -> LiveResult:
        """
        Run one case through the qualifier and produce a LiveResult.
        """

        started_at = time.perf_counter()

        try:
            verdict = await qualifier.qualify(intent=case.intent)

            latency = time.perf_counter() - started_at
            blocked = verdict.should_block(floor=floor)

            return LiveResult(
                case=case,
                blocked=blocked,
                latency=latency,
                label=verdict.label,
                confidence=verdict.confidence,
                correct=blocked == case.must_block,
                category=verdict.rationale.category.value,
            )
        except Exception as exception:
            return LiveResult(
                case=case,
                blocked=False,
                correct=False,
                confidence=0.0,
                category="error",
                error=str(exception),
                latency=time.perf_counter() - started_at,
                label=QualificationLabel.PROBABLY_EXECUTABLE,
            )

    async def test_every_intent_classifies_correctly(self) -> None:
        """
        Run the corpus through Gemini and report per-intent and aggregate outcomes.
        """

        cases = IntentCorpus.cases()
        configuration = QualifierConfiguration()

        qualifier = ProductionQualifierWiring.build(configuration=configuration)

        results: List[LiveResult] = []
        floor = configuration.confidence

        for case in cases:
            results.append(await self.__classify(qualifier=qualifier, case=case, floor=floor))
            await asyncio.sleep(0.05)

        print(LiveReport.render(results=results))

        executable_failures = [
            result for result in results if not result.case.must_block and not result.correct
        ]
        self.assertEqual(
            executable_failures,
            [],
            msg=(
                "False rejections detected (executable intents were blocked): "
                + ", ".join(f"{result.case.intent!r}" for result in executable_failures)
            ),
        )


if __name__ == "__main__":
    unittest.main()
