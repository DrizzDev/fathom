"""
Live test for LLMIntentQualifier.

Calls the real Gemini backend (configured via .env -> FathomSettings) for every
distinct intent observed in /logs, /debug, /debugging plus the requirement's
borderline executable cases and clearly non-executable cases. Asserts the
threshold gate's correctness and prints a single summary table.

Run:
    conda run -n Fathom-ENV python -m pytest tests/live/test_intent_qualifier_live.py -s
"""

from __future__ import annotations

import asyncio
import logging
import unittest
from dataclasses import dataclass
from typing import List, Optional, Tuple

from fathom.constants.qualification import QualificationLabel
from fathom.core.services.qualifier.llm import LLMIntentQualifier
from fathom.runtime.assembly import RunAssemblyBuilder
from fathom.runtime.factories import LLMFactory
from fathom.schemas.configuration import QualifierConfiguration
from fathom.settings.env import FathomSettings

logging.getLogger("fathom.core.services.qualifier.llm").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)


@dataclass(frozen=True)
class IntentCase:
    """
    A single live-test case: an intent plus whether the threshold gate must block it.
    """

    intent: str
    must_block: bool
    source: str


class IntentCorpus:
    """
    Static corpus of intent test cases assembled from real logs and the requirement spec.
    """

    @staticmethod
    def cases() -> List[IntentCase]:
        """
        Return every intent case to run through the live qualifier.
        """

        observed_executable: Tuple[str, ...] = (
            "Open Meesho and then scroll down until you find Jars & containers on the screen",
            (
                "Open Swiggy app. Tap on Location dropdown. Select 'HSR Layout'. "
                "Tap on search bar. Enter 'Biryani' in search bar. Scroll up 40% auto suggest "
                "page. Tap on Show results. Validate srp page is loaded. "
                "Tap on heart icon in restaurant list. Validate login half card is displayed"
            ),
            (
                "Open Swiggy app. Tap on skip. Select 'HSR Layout' from the bottom-sheet option. "
                "Tap on search bar. Enter 'Biryani' in search bar. Scroll up 40% auto suggest "
                "page. Tap on Show results. Validate srp page is loaded."
            ),
            "Scroll down until you find Jars & containers on the screen",
            "Scroll the food catalog section until you find Salad on the screen",
            "Scroll until you find Ashsa Tiffin on the screen",
            "Scroll vertically until you find Asha Tiffin on the screen",
            "Scroll vertically until you find Grill Daddy on the screen",
            "Search for McPuff",
            "Search for a product, add it to cart, and complete checkout.",
            "open the contact app add new contact to it",
        )
        borderline_executable: Tuple[str, ...] = (
            "Find my invoices",
            "Check my settings",
            "Look for failed deployments",
            "See open pull requests",
            "Open billing page",
            "Find the pricing section",
            "Review pending approvals",
            "Create a Jira ticket",
            "Upload a PDF",
            "Change profile picture",
            "Create a project",
            "Invite a user",
            "Add item to cart",
            "Submit a form",
            "Download a report",
            "Navigate to a page",
        )
        clearly_non_executable: Tuple[str, ...] = (
            "what is abc?",
            "tell me 2 + 2",
            "who founded google?",
            "write me a poem",
            "can you explain kubernetes?",
            "What is Kubernetes?",
            "Explain recursion",
            "Why is the sky blue?",
            "Tell me a joke",
            "Compare React and Vue",
            "Summarize this concept",
            "asdkfjhqwoeiruzxcv",
        )

        cases: List[IntentCase] = []
        for intent in observed_executable:
            cases.append(IntentCase(intent=intent, must_block=False, source="logs"))
        for intent in borderline_executable:
            cases.append(IntentCase(intent=intent, must_block=False, source="borderline"))
        for intent in clearly_non_executable:
            cases.append(IntentCase(intent=intent, must_block=True, source="non-executable"))
        return cases


@dataclass
class LiveResult:
    """
    Outcome of running a single intent through the live qualifier.
    """

    case: IntentCase
    label: QualificationLabel
    confidence: float
    category: str
    blocked: bool
    correct: bool
    latency: float
    error: Optional[str] = None


class LiveReport:
    """
    Pretty-printer for a batch of live qualification results.
    """

    HEADER_INTENT_WIDTH: int = 70

    @classmethod
    def render(cls, *, results: List[LiveResult]) -> str:
        """
        Render a markdown-ish table summarising every case alongside an aggregate.
        """

        lines: List[str] = []
        lines.append("")
        lines.append("Live Intent Qualifier Results")
        lines.append("=" * 110)
        header = (
            f"{'#':<3} {'OK':<3} {'Intent':<{cls.HEADER_INTENT_WIDTH}} "
            f"{'Label':<24} {'Conf':>5} {'Block':<5} {'Lat(s)':>7}"
        )
        lines.append(header)
        lines.append("-" * 110)

        for index, result in enumerate(results, start=1):
            ok = "✓" if result.correct else "✗"
            label = result.label.value if result.label else "ERROR"
            intent = result.case.intent
            if len(intent) > cls.HEADER_INTENT_WIDTH:
                intent = intent[: cls.HEADER_INTENT_WIDTH - 1] + "…"
            blocked = "yes" if result.blocked else "no"
            lines.append(
                f"{index:<3} {ok:<3} {intent:<{cls.HEADER_INTENT_WIDTH}} "
                f"{label:<24} {result.confidence:>5.2f} {blocked:<5} {result.latency:>7.2f}"
            )
            if result.error:
                lines.append(f"    ! error: {result.error}")

        lines.append("-" * 110)
        passed = sum(1 for r in results if r.correct)
        total = len(results)
        lines.append(f"Summary: {passed}/{total} correct")
        executable_total = sum(1 for r in results if not r.case.must_block)
        executable_passed = sum(1 for r in results if not r.case.must_block and r.correct)
        non_exec_total = sum(1 for r in results if r.case.must_block)
        non_exec_passed = sum(1 for r in results if r.case.must_block and r.correct)
        lines.append(f"  Executable (must pass through):    {executable_passed}/{executable_total}")
        lines.append(f"  Non-executable (must be blocked):  {non_exec_passed}/{non_exec_total}")
        latencies = [r.latency for r in results if r.error is None]
        if latencies:
            lines.append(
                f"  Latency seconds: avg={sum(latencies) / len(latencies):.2f} "
                f"min={min(latencies):.2f} max={max(latencies):.2f}"
            )
        lines.append("=" * 110)
        return "\n".join(lines)


class GeminiQualifierBuilder:
    """
    Constructor for an LLMIntentQualifier wired against the real Gemini backend.
    """

    @staticmethod
    def build() -> LLMIntentQualifier:
        """
        Build a live qualifier through the same assembly seam the builder uses in production.
        """

        configuration = QualifierConfiguration()
        assembly = RunAssemblyBuilder(settings=FathomSettings())
        llm = LLMFactory().create(
            configuration=assembly.build_qualifier_model_configuration(configuration=configuration)
        )
        return LLMIntentQualifier(llm=llm, configuration=configuration)


class IntentQualifierLiveTest(unittest.IsolatedAsyncioTestCase):
    """
    End-to-end check that the real Gemini qualifier classifies every observed and
    requirement-mandated intent correctly under the threshold gate.
    """

    async def test_every_intent_classifies_correctly(self) -> None:
        """
        Run the corpus through Gemini and report per-intent and aggregate outcomes.
        """

        cases = IntentCorpus.cases()
        qualifier = GeminiQualifierBuilder.build()
        floor = QualifierConfiguration().confidence_floor
        results: List[LiveResult] = []

        for case in cases:
            import time

            started_at = time.perf_counter()
            try:
                verdict = await qualifier.qualify(intent=case.intent)
                latency = time.perf_counter() - started_at
                blocked = verdict.should_block(floor=floor)
                correct = blocked == case.must_block
                results.append(
                    LiveResult(
                        case=case,
                        label=verdict.label,
                        confidence=verdict.confidence,
                        category=verdict.rationale.category.value,
                        blocked=blocked,
                        correct=correct,
                        latency=latency,
                    )
                )
            except Exception as exception:
                latency = time.perf_counter() - started_at
                results.append(
                    LiveResult(
                        case=case,
                        label=QualificationLabel.PROBABLY_EXECUTABLE,
                        confidence=0.0,
                        category="error",
                        blocked=False,
                        correct=False,
                        latency=latency,
                        error=str(exception),
                    )
                )
            await asyncio.sleep(0.05)

        print(LiveReport.render(results=results))

        executable_failures = [r for r in results if not r.case.must_block and not r.correct]
        self.assertEqual(
            executable_failures,
            [],
            msg=(
                "False rejections detected (executable intents were blocked): "
                + ", ".join(f"{r.case.intent!r}" for r in executable_failures)
            ),
        )


if __name__ == "__main__":
    unittest.main()
