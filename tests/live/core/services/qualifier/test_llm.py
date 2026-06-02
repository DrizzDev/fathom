from __future__ import annotations

import asyncio
import time
import unittest
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.qualification import QualificationLabel
from fathom.core.services.qualifier.gate import QualificationGatePolicy
from fathom.interfaces.qualifier import IntentQualifierPort
from fathom.runtime.assembly import RunAssemblyBuilder
from fathom.runtime.factories import LLMFactory
from fathom.runtime.qualifier import QualifierComposer
from fathom.schemas.configuration import QualifierConfiguration
from fathom.settings.env import FathomSettings


class IntentCase(BaseModel):
    """
    Single live-test case: an intent plus whether the binary gate must block it.
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
    blocked: bool = Field(description="Whether the binary gate blocked the run.")

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
    REAL_USER_INTENTS: List[str] = [
        (
            "Hi. I would like you to open the Fairway Solitaire app. go through the Big Fish "
            "Splash Screen, and await until you are on the main menu. If it asks to send "
            "notifications, allow. If it asks for Terms of Use, accept. If it asks to get a "
            "more personalized Experience, select Next. If it asks to track your activities "
            "across partner companies, allow.\nYou should be on the main screen, select Play "
            "to begin FTUE."
        ),
        "Ok lets simplify this, open the Fairway app, that is all",
        "Hello there, can you open the fair way app and get the the main menu?",
        "ok hello, can you open the Cooking Craze app, and then proceed to claim the daily reward.",
        "Open the Cooking Craze app and claim the daily reward",
        (
            "ok we are gonna take it up a notch\nOpen the Cooking Craze app. navigate to the "
            "main menu and open the settings. The settings will have a gear icon. When in "
            "settings select the support tab and open the terms of use. Take a screenshot of "
            "the terms of use. return back to the main menu and stop testing."
        ),
        (
            "Open the Cooking Craze app.\nSelect the play button.\nSelect the Settings button, "
            "it looks like a gear icon.\nOpen the support tab.\nOpen the terms of use page.\n"
            "Take a screenshot of the terms of use.\nReturn back to game.\nClose the settings "
            "menu.\nEnd Testing."
        ),
        (
            "Open the Cooking Craze App\nPress the Play button\nSelect the spoons Icon in the "
            "bottom right tiles\nOpen the offers to get spoons\nVerify that the player is "
            "re-directed to the offerwall.\nReturn to game\nClose the Survey/Offer wall menu.\n"
            "End test"
        ),
        (
            "If any of the attempts below are made more than 5 times and it still does not "
            "yield the desired results, mark the test as a fail and stop testing.\n\nOpen the "
            "Cooking Craze App\nPress the Play button\nSelect the spoons Icon in the bottom "
            "right tiles\nOpen the offers to get spoons\nVerify that the player is re-directed "
            "to the offerwall.\nReturn to game\nClose the Survey/Offer wall menu.\nEnd test"
        ),
        (
            "Open Cooking Craze, select play and open the settings menu. From there open the "
            "Support tab and open the terms of use. Validate that they are displayed. Return "
            "to game and close the settings menu."
        ),
        (
            "If any step takes more than 5 attempts immediately stop testing and fail the test.\n"
            "Main Test: Can the player select the offerwall and be directed to the offer wall\n"
            "- Open the Cooking Craze app.\n- Select the play button.\n- Select the Spoons icon "
            "in the bottom right tiles.\n- Select Offer.\n- Observe the player is directed to "
            "the offerwall.\n- Close Offerwall and return to game.\n- Close the Spoons menu.\n"
            "- End Test."
        ),
        (
            "If any step takes more than 5 attempts immediately stop testing. No system "
            "override allowed, STOP TESTING.\n\nMain Test: Can the player select the offerwall "
            "and be directed to the offer wall\n- Open the Cooking Craze app.\n- Select the "
            "play button.\n- Select the Spoons icon in the bottom right tiles.\n- Select Offer.\n"
            "- Observe the player is directed to the offerwall.\n- Close Offerwall and return "
            "to game.\n- Close the Spoons menu.\n- End Test."
        ),
    ]
    ANCHORED_EXECUTABLE: List[str] = [
        "go back",
        "scroll down",
        "On the Cart screen, tap Checkout",
        "Open Settings app and toggle Bluetooth",
    ]
    UNDER_SPECIFIED_UI_EXECUTABLE: List[str] = [
        "Login",
        "Open photos",
        "Open browser",
        "Upload a PDF",
        "Submit a form",
        "Check my settings",
        "Find my invoices",
        "Add item to cart",
        "Open billing page",
        "Login with abc",
    ]
    QUESTION_FORM_EXECUTABLE: List[str] = [
        "Can you open Swiggy and search for Biryani?",
        "Can you tap the checkout button?",
        "Could you scroll down and find the price?",
        "Can you go back to the previous screen?",
        "Can you open Chrome and search for weather today?",
        "Could you add this item to cart?",
        "Can you tap Continue?",
        "Can you open Settings and turn on Bluetooth?",
    ]
    CONVERSATIONAL_MUST_BLOCK: List[str] = [
        "Why is this not done yet?",
        "Why did this fail?",
        "Why is checkout not working?",
        "Why am I seeing this screen?",
        "Why can't I log in?",
        "How does this app work?",
        "How do I complete this?",
        "How can I fix this issue?",
        "How do I get my refund?",
        "What should I do next?",
        "What is this screen asking me to do?",
        "What does this error mean?",
        "What happens if I press continue?",
        "Can you explain why this failed?",
        "Can you tell me what went wrong?",
        "Can you help me understand this page?",
        "Is this completed already?",
        "Is my order confirmed?",
        "Do I need to fill this form?",
        "Should I click continue?",
        "Tell me what to do here.",
        "Explain this screen to me.",
    ]
    CLEARLY_NON_EXECUTABLE: List[str] = [
        "+",
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

        Binary contract is asserted end-to-end:
          - EXECUTABLE intents must never block. This includes under-specified
            UI actions ('Open browser', 'Submit a form') and polite-question UI
            commands ('Can you tap Continue?'); both are the strategy's problem.
          - NOT_EXECUTABLE intents must always block: gibberish, factual or
            creative requests, and answer-seeking / meta-conversation about
            the app, screen, or state.
        """

        cases: List[IntentCase] = []

        executable_intents = (
            cls.OBSERVED_EXECUTABLE
            + cls.REAL_USER_INTENTS
            + cls.ANCHORED_EXECUTABLE
            + cls.UNDER_SPECIFIED_UI_EXECUTABLE
            + cls.QUESTION_FORM_EXECUTABLE
        )
        for intent in executable_intents:
            cases.append(IntentCase(intent=intent, must_block=False))

        non_executable_intents = cls.CONVERSATIONAL_MUST_BLOCK + cls.CLEARLY_NON_EXECUTABLE
        for intent in non_executable_intents:
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
    async def build(*, configuration: QualifierConfiguration) -> IntentQualifierPort:
        """
        Construct the qualifier port through QualifierComposer with real settings + factory.

        The composer now returns a QualifierComposition; for the live test we
        only need the qualifier port. Owned resources would be drained by the
        composition root in production — this single-process test relies on
        process teardown to release the LLM client.
        """

        llm_factory = LLMFactory()
        assembly = RunAssemblyBuilder(settings=FathomSettings())

        planner_llm = llm_factory.create(
            configuration=assembly.build_qualifier_model_configuration(configuration=configuration)
        )
        composition = await QualifierComposer(assembly=assembly, llm_factory=llm_factory).compose(
            planner_llm=planner_llm, configuration=configuration
        )
        return composition.qualifier


class LLMIntentQualifierLiveTest(unittest.IsolatedAsyncioTestCase):
    """
    End-to-end check that the real Gemini qualifier classifies every observed and
    requirement-mandated intent correctly under the binary gate.
    """

    async def __classify(
        self,
        *,
        case: IntentCase,
        qualifier: IntentQualifierPort,
        policy: QualificationGatePolicy,
    ) -> LiveResult:
        """
        Run one case through the qualifier and produce a LiveResult.
        """

        started_at = time.perf_counter()

        try:
            verdict = await qualifier.qualify(intent=case.intent)

            latency = time.perf_counter() - started_at
            blocked = policy.should_block(verdict=verdict)

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
                label=QualificationLabel.EXECUTABLE,
            )

    async def test_every_intent_classifies_correctly(self) -> None:
        """
        Run the corpus through Gemini and report per-intent and aggregate outcomes.
        """

        cases = IntentCorpus.cases()
        configuration = QualifierConfiguration()

        qualifier = await ProductionQualifierWiring.build(configuration=configuration)
        policy = QualificationGatePolicy(configuration=configuration)

        results: List[LiveResult] = []

        for case in cases:
            results.append(await self.__classify(qualifier=qualifier, policy=policy, case=case))
            await asyncio.sleep(0.05)

        print(LiveReport.render(results=results))

        false_rejections = [
            result for result in results if not result.case.must_block and not result.correct
        ]
        false_acceptances = [
            result for result in results if result.case.must_block and not result.correct
        ]
        self.assertEqual(
            false_rejections,
            [],
            msg=(
                "False rejections detected (executable intents were blocked): "
                + ", ".join(f"{result.case.intent!r}" for result in false_rejections)
            ),
        )
        self.assertEqual(
            false_acceptances,
            [],
            msg=(
                "False acceptances detected (non-executable intents passed the gate): "
                + ", ".join(f"{result.case.intent!r}" for result in false_acceptances)
            ),
        )


if __name__ == "__main__":
    unittest.main()
