from __future__ import annotations

import unittest
from typing import Any, List, Optional, Sequence, Tuple, Union

from fathom.constants import ActionType
from fathom.constants.success import CaptureNameProvenance
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.core.exceptions import ConfigurationError, DecompositionError
from fathom.core.services.decomposer import IntentDecomposer
from fathom.core.services.translation import ProposalTranslator
from fathom.interfaces.llm import LLMPort
from fathom.schemas.decomposition import DecomposedTask, DecompositionSchema
from fathom.schemas.proposal import (
    CaptureProposal,
    CommandProposal,
    DecompositionProposal,
    ObservedProposal,
)
from fathom.schemas.requirement import PressRequirement
from fathom.schemas.results import GenerateResult
from fathom.schemas.subgoal import SubGoal
from fathom.schemas.success import CaptureSuccess, CommandSuccess, ObservedSuccess


class _ScriptedLLM(LLMPort):
    """
    Test LLM returning queued contents or raising queued errors, counting generate() calls.
    """

    def __init__(self, *, responses: Sequence[Union[str, BaseException]]) -> None:
        """
        Queue one response (content or exception) per expected generate() call.
        """

        self.__responses = list(responses)
        self.calls: int = 0

    @property
    def model_name(self) -> str:
        """
        Report a Gemini-family model so the decomposition prompt builder resolves.
        """

        return "gemini-stub"

    async def generate(
        self,
        *,
        use_cache: bool,
        prompt: Any,
        tools: Optional[Any] = None,
        structured_output: Optional[Any] = None,
        system_instruction: Optional[Any] = None,
        conversation_history: Optional[Any] = None,
    ) -> GenerateResult:
        """
        Return the next queued content or raise the next queued exception.
        """

        _ = (use_cache, prompt, tools, structured_output, system_instruction, conversation_history)
        response = self.__responses[self.calls]
        self.calls += 1

        if isinstance(response, BaseException):
            raise response

        return GenerateResult(content=response, tool_calls=[], metrics={})

    async def cleanup(self) -> None:
        """
        Release resources (no-op for the scripted stub).
        """

        return None


class IntentDecomposerContractTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the one-DecompositionPhase contract over the proposal union: one generation, at most one
    repair, translation to canonical Success before any SubGoal, and never a fallback.
    """

    __UNPARSEABLE: str = "not json at all"
    __RAW_STRING_PLAN: str = '{"confidence": 0.9, "sub_goals": ["just do the thing"]}'

    @staticmethod
    def __task(
        *,
        objective: str = "Tap the login button",
        proposal: Optional[DecompositionProposal] = None,
    ) -> DecomposedTask:
        """
        Build one valid typed decomposition task, defaulting to an observed proposal.
        """

        return DecomposedTask(
            objective=objective,
            proposal=proposal
            if proposal is not None
            else ObservedProposal(assertion="The home screen is displayed"),
        )

    @staticmethod
    def __plan(*tasks: DecomposedTask, confidence: float = 0.9) -> str:
        """
        Render a valid decomposition payload from typed tasks as the LLM's JSON content.
        """

        return DecompositionSchema(confidence=confidence, sub_goals=list(tasks)).model_dump_json()

    def __decomposer(
        self, *, responses: Sequence[Union[str, BaseException]]
    ) -> Tuple[IntentDecomposer, _ScriptedLLM]:
        """
        Build a decomposer over a scripted LLM and the real proposal translator.
        """

        llm = _ScriptedLLM(responses=responses)
        translator = ProposalTranslator(catalog=CommandCatalogProvider().build())
        return IntentDecomposer(llm=llm, translator=translator), llm

    async def test_structured_success_yields_canonical_observed_sub_goal(self) -> None:
        """
        A single valid observed plan is accepted in one generation and translated to ObservedSuccess.
        """

        decomposer, llm = self.__decomposer(responses=[self.__plan(self.__task())])

        sub_goals = await decomposer.decompose(intent="Tap the login button")

        self.assertEqual(llm.calls, 1)
        self.assertEqual(len(sub_goals), 1)
        goal = sub_goals[0]
        self.assertEqual(goal.objective, "Tap the login button")
        self.assertIsInstance(goal.success, ObservedSuccess)
        assert isinstance(goal.success, ObservedSuccess)
        self.assertEqual(goal.success.observation.assertion, "The home screen is displayed")

    async def test_command_proposal_binds_to_command_success(self) -> None:
        """
        A command proposal citing an exact intent quote translates to a source-bound CommandSuccess.
        """

        intent = "tap the Login button"
        proposal = CommandProposal(
            requirement=PressRequirement(operation=ActionType.TAP, target="Login"),
            quote="tap",
        )
        decomposer, _ = self.__decomposer(
            responses=[self.__plan(self.__task(objective=intent, proposal=proposal))]
        )

        sub_goals = await decomposer.decompose(intent=intent)

        self.assertIsInstance(sub_goals[0].success, CommandSuccess)

    async def test_capture_proposal_binds_to_capture_success(self) -> None:
        """
        A capture proposal translates to a canonical CaptureSuccess identity.
        """

        proposal = CaptureProposal(
            subject="account balance", name="balance", provenance=CaptureNameProvenance.USER
        )
        decomposer, _ = self.__decomposer(
            responses=[self.__plan(self.__task(objective="Store the balance", proposal=proposal))]
        )

        sub_goals = await decomposer.decompose(intent="Store the balance")

        goal = sub_goals[0]
        self.assertIsInstance(goal.success, CaptureSuccess)
        assert isinstance(goal.success, CaptureSuccess)
        self.assertEqual(goal.success.target.name, "balance")

    async def test_command_quote_absence_no_longer_fails_closed(self) -> None:
        """
        An absent cited quote is diagnostic only; the command still translates, so decomposition succeeds.
        """

        proposal = CommandProposal(
            requirement=PressRequirement(operation=ActionType.TAP, target="Login"),
            quote="a phrase that is not present",
        )
        decomposer, _ = self.__decomposer(
            responses=[self.__plan(self.__task(objective="Tap login", proposal=proposal))]
        )

        sub_goals = await decomposer.decompose(intent="do the thing")

        self.assertIsInstance(sub_goals[0].success, CommandSuccess)

    async def test_initial_invalid_then_repair_valid_is_accepted(self) -> None:
        """
        An invalid first response triggers exactly one repair; a valid repair is accepted.
        """

        decomposer, llm = self.__decomposer(
            responses=[self.__UNPARSEABLE, self.__plan(self.__task())]
        )

        sub_goals = await decomposer.decompose(intent="Tap the login button")

        self.assertEqual(llm.calls, 2)
        self.assertEqual(len(sub_goals), 1)

    async def test_both_attempts_invalid_fails_closed(self) -> None:
        """
        When the initial output and the single repair are both invalid, decomposition fails closed.
        """

        decomposer, llm = self.__decomposer(responses=[self.__UNPARSEABLE, self.__UNPARSEABLE])

        with self.assertRaises(DecompositionError):
            await decomposer.decompose(intent="Tap the login button")

        self.assertEqual(llm.calls, 2)

    async def test_generation_failure_fails_closed_without_repair(self) -> None:
        """
        A provider error on the initial generation fails closed; there is nothing to repair.
        """

        decomposer, llm = self.__decomposer(responses=[RuntimeError("provider down")])

        with self.assertRaises(DecompositionError):
            await decomposer.decompose(intent="Tap the login button")

        self.assertEqual(llm.calls, 1)

    async def test_raw_string_entries_are_never_accepted(self) -> None:
        """
        Raw-string sub-goals are rejected at the boundary and, unrepaired, fail closed.
        """

        decomposer, llm = self.__decomposer(
            responses=[self.__RAW_STRING_PLAN, self.__RAW_STRING_PLAN]
        )

        with self.assertRaises(DecompositionError):
            await decomposer.decompose(intent="Tap the login button")

        self.assertEqual(llm.calls, 2)

    async def test_low_confidence_triggers_repair_then_fails_closed(self) -> None:
        """
        A plan below the confidence floor is a validation finding: one repair, then fail closed.
        """

        low = self.__plan(self.__task(), confidence=0.2)
        decomposer, llm = self.__decomposer(responses=[low, low])

        with self.assertRaises(DecompositionError):
            await decomposer.decompose(intent="Tap the login button")

        self.assertEqual(llm.calls, 2)

    async def test_low_confidence_recovers_when_repair_is_confident(self) -> None:
        """
        A confident repair after a low-confidence first attempt is accepted.
        """

        decomposer, llm = self.__decomposer(
            responses=[
                self.__plan(self.__task(), confidence=0.2),
                self.__plan(self.__task(), confidence=0.9),
            ]
        )

        sub_goals = await decomposer.decompose(intent="Tap the login button")

        self.assertEqual(llm.calls, 2)
        self.assertEqual(len(sub_goals), 1)

    async def test_empty_intent_is_rejected_before_any_generation(self) -> None:
        """
        An empty intent is a configuration error and never reaches the LLM.
        """

        decomposer, llm = self.__decomposer(responses=[])

        with self.assertRaises(ConfigurationError):
            await decomposer.decompose(intent="   ")

        self.assertEqual(llm.calls, 0)

    async def test_multi_task_plan_preserves_order_and_indices(self) -> None:
        """
        A multi-task plan materializes sequential, zero-based, contiguous sub-goals.
        """

        decomposer, _ = self.__decomposer(
            responses=[
                self.__plan(
                    self.__task(objective="Open Settings app"),
                    self.__task(objective="Confirm the search screen"),
                    self.__task(
                        objective="Store the balance",
                        proposal=CaptureProposal(
                            subject="balance",
                            name="account_balance",
                            provenance=CaptureNameProvenance.MODEL,
                        ),
                    ),
                )
            ]
        )

        sub_goals: List[SubGoal] = await decomposer.decompose(
            intent="Open Settings and store balance"
        )

        self.assertEqual([goal.index for goal in sub_goals], [0, 1, 2])
        self.assertIsInstance(sub_goals[-1].success, CaptureSuccess)


if __name__ == "__main__":
    unittest.main()
