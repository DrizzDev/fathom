"""
Provider exhaustion must survive the whole path, not just the last hop.

A terminal 429 is the one failure a benchmark host must never score against a model:
nothing the model did caused it and nothing it could have done would have avoided it.
For the host to exclude the cell it has to *see* the exhaustion, which means the typed
error has to cross four boundaries - the adapter, the vision service, the graph node,
and the process - without any of them turning it into ordinary prose.

The dangerous boundary is the graph. ``AnalyzeNode`` ends the run gracefully on a broad
``except Exception``, recording ``CompletionReason.FAILED``; a provider outage caught
there becomes a workflow that finished and failed its task, which is indistinguishable
from a model that could not do the work. These tests drive the real ``VisionService``,
the real ``StepPlanner`` and the real ``AnalyzeNode`` against a fake adapter that raises
the way an exhausted provider does, and then drive the real ``CommandExecutor.run`` to
the process boundary.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import unittest
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from tests.builders import ScreenFixtures

from fathom.constants.llm import ModelProvider
from fathom.constants.state import CommonStateKey
from fathom.core.agent.planner import StepPlanner
from fathom.core.agent.state import AgentState
from fathom.core.exceptions import ProviderExhaustedError, VisionError
from fathom.core.services.timing import RunClock
from fathom.core.services.vision import VisionService
from fathom.interfaces.llm import LLMPort, PromptPart
from fathom.runtime.command.executor import CommandExecutor
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.llm import ModelIdentity, StructuredOutput
from fathom.schemas.results import GenerateResult, IntentResult
from fathom.schemas.retries import RetryLimits


class _ExhaustedProviderAdapter(LLMPort):
    """
    A deterministic adapter standing in for one whose retries have run out.

    It raises the same type the real adapters raise once they stop retrying, so the
    test exercises the contract rather than a message. Nothing here is random or
    time-dependent: the call that exhausts is fixed by construction.
    """

    def __init__(self, *, serve: int = 0) -> None:
        """
        Serve ``serve`` successful generations, then refuse every later call.
        """

        self.__serve = serve
        self.calls = 0

    @property
    def model_name(self) -> str:
        """
        A model name the prompt factory can resolve a builder for.
        """

        return "gpt-5-2"

    @property
    def identity(self) -> ModelIdentity:
        """
        Declared provider, so the real prompt builder is selected rather than guessed.
        """

        return ModelIdentity(provider=ModelProvider.OPENAI, model=self.model_name)

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
        Refuse the call the way an adapter does once its retry budget is spent.
        """

        self.calls += 1
        if self.calls > self.__serve:
            raise ProviderExhaustedError(
                f"OpenAI rate limit after {self.calls} attempts: 429 Too Many Requests"
            )
        return GenerateResult(content="", tool_calls=[], metrics={})

    async def prewarm(
        self, *, system_instruction: Optional[str], tools: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        No cache to warm.
        """

        return None

    async def cleanup(self) -> None:
        """
        No client to close.
        """

        return None


def _context_manager() -> Any:
    """
    A context manager whose accessors return real containers.

    Specced against the production class so a rename breaks the test rather than
    silently answering every attribute, and returning real lists and dicts so the
    vision service measures a shape instead of a Mock.
    """

    from fathom.core.context.manager import ContextManager

    stub = Mock(spec=ContextManager)
    stub.get_full_context = Mock(return_value={"trace": [], "milestones": []})
    stub.get_user_guidance = Mock(return_value=[])
    stub.consume_user_guidance = Mock()
    stub.clear_user_guidance = Mock()
    stub.clear_verifier_feedback = Mock()
    stub.clear_completion_feedback = Mock()
    return stub


def _analyze_node(*, llm: LLMPort) -> Any:
    """
    Build an ``AnalyzeNode`` over the real vision service and the real step planner.

    Only the device, telemetry and memory edges are doubled. The LLM boundary is the
    fake adapter and everything between it and the node is production code, which is
    the point: the absorption under test happens in that stretch.
    """

    from fathom.strategies.graph.intent.nodes.analyze import AnalyzeNode

    vision = VisionService(
        llm=llm,
        memory=SimpleNamespace(
            get_all=AsyncMock(return_value={}),
            retrieve_knowledge=AsyncMock(return_value={}),
        ),
        telemetry=SimpleNamespace(
            info=AsyncMock(), error=AsyncMock(), debug=AsyncMock(), warning=AsyncMock()
        ),
        use_cache=False,
        capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
    )
    agent_state = AgentState(
        max_steps=10,
        intent="Turn on airplane mode",
        retries=RetryLimits(planner=5),
        capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
    )
    provider = SimpleNamespace(
        persistence=SimpleNamespace(restore=Mock(), persist=Mock()),
        hitl=SimpleNamespace(prompt=AsyncMock()),
        is_cancelled=AsyncMock(return_value=False),
        context=SimpleNamespace(
            max_steps=10,
            use_xml=True,
            workflow_id="run-provider-exhaustion",
            intent="Turn on airplane mode",
            phase=AsyncMock(),
            clock=RunClock(),
            planner=StepPlanner(vision_tool=vision),
            reasoner=SimpleNamespace(select_best_action=Mock(return_value=None)),
            agent_state=agent_state,
            context_manager=_context_manager(),
            metrics=SimpleNamespace(record=Mock(), record_tokens=Mock()),
            memory=SimpleNamespace(set=AsyncMock()),
            telemetry=SimpleNamespace(info=AsyncMock(), error=AsyncMock()),
            signal=SimpleNamespace(supports_interruption=Mock(return_value=False)),
            device=SimpleNamespace(get_dimensions=AsyncMock(return_value=(1080, 2400))),
            configuration=SimpleNamespace(intent=SimpleNamespace(prompt_user_if_stuck=False)),
        ),
    )
    return AnalyzeNode(provider=provider), provider


@contextlib.asynccontextmanager
async def _null_context() -> Any:
    """
    An async context yielding nothing, standing in for the checkpoint store.
    """

    yield SimpleNamespace()


class GraphPropagationTest(unittest.IsolatedAsyncioTestCase):
    """
    The graph must not convert a provider outage into a completion reason.
    """

    async def test_exhaustion_escapes_the_analyze_node(self) -> None:
        """
        The regression this fix exists for.

        ``AnalyzeNode``'s broad handler used to catch this, mark the run complete with
        ``CompletionReason.FAILED`` and return a state, so the run finished normally
        and the outcome line said COMPLETED. The host then read a provider outage as a
        model that failed the task and scored it.
        """

        llm = _ExhaustedProviderAdapter()
        node, provider = _analyze_node(llm=llm)

        with self.assertRaises(ProviderExhaustedError):
            await node.run(
                state={
                    CommonStateKey.CAPTURE: ScreenFixtures.capture(
                        activity="com.android.settings", width=1080, height=2400
                    )
                }
            )

        self.assertEqual(llm.calls, 1, "the adapter was reached through the real path")
        self.assertFalse(
            provider.context.agent_state.is_complete,
            "the run must not be marked complete by a provider outage",
        )

    async def test_the_run_is_not_recorded_as_stuck_or_failed(self) -> None:
        """
        The specific mislabelling: no completion reason may be persisted at all.

        Absorbing the error wrote FAILED into the graph state, and a run that then ran
        out of usable steps surfaced as "Stuck: No progress" - a phrase describing the
        model, for something the model did not do.
        """

        node, provider = _analyze_node(llm=_ExhaustedProviderAdapter())

        with self.assertRaises(ProviderExhaustedError):
            await node.run(
                state={
                    CommonStateKey.CAPTURE: ScreenFixtures.capture(
                        activity="com.android.settings", width=1080, height=2400
                    )
                }
            )

        provider.persistence.persist.assert_not_called()
        self.assertIsNone(provider.context.agent_state.completion_reason)

    async def test_cancellation_still_propagates_from_analyze(self) -> None:
        """
        The behaviour the widened handler inherited, pinned.

        ``UNABSORBABLE_ERRORS`` replaced a clause that caught ``CancelledError`` alone.
        Adding a member to a tuple cannot remove one, but nothing else in the suite
        would notice if the tuple were later rewritten, and cancellation that stopped
        unwinding would hang the LangGraph task tree rather than fail loudly.
        """

        class _Cancelled(_ExhaustedProviderAdapter):
            async def generate(self, **kwargs: Any) -> GenerateResult:
                raise asyncio.CancelledError()

        node, provider = _analyze_node(llm=_Cancelled())

        with self.assertRaises(asyncio.CancelledError):
            await node.run(
                state={
                    CommonStateKey.CAPTURE: ScreenFixtures.capture(
                        activity="com.android.settings", width=1080, height=2400
                    )
                }
            )

        self.assertFalse(provider.context.agent_state.is_complete)

    async def test_an_ordinary_vision_failure_is_still_absorbed(self) -> None:
        """
        The fix must stay narrow.

        A generic ``VisionError`` is a fault the agent can end gracefully on, and
        turning every LLM problem into a crashed process would trade one
        misclassification for another.
        """

        class _Failing(_ExhaustedProviderAdapter):
            async def generate(self, **kwargs: Any) -> GenerateResult:
                raise VisionError("LLM fail: malformed response")

        node, provider = _analyze_node(llm=_Failing())

        state = await node.run(
            state={
                CommonStateKey.CAPTURE: ScreenFixtures.capture(
                    activity="com.android.settings", width=1080, height=2400
                )
            }
        )

        self.assertTrue(state.get(CommonStateKey.IS_COMPLETE))


class ProcessBoundaryTest(unittest.IsolatedAsyncioTestCase):
    """
    What a benchmark host reading stdout actually receives.
    """

    @staticmethod
    def __request() -> SimpleNamespace:
        """
        The fields ``run`` and ``__invoke_runner`` read off the request.
        """

        return SimpleNamespace(
            principal=None,
            objective=SimpleNamespace(
                intent="Turn on airplane mode",
                use_xml=True,
                max_steps=40,
                package_name="com.android.settings",
            ),
            runtime=SimpleNamespace(
                interactive=False, session_id="s-1", execution_id="e-1", signal_type="none"
            ),
            memory=SimpleNamespace(context_scope=None),
            interaction=SimpleNamespace(realignment=None),
        )

    async def __run(self, *, outcome: Any) -> tuple[int, dict]:
        """
        Drive the real ``CommandExecutor.run`` with the runner boundary substituted.

        ``run`` owns the emission, so the assembly below it is doubled and everything
        the test asserts on - the handler that fires, the status translated, the line
        written - is production code.
        """

        executor = CommandExecutor.__new__(CommandExecutor)
        executor._CommandExecutor__runner = None
        executor._CommandExecutor__composition = None

        runner = SimpleNamespace(run_intent=AsyncMock(**outcome), cleanup=AsyncMock())
        composition = SimpleNamespace(runner=runner, resources=[])

        with (
            patch.object(CommandExecutor, "_CommandExecutor__setup_signals", Mock()),
            patch.object(CommandExecutor, "_CommandExecutor__create_signal_adapter", Mock()),
            patch.object(
                CommandExecutor,
                "_CommandExecutor__create_runner",
                AsyncMock(return_value=composition),
            ),
            patch.dict("os.environ", {"PLUMB_RUN_ID": "run-1"}),
            patch("fathom.runtime.plumb_stream.sys.stdout") as stdout,
        ):
            code = await executor.run(request=self.__request())

        written = "".join(str(item.args[0]) for item in stdout.write.call_args_list)
        lines = [part for part in written.splitlines() if part.startswith("PLUMB_OUTCOME ")]
        self.assertEqual(len(lines), 1, "exactly one terminal outcome line")
        return code, json.loads(lines[0][len("PLUMB_OUTCOME ") :])

    async def test_exhaustion_reaches_the_process_boundary(self) -> None:
        """
        The four properties the host depends on, asserted on the emitted line itself.
        """

        code, payload = await self.__run(
            outcome={"side_effect": ProviderExhaustedError("OpenAI rate limit: 429")}
        )

        self.assertEqual(payload["status"], "PROVIDER_EXHAUSTED")
        self.assertEqual(payload["status_source"], "PAYLOAD")
        self.assertEqual(payload["completion_reason"], "ProviderExhaustedError")
        self.assertNotIn("Stuck", payload["completion_reason"])
        self.assertEqual(code, 1)

    async def test_cancellation_reaches_the_boundary_as_cancelled(self) -> None:
        """
        ``run`` catches ``CancelledError`` alongside ``KeyboardInterrupt`` and returns
        1. That handler now also emits, so the value it emits is worth pinning: a run an
        operator stopped is not a run the provider stopped, and neither is a task the
        model failed.
        """

        _, payload = await self.__run(outcome={"side_effect": asyncio.CancelledError()})

        self.assertEqual(payload["status"], "CANCELLED")
        self.assertEqual(payload["completion_reason"], "CancelledError")

    async def test_a_task_failure_stays_distinguishable_from_an_outage(self) -> None:
        """
        The contrast that gives the previous test its meaning.

        Both exit 1 - which is exactly why the exit status cannot be the signal, and
        why the payload has to carry the distinction.
        """

        failed = IntentResult(
            intent="Turn on airplane mode",
            status="failed",
            success=False,
            steps_taken=19,
            duration=275.4,
            completion_reason="Stuck: No progress",
        )
        code, payload = await self.__run(outcome={"return_value": failed})

        self.assertEqual(payload["status"], "COMPLETED")
        self.assertEqual(payload["completion_reason"], "Stuck: No progress")
        self.assertEqual(code, 1)

        outage_code, outage = await self.__run(
            outcome={"side_effect": ProviderExhaustedError("429")}
        )

        self.assertEqual(code, outage_code, "the exit status alone cannot tell them apart")
        self.assertNotEqual(payload["status"], outage["status"])


class VerifyBoundaryTest(unittest.IsolatedAsyncioTestCase):
    """
    ``VerifyNode`` calls the LLM directly, on its own handler.

    Its broad ``except Exception`` records a failed verification and lets the run carry
    on, which is right for a malformed answer and wrong for a provider that stopped
    answering: the verification never happened, so recording that it failed is a claim
    about the model nobody is entitled to make.
    """

    def __node(self, *, llm: Any) -> Any:
        from fathom.strategies.graph.intent.nodes.verify import VerifyNode

        agent_state = AgentState(
            max_steps=10,
            intent="Turn on airplane mode",
            retries=RetryLimits(planner=5),
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        capture = ScreenFixtures.capture(activity="com.android.settings", width=1080, height=2400)
        provider = SimpleNamespace(
            persistence=SimpleNamespace(restore=Mock(), persist=Mock()),
            is_cancelled=AsyncMock(return_value=False),
            context=SimpleNamespace(
                llm=llm,
                intent="Turn on airplane mode",
                workflow_id="run-verify",
                agent_state=agent_state,
                phase=AsyncMock(),
                clock=RunClock(),
                context_manager=_context_manager(),
                telemetry=SimpleNamespace(info=AsyncMock(), error=AsyncMock()),
                perception=SimpleNamespace(perceive=AsyncMock(return_value=capture)),
            ),
        )
        return VerifyNode(provider=provider), provider

    async def __run(self, *, llm: Any) -> Any:
        from fathom.constants.state import IntentStateKey, VerifyMode

        node, provider = self.__node(llm=llm)
        state = {IntentStateKey.VERIFY_MODE: VerifyMode.FULL_INTENT.value}
        return node, provider, state

    async def test_exhaustion_escapes_the_verify_node(self) -> None:
        """
        The provider failure must leave the node rather than become a verdict.
        """

        node, provider, state = await self.__run(llm=_ExhaustedProviderAdapter())

        with self.assertRaises(ProviderExhaustedError):
            await node.run(state=state)

        self.assertFalse(
            provider.context.agent_state.is_complete,
            "a provider outage must not complete the run",
        )

    async def test_an_ordinary_verify_failure_is_still_absorbed(self) -> None:
        """
        A malformed verification answer stays a verification failure, as before.
        """

        class _Failing(_ExhaustedProviderAdapter):
            async def generate(self, **kwargs: Any) -> GenerateResult:
                raise VisionError("LLM fail: unparseable verdict")

        node, _, state = await self.__run(llm=_Failing())
        result = await node.run(state=state)

        self.assertIsInstance(result, dict)

    async def test_cancellation_still_propagates_from_verify(self) -> None:
        """
        The behaviour that already existed at this site, pinned so widening the handler
        cannot quietly narrow it.
        """

        class _Cancelled(_ExhaustedProviderAdapter):
            async def generate(self, **kwargs: Any) -> GenerateResult:
                raise asyncio.CancelledError()

        node, _, state = await self.__run(llm=_Cancelled())

        with self.assertRaises(asyncio.CancelledError):
            await node.run(state=state)


class StrategyBoundaryTest(unittest.IsolatedAsyncioTestCase):
    """
    The two handlers between the graph and the runner.

    ``__run_executor`` sees whatever LangGraph propagates out of a node, and ``execute``
    sees whatever ``__run_executor`` re-raises. Both end the run gracefully on a broad
    handler, so both had to be taught the difference.
    """

    @staticmethod
    def __strategy() -> Any:
        from fathom.strategies.intent import IntentStrategy

        strategy = IntentStrategy.__new__(IntentStrategy)
        strategy._IntentStrategy__workflow_id = "run-strategy"
        return strategy

    async def test_run_executor_reraises_exhaustion(self) -> None:
        """
        Its broad handler returns ``RunOutcome.FAILED`` - a graded result for something
        that was never graded.
        """

        strategy = self.__strategy()
        executor = SimpleNamespace(
            run=AsyncMock(side_effect=ProviderExhaustedError("429 after 5 attempts"))
        )

        with self.assertRaises(ProviderExhaustedError):
            await strategy._IntentStrategy__run_executor(executor=executor)

    async def test_run_executor_still_absorbs_an_ordinary_failure(self) -> None:
        """
        An ordinary crash inside the graph still becomes a FAILED outcome.
        """

        from fathom.constants.state import RunOutcome

        strategy = self.__strategy()
        executor = SimpleNamespace(run=AsyncMock(side_effect=RuntimeError("boom")))

        outcome = await strategy._IntentStrategy__run_executor(executor=executor)
        self.assertIs(outcome, RunOutcome.FAILED)

    async def test_run_executor_still_reraises_cancellation(self) -> None:
        """
        Unchanged behaviour, pinned.
        """

        strategy = self.__strategy()
        executor = SimpleNamespace(run=AsyncMock(side_effect=asyncio.CancelledError()))

        with self.assertRaises(asyncio.CancelledError):
            await strategy._IntentStrategy__run_executor(executor=executor)

    async def test_execute_reraises_exhaustion_without_finalizing(self) -> None:
        """
        ``execute``'s clause sits ahead of the cancellation handler, which finalizes the
        run *as cancelled*. A provider outage must take neither that path nor the broad
        one that assembles a FAILED result: there is no honest ending to record.
        """

        strategy = self.__strategy()
        context = SimpleNamespace(
            configuration=SimpleNamespace(
                intent=SimpleNamespace(
                    finalization=SimpleNamespace(
                        graph=SimpleNamespace(checkpointer_close=1.0),
                        runtime=SimpleNamespace(background_drain=1.0, context_shutdown=1.0),
                    )
                )
            ),
            vision=SimpleNamespace(prewarm=AsyncMock()),
            abort_detector=SimpleNamespace(warmup=AsyncMock()),
            realignment=SimpleNamespace(immediate=False),
            signal=SimpleNamespace(supports_interruption=Mock(return_value=False)),
            cancel=Mock(),
        )
        strategy._IntentStrategy__graph_context = context
        strategy._IntentStrategy__interrupt_nodes = ()
        strategy._IntentStrategy__checkpoint_store = SimpleNamespace(
            open=Mock(return_value=_null_context())
        )
        strategy._IntentStrategy__graph_builder = SimpleNamespace(build=Mock())
        strategy._IntentStrategy__resolve_plan = AsyncMock()
        strategy._IntentStrategy__shutdown_graph_context = AsyncMock()
        strategy._IntentStrategy__cleanup_background_task = AsyncMock()
        strategy._IntentStrategy__run_executor = AsyncMock(
            side_effect=ProviderExhaustedError("429")
        )
        finalize = AsyncMock()
        strategy._IntentStrategy__finalize_run = finalize
        strategy._IntentStrategy__finalize_cancelled_run = finalize
        strategy._IntentStrategy__finalize_failed_run = finalize

        with self.assertRaises(ProviderExhaustedError):
            await strategy.execute()

        finalize.assert_not_awaited()
        context.cancel.assert_not_called()


class OutcomeEvidenceTest(unittest.IsolatedAsyncioTestCase):
    """
    The post-action verdict is evidence, so a refused call is not an unclear one.

    ``OutcomeObserver.assess`` degraded every failure to ``VisualVerdict.UNCLEAR``, and
    ``SubGoalEvaluator`` consumes that verdict as the visual proof of an observed
    sub-goal. UNCLEAR therefore asserts something specific: the model looked and the
    screen did not settle the question. When the provider refused the call nobody
    looked. The sub-goal does not advance, the agent retries the same step, and the run
    ends "Stuck: No progress" - a verdict about the model, produced by the provider.
    """

    @staticmethod
    def __requirement() -> Any:
        from fathom.schemas.success import ObservationRequirement

        return ObservationRequirement(assertion="airplane mode is on")

    async def test_exhaustion_escapes_the_outcome_observer(self) -> None:
        """
        The defect: this is the handler that turned a provider outage into evidence.
        """

        from fathom.core.services.outcome import OutcomeObserver

        observer = OutcomeObserver(llm=_ExhaustedProviderAdapter())

        with self.assertRaises(ProviderExhaustedError):
            await observer.assess(
                requirement=self.__requirement(), before=b"before", after=b"after"
            )

    async def test_an_ordinary_adjudication_failure_still_degrades_to_unclear(self) -> None:
        """
        The degradation is the right answer for a model that answered badly, and must
        survive. A malformed verdict means the question genuinely was not settled.
        """

        from fathom.constants.assessment import VisualVerdict
        from fathom.core.services.outcome import OutcomeObserver

        class _Failing(_ExhaustedProviderAdapter):
            async def generate(self, **kwargs: Any) -> GenerateResult:
                raise VisionError("LLM fail: unparseable assessment")

        observer = OutcomeObserver(llm=_Failing())
        assessment = await observer.assess(
            requirement=self.__requirement(), before=b"before", after=b"after"
        )

        self.assertIs(assessment.verdict, VisualVerdict.UNCLEAR)
        self.assertEqual(assessment.confidence, 0.0)

    async def test_cancellation_still_propagates_from_the_outcome_observer(self) -> None:
        """
        ``CancelledError`` derives from ``BaseException``, so the broad handler never
        caught it and this behaviour is unchanged. Pinned because the new clause names
        it explicitly, and a later edit narrowing that clause would start absorbing it.
        """

        from fathom.core.services.outcome import OutcomeObserver

        class _Cancelled(_ExhaustedProviderAdapter):
            async def generate(self, **kwargs: Any) -> GenerateResult:
                raise asyncio.CancelledError()

        observer = OutcomeObserver(llm=_Cancelled())

        with self.assertRaises(asyncio.CancelledError):
            await observer.assess(
                requirement=self.__requirement(), before=b"before", after=b"after"
            )


class CompletionPathTest(unittest.IsolatedAsyncioTestCase):
    """
    The real evaluator, over the real observer, over a refusing adapter.

    This is the chain the defect actually travels: RecordNode asks the evaluator whether
    the sub-goal is done, the evaluator asks the observer, the observer asks the model.
    """

    @staticmethod
    def __evaluator(*, llm: Any) -> Any:
        from tests.builders import SubGoalFixtures, SuccessFixtures

        from fathom.core.services.criterion import CriterionObserver
        from fathom.core.services.outcome import OutcomeObserver
        from fathom.schemas.criterion import (
            CriterionDecision,
            CriterionSource,
            CriterionVerdict,
        )
        from fathom.strategies.graph.intent.nodes.completion import SubGoalEvaluator

        state = AgentState(
            intent="Turn on airplane mode",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        state.set_sub_goals(
            [
                SubGoalFixtures.make(
                    index=0,
                    description="open quick settings",
                    success=SuccessFixtures.observed(assertion="airplane mode is on"),
                ),
                SubGoalFixtures.make(index=1, description="toggle airplane mode"),
            ]
        )

        class _Checker(CriterionObserver):
            """
            A symbolic criterion that always agrees, so the visual verdict is the only
            thing left that can decide the sub-goal.
            """

            def __init__(self) -> None:
                pass

            async def check(self, **kwargs: Any) -> CriterionDecision:
                return CriterionDecision(
                    verdict=CriterionVerdict.SATISFIED,
                    source=CriterionSource.SYMBOLIC,
                    confidence=0.95,
                    evidence=(),
                    notes=None,
                )

        context = Mock(name="GraphContext")
        context.agent_state = state
        context.workflow_id = "wf"
        context.intent = "Turn on airplane mode"
        context.phase = AsyncMock()
        context.clock = RunClock()
        context.context_manager = AsyncMock()

        return SubGoalEvaluator(
            context=context,
            criterion_observer=_Checker(),
            outcome=OutcomeObserver(llm=llm),
        ), state

    @staticmethod
    def __plan() -> Any:
        from fathom.constants import ActionType
        from fathom.schemas.actions import Action
        from fathom.schemas.results import AnalysisResult, PlanContext, PlanResult

        analysis = AnalysisResult(
            action=Action(action_type=ActionType.TAP, target="t", rationale="r", confidence=1.0),
            reasoning="r",
            screen_description="s",
            metadata={"tool_args": {}},
        )
        return PlanResult(
            step=None, is_complete=False, reason="t", context=PlanContext(analysis=analysis)
        )

    @staticmethod
    def __step_result() -> Any:
        from fathom.constants import ActionType
        from fathom.schemas.actions import Action
        from fathom.schemas.artifacts import (
            ScreenArtifact,
            ScreenArtifactBundle,
            StepArtifacts,
        )
        from fathom.schemas.steps import Step, StepResult

        # Artifacts matter: OutcomeObserver.frames returns None without before/after
        # bytes, and a None there would skip adjudication entirely and pass the test
        # for the wrong reason.
        step = Step(
            action=Action(
                action_type=ActionType.TAP,
                target="element",
                natural_language_target="Airplane mode",
                rationale="r",
                confidence=1.0,
            ),
            step_number=0,
            screen_hash="pre",
        )
        return StepResult(
            step=step,
            success=True,
            executed=True,
            artifacts=StepArtifacts(
                screen=ScreenArtifactBundle(
                    before=ScreenArtifact(image=b"before"),
                    after=ScreenArtifact(image=b"after"),
                )
            ),
            pre_hash="pre",
            post_hash="post",
            screen_changed=True,
            duration=1,
        )

    async def test_exhaustion_escapes_the_completion_evaluation(self) -> None:
        """
        Nothing between RecordNode and the adapter may turn this into a verdict.
        """

        llm = _ExhaustedProviderAdapter()
        evaluator, state = self.__evaluator(llm=llm)

        with self.assertRaises(ProviderExhaustedError):
            await evaluator.evaluate(
                plan=self.__plan(), step_result=self.__step_result(), accumulated=[]
            )

        self.assertEqual(llm.calls, 1, "the adapter was reached through the real chain")
        self.assertEqual(
            state.current_sub_goal_index,
            0,
            "an unproven sub-goal must not advance on a call that never happened",
        )


class RecordBoundaryTest(unittest.IsolatedAsyncioTestCase):
    """
    ``RecordNode`` is the graph node that owns the completion evaluation.
    """

    @staticmethod
    def __state() -> dict:
        """
        The graph state that carries RecordNode as far as the completion evaluation.
        """

        from fathom.constants import ActionType
        from fathom.constants.state import CommonStateKey, IntentStateKey
        from fathom.schemas.actions import Action
        from fathom.schemas.results import PlanResult
        from fathom.schemas.screens import ScreenState
        from fathom.schemas.steps import Step, StepResult

        step_result = StepResult(
            step=Step(
                action=Action(
                    action_type=ActionType.TAP,
                    target="Airplane mode",
                    rationale="toggle it",
                    confidence=1.0,
                ),
                step_number=0,
                screen_hash="pre",
                metadata={"started_at": "2026-09-02T10:00:00+00:00"},
            ),
            success=True,
            duration=12,
            screen_changed=True,
            pre_hash="pre",
            post_hash="post",
        )
        screen = ScreenState(
            timestamp=0,
            activity="com.android.settings/.Settings",
            activity_hash="a" * 16,
            visual_hash="b" * 16,
        )
        return {
            CommonStateKey.STEP_RESULT: step_result,
            CommonStateKey.SCREEN_STATE: screen,
            CommonStateKey.IS_NEW_SCREEN: False,
            IntentStateKey.POST_ACTIVITY: "com.android.settings/.Settings",
            IntentStateKey.PLAN: PlanResult(is_complete=False, reason="continue", step=None),
        }

    def __node(self, *, effect: BaseException) -> Any:
        from fathom.strategies.graph.intent.nodes.record import RecordNode

        agent_state = AgentState(
            intent="Turn on airplane mode",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )

        provider = MagicMock(name="IntentNodeProvider")
        provider.is_cancelled = AsyncMock(return_value=False)
        provider.context.workflow_id = "run-record"
        provider.context.agent_state = agent_state
        provider.context.telemetry.info = AsyncMock()
        provider.context.telemetry.error = AsyncMock()
        provider.context.telemetry.warning = AsyncMock()
        provider.context.memory.store_experience = AsyncMock()
        provider.context.context_manager.commit = AsyncMock()
        provider.context.context_manager.branch = AsyncMock()
        provider.context.context_manager.get_full_context.return_value = {"active_count": 0}
        provider.context.auditor.log_step = MagicMock()
        provider.persistence.persist = MagicMock()
        provider.persistence.enqueue_history = MagicMock()
        provider.persistence.should_skip_launcher.return_value = False
        provider.completion.evaluate = AsyncMock(side_effect=effect)
        return RecordNode(provider=provider), provider

    async def test_record_node_reraises_exhaustion_from_completion(self) -> None:
        """
        Its broad handler would end the run gracefully, which for this error means
        recording a step that failed to prove its sub-goal.
        """

        node, _ = self.__node(effect=ProviderExhaustedError("429 after 5 attempts"))

        with self.assertRaises(ProviderExhaustedError):
            await node.run(state=self.__state())

    async def test_record_node_still_reraises_cancellation(self) -> None:
        """
        Unchanged behaviour at the site the clause replaced.
        """

        node, _ = self.__node(effect=asyncio.CancelledError())

        with self.assertRaises(asyncio.CancelledError):
            await node.run(state=self.__state())

    async def test_record_node_still_absorbs_an_ordinary_failure(self) -> None:
        """
        A crash inside the evaluation is still ended gracefully by the node.
        """

        node, provider = self.__node(effect=RuntimeError("boom"))
        result = await node.run(state=self.__state())

        provider.completion.evaluate.assert_awaited()
        self.assertIsInstance(result, dict)


class RecordToBoundaryTest(ProcessBoundaryTest):
    """
    The whole point: what the host reads when the completion evaluation is refused.
    """

    async def test_completion_exhaustion_emits_provider_exhausted(self) -> None:
        """
        The error the completion path now raises, carried to the process boundary.

        Before this fix that same run reached the host as COMPLETED with
        "Stuck: No progress", and was scored against the model.
        """

        _, payload = await self._ProcessBoundaryTest__run(
            outcome={
                "side_effect": ProviderExhaustedError(
                    "OpenAI rate limit after 5 attempts: 429 Too Many Requests"
                )
            }
        )

        self.assertEqual(payload["status"], "PROVIDER_EXHAUSTED")
        self.assertNotEqual(payload["completion_reason"], "Stuck: No progress")


if __name__ == "__main__":
    unittest.main()
