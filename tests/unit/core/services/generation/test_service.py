from __future__ import annotations

import unittest
from typing import Any, List, Sequence, Tuple

from fathom.adapters.dialect.drizz.factory import DrizzDialectFactory
from fathom.constants.dialect import DialectName
from fathom.constants.flow import CheckKind, IssueCode, LaunchProvenance
from fathom.core.dialect.policy import Policy
from fathom.core.exceptions import LanguageComplianceError
from fathom.core.services.generation.binder import LaunchBinder
from fathom.core.services.generation.service import ScriptGenerationService
from fathom.interfaces.checker import Checker
from fathom.interfaces.dialect import Dialect
from fathom.interfaces.evidence import EvidenceSource
from fathom.interfaces.generator import FlowGenerator
from fathom.interfaces.renderer import Renderer
from fathom.schemas.flow import (
    Check,
    CheckNode,
    Evidence,
    EvidenceStep,
    Flow,
    Issue,
    LaunchNode,
    Report,
    RunObjective,
    Selector,
    StepLaunch,
    StepTarget,
    TapNode,
    TypeNode,
)


class FixedRenderer(Renderer):
    """
    Renderer that returns fixed text regardless of the flow.
    """

    def __init__(self, *, text: str) -> None:
        """
        Hold the text to return.
        """

        self.__text = text

    def render(self, *, flow: Flow) -> str:
        """
        Return the held text.
        """

        return self.__text


class MismatchChecker(Checker):
    """
    Checker that always reports a line-level round-trip mismatch.
    """

    def check(self, *, text: str) -> Report:
        """
        Report a single round-trip mismatch carrying a line-level diagnostic.
        """

        return Report(
            issues=(
                Issue(
                    code=IssueCode.ROUND_TRIP_MISMATCH,
                    message="Rendered text is not canonical Drizz at line 2: "
                    "expected 'Tap on Login', got 'Tap on  Login'.",
                ),
            )
        )


class StubDialect(Dialect):
    """
    Dialect binding a fixed renderer and a mismatch checker for round-trip logging tests.
    """

    def __init__(self, *, renderer: Renderer, checker: Checker) -> None:
        """
        Hold the renderer and checker to expose.
        """

        self.__renderer = renderer
        self.__checker = checker

    @property
    def name(self) -> DialectName:
        """
        Identify this dialect as Drizz.
        """

        return DialectName.DRIZZ

    @property
    def renderer(self) -> Renderer:
        """
        Return the bound renderer.
        """

        return self.__renderer

    @property
    def checker(self) -> Checker:
        """
        Return the bound checker.
        """

        return self.__checker


class StubEvidenceSource(EvidenceSource):
    """
    Returns a fixed evidence aggregate for any run.
    """

    def __init__(self, *, evidence: Evidence) -> None:
        """
        Hold the evidence to return.
        """

        self.__evidence = evidence

    async def read(self, *, run: str, objective: RunObjective) -> Evidence:
        """
        Return the held evidence.
        """

        return self.__evidence


class QueueFlowGenerator(FlowGenerator):
    """
    Returns queued flows in order, recording the feedback received on each call.
    """

    def __init__(self, *, flows: Sequence[Flow]) -> None:
        """
        Hold the flows to emit per attempt.
        """

        self.__flows = list(flows)
        self.received: List[Tuple[Issue, ...]] = []

    async def generate(self, *, evidence: Evidence, feedback: Tuple[Issue, ...] = ()) -> Flow:
        """
        Emit the next queued flow, repeating the last once the queue is exhausted.
        """

        self.received.append(feedback)
        index = min(len(self.received) - 1, len(self.__flows) - 1)
        return self.__flows[index]


class ScriptGenerationServiceTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover the evidence-to-script pipeline, including the bounded repair loop.
    """

    def setUp(self) -> None:
        """
        Build shared evidence, a real policy, the real Drizz dialect, and the run objective.
        """

        self.__evidence = Evidence(
            intent="open and verify",
            goal="home visible",
            package="com.example",
            steps=(
                EvidenceStep(
                    index=0,
                    event="launch",
                    action="launch",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                    ),
                ),
                EvidenceStep(
                    index=1,
                    event="action",
                    action="tap",
                    target=StepTarget(export="Login"),
                ),
                EvidenceStep(
                    index=2,
                    event="validation",
                    action="complete",
                    target=StepTarget(export="home"),
                ),
                EvidenceStep(
                    index=3,
                    event="action",
                    action="type",
                    text="\"'`",
                    target=StepTarget(export="search bar"),
                ),
            ),
        )
        self.__policy = Policy()
        self.__dialect = DrizzDialectFactory().create()
        self.__objective = RunObjective(
            intent="open and verify", goal="home visible", package="com.example"
        )

    def __service(
        self, *, flows: Sequence[Flow]
    ) -> Tuple[ScriptGenerationService, QueueFlowGenerator]:
        """
        Assemble the service with a stub evidence source around the queued flows.
        """

        generator = QueueFlowGenerator(flows=flows)
        service = ScriptGenerationService(
            evidence=StubEvidenceSource(evidence=self.__evidence),
            generator=generator,
            policy=self.__policy,
            dialect=self.__dialect,
            binder=LaunchBinder(),
        )
        return service, generator

    def __valid_flow(self) -> Flow:
        """
        Build a launch-first, grounded, terminally-validated flow.
        """

        return Flow(
            intent="open and verify",
            package="com.example",
            nodes=(
                LaunchNode(package="com.example", source_steps=()),
                TapNode(selector=Selector(text="Login"), source_steps=(1,)),
                CheckNode(
                    checks=(Check(kind=CheckKind.VISIBLE, subject="home"),), source_steps=(2,)
                ),
            ),
        )

    def __unterminated_flow(self) -> Flow:
        """
        Build a grounded flow that omits the terminal validation.
        """

        return Flow(
            intent="open and verify",
            package="com.example",
            nodes=(
                LaunchNode(package="com.example", source_steps=()),
                TapNode(selector=Selector(text="Login"), source_steps=(1,)),
            ),
        )

    def __unrenderable_flow(self) -> Flow:
        """
        Build a grounded flow whose typed value contains every quote and cannot render.
        """

        return Flow(
            intent="open and verify",
            package="com.example",
            nodes=(
                LaunchNode(package="com.example", source_steps=()),
                TypeNode(text="\"'`", field=Selector(text="search bar"), source_steps=(3,)),
                CheckNode(
                    checks=(Check(kind=CheckKind.VISIBLE, subject="home"),), source_steps=(2,)
                ),
            ),
        )

    async def test_grounded_flow_returns_script_on_first_attempt(self) -> None:
        """
        A valid flow is rendered, gated, and returned on the first attempt.
        """

        service, generator = self.__service(flows=(self.__valid_flow(),))
        result = await service.generate(run="run-1", objective=self.__objective)

        self.assertEqual(result.attempts, 1)
        self.assertIn("OPEN_APP: com.example", result.text)
        self.assertEqual(generator.received[0], ())

    async def test_repairs_then_succeeds(self) -> None:
        """
        A first-attempt gate failure feeds issues back and the repaired flow is returned.
        """

        service, generator = self.__service(flows=(self.__unterminated_flow(), self.__valid_flow()))
        result = await service.generate(run="run-2", objective=self.__objective)

        self.assertEqual(result.attempts, 2)
        self.assertEqual(generator.received[0], ())
        self.assertTrue(generator.received[1])

    async def test_unrenderable_value_is_repaired(self) -> None:
        """
        A flow that cannot render is fed back as an issue and the repaired flow is returned.
        """

        service, _ = self.__service(flows=(self.__unrenderable_flow(), self.__valid_flow()))
        result = await service.generate(run="run-3", objective=self.__objective)

        self.assertEqual(result.attempts, 2)

    async def test_partial_run_carries_partial_metadata(self) -> None:
        """
        A partial run's flag, discarded steps, and reason surface on the result.
        """

        evidence = Evidence(
            intent="open and verify",
            goal="home visible",
            package="com.example",
            partial=True,
            discarded=(7,),
            reason="loop thrash distilled",
            steps=(
                EvidenceStep(
                    index=0,
                    event="launch",
                    action="launch",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                    ),
                ),
                EvidenceStep(
                    index=1,
                    event="action",
                    action="tap",
                    target=StepTarget(export="Login"),
                ),
            ),
        )
        flow = Flow(
            intent="open and verify",
            package="com.example",
            partial=True,
            nodes=(
                LaunchNode(package="com.example", source_steps=()),
                TapNode(selector=Selector(text="Login"), source_steps=(1,)),
            ),
        )
        service = ScriptGenerationService(
            evidence=StubEvidenceSource(evidence=evidence),
            generator=QueueFlowGenerator(flows=(flow,)),
            policy=self.__policy,
            dialect=self.__dialect,
            binder=LaunchBinder(),
        )

        result = await service.generate(run="run-partial", objective=self.__objective)

        self.assertTrue(result.review.partial)
        self.assertEqual(result.review.discarded, (7,))
        self.assertEqual(result.review.reason, "loop thrash distilled")

    async def test_exhausts_repair_budget_and_raises(self) -> None:
        """
        A flow that never passes the gates raises after the bounded repair budget.
        """

        service, generator = self.__service(flows=(self.__unterminated_flow(),))

        with self.assertRaises(LanguageComplianceError):
            await service.generate(run="run-4", objective=self.__objective)

        self.assertEqual(len(generator.received), 3)

    @staticmethod
    def __events(records: List[Any]) -> List[Any]:
        """
        Extract the structured event identifiers from captured log records.
        """

        return [getattr(record, "event", None) for record in records]

    async def test_binder_application_is_logged_with_stamp_counts(self) -> None:
        """
        Binding the launch nodes logs the marker, launch, and stamped counts.
        """

        service, _ = self.__service(flows=(self.__valid_flow(),))

        with self.assertLogs(ScriptGenerationService.__module__, level="INFO") as captured:
            await service.generate(run="run-binder", objective=self.__objective)

        applied = next(
            record
            for record in captured.records
            if getattr(record, "event", None) == "script.binder.applied"
        )
        self.assertEqual(applied.__dict__["script.stamped_count"], 1)

    async def test_policy_failure_logs_issue_codes(self) -> None:
        """
        A flow rejected by the fidelity gate logs the policy failure with issue codes.
        """

        service, _ = self.__service(flows=(self.__unterminated_flow(),))

        with (
            self.assertLogs(ScriptGenerationService.__module__, level="INFO") as captured,
            self.assertRaises(LanguageComplianceError),
        ):
            await service.generate(run="run-policy", objective=self.__objective)

        self.assertIn("script.policy.failed", self.__events(captured.records))

    async def test_round_trip_mismatch_logs_line_level_diagnostic(self) -> None:
        """
        A syntax-checker round-trip mismatch logs the check failure carrying the line-level message.
        """

        service = ScriptGenerationService(
            evidence=StubEvidenceSource(evidence=self.__evidence),
            generator=QueueFlowGenerator(flows=(self.__valid_flow(),)),
            policy=self.__policy,
            dialect=StubDialect(
                renderer=FixedRenderer(text="OPEN_APP: com.example\nTap on  Login\n"),
                checker=MismatchChecker(),
            ),
            binder=LaunchBinder(),
        )

        with (
            self.assertLogs(ScriptGenerationService.__module__, level="INFO") as captured,
            self.assertRaises(LanguageComplianceError),
        ):
            await service.generate(run="run-check", objective=self.__objective)

        failed = next(
            record
            for record in captured.records
            if getattr(record, "event", None) == "script.check.failed"
        )
        self.assertTrue(
            any("line 2" in message for message in failed.__dict__["script.issue_messages"])
        )
