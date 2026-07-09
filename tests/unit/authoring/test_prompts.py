from __future__ import annotations

import unittest

from fathom.authoring.agent.packet import AuthoringPacketBuilder
from fathom.authoring.agent.prompts import AuthoringPromptFactory
from fathom.authoring.agent.reference import AuthoringReferenceProvider
from fathom.authoring.evidence import AuthoringEvidenceBuilder
from fathom.constants.authoring import AuthoringExampleKind, AuthoringKind
from fathom.constants.dialect import DialectName
from fathom.schemas.authoring import (
    AuthoringEvidence,
    AuthoringTask,
    RepairAuthoringEvidence,
)
from fathom.schemas.flow import Evidence, EvidenceStep, StepTarget, TargetAnchors, TargetClaim


class AuthoringPromptFactoryTest(unittest.TestCase):
    """
    Cover task-specific authoring prompt strategies.
    """

    def test_run_step_and_repair_prompts_share_contract_but_have_distinct_objectives(self) -> None:
        """
        The single agent must select distinct prompt strategies by task kind.
        """

        factory = AuthoringPromptFactory()

        self.assertIn("complete run script", factory.prompt(kind=AuthoringKind.RUN).objective())
        self.assertIn("selected step", factory.prompt(kind=AuthoringKind.STEP).objective())
        self.assertIn("repair", factory.prompt(kind=AuthoringKind.REPAIR).objective())

    def test_prompt_is_generic_and_packet_based(self) -> None:
        """
        Prompt text must be task-generic and carry the typed packet.
        """

        evidence = Evidence(
            intent="open app",
            goal="home visible",
            package="com.example",
            steps=(
                EvidenceStep(
                    action="tap",
                    event="action",
                    index=1,
                    target=StepTarget(
                        claim=TargetClaim(text="wrapper text", verified=False),
                        anchors=TargetAnchors(accessibility=("product card",)),
                    ),
                ),
            ),
        )
        task = AuthoringTask(
            kind=AuthoringKind.RUN,
            intent="open app",
            step_number=2,
            execution_id="execution-1",
            evidence=AuthoringEvidenceBuilder().build_run(evidence=evidence),
        )
        reference = AuthoringReferenceProvider().reference(dialect=DialectName.DRIZZ)
        packet = AuthoringPacketBuilder().build(task=task, dialect=reference)
        prompt = AuthoringPromptFactory().prompt(kind=AuthoringKind.RUN)

        system = prompt.system_instruction()
        user = prompt.user_prompt(packet=packet)

        self.assertIn("# Identity", system)
        self.assertIn("# Contract", system)
        self.assertIn("# Method", system)
        self.assertIn("# Examples", system)
        self.assertIn("# Output", system)
        self.assertIn("dialect reference", system)
        self.assertIn("evidence", system)
        self.assertIn("few-shot examples", system)
        self.assertIn("target.anchors", system)
        self.assertIn("target.structure", system)
        self.assertIn("never use target.claim.text", system)
        self.assertIn("actual visible UI role", system)
        self.assertIn("dropdown, not bar", system)
        self.assertIn("partial Flow", system)
        self.assertIn("# Dialect Reference", user)
        self.assertIn("# Evidence", user)
        self.assertIn('"guide"', user)
        self.assertIn('"examples"', user)
        self.assertIn('"scenarios"', user)
        self.assertIn('"lexicon"', user)
        self.assertIn('"anchors"', user)
        self.assertIn('"claim"', user)
        self.assertIn('"verified":false', user)
        self.assertIn("# Task", user)

    def test_reference_exposes_plain_scroll_and_scroll_until_separately(self) -> None:
        """
        Drizz reference must expose plain scroll and scroll-until as distinct commands.
        """

        reference = AuthoringReferenceProvider().reference(dialect=DialectName.DRIZZ)
        commands = {command.name: command for command in reference.commands}

        self.assertIn("scroll", commands)
        self.assertIn("scroll_until", commands)
        self.assertEqual(commands["scroll"].syntax, "Scroll <direction>")
        self.assertEqual(
            commands["scroll_until"].syntax,
            "Scroll <direction> until <target>",
        )

    def test_reference_carries_drizz_guide_and_examples(self) -> None:
        """
        Drizz reference must teach command semantics, not only list syntax.
        """

        reference = AuthoringReferenceProvider().reference(dialect=DialectName.DRIZZ)
        commands = {command.name: command for command in reference.commands}

        self.assertIn("replayable script", " ".join(reference.guide.principles))
        self.assertIn("dynamic target", " ".join(reference.guide.selection))
        self.assertIn("candidate aliases", " ".join(reference.guide.selection))
        self.assertIn("target.anchors", " ".join(reference.guide.selection))
        self.assertIn("target.claim.verified", " ".join(reference.guide.selection))
        self.assertIn("placeholder", " ".join(reference.guide.selection))
        self.assertTrue(any(term.term == "product grid" for term in reference.guide.lexicon))
        self.assertTrue(any(term.term == "search field" for term in reference.guide.lexicon))
        self.assertIn("Merge repeated attempts", " ".join(reference.guide.composition))
        self.assertIn("partial Flow", " ".join(reference.guide.completion))
        self.assertGreaterEqual(len(reference.guide.scenarios), 4)
        self.assertTrue(
            any("assertion_ids" in scenario.preferred for scenario in reference.guide.scenarios)
        )
        self.assertTrue(
            any("Flow.partial" in scenario.preferred for scenario in reference.guide.scenarios)
        )
        self.assertTrue(any("UI role" in scenario.reason for scenario in reference.guide.scenarios))
        self.assertTrue(
            any("visible state" in scenario.reason for scenario in reference.guide.scenarios)
        )
        self.assertTrue(
            any("Conditional guards" in scenario.reason for scenario in reference.guide.scenarios)
        )
        self.assertTrue(
            any(
                "actual visible UI role" in scenario.preferred
                for scenario in reference.guide.scenarios
            )
        )

        store_examples = commands["store"].examples
        validate_examples = commands["validate"].examples
        tap_examples = commands["tap"].examples
        scroll_until_examples = commands["scroll_until"].examples

        self.assertTrue(
            any(example.kind is AuthoringExampleKind.PREFERRED for example in store_examples)
        )
        self.assertTrue(
            any(example.kind is AuthoringExampleKind.AVOID for example in store_examples)
        )
        self.assertTrue(
            any(example.kind is AuthoringExampleKind.AVOID for example in validate_examples)
        )
        self.assertTrue(any("selection context" in example.reason for example in tap_examples))
        self.assertTrue(any("UI role" in example.reason for example in tap_examples))
        self.assertTrue(any("location dropdown" in example.command for example in tap_examples))
        self.assertTrue(any("null minutes" in example.command for example in tap_examples))
        self.assertTrue(any("visible state" in example.reason for example in scroll_until_examples))
        self.assertTrue(
            any("concatenates aliases" in example.reason for example in commands["type"].examples)
        )
        self.assertTrue(
            any(
                "visible object and the condition" in example.reason
                for example in scroll_until_examples
            )
        )

    def test_strategy_accepts_all_evidence_views(self) -> None:
        """
        Prompt strategies must accept run, step, and repair evidence packets.
        """

        evidence = Evidence(
            intent="open app",
            goal="home visible",
            package="com.example",
            steps=(EvidenceStep(action="tap", event="action", index=1),),
        )
        reference = AuthoringReferenceProvider().reference(dialect=DialectName.DRIZZ)
        tasks = (
            AuthoringTask(
                kind=AuthoringKind.RUN,
                intent="open app",
                step_number=2,
                execution_id="execution-1",
                evidence=AuthoringEvidenceBuilder().build_run(evidence=evidence),
            ),
            AuthoringTask(
                kind=AuthoringKind.STEP,
                intent="open app",
                step_number=2,
                execution_id="execution-1",
                evidence=AuthoringEvidenceBuilder().build_step(evidence=evidence, step_index=1),
            ),
            AuthoringTask(
                kind=AuthoringKind.REPAIR,
                intent="repair",
                step_number=2,
                execution_id="execution-1",
                evidence=AuthoringEvidence(repair=RepairAuthoringEvidence(script="Tap on Search")),
            ),
        )

        for task in tasks:
            packet = AuthoringPacketBuilder().build(task=task, dialect=reference)
            prompt = AuthoringPromptFactory().prompt(kind=task.kind)
            self.assertIn("# Evidence", prompt.user_prompt(packet=packet))
