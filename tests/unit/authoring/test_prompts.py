from __future__ import annotations

import unittest

from fathom.authoring.agent.packet import AuthoringPacketBuilder
from fathom.authoring.agent.prompts import AuthoringPromptFactory
from fathom.authoring.agent.reference import AuthoringReferenceProvider
from fathom.constants.authoring import AuthoringExampleKind, AuthoringKind
from fathom.constants.dialect import DialectName
from fathom.schemas.authoring import (
    AuthoringEvidence,
    AuthoringTask,
    RepairAuthoringEvidence,
    RunAuthoringEvidence,
    StepAuthoringEvidence,
)
from fathom.schemas.flow import Evidence, EvidenceStep


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
            steps=(EvidenceStep(action="tap", event="action", index=1),),
        )
        task = AuthoringTask(
            kind=AuthoringKind.RUN,
            intent="open app",
            step_number=2,
            workflow_id="workflow-1",
            evidence=AuthoringEvidence(run=RunAuthoringEvidence(evidence=evidence)),
        )
        reference = AuthoringReferenceProvider().reference(dialect=DialectName.DRIZZ)
        packet = AuthoringPacketBuilder().build(task=task, dialect=reference)
        prompt = AuthoringPromptFactory().prompt(kind=AuthoringKind.RUN)

        system = prompt.system_instruction()
        user = prompt.user_prompt(packet=packet)

        self.assertIn("# Identity", system)
        self.assertIn("# Contract", system)
        self.assertIn("# Method", system)
        self.assertIn("# Output", system)
        self.assertIn("dialect reference", system)
        self.assertIn("evidence", system)
        self.assertIn("partial Flow", system)
        self.assertIn("# Context", user)
        self.assertIn("dialect guide", user)
        self.assertIn("few-shot examples", user)
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
        self.assertIn("Merge repeated attempts", " ".join(reference.guide.composition))
        self.assertIn("partial Flow", " ".join(reference.guide.completion))

        store_examples = commands["store"].examples
        validate_examples = commands["validate"].examples
        tap_examples = commands["tap"].examples

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
                workflow_id="workflow-1",
                evidence=AuthoringEvidence(run=RunAuthoringEvidence(evidence=evidence)),
            ),
            AuthoringTask(
                kind=AuthoringKind.STEP,
                intent="open app",
                step_number=2,
                workflow_id="workflow-1",
                evidence=AuthoringEvidence(
                    step=StepAuthoringEvidence(evidence=evidence, step_index=1)
                ),
            ),
            AuthoringTask(
                kind=AuthoringKind.REPAIR,
                intent="repair",
                step_number=2,
                workflow_id="workflow-1",
                evidence=AuthoringEvidence(repair=RepairAuthoringEvidence(script="Tap on Search")),
            ),
        )

        for task in tasks:
            packet = AuthoringPacketBuilder().build(task=task, dialect=reference)
            prompt = AuthoringPromptFactory().prompt(kind=task.kind)
            self.assertIn("# Context", prompt.user_prompt(packet=packet))
