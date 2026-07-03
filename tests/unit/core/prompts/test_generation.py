from __future__ import annotations

import unittest

from fathom.constants.flow import IssueCode, LaunchProvenance
from fathom.core.prompts.generation import FlowPromptBuilder
from fathom.schemas.flow import Evidence, EvidenceStep, Issue, StepCapture, StepLaunch, StepTarget


class FlowPromptBuilderTest(unittest.TestCase):
    """
    Cover the system instruction and evidence-grounded user prompt.
    """

    def setUp(self) -> None:
        """
        Build a shared prompt builder and evidence.
        """

        self.__builder = FlowPromptBuilder()

        self.__evidence = Evidence(
            goal="home visible",
            package="com.example",
            intent="open and verify",
            steps=(EvidenceStep(index=0, event="action", action="tap"),),
        )

    def test_system_instruction_states_grounding_rules(self) -> None:
        """
        The system instruction names the provenance and recovery grounding rules.
        """

        instruction = self.__builder.system_instruction()

        self.assertIn("recovery", instruction)
        self.assertIn("source_steps", instruction)

    def test_system_instruction_defers_to_llm_judgment_and_states_invariants(self) -> None:
        """
        The instruction asks the model to judge over intent and evidence, and states the launch
        invariant, rather than prescribing heuristics or examples.
        """

        instruction = self.__builder.system_instruction()

        self.assertIn("judgment", instruction)
        self.assertIn("BOTH the user's intent and the recorded evidence", instruction)
        self.assertIn("begins with a launch", instruction)
        self.assertNotIn("Examples", instruction)

    def test_system_instruction_teaches_evidence_fields(self) -> None:
        """
        The instruction tells the model which evidence fields to read for the target and goal.
        """

        instruction = self.__builder.system_instruction()

        self.assertIn("goal.description", instruction)
        self.assertIn("target.export", instruction)
        self.assertIn("target.name", instruction)
        self.assertIn("recorded script target", instruction)
        self.assertIn("guard.conditional", instruction)

    def test_system_instruction_teaches_episode_level_authoring(self) -> None:
        """
        The prompt asks the model to author from episode intent, not transcribe every attempt.
        """

        instruction = self.__builder.system_instruction()

        self.assertIn("Repeated actions inside one episode", instruction)
        self.assertIn("Collapse them into the clean replay command", instruction)
        self.assertIn("exact labels for stable controls", instruction)
        self.assertIn("relative or dynamic phrases", instruction)

    def test_system_instruction_does_not_force_scroll_until(self) -> None:
        """
        The quality prompt leaves scroll-until as an authoring choice verified by truth gates.
        """

        instruction = self.__builder.system_instruction()

        self.assertIn("otherwise use a plain Scroll <dir>", instruction)
        self.assertNotIn("must use the Scroll <dir> until", instruction)
        self.assertNotIn("never a bare scroll", instruction)

    def test_user_prompt_embeds_compact_evidence_packet(self) -> None:
        """
        The user prompt carries the typed compact evidence packet.
        """

        prompt = self.__builder.user_prompt(evidence=self.__evidence)

        self.assertIn("com.example", prompt)
        self.assertIn("open and verify", prompt)
        self.assertIn("step_id", prompt)
        self.assertIn("episodes", prompt)
        self.assertIn("steps", prompt)

    def test_user_prompt_carries_capture_and_launch_without_noisy_artifacts(self) -> None:
        """
        The packet keeps authoring truth while dropping bulky artifact references.
        """

        evidence = Evidence(
            goal="product visible",
            package="com.example",
            intent="store price",
            artifacts=("blob://large",),
            steps=(
                EvidenceStep(
                    index=0,
                    event="launch",
                    action="launch",
                    screenshot="screenshot://ignored",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                        source_steps=(0,),
                    ),
                ),
                EvidenceStep(
                    index=1,
                    event="action",
                    action="store",
                    target=StepTarget(export="Price label"),
                    capture=StepCapture(
                        name="item_price",
                        subject="price",
                        success=True,
                        value="₹86",
                    ),
                ),
            ),
        )

        prompt = self.__builder.user_prompt(evidence=evidence)

        self.assertIn("item_price", prompt)
        self.assertIn("₹86", prompt)
        self.assertIn("source_steps", prompt)
        self.assertNotIn("blob://large", prompt)
        self.assertNotIn("screenshot://ignored", prompt)

    def test_user_prompt_lists_feedback(self) -> None:
        """
        Prior gate issues are appended to the prompt for repair.
        """

        feedback = (
            Issue(code=IssueCode.MISSING_GOAL_VALIDATION, message="needs a terminal check"),
        )
        prompt = self.__builder.user_prompt(evidence=self.__evidence, feedback=feedback)

        self.assertIn("needs a terminal check", prompt)
        self.assertIn("must fix every issue", prompt)
        self.assertIn(str(IssueCode.MISSING_GOAL_VALIDATION), prompt)
