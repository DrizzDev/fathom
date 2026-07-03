from __future__ import annotations

import unittest
from typing import Tuple

from fathom.adapters.dialect.drizz.factory import DrizzDialectFactory
from fathom.constants.flow import IssueCode, LaunchProvenance
from fathom.constants.generation import ScriptSource, ScriptStatus, SkipReason
from fathom.core.dialect.policy import Policy
from fathom.core.services.generation.baseline import BaselineScriptService
from fathom.core.services.generation.projector import DeterministicFlowGenerator
from fathom.schemas.flow import (
    Evidence,
    EvidenceStep,
    StepLaunch,
    StepOutcome,
    StepTarget,
)


class BaselineScriptServiceTest(unittest.TestCase):
    """
    Pins the baseline service: faithful generated artifact, carried skip diagnostics, and loud typed failure.
    """

    def __service(self) -> BaselineScriptService:
        """
        Build the baseline service wired to the projector, policy, and Drizz dialect.
        """

        return BaselineScriptService(
            policy=Policy(),
            generator=DeterministicFlowGenerator(),
            dialect=DrizzDialectFactory().create(),
        )

    @staticmethod
    def __evidence(*, steps: Tuple[EvidenceStep, ...], partial: bool = False) -> Evidence:
        """
        Wrap evidence steps for a baseline build.
        """

        return Evidence(
            steps=steps,
            partial=partial,
            intent="order a burger",
            goal="burger added to cart",
            package="in.swiggy.android",
        )

    @staticmethod
    def __launch() -> EvidenceStep:
        """
        Build a launcher-transition launch step.
        """

        return EvidenceStep(
            index=0,
            event="launch",
            action="launch",
            launch=StepLaunch(
                source_steps=(0,),
                package="in.swiggy.android",
                provenance=LaunchProvenance.LAUNCHER_TRANSITION,
            ),
        )

    @staticmethod
    def __validation(*, index: int) -> EvidenceStep:
        """
        Build a successful goal validation step.
        """

        return EvidenceStep(
            index=index,
            event="validation",
            action="complete",
            target=StepTarget(export="Cart screen"),
            outcome=StepOutcome(success=True),
        )

    @staticmethod
    def __tap(*, index: int, export: str, success: bool = True) -> EvidenceStep:
        """
        Build a tap step grounded on an export phrase.
        """

        return EvidenceStep(
            index=index,
            action="tap",
            event="action",
            target=StepTarget(export=export),
            outcome=StepOutcome(success=success),
        )

    def test_clean_run_builds_generated_baseline_artifact(self) -> None:
        """
        A clean run yields a generated, baseline-sourced artifact with rendered script text.
        """

        artifact = self.__service().build(
            evidence=self.__evidence(
                steps=(
                    self.__launch(),
                    self.__tap(index=1, export="Search box"),
                    self.__validation(index=2),
                )
            )
        )

        self.assertIs(artifact.metadata.status, ScriptStatus.GENERATED)
        self.assertIs(artifact.metadata.source, ScriptSource.BASELINE)

        self.assertIsNotNone(artifact.text)
        assert artifact.text is not None

        self.assertFalse(artifact.metadata.review.partial)
        self.assertTrue(artifact.text.startswith("OPEN_APP"))

    def test_skipped_steps_are_carried_into_metadata(self) -> None:
        """
        A generated baseline still records the evidence steps the projector dropped.
        """

        artifact = self.__service().build(
            evidence=self.__evidence(
                steps=(
                    self.__launch(),
                    self.__tap(index=1, export="broken", success=False),
                    self.__validation(index=2),
                )
            )
        )

        reasons = {skip.index: skip.reason for skip in artifact.metadata.skipped}

        self.assertIs(artifact.metadata.status, ScriptStatus.GENERATED)
        self.assertEqual(reasons, {1: SkipReason.FAILED})

    def test_unrenderable_target_yields_failed_artifact(self) -> None:
        """
        A target colliding with every quote delimiter fails loudly with no script text, never empty success.
        """

        artifact = self.__service().build(
            evidence=self.__evidence(
                steps=(
                    self.__launch(),
                    self.__tap(index=1, export="a'b\"c`"),
                    self.__validation(index=2),
                )
            )
        )

        self.assertIs(artifact.metadata.status, ScriptStatus.FAILED)
        self.assertIs(artifact.metadata.source, ScriptSource.BASELINE)

        self.assertIsNone(artifact.text)
        self.assertEqual(artifact.metadata.issues[0].code, IssueCode.UNRENDERABLE_VALUE)

    def test_run_with_no_scriptable_steps_fails_not_empty_generated(self) -> None:
        """
        A run where every step was dropped fails loudly with text=None, never an empty generated script.
        """

        artifact = self.__service().build(
            evidence=self.__evidence(
                partial=True,
                steps=(self.__tap(index=0, export="broken", success=False),),
            )
        )

        self.assertIsNone(artifact.text)
        self.assertIs(artifact.metadata.status, ScriptStatus.FAILED)
        self.assertEqual(artifact.metadata.issues[0].code, IssueCode.EMPTY_SCRIPT)
