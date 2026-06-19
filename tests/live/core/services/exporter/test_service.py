from __future__ import annotations

from typing import Sequence

import pytest

from fathom.constants import ActionType
from fathom.core.services.exporter.service import ScriptExporter
from fathom.interfaces.llm import LLMPort
from fathom.schemas.actions import Action
from fathom.schemas.steps import Step, StepResult

pytestmark = pytest.mark.release

EMAIL = "dev+test+Ilu+z2O5@varomoney.com"
PASSWORD = "Password1!"
PACKAGE_NAME = "com.varomoney.varo"
VARO_INTENT = (
    "Launch the Varo app, log in with the provided credentials, complete onboarding, "
    "and reach the main Home screen."
)


class StepResultFactory:
    """
    Builds representative StepResult values for live exporter tests.
    """

    @staticmethod
    def create(
        *,
        step_number: int,
        action_type: ActionType,
        target: str,
        text: str | None = None,
        activity: str = PACKAGE_NAME,
        validation_subject: str | None = None,
        is_app_launcher: bool = False,
    ) -> StepResult:
        """
        Build one successful StepResult with exporter-friendly action fields.
        """

        action = Action(
            action_type=action_type,
            rationale=f"Perform {target}",
            target=target,
            natural_language_target=target,
            export_target=target,
            text=text,
            confidence=0.95,
            validation_subject=validation_subject,
            is_app_launcher=is_app_launcher,
        )
        step = Step(
            action=action,
            screen_hash=f"screen-{step_number}",
            step_number=step_number,
            metadata={"activity": activity},
        )
        return StepResult(
            step=step,
            success=True,
            pre_hash=f"pre-{step_number}",
            post_hash=f"post-{step_number}",
            screen_changed=True,
            duration=250,
        )

    @staticmethod
    def varo_onboarding_trace() -> Sequence[StepResult]:
        """
        Return a Varo login/onboarding trace that must not lose steps.
        """

        launcher = "com.google.android.apps.nexuslauncher"
        return (
            StepResultFactory.create(
                step_number=1,
                action_type=ActionType.TAP,
                target="Varo app icon",
                activity=launcher,
                is_app_launcher=True,
            ),
            StepResultFactory.create(
                step_number=2,
                action_type=ActionType.TYPE,
                target="Email field",
                text=EMAIL,
            ),
            StepResultFactory.create(
                step_number=3,
                action_type=ActionType.TYPE,
                target="Password field",
                text=PASSWORD,
            ),
            StepResultFactory.create(
                step_number=4,
                action_type=ActionType.TAP,
                target="Log in button",
            ),
            StepResultFactory.create(
                step_number=5,
                action_type=ActionType.TAP,
                target="Continue onboarding button",
            ),
            StepResultFactory.create(
                step_number=6,
                action_type=ActionType.VALIDATE,
                target="Home screen",
                validation_subject="main Home screen is visible",
            ),
        )


class TestScriptExporter:
    """
    Live LLM checks for ScriptExporter.
    """

    async def test_varo_trace_export_keeps_required_actions(self, llm: LLMPort) -> None:
        """
        Varo export must include launch, login credentials, onboarding, and final validation.
        """

        exporter = ScriptExporter(llm=llm, use_cache=False)

        script = await exporter.export_with_llm(
            step_results=StepResultFactory.varo_onboarding_trace(),
            intent=VARO_INTENT,
            goal_state='main "Home" screen is visible',
            package_name=PACKAGE_NAME,
        )

        assert script is not None
        lowered = script.casefold()
        assert f"open_app {PACKAGE_NAME}".casefold() in lowered
        assert EMAIL.casefold() in lowered
        assert PASSWORD.casefold() in lowered
        assert "log in" in lowered or "login" in lowered
        assert "continue onboarding" in lowered
        assert "home screen" in lowered
