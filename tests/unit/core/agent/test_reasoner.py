from __future__ import annotations

import unittest
from typing import Optional

from fathom.constants import ActionType, StepEvent
from fathom.core.agent.opener import OpenerSignalPolicy
from fathom.core.agent.reasoner import Reasoner
from fathom.schemas.actions import Action
from fathom.schemas.results import AnalysisResult


class ReasonerAnalyzeCompletionTest(unittest.TestCase):
    """
    Pins the surviving non-sub-goal completion heuristic (analyze_completion) used by the planner.
    """

    @staticmethod
    def __reasoner(intent: str = "open meesho and find Jars & containers") -> Reasoner:
        """
        Build a Reasoner with a representative intent string.
        """

        return Reasoner(intent=intent, opener_policy=OpenerSignalPolicy())

    @staticmethod
    def __analysis(
        *,
        action_type: ActionType = ActionType.TAP,
        reasoning: str = "Submit tapped; new screen visible.",
        validation_subject: Optional[str] = None,
        event_type: Optional[StepEvent] = None,
    ) -> AnalysisResult:
        """
        Build an AnalysisResult fixture with the requested action.
        """

        return AnalysisResult(
            action=Action(
                action_type=action_type,
                event_type=event_type,
                target="t",
                rationale="r",
                confidence=1.0,
                validation_subject=validation_subject,
            ),
            reasoning=reasoning,
            screen_description="post-action screen",
            is_sub_goal_complete=False,
            is_goal_complete=False,
            metadata={"tool_args": {}},
        )

    def test_opening_sub_goal_completes_for_next_phase_actions(self) -> None:
        """
        Next-phase action types complete opener sub-goals when reasoning confirms follow-up work.
        """

        for action_type in (
            ActionType.SWIPE,
            ActionType.SCROLL,
            ActionType.VALIDATE,
            ActionType.SWIPE_UP,
            ActionType.SWIPE_DOWN,
            ActionType.SWIPE_LEFT,
            ActionType.SWIPE_RIGHT,
        ):
            with self.subTest(action_type=action_type):
                signal = self.__reasoner().analyze_completion(
                    analysis=self.__analysis(
                        action_type=action_type,
                        reasoning="Swipe and check the main content after app launch.",
                    ),
                    current_sub_goal="Open Tata 1mg app",
                )

                self.assertTrue(signal.success_indicator)
                self.assertIn(action_type.value, signal.evidence)


if __name__ == "__main__":
    unittest.main()
