from __future__ import annotations

from typing import Any, Optional

from fathom.constants.state import CommonStateKey, ExplorationStateKey
from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import StepResult


class ExplorationGraphState(dict[str, Any]):
    """
    State flowing through the Exploration Graph.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize exploration graph state.
        """

        super().__init__(**kwargs)

    def get_capture(self) -> Optional[ScreenCapture]:
        """
        Get capture from state.
        """

        return self.get(CommonStateKey.CAPTURE)

    def get_screen_state(self) -> Optional[ScreenState]:
        """
        Get screen state from state.
        """

        return self.get(CommonStateKey.SCREEN_STATE)

    def get_action(self) -> Optional[Action]:
        """
        Get action from state.
        """

        return self.get(ExplorationStateKey.ACTION)

    def get_step_result(self) -> Optional[StepResult]:
        """
        Get step result from state.
        """

        return self.get(CommonStateKey.STEP_RESULT)

    def is_complete(self) -> bool:
        """
        Check if execution is complete.
        """

        return bool(self.get(CommonStateKey.IS_COMPLETE, False))

    def is_content_exhausted(self) -> bool:
        """
        Check if content is exhausted.
        """

        return bool(self.get(ExplorationStateKey.CONTENT_EXHAUSTED, False))
