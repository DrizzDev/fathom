from __future__ import annotations

from fathom.constants import ActionType
from fathom.schemas.actions import Action
from fathom.schemas.state import ActionHistory


class TestActionHistoryRecentDescriptors:
    """
    Pins the trailing-window descriptor accessor used by the
    inert-action-repetition detector on :class:`AgentState`.
    """

    @staticmethod
    def __action(*, target: str) -> Action:
        """
        Build a minimal :class:`Action` for history tests.
        """

        return Action(
            action_type=ActionType.TAP,
            rationale="probe",
            target=target,
            natural_language_target=target,
        )

    def test_empty_history_returns_no_descriptors(self) -> None:
        """
        Empty history must produce an empty descriptor list.
        """

        history = ActionHistory(max_size=10)

        assert history.recent_action_descriptors(count=3) == []

    def test_zero_or_negative_count_returns_no_descriptors(self) -> None:
        """
        Non-positive ``count`` must return an empty list, not the whole history.
        """

        history = ActionHistory(max_size=10)
        history.record_action(action=self.__action(target="A"), success=True, activity="screen")

        assert history.recent_action_descriptors(count=0) == []
        assert history.recent_action_descriptors(count=-1) == []

    def test_returns_trailing_window_in_execution_order(self) -> None:
        """
        Returned descriptors must be the last ``count`` items in execution order.
        """

        history = ActionHistory(max_size=10)
        for letter in ("A", "B", "C", "D"):
            history.record_action(
                action=self.__action(target=letter), success=True, activity="screen"
            )

        descriptors = history.recent_action_descriptors(count=2)

        assert len(descriptors) == 2
        assert descriptors[0].endswith("C")
        assert descriptors[1].endswith("D")

    def test_count_larger_than_history_returns_all(self) -> None:
        """
        Asking for more than recorded must return everything available.
        """

        history = ActionHistory(max_size=10)
        history.record_action(action=self.__action(target="solo"), success=True, activity="screen")

        descriptors = history.recent_action_descriptors(count=5)

        assert len(descriptors) == 1
