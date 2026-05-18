from __future__ import annotations

from typing import List, Optional

from fathom.constants.runtime import DEFAULT_LOOP_THRESHOLD, DEFAULT_LOOP_WINDOW
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.screens import ScreenState
from fathom.schemas.state import LoopDetector


class ScreenRuntimeState:
    """
    Owns current screen identity, observation history, and loop detector state.
    """

    def __init__(
        self,
        *,
        loop_detector: Optional[LoopDetector] = None,
        loop_threshold: int = DEFAULT_LOOP_THRESHOLD,
        loop_window: int = DEFAULT_LOOP_WINDOW,
    ) -> None:
        """
        Initialize screen runtime state with the loop-detection bounds.
        """

        self.__current: Optional[ScreenState] = None
        self.__previous: Optional[ScreenState] = None
        self.__observation: Optional[ScreenObservation] = None
        self.__seen: List[ScreenState] = []
        self.__loop_detector = loop_detector or LoopDetector(
            threshold=loop_threshold,
            window_size=loop_window,
        )

    @property
    def current(self) -> Optional[ScreenState]:
        """
        Return the current screen state.
        """

        return self.__current

    @property
    def previous(self) -> Optional[ScreenState]:
        """
        Return the previous screen state.
        """

        return self.__previous

    @property
    def observation(self) -> Optional[ScreenObservation]:
        """
        Return the current screen observation.
        """

        return self.__observation

    @property
    def detector(self) -> LoopDetector:
        """
        Return the loop detector.
        """

        return self.__loop_detector

    @property
    def seen(self) -> List[ScreenState]:
        """
        Return the cumulative list of distinct screens observed in this run.
        """

        return list(self.__seen)

    def is_new(self, *, screen: ScreenState) -> bool:
        """
        Return whether the supplied screen is structurally new to this run.
        """

        return all(not previously.is_same_screen(screen) for previously in self.__seen)

    def update(self, *, screen: ScreenState, observation: Optional[ScreenObservation]) -> None:
        """
        Update current screen and observation state.
        """

        self.__previous = self.__current
        self.__current = screen
        self.__observation = observation

    def remember(self, *, screen: ScreenState) -> None:
        """
        Append a structurally-new screen to the seen-set.
        """

        if self.is_new(screen=screen):
            self.__seen.append(screen)

    def load_seen(self, *, screens: List[ScreenState]) -> None:
        """
        Replace the seen-screens history with a restored checkpoint window.
        """

        self.__seen = list(screens)

    def reset_loop_history(self) -> None:
        """
        Reset loop detector history without disturbing seen-screens history.
        """

        self.__loop_detector.reset()
