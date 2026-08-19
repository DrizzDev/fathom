from __future__ import annotations

from typing import Mapping, Optional

from fathom.constants import ActionType
from fathom.constants.flow import SwipeDirection
from fathom.schemas.requirement import SwipeRequirement


class GestureNormalizer:
    """
    Normalizes legacy flattened swipe aliases into a canonical swipe requirement with finger direction.
    """

    __FINGER_DIRECTION: Mapping[ActionType, SwipeDirection] = {
        ActionType.SWIPE_UP: SwipeDirection.UP,
        ActionType.SWIPE_DOWN: SwipeDirection.DOWN,
        ActionType.SWIPE_LEFT: SwipeDirection.LEFT,
        ActionType.SWIPE_RIGHT: SwipeDirection.RIGHT,
    }

    def canonical(self, *, operation: ActionType, target: Optional[str] = None) -> SwipeRequirement:
        """
        Return the canonical swipe requirement for a flattened directional alias, or fail closed.
        """

        direction = self.__FINGER_DIRECTION.get(operation)
        if direction is None:
            raise ValueError(f"'{operation.value}' is not a flattened directional swipe alias.")

        return SwipeRequirement(operation=ActionType.SWIPE, direction=direction, target=target)
