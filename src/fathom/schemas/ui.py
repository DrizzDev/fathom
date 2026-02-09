from __future__ import annotations

from typing import Any, Dict, Tuple, Union

from pydantic import BaseModel, ConfigDict


class UIBounds(BaseModel):
    """
    Represents bounds with top-left and bottom-right absolute pixel coordinates.
    Distinct from the normalized Bounds used in actions.
    """

    x1: Union[int, float]
    y1: Union[int, float]
    x2: Union[int, float]
    y2: Union[int, float]

    model_config = ConfigDict(frozen=True)

    def __hash__(self) -> int:
        """
        Make bounds hashable for use in sets.
        """

        return hash((self.x1, self.y1, self.x2, self.y2))

    def __eq__(self, other: Any) -> bool:
        """
        Check equality with tolerance for floating point comparison.
        """

        if not isinstance(other, UIBounds):
            return False

        tolerance = 0.01
        return (
            abs(self.x1 - other.x1) < tolerance
            and abs(self.y1 - other.y1) < tolerance
            and abs(self.x2 - other.x2) < tolerance
            and abs(self.y2 - other.y2) < tolerance
        )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """
        Calculates the area.
        """

        return self.width * self.height

    def to_rectangle(self) -> Tuple[int, int, int, int]:
        """
        Converts to integer rectangle coordinates.
        """

        return (
            int(self.x1),
            int(self.y1),
            int(self.x2),
            int(self.y2),
        )


class LabeledElement(BaseModel):
    """
    Represents a processed, drawable UI element with a label.
    """

    label: str
    color: str = "red"

    bounds: UIBounds
    attributes: Dict[str, Any]

    model_config = ConfigDict(frozen=True)
