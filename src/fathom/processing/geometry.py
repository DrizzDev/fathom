from __future__ import annotations

from typing import Tuple

from fathom.schemas.ui import UIBounds


class GeometryUtils:
    """
    Utility class for geometric operations.
    """

    @staticmethod
    def calculate_iou(bounds1: UIBounds, bounds2: UIBounds) -> float:
        """
        Calculate the Intersection over Union (IoU) of two bounds.
        """

        left = max(bounds1.x1, bounds2.x1)
        top = max(bounds1.y1, bounds2.y1)
        right = min(bounds1.x2, bounds2.x2)
        bottom = min(bounds1.y2, bounds2.y2)

        if right < left or bottom < top:
            return 0.0

        intersection = (right - left) * (bottom - top)
        area1 = (bounds1.x2 - bounds1.x1) * (bounds1.y2 - bounds1.y1)
        area2 = (bounds2.x2 - bounds2.x1) * (bounds2.y2 - bounds2.y1)
        union = float(area1 + area2 - intersection)

        return 0.0 if union == 0 else intersection / union

    @staticmethod
    def is_box_contained(box1: UIBounds, box2: UIBounds) -> bool:
        """
        Checks if box1 is completely contained within box2.
        """

        return (
            box2.x1 <= box1.x1 and box2.y1 <= box1.y1 and box2.x2 >= box1.x2 and box2.y2 >= box1.y2
        )

    @staticmethod
    def boxes_overlap(
        first: Tuple[float, float, float, float], second: Tuple[float, float, float, float]
    ) -> bool:
        """
        Checks if two boxes overlap.
        """

        return not (
            first[2] <= second[0]
            or first[0] >= second[2]
            or first[3] <= second[1]
            or first[1] >= second[3]
        )

    @staticmethod
    def get_line_endpoints(
        label: Tuple[float, float, float, float], element: Tuple[float, float, float, float]
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Calculates the optimal start and end points for a leader line.
        """

        center = (
            (label[0] + label[2]) / 2,
            (label[1] + label[3]) / 2,
        )

        point_on_element = (
            max(element[0], min(center[0], element[2])),
            max(element[1], min(center[1], element[3])),
        )

        point_on_label = (
            max(label[0], min(point_on_element[0], label[2])),
            max(label[1], min(point_on_element[1], label[3])),
        )

        return point_on_label, point_on_element
