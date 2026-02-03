from typing import Tuple

from fathom.schemas.ui import UIBoundingBox


class GeometryUtils:
    """
    Utility class with methods for geometry related operations on bounding boxes.
    """

    @staticmethod
    def calculate_iou(box_a: UIBoundingBox, box_b: UIBoundingBox) -> float:
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        """

        x_left = max(box_a.x1, box_b.x1)
        y_top = max(box_a.y1, box_b.y1)
        x_right = min(box_a.x2, box_b.x2)
        y_bottom = min(box_a.y2, box_b.y2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box_a_area = (box_a.x2 - box_a.x1) * (box_a.y2 - box_a.y1)
        box_b_area = (box_b.x2 - box_b.x1) * (box_b.y2 - box_b.y1)
        union_area = float(box_a_area + box_b_area - intersection_area)

        return 0.0 if union_area == 0 else intersection_area / union_area

    @staticmethod
    def is_box_contained(inner_box: UIBoundingBox, outer_box: UIBoundingBox) -> bool:
        """
        Checks if inner_box is completely contained within outer_box.
        """

        return (
            outer_box.x1 <= inner_box.x1
            and outer_box.y1 <= inner_box.y1
            and outer_box.x2 >= inner_box.x2
            and outer_box.y2 >= inner_box.y2
        )

    @staticmethod
    def boxes_overlap(
        box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]
    ) -> bool:
        """
        Checks if two boxes (x1, y1, x2, y2) overlap.
        """

        return not (
            box1[2] <= box2[0] or box1[0] >= box2[2] or box1[3] <= box2[1] or box1[1] >= box2[3]
        )

    @staticmethod
    def get_line_endpoints(
        label_box: Tuple[float, float, float, float], element_box: Tuple[float, float, float, float]
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Calculates the optimal start and end points for a leader line.
        """

        label_center = (
            (label_box[0] + label_box[2]) / 2,
            (label_box[1] + label_box[3]) / 2,
        )

        point_on_element_box = (
            max(element_box[0], min(label_center[0], element_box[2])),
            max(element_box[1], min(label_center[1], element_box[3])),
        )

        point_on_label_box = (
            max(label_box[0], min(point_on_element_box[0], label_box[2])),
            max(label_box[1], min(point_on_element_box[1], label_box[3])),
        )

        return point_on_label_box, point_on_element_box
