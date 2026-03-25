from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    import xml.etree.ElementTree as ET  # nosec

from fathom.schemas.hierarchy import NormalizedHierarchyNodeSignature
from fathom.schemas.ui import LabeledElement


class PlatformParser(ABC):
    """
    Base class for platform-specific XML parsers.
    """

    @classmethod
    @abstractmethod
    def is_platform_match(cls, root: ET.Element) -> bool:
        """
        Check if the root element matches this platform.
        """

        raise NotImplementedError

    @abstractmethod
    def get_scale_factor(self, root: ET.Element, screenshot_width: int) -> float:
        """
        Calculate scale factor between XML coordinates and screenshot pixels.
        """

        raise NotImplementedError

    @abstractmethod
    def find_all_elements(self, root: ET.Element, **kwargs: Any) -> List[LabeledElement]:
        """
        Find all potential elements in the XML tree.
        """

        raise NotImplementedError

    @abstractmethod
    def filter_and_deduplicate(
        self,
        elements: List[LabeledElement],
        *,
        iou_threshold: float = 0.4,
        action: Optional[Any] = None,
    ) -> List[LabeledElement]:
        """
        Filter and deduplicate elements.
        """

        raise NotImplementedError

    @abstractmethod
    def filter_by_action(self, elements: List[LabeledElement], action: Any) -> List[LabeledElement]:
        """
        Filter elements relevant to a specific action.
        """

        raise NotImplementedError

    @abstractmethod
    def build_signature_metadata(self, *, node: ET.Element) -> NormalizedHierarchyNodeSignature:
        """
        Normalize one hierarchy node into structural-signature metadata.
        """

        raise NotImplementedError

    @abstractmethod
    def should_ignore_signature_node(self, *, metadata: NormalizedHierarchyNodeSignature) -> bool:
        """
        Return whether a normalized node should be ignored in structural signatures.
        """

        raise NotImplementedError
