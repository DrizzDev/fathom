from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.schemas.screens import ScreenCapture
from fathom.schemas.ui import LabeledElement


class NormalizedHierarchyNodeSignature(BaseModel):
    """
    Platform-normalized signature data for one hierarchy node.
    """

    model_config = ConfigDict(frozen=True)

    identifier: str = Field(description="Stable element identifier")
    class_name: str = Field(description="Normalized platform-neutral class name")
    source_type: str = Field(description="Original platform-specific type or class identifier")

    text: str = Field(description="Stable visible text for the node")
    content_description: str = Field(description="Stable accessibility description")

    is_focused: bool = Field(description="Whether the node is focused")
    is_selected: bool = Field(description="Whether the node is selected")
    is_checked: bool = Field(description="Whether the node is checked or toggled on")

    bounds: str = Field(description="Serialized bounds for positional diffing")
    is_scrollable: bool = Field(description="Whether the node acts as a scroll container")
    raw_value: str = Field(description="Small stable value payload for value-sensitive controls")
    include_value_in_signature: bool = Field(
        description="Whether the raw value should contribute to the structural signature"
    )


class HierarchyProcessingResult(BaseModel):
    """
    Typed result of XML processing for one capture.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    annotated_capture: Optional[ScreenCapture] = Field(
        default=None,
        description="Capture carrying any annotation artifacts while preserving raw screen bytes",
    )
    labeled_elements: List[LabeledElement] = Field(
        default_factory=list,
        description="Typed interactive elements extracted from the hierarchy",
    )
    label_map: Dict[str, Any] = Field(
        default_factory=dict,
        description="Rendered element manifest keyed by label for prompt grounding",
    )


class HierarchyElementExtraction(BaseModel):
    """
    Typed extraction result for one XML parse pass.
    """

    model_config = ConfigDict(frozen=True)

    labeled_elements: List[LabeledElement] = Field(
        default_factory=list,
        description="Typed interactive elements extracted from the hierarchy",
    )
    label_map: Dict[str, Any] = Field(
        default_factory=dict,
        description="Rendered element manifest keyed by label for prompt grounding",
    )
