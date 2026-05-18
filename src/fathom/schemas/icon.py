from __future__ import annotations

from enum import StrEnum
from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field

from fathom.schemas.actions import Bounds


class IconKind(StrEnum):
    """
    Canonical icon vocabulary used by the icon-detection layer.
    """

    BACK = "back"
    CART = "cart"
    HOME = "home"
    MENU = "menu"
    CHECK = "check"
    CLOSE = "close"
    HEART = "heart"
    SHARE = "share"
    SEARCH = "search"
    SETTINGS = "settings"


class IconMatch(BaseModel):
    """
    One detected icon with its executable bounds and confidence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: IconKind = Field(description="Canonical icon classification.")
    bounds: Bounds = Field(description="Pixel bounds of the matched region.")

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Match confidence in the closed unit interval.",
    )


class IconDetectionResult(BaseModel):
    """
    Result of one icon-detection pass over a screen capture.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    matches: Tuple[IconMatch, ...] = Field(description="Detected icon matches.")
    duration: int = Field(ge=0, description="Detection duration in milliseconds.")


class IconTemplate(BaseModel):
    """
    One named icon template image used by the template detector.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    image: bytes = Field(description="PNG-encoded template bytes.", repr=False)
    kind: IconKind = Field(description="Canonical icon kind this template represents.")
