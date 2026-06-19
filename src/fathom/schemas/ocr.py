from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.ocr import OcrConfidence, OcrLevel
from fathom.schemas.actions import Bounds


class OcrToken(BaseModel):
    """
    One OCR-detected element with executable pixel bounds.
    May represent a single word, a row-merged phrase, or a multi-row semantic block — distinguished by :attr:`level`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1, description="Recognized text.")
    bounds: Bounds = Field(description="Pixel bounds in the source screenshot.")

    confidence: OcrConfidence = Field(
        description="Coarse confidence band derived from the provider score.",
    )
    raw_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Provider-reported numeric confidence in the closed unit interval.",
    )
    level: OcrLevel = Field(
        default=OcrLevel.TOKEN,
        description="Layout hierarchy level this element was extracted from.",
    )


class OcrResult(BaseModel):
    """
    Result of one OCR pass over a screen capture.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tokens: Tuple[OcrToken, ...] = Field(description="Detected tokens in reading order.")

    duration: int = Field(ge=0, description="Provider call duration in milliseconds.")
    raw_response: Optional[str] = Field(
        default=None,
        description="Provider raw response serialized as JSON for debugging artifacts.",
    )


class DocumentAiConfiguration(BaseModel):
    """
    Configuration for the Document AI OCR adapter.

    ``credentials`` carries explicit service-account material so the
    adapter authenticates against the same identity Gemini already uses.
    The field is :data:`Optional` so test fixtures can omit it; in the
    runtime composition root the assembly always populates it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    project: str = Field(min_length=1, description="GCP project identifier.")

    location: str = Field(
        min_length=1,
        description="Document AI location id such as 'us' or 'eu'.",
    )
    processor: str = Field(min_length=1, description="Document AI processor identifier.")
    credentials: Optional[Union[Dict[str, Any], str]] = Field(
        default=None,
        description=(
            "Service-account credentials: inline JSON dict or absolute "
            "path to the key file. None falls back to library defaults."
        ),
    )

    @property
    def processor_path(self) -> str:
        """
        Return the fully-qualified Document AI processor resource path.
        """

        return f"projects/{self.project}/locations/{self.location}/processors/{self.processor}"
