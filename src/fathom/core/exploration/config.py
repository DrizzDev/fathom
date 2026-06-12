"""
Configuration models for the exploration decision policies.
"""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class DepthFloorConfig(BaseModel):
    """
    Tuning for the DFS depth-floor guardrail.
    """

    minimum: int = Field(
        default=4,
        ge=0,
        description="Shortest DFS path that may honour a content-exhaustion signal",
    )


class DedupConfig(BaseModel):
    """
    Tuning for the action-deduplication guard.
    """

    retries: int = Field(
        default=2,
        ge=0,
        description="Re-prompts allowed when the model repeats an already-tried action",
    )


class SamplingConfig(BaseModel):
    """
    Per-category tap caps that bound enumeration of long lists.
    """

    limits: Dict[str, int] = Field(
        default_factory=lambda: {"content_item": 3, "filter_or_category": 4},
        description="Maximum taps per element category on a single screen",
    )


class ExplorationPolicyConfig(BaseModel):
    """
    Aggregate tuning for the exploration decision policies.
    """

    depth: DepthFloorConfig = Field(
        default_factory=DepthFloorConfig,
        description="Depth-floor guardrail tuning",
    )
    dedup: DedupConfig = Field(
        default_factory=DedupConfig,
        description="Action-deduplication tuning",
    )
    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig,
        description="List-sampling caps",
    )
