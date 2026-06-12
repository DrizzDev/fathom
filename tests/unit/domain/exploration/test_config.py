"""
Unit tests for the exploration policy configuration models.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fathom.domain.exploration.config import (
    DepthFloorConfig,
    ExplorationPolicyConfig,
    SamplingConfig,
)


class TestExplorationPolicyConfig:
    """
    Defaults, nested overrides, validation, and instance isolation.
    """

    def test_defaults(self) -> None:
        config = ExplorationPolicyConfig()
        assert config.depth.minimum == 4
        assert config.dedup.retries == 2
        assert config.sampling.limits == {"content_item": 3, "filter_or_category": 4}

    def test_nested_override(self) -> None:
        config = ExplorationPolicyConfig(depth=DepthFloorConfig(minimum=6))
        assert config.depth.minimum == 6
        assert config.dedup.retries == 2

    def test_negative_minimum_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DepthFloorConfig(minimum=-1)

    def test_sampling_limits_are_not_shared_across_instances(self) -> None:
        first = SamplingConfig()
        second = SamplingConfig()
        first.limits["content_item"] = 99
        assert second.limits["content_item"] == 3
