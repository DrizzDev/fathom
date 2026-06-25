from __future__ import annotations

import unittest

from fathom.constants.exploration import SCROLL_PROBE_MAX_PROBES
from fathom.core.exploration.config import (
    DepthFloorConfig,
    ExplorationPolicyConfig,
    SamplingConfig,
    ScrollProbeConfig,
)


class TestExplorationPolicyConfig(unittest.TestCase):
    """Aggregate exploration tuning exposes sane defaults and nested overrides."""

    def test_defaults(self) -> None:
        config = ExplorationPolicyConfig()

        self.assertEqual(config.depth.minimum, 4)
        self.assertEqual(config.dedup.retries, 2)
        self.assertEqual(config.sampling.limits["content_item"], 3)
        self.assertEqual(config.sampling.limits["filter_or_category"], 4)
        self.assertEqual(config.scroll.maximum, SCROLL_PROBE_MAX_PROBES)

    def test_nested_override_preserves_other_defaults(self) -> None:
        config = ExplorationPolicyConfig(depth=DepthFloorConfig(minimum=6))

        self.assertEqual(config.depth.minimum, 6)
        self.assertEqual(config.dedup.retries, 2)

    def test_sampling_limits_override(self) -> None:
        config = ExplorationPolicyConfig(sampling=SamplingConfig(limits={"content_item": 1}))

        self.assertEqual(config.sampling.limits, {"content_item": 1})

    def test_scroll_probe_override_and_kill_switch(self) -> None:
        config = ExplorationPolicyConfig(scroll=ScrollProbeConfig(maximum=0))

        self.assertEqual(config.scroll.maximum, 0)
        self.assertEqual(config.depth.minimum, 4)
