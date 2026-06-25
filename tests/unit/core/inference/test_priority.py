from __future__ import annotations

import unittest

from fathom.constants.llm import (
    InferencePriorityMode,
    InferencePriorityTransitionReason,
    InferenceTier,
)
from fathom.core.inference.priority import PriorityInferencePolicy
from fathom.schemas.base.common import ThresholdConfiguration
from fathom.schemas.configuration import (
    AdaptivePriorityConfiguration,
    PriorityInferenceConfiguration,
)
from fathom.schemas.llm import PriorityInferenceSignal


class PriorityInferencePolicyTest(unittest.TestCase):
    """
    Covers provider-neutral priority tier selection.
    """

    def test_always_mode_selects_priority(self) -> None:
        """
        Always mode must request priority before any history exists.
        """

        policy = PriorityInferencePolicy(configuration=PriorityInferenceConfiguration())

        self.assertEqual(policy.select(), InferenceTier.PRIORITY)

    def test_disabled_policy_selects_standard(self) -> None:
        """
        Disabled policy must never request elevated capacity.
        """

        policy = PriorityInferencePolicy(
            configuration=PriorityInferenceConfiguration(enabled=False),
        )

        self.assertEqual(policy.select(), InferenceTier.STANDARD)

    def test_adaptive_scales_up_after_transient_failures(self) -> None:
        """
        Adaptive mode escalates after enough recent transient failures.
        """

        policy = PriorityInferencePolicy(
            configuration=PriorityInferenceConfiguration(
                mode=InferencePriorityMode.ADAPTIVE,
                adaptive=AdaptivePriorityConfiguration(
                    threshold=ThresholdConfiguration(
                        failures=2,
                        slows=3,
                        latency=15.0,
                        recovery=5,
                    ),
                ),
            ),
        )

        self.assertEqual(policy.select(), InferenceTier.STANDARD)
        transition = policy.record(
            signal=PriorityInferenceSignal(
                tier=InferenceTier.STANDARD,
                success=False,
                transient=True,
            ),
        )
        self.assertIsNone(transition)
        self.assertEqual(policy.select(), InferenceTier.STANDARD)
        transition = policy.record(
            signal=PriorityInferenceSignal(
                tier=InferenceTier.STANDARD,
                success=False,
                transient=True,
            ),
        )

        self.assertIsNotNone(transition)
        assert transition is not None
        self.assertEqual(transition.previous, InferenceTier.STANDARD)
        self.assertEqual(transition.current, InferenceTier.PRIORITY)
        self.assertEqual(transition.reason, InferencePriorityTransitionReason.TRANSIENT)
        self.assertEqual(transition.evidence.failures, 2)
        self.assertEqual(transition.evidence.threshold.failures, 2)
        self.assertEqual(policy.select(), InferenceTier.PRIORITY)

    def test_adaptive_scales_up_after_slow_successes(self) -> None:
        """
        Adaptive mode escalates when successful calls repeatedly cross the threshold.
        """

        policy = PriorityInferencePolicy(
            configuration=PriorityInferenceConfiguration(
                mode=InferencePriorityMode.ADAPTIVE,
                adaptive=AdaptivePriorityConfiguration(
                    threshold=ThresholdConfiguration(
                        failures=2,
                        slows=2,
                        latency=1.0,
                        recovery=5,
                    ),
                ),
            ),
        )

        transition = policy.record(
            signal=PriorityInferenceSignal(
                tier=InferenceTier.STANDARD,
                success=True,
                latency=1.2,
            ),
        )
        self.assertIsNone(transition)
        transition = policy.record(
            signal=PriorityInferenceSignal(
                tier=InferenceTier.STANDARD,
                success=True,
                latency=1.1,
            ),
        )

        self.assertIsNotNone(transition)
        assert transition is not None
        self.assertEqual(transition.previous, InferenceTier.STANDARD)
        self.assertEqual(transition.current, InferenceTier.PRIORITY)
        self.assertEqual(transition.reason, InferencePriorityTransitionReason.SLOW)
        self.assertEqual(transition.evidence.slows, 2)
        self.assertEqual(transition.evidence.threshold.slows, 2)
        self.assertEqual(policy.select(), InferenceTier.PRIORITY)

    def test_adaptive_recovers_after_healthy_priority_streak(self) -> None:
        """
        Adaptive mode returns to standard after enough healthy priority attempts.
        """

        policy = PriorityInferencePolicy(
            configuration=PriorityInferenceConfiguration(
                mode=InferencePriorityMode.ADAPTIVE,
                adaptive=AdaptivePriorityConfiguration(
                    threshold=ThresholdConfiguration(
                        failures=1,
                        slows=3,
                        latency=2.0,
                        recovery=2,
                    ),
                ),
            ),
        )

        transition = policy.record(
            signal=PriorityInferenceSignal(
                tier=InferenceTier.STANDARD,
                success=False,
                transient=True,
            ),
        )
        self.assertIsNotNone(transition)
        self.assertEqual(policy.select(), InferenceTier.PRIORITY)
        transition = policy.record(
            signal=PriorityInferenceSignal(
                tier=InferenceTier.PRIORITY,
                success=True,
                latency=1.0,
            ),
        )
        self.assertIsNone(transition)
        self.assertEqual(policy.select(), InferenceTier.PRIORITY)
        transition = policy.record(
            signal=PriorityInferenceSignal(
                tier=InferenceTier.PRIORITY,
                success=True,
                latency=1.0,
            ),
        )

        self.assertIsNotNone(transition)
        assert transition is not None
        self.assertEqual(transition.previous, InferenceTier.PRIORITY)
        self.assertEqual(transition.current, InferenceTier.STANDARD)
        self.assertEqual(transition.reason, InferencePriorityTransitionReason.RECOVERY)
        self.assertEqual(transition.evidence.healthy, 2)
        self.assertEqual(transition.evidence.threshold.recovery, 2)
        self.assertEqual(policy.select(), InferenceTier.STANDARD)
        policy.record(
            signal=PriorityInferenceSignal(
                tier=InferenceTier.STANDARD,
                success=True,
                latency=1.0,
            ),
        )
        self.assertEqual(policy.select(), InferenceTier.STANDARD)


if __name__ == "__main__":
    unittest.main()
