import unittest
from datetime import timedelta

from fathom.runtime.temporal.workflow import FathomBaseWorkflow
from fathom.schemas.configuration import WorkflowHostPolicyConfiguration


class _WorkflowDouble(FathomBaseWorkflow):
    """
    Minimal workflow double for Temporal policy calculations.
    """


class TestTemporalWorkflowPolicy(unittest.TestCase):
    """
    Cover Temporal activity policy calculation helpers.
    """

    def test_compute_activity_timeout_uses_policy_values(self) -> None:
        """
        Build start-to-close timeout from the configured policy values.
        """

        workflow = _WorkflowDouble()
        policy = WorkflowHostPolicyConfiguration(
            timeout_floor=45,
            timeout_per_step=3,
            timeout_overhead=7,
            heartbeat_seconds=120,
        )

        timeout = workflow.timeout(max_steps=20, policy=policy)

        self.assertEqual(timeout, timedelta(minutes=67))

    def test_compute_activity_heartbeat_uses_policy_value(self) -> None:
        """
        Build heartbeat timeout from the configured policy value.
        """

        workflow = _WorkflowDouble()
        policy = WorkflowHostPolicyConfiguration(heartbeat_seconds=180)

        heartbeat = workflow.heartbeat(policy=policy)

        self.assertEqual(heartbeat, timedelta(seconds=180))
