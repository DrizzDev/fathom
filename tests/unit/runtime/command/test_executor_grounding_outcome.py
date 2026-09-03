"""
The effective grounding policy must reach the benchmark outcome line.

A host grades a run against the policy it requested, so it has to be told the policy
that actually ran. Reporting nothing is the honest answer when the run never got a
policy — it broke while assembling — and the host records UNVERIFIED for it. Reporting
a *default* would be the fabrication this contract exists to remove: it reads as
agreement, and agreement is precisely the claim nobody has evidence for.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any, Dict, Optional

from fathom.constants.localization import GroundingPolicy
from fathom.runtime.command.executor import CommandExecutor
from fathom.runtime.plumb_stream import OUTCOME_PREFIX


class _CapturingStream:
    """
    A trajectory stream that keeps what was emitted instead of writing to stdout.
    """

    def __init__(self) -> None:
        self.enabled = True
        self.events: list[tuple[str, Dict[str, Any]]] = []
        self.payload: Optional[Dict[str, Any]] = None

    def event(self, *, kind: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self.events.append((kind, payload or {}))

    def outcome(self, *, payload: Dict[str, Any]) -> None:
        self.payload = payload


def _result(**overrides: Any) -> SimpleNamespace:
    """
    A finished workflow result shaped the way the emitter reads it.
    """

    values: Dict[str, Any] = {
        "status": "completed",
        "steps_taken": 3,
        "max_steps": 40,
        "completion_reason": "Goal reached",
        "success": True,
        "duration": 12.5,
        "error": None,
        "counters": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _emit_trajectory(*, stream: Any, result: Any, policy: Optional[GroundingPolicy]) -> None:
    """
    Call the private emitter under its mangled name, as the run flow does.
    """

    CommandExecutor._CommandExecutor__emit_trajectory(  # type: ignore[attr-defined]
        stream=stream, result=result, grounding_policy=policy
    )


class EffectiveGroundingPolicyReachesTheOutcomeTest(unittest.TestCase):
    """
    Verify the field a host compares against its request.
    """

    def test_the_policy_the_run_used_is_reported(self) -> None:
        stream = _CapturingStream()

        _emit_trajectory(stream=stream, result=_result(), policy=GroundingPolicy.PERMISSIVE)

        assert stream.payload is not None
        self.assertEqual(stream.payload["grounding_policy"], "PERMISSIVE")

    def test_strict_is_reported_as_itself(self) -> None:
        """
        Both values must travel; a field that only ever says PERMISSIVE proves nothing.
        """

        stream = _CapturingStream()

        _emit_trajectory(stream=stream, result=_result(), policy=GroundingPolicy.STRICT)

        assert stream.payload is not None
        self.assertEqual(stream.payload["grounding_policy"], "STRICT")

    def test_an_unknown_policy_is_omitted_rather_than_defaulted(self) -> None:
        """
        Absent means "never confirmed". A default here would read as agreement.
        """

        stream = _CapturingStream()

        _emit_trajectory(stream=stream, result=_result(), policy=None)

        assert stream.payload is not None
        self.assertNotIn("grounding_policy", stream.payload)

    def test_the_rest_of_the_contract_is_unchanged(self) -> None:
        """
        Adding a field must not disturb the ones a host already parses.
        """

        stream = _CapturingStream()

        _emit_trajectory(stream=stream, result=_result(), policy=GroundingPolicy.PERMISSIVE)

        assert stream.payload is not None
        self.assertEqual(stream.payload["status"], "COMPLETED")
        self.assertEqual(stream.payload["steps_taken"], 3)
        self.assertEqual(stream.payload["effective_max_steps"], 40)
        self.assertEqual(stream.payload["status_source"], "PAYLOAD")
        self.assertIs(stream.payload["self_reported_success"], True)

    def test_the_payload_is_json_serialisable_behind_the_prefix(self) -> None:
        """
        The host reads one line of JSON after a fixed prefix; an enum that serialised
        as ``GroundingPolicy.PERMISSIVE`` would parse into a value it cannot match.
        """

        stream = _CapturingStream()

        _emit_trajectory(stream=stream, result=_result(), policy=GroundingPolicy.PERMISSIVE)

        assert stream.payload is not None
        line = OUTCOME_PREFIX + json.dumps(stream.payload, default=str)
        decoded = json.loads(line[len(OUTCOME_PREFIX) :])
        self.assertEqual(decoded["grounding_policy"], "PERMISSIVE")


if __name__ == "__main__":
    unittest.main()
