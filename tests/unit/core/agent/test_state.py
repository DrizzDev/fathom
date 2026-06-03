from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.core.agent.state import AgentState
from fathom.schemas.actions import Action
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.screens import ScreenState


class AgentStateContinuationTest(unittest.TestCase):
    """
    Covers continuation policy across autonomous and HITL runtimes via state.capabilities.
    """

    @staticmethod
    def __screen() -> ScreenState:
        """
        Return a stable screen used to create loop-detector evidence.
        """

        return ScreenState(
            timestamp=0,
            activity="app",
            visual_hash="b" * 16,
            activity_hash="a" * 16,
        )

    def test_hitl_capability_can_continue_after_autonomous_recovery_exhausted(self) -> None:
        """
        HITL-capable runtime is governed by the realignment budget, not the
        autonomous recovery budget. The capability flag on AgentState picks
        the correct stuck-rescue budget.
        """

        autonomous = AgentState(
            intent="complete onboarding",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        detector = autonomous.runtime.screen.detector

        for _ in range(detector.threshold):
            detector.record(
                action_type="tap",
                screen=self.__screen(),
                action_description="Tap Continue",
            )

        self.assertTrue(autonomous.is_stuck)

        while detector.can_recover():
            detector.record_recovery_attempt()

        self.assertFalse(autonomous.can_continue)

        hitl = AgentState(
            intent="complete onboarding",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
        )
        hitl_detector = hitl.runtime.screen.detector

        for _ in range(hitl_detector.threshold):
            hitl_detector.record(
                action_type="tap",
                screen=self.__screen(),
                action_description="Tap Continue",
            )
        while hitl_detector.can_recover():
            hitl_detector.record_recovery_attempt()

        self.assertTrue(hitl.can_continue)

    def test_from_checkpoint_uses_live_capabilities_not_stale(self) -> None:
        """
        Restoration must bind capabilities from the resuming runtime, not the checkpoint.
        """

        original = AgentState(
            intent="x",
            max_steps=7,
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
        )
        original.mark_complete(reason="done")
        payload = original.to_checkpoint()

        self.assertNotIn("hitl", payload)
        self.assertNotIn("capabilities", payload)

        restored = AgentState.from_checkpoint(
            payload,
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )

        self.assertTrue(restored.is_complete)
        self.assertEqual(restored.intent, "x")
        self.assertFalse(restored.capabilities.hitl.enabled)
        self.assertEqual(restored.completion_reason, "done")

    def test_from_checkpoint_requires_capabilities_argument(self) -> None:
        """
        from_checkpoint must reject a missing capabilities kwarg fail-fast.
        """

        payload = AgentState(
            intent="x",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        ).to_checkpoint()

        with self.assertRaises(TypeError):
            AgentState.from_checkpoint(payload)  # type: ignore[call-arg]

    def test_hitl_capability_stops_when_realignment_budget_is_exhausted(self) -> None:
        """
        HITL-capable runtime terminates once the realignment budget is exhausted.
        """

        state = AgentState(
            realignment_budget=1,
            intent="complete onboarding",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
        )
        detector = state.runtime.screen.detector

        for _ in range(detector.threshold):
            detector.record(
                action_type="tap",
                screen=self.__screen(),
                action_description="Tap Continue",
            )

        state.bump_realignment_budget()

        self.assertTrue(state.is_stuck)
        self.assertFalse(state.can_continue)


class AgentStateLastActionPersistenceTest(unittest.TestCase):
    """
    Behavioral pins for the last-action descriptor that the LoopDetector consumes via update_screen on the next graph turn.
    """

    @staticmethod
    def __caps() -> RuntimeCapabilities:
        """
        Return autonomous capabilities — HITL is irrelevant for these tests.
        """

        return RuntimeCapabilities(hitl=HITLCapability(enabled=False))

    @staticmethod
    def __screen(*, visual_hash: str = "a" * 16) -> ScreenState:
        """
        Return a stable :class:`ScreenState` for recording.
        """

        return ScreenState(
            timestamp=0,
            activity="app",
            activity_hash="0" * 16,
            visual_hash=visual_hash,
        )

    @staticmethod
    def __tap_action(*, target: str) -> Action:
        """
        Return a tap :class:`Action` with the requested target descriptor.
        """

        return Action(
            target=target,
            confidence=1.0,
            rationale="test fixture",
            action_type=ActionType.TAP,
        )

    def test_last_action_round_trips_through_checkpoint(self) -> None:
        """
        ``to_checkpoint`` / ``from_checkpoint`` must preserve
        ``__last_action_type`` and ``__last_action_description`` so a
        graph-state restore between nodes does not wipe the descriptor the
        LoopDetector will consume on the next ``update_screen``.
        """

        state = AgentState(intent="test", capabilities=self.__caps())

        state.record_attempt(
            reason="test_seed",
            action=self.__tap_action(target="Play button"),
        )

        checkpoint = state.to_checkpoint()
        restored = AgentState.from_checkpoint(checkpoint, capabilities=self.__caps())

        self.assertEqual(restored.last_action_type, "tap")
        self.assertEqual(
            restored._AgentState__last_action_description,  # type: ignore[attr-defined]
            "Tap on Play button",
        )

    def test_legacy_checkpoint_without_last_action_keys_restores_to_none(self) -> None:
        """
        Old checkpoints written before this change have no ``last_action_*`` keys.
        Restore must default to ``None`` to match the original first-turn semantics.
        """

        state = AgentState(intent="test", capabilities=self.__caps())

        payload = state.to_checkpoint()

        payload.pop("last_action_type")
        payload.pop("last_action_description")

        restored = AgentState.from_checkpoint(payload, capabilities=self.__caps())

        self.assertIsNone(restored.last_action_type)

    def test_record_attempt_pushes_intent_into_loop_detector(self) -> None:
        """
        Four rejected attempts of the same target are enough to flip ``is_stuck`` via the LoopDetector.
        Without ``record_attempt`` the intent never accumulates because no ``record_step`` runs on SUPERVISE-rejected paths.
        """

        state = AgentState(intent="test", capabilities=self.__caps())
        state.update_screen(screen=self.__screen())

        for _ in range(4):
            state.record_attempt(
                reason="supervise_spatial_unresolved",
                action=self.__tap_action(target="Play button"),
            )

        self.assertTrue(state.is_stuck)

    def test_record_attempt_updates_last_action(self) -> None:
        """
        After a rejected attempt the agent's intent must be the last
        recorded action so the next ``update_screen`` capture inherits it and the LoopDetector window stays in sync.
        """

        state = AgentState(intent="test", capabilities=self.__caps())

        state.record_attempt(
            reason="low_confidence",
            action=self.__tap_action(target="Submit"),
        )

        self.assertEqual(state.last_action_type, "tap")
        self.assertEqual(
            state._AgentState__last_action_description,  # type: ignore[attr-defined]
            "Tap on Submit",
        )
