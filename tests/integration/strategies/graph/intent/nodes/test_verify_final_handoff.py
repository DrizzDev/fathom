from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock

from tests.builders import SubGoalFixtures

from fathom.constants import ActionType
from fathom.constants.state import CommonStateKey, IntentStateKey, VerifyMode
from fathom.core.agent.state import AgentState
from fathom.core.services.timing import RunClock
from fathom.schemas.actions import Action
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.results import PlanResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.strategies.graph.intent.nodes.record import RecordNode
from fathom.strategies.graph.intent.nodes.verify import VerifyNode


class _SequenceLlm:
    """
    Deterministic LLM double returning verifier verdicts in order.
    """

    def __init__(self, *, contents: List[str]) -> None:
        """
        Bind the ordered verifier payloads and prompt capture list.
        """

        self.prompts: List[str] = []
        self.__contents = list(contents)

    async def generate(self, **kwargs: object) -> SimpleNamespace:
        """
        Return the next scripted verifier response.
        """

        prompt = kwargs.get("prompt")
        if isinstance(prompt, list) and prompt:
            self.prompts.append(str(prompt[0]))

        return SimpleNamespace(content=self.__contents.pop(0))


class _Perception:
    """
    Deterministic perception double used by VerifyNode.
    """

    async def perceive(self, **_: object) -> ScreenCapture:
        """
        Return a valid screenshot capture.
        """

        return ScreenCapture(
            width=100,
            height=200,
            image=b"png",
            timestamp=1,
            activity="com.delivery/.HomeActivity",
        )


class _ContextManager:
    """
    Context manager double spanning RecordNode and VerifyNode.
    """

    def __init__(self) -> None:
        """
        Initialise feedback and prompt-context capture.
        """

        self.cleared = False
        self.commits: List[str] = []
        self.feedback: List[str] = []

    def get_user_guidance(self) -> List[object]:
        """
        Return no operator guidance for this deterministic replay.
        """

        return []

    def get_full_context(self) -> Dict[str, object]:
        """
        Return trace context expected by RecordNode and sub-goal verifier mode.
        """

        return {"active_count": 0, "trace": []}

    async def commit(self, *, observation: str, action: Action, thought: str) -> None:
        """
        Record that RecordNode committed the executed action to trace.
        """

        _ = action, thought
        self.commits.append(observation)

    async def inject_verifier_feedback(self, *, feedback: str, step: int | None = None) -> None:
        """
        Capture verifier feedback for the next planner turn.
        """

        _ = step
        self.feedback.append(feedback)

    def clear_verifier_feedback(self) -> None:
        """
        Capture feedback clearing after accepted verification.
        """

        self.cleared = True


class _Persistence:
    """
    Persistence double that records every graph patch.
    """

    def __init__(self) -> None:
        """
        Initialise the persisted patch stream.
        """

        self.patches: List[Dict[object, object]] = []

    def restore(self, *, state: Dict[object, object]) -> None:
        """
        Restore is a no-op; the real AgentState instance is shared.
        """

        _ = state

    def persist(self, *, result: Dict[object, object]) -> None:
        """
        Snapshot a graph-state patch.
        """

        self.patches.append(dict(result))


class _Provider:
    """
    Shared provider fixture for RecordNode and VerifyNode integration.
    """

    def __init__(
        self,
        *,
        agent_state: AgentState,
        llm: _SequenceLlm,
        intent: str = "change the address to salary-se office",
        workflow_id: str = "salary-se-replay",
    ) -> None:
        """
        Bind the shared context and infrastructure doubles.
        """

        self.persistence = _Persistence()
        self.context_manager = _ContextManager()
        self.completion = SimpleNamespace(evaluate=AsyncMock(return_value=None))

        self.context = SimpleNamespace(
            llm=llm,
            max_steps=20,
            phase=AsyncMock(),
            clock=RunClock(),
            artifact_pipeline=None,
            agent_state=agent_state,
            perception=_Perception(),
            workflow_id=workflow_id,
            context_manager=self.context_manager,
            auditor=SimpleNamespace(log_step=MagicMock()),
            intent=intent,
            memory=SimpleNamespace(store_experience=AsyncMock()),
            telemetry=SimpleNamespace(info=AsyncMock(), warning=AsyncMock(), error=AsyncMock()),
            history=SimpleNamespace(save_completion_assertions=MagicMock()),
        )
        self.persistence.should_skip_launcher = MagicMock(return_value=False)  # type: ignore[attr-defined]
        self.persistence.enqueue_history = MagicMock()  # type: ignore[attr-defined]

    async def is_cancelled(self) -> bool:
        """
        Keep the replay on the normal non-cancelled path.
        """

        return False


class VerifyFinalHandoffIntegrationTest(unittest.IsolatedAsyncioTestCase):
    """
    Replays the final-subgoal VERIFY rejection/acceptance handoff.
    """

    @staticmethod
    def __step_result(*, step_number: int) -> StepResult:
        """
        Build a successful tap StepResult resembling the corrective modal action.
        """

        action = Action(
            confidence=1.0,
            action_type=ActionType.TAP,
            target="Yes, continue",
            rationale="Dismiss the blocker before confirming the address",
        )
        return StepResult(
            success=True,
            duration=25,
            pre_hash="before",
            post_hash="after",
            screen_changed=True,
            step=Step(action=action, step_number=step_number, screen_hash="before"),
        )

    @staticmethod
    def __screen() -> ScreenState:
        """
        Build the post-action screen state used by RecordNode.
        """

        return ScreenState(
            timestamp=1,
            visual_hash="b" * 16,
            activity_hash="a" * 16,
            activity="com.delivery/.HomeActivity",
        )

    def __record_state(self, *, step_number: int) -> Dict[object, object]:
        """
        Build the graph state for a RecordNode plan-complete turn.
        """

        return {
            CommonStateKey.IS_NEW_SCREEN: True,
            CommonStateKey.SCREEN_STATE: self.__screen(),
            IntentStateKey.POST_ACTIVITY: "com.delivery/.HomeActivity",
            CommonStateKey.STEP_RESULT: self.__step_result(step_number=step_number),
            IntentStateKey.PLAN: PlanResult(
                step=None,
                is_complete=True,
                reason="Finance office appears selected",
            ),
        }

    @staticmethod
    def __hsr_agent_state() -> AgentState:
        """
        Build AgentState at the HSR prod failure point with the final sub-goal still active.
        """

        agent_state = AgentState(
            intent="Change the address to HSR Layout",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        agent_state.set_sub_goals(
            [
                SubGoalFixtures.make(
                    index=0, description="Tap on the current address or change address option"
                ),
                SubGoalFixtures.make(
                    index=1, description="Type HSR Layout into the address search field"
                ),
                SubGoalFixtures.make(index=2, description="Tap HSR Layout from the search results"),
                SubGoalFixtures.make(
                    index=3, description="Tap the button to confirm or save the address change"
                ),
            ]
        )
        for _ in range(3):
            agent_state.advance_current_sub_goal()
        return agent_state

    @staticmethod
    def __hsr_corrective_step_result() -> StepResult:
        """
        Build the prod corrective tap on the confirmation sheet's affirmative action.
        """

        action = Action(
            confidence=1.0,
            action_type=ActionType.TAP,
            target="Yes, continue with this location",
            rationale="Tap the confirmation sheet action to finalize HSR Layout",
        )
        return StepResult(
            success=True,
            duration=25,
            pre_hash="modal",
            post_hash="home",
            screen_changed=True,
            step=Step(action=action, step_number=6, screen_hash="modal"),
        )

    def __hsr_record_state(self) -> Dict[object, object]:
        """
        Build the graph state seen by RECORD after the HSR corrective tap.
        """

        return {
            CommonStateKey.IS_NEW_SCREEN: True,
            CommonStateKey.SCREEN_STATE: self.__screen(),
            IntentStateKey.POST_ACTIVITY: "in.delivery.android",
            CommonStateKey.STEP_RESULT: self.__hsr_corrective_step_result(),
            IntentStateKey.PLAN: PlanResult(
                step=None,
                is_complete=True,
                reason="HSR Layout is visible in the delivery address header",
            ),
        }

    async def test_final_subgoal_rejection_remains_active_until_verify_accepts(self) -> None:
        """
        VERIFY rejection must not move the cursor past the final sub-goal; acceptance commits it.
        """

        agent_state = AgentState(
            intent="change the address to salary-se office",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        agent_state.set_sub_goals(
            [
                SubGoalFixtures.make(index=0, description="Tap address selector"),
                SubGoalFixtures.make(index=1, description="Confirm Finance office address"),
            ]
        )
        agent_state.advance_current_sub_goal()
        llm = _SequenceLlm(
            contents=[
                '{"is_complete": false, "reason": "Tap Yes, continue first"}',
                (
                    '{"is_complete": true, "reason": "Finance office is selected", '
                    '"assertions": [{"kind": "VISIBLE", "subject": "Finance office address header"}]}'
                ),
            ]
        )
        provider = _Provider(agent_state=agent_state, llm=llm)
        record = RecordNode(provider=provider)  # type: ignore[arg-type]
        verify = VerifyNode(provider=provider)  # type: ignore[arg-type]

        pending = await record.run(state=self.__record_state(step_number=0))  # type: ignore[arg-type]

        self.assertTrue(pending[CommonStateKey.IS_COMPLETE])
        self.assertEqual(
            pending[IntentStateKey.VERIFY_MODE],
            VerifyMode.PENDING_FINAL_COMMIT.value,
        )
        self.assertFalse(pending[IntentStateKey.SHOULD_RETRY])
        self.assertFalse(agent_state.is_complete)
        self.assertEqual(agent_state.current_sub_goal_index, 1)

        rejected = await verify.run(state=pending)  # type: ignore[arg-type]

        self.assertFalse(rejected[CommonStateKey.IS_COMPLETE])
        self.assertTrue(rejected[IntentStateKey.SHOULD_RETRY])
        self.assertIsNone(rejected[IntentStateKey.VERIFY_MODE])
        self.assertFalse(agent_state.is_complete)
        self.assertEqual(agent_state.current_sub_goal_index, 1)
        self.assertEqual(
            provider.context_manager.feedback,
            ["Verification failed: Tap Yes, continue first"],
        )
        self.assertIsNotNone(agent_state.verification_loop)

        pending_again = await record.run(state=self.__record_state(step_number=1))  # type: ignore[arg-type]
        self.assertFalse(pending_again[IntentStateKey.SHOULD_RETRY])
        self.assertIsNone(agent_state.verification_loop)
        accepted = await verify.run(state=pending_again)  # type: ignore[arg-type]

        self.assertTrue(accepted[CommonStateKey.IS_COMPLETE])
        self.assertIsNone(accepted[IntentStateKey.VERIFY_MODE])
        self.assertFalse(accepted[IntentStateKey.SHOULD_RETRY])

        self.assertTrue(agent_state.is_complete)
        self.assertTrue(provider.context_manager.cleared)
        self.assertTrue(agent_state.all_sub_goals_complete())

        self.assertNotIn("Step:", llm.prompts[0])
        self.assertIn("User Intent: change the address to salary-se office", llm.prompts[0])

    async def test_replay_rejects_corrects_then_verifies_final_intent(self) -> None:
        """
        Replay 6f72ec86: final VERIFY rejects the modal, corrective tap routes back to final VERIFY, then acceptance completes.
        """

        agent_state = self.__hsr_agent_state()
        llm = _SequenceLlm(
            contents=[
                (
                    '{"is_complete": false, "reason": "Although HSR Layout is visible, '
                    'the confirmation modal is still present. Tap Yes, continue with this location."}'
                ),
                (
                    '{"is_complete": true, "reason": "HSR Layout is selected in the address header.", '
                    '"assertions": [{"kind": "VISIBLE", "subject": "HSR Layout address header"}]}'
                ),
            ]
        )
        provider = _Provider(
            agent_state=agent_state,
            llm=llm,
            intent="Change the address to HSR Layout",
            workflow_id="6f72ec86-regression",
        )
        record = RecordNode(provider=provider)  # type: ignore[arg-type]
        verify = VerifyNode(provider=provider)  # type: ignore[arg-type]

        rejected = await verify.run(
            state={IntentStateKey.VERIFY_MODE: VerifyMode.PENDING_FINAL_COMMIT.value}
        )  # type: ignore[arg-type]

        self.assertFalse(rejected[CommonStateKey.IS_COMPLETE])
        self.assertTrue(rejected[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(agent_state.current_sub_goal_index, 3)
        self.assertFalse(agent_state.is_complete)
        self.assertEqual(
            provider.context_manager.feedback,
            [
                "Verification failed: Although HSR Layout is visible, "
                "the confirmation modal is still present. Tap Yes, continue with this location."
            ],
        )

        pending_again = await record.run(state=self.__hsr_record_state())  # type: ignore[arg-type]

        self.assertTrue(pending_again[CommonStateKey.IS_COMPLETE])
        self.assertFalse(pending_again[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(
            pending_again[IntentStateKey.VERIFY_MODE],
            VerifyMode.PENDING_FINAL_COMMIT.value,
        )
        self.assertFalse(agent_state.is_complete)
        self.assertEqual(agent_state.current_sub_goal_index, 3)

        accepted = await verify.run(state=pending_again)  # type: ignore[arg-type]

        self.assertTrue(accepted[CommonStateKey.IS_COMPLETE])
        self.assertFalse(accepted[IntentStateKey.SHOULD_RETRY])
        self.assertIsNone(accepted[IntentStateKey.VERIFY_MODE])
        self.assertTrue(agent_state.is_complete)
        self.assertTrue(agent_state.all_sub_goals_complete())
        self.assertTrue(provider.context_manager.cleared)
        self.assertNotIn("Step:", llm.prompts[0])
        self.assertIn("User Intent: Change the address to HSR Layout", llm.prompts[0])
