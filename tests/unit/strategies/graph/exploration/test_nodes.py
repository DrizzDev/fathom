from __future__ import annotations

import unittest
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, Mock, patch

from fathom.constants import ActionType
from fathom.constants.defect import (
    DEFECT_DETECTED_EVENT,
    DefectSignal,
    DefectSource,
    DefectVerification,
)
from fathom.constants.exploration import (
    EXPLORATION_PROGRESS_EVENT,
    MAX_ROUTES_WITHOUT_PROGRESS,
    MAX_STEPS_WITHOUT_NEW_SCREEN,
    BFSPhase,
    ExpectedOutcome,
    FocusRelevance,
)
from fathom.constants.graph import NodeName
from fathom.constants.screen import ScreenCategory
from fathom.constants.state import CommonStateKey as CKey
from fathom.constants.state import CompletionReason
from fathom.constants.state import ExplorationStateKey as EKey
from fathom.core.exceptions import DeviceError
from fathom.core.safety.guard import TraversalGuard
from fathom.schemas.actions import Action
from fathom.schemas.defect import Defect, DefectEvidence
from fathom.schemas.exploration import TriedAction
from fathom.schemas.results import AnalysisResult, ExecutionResult
from fathom.schemas.screens import ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.strategies.graph.exploration.dfs import DfsState
from fathom.strategies.graph.exploration.nodes import ExplorationNodeProvider


def _screen_state(visual_hash: str = "ffffffffffffffff") -> ScreenState:
    return ScreenState(
        activity="com.app/.Home",
        timestamp=0,
        activity_hash="acthash",
        visual_hash=visual_hash,
    )


def _action(target: str = "Home", action_type: ActionType = ActionType.TAP) -> Action:
    return Action(action_type=action_type, rationale="r", natural_language_target=target)


def _analysis(
    *,
    action: Action,
    content_exhausted: bool = False,
    focus_relevance: FocusRelevance | None = None,
    category: ScreenCategory | None = None,
) -> AnalysisResult:
    return AnalysisResult(
        action=action,
        reasoning="r",
        screen_description="s",
        content_exhausted=content_exhausted,
        focus_relevance=focus_relevance,
        category=category,
    )


def _graph_mock(**overrides: Any) -> Mock:
    graph = Mock(
        nodes={},
        resolve_hash=Mock(side_effect=lambda value: value),
        build_exploration_context=Mock(return_value="CONTEXT"),
        get_tried_actions=Mock(return_value=[]),
        get_screen=Mock(return_value=None),
        count_category_taps=Mock(return_value=0),
        has_screen=Mock(return_value=False),
        exhausted_hashes=Mock(return_value=set()),
        node_count=0,
        add_screen=AsyncMock(),
        update_rich_description=AsyncMock(),
        record_transition=AsyncMock(),
        mark_exhausted=AsyncMock(),
        record_relevance=AsyncMock(),
        record_category=AsyncMock(),
    )
    for key, value in overrides.items():
        setattr(graph, key, value)
    return graph


def _quiet_hitl() -> Mock:
    return Mock(
        check_signal=AsyncMock(return_value=None),
        is_pause_requested=AsyncMock(return_value=False),
        wait_for_resume=AsyncMock(),
    )


class TestGroundNode(unittest.IsolatedAsyncioTestCase):
    """Ground captures and resets per-step fields without registering the screen."""

    async def test_ground_captures_and_resets(self) -> None:
        capture = Mock()
        capture.model_copy = Mock(return_value=capture)
        context = Mock(
            is_cancelled=False,
            workflow_id="wf",
            exploration_graph=_graph_mock(),
            hitl=_quiet_hitl(),
        )
        context.perception = Mock(
            perceive=AsyncMock(return_value=capture),
            build_state=Mock(return_value=_screen_state()),
        )
        context.agent_state = Mock(update_screen=Mock(return_value=True))

        result = await ExplorationNodeProvider(context=context, vision=Mock()).ground(
            {EKey.ACTION: _action()}
        )

        context.perception.build_state.assert_called_once_with(capture=capture)
        # Screen registration is deferred to scan/record, not ground.
        context.exploration_graph.add_screen.assert_not_called()
        self.assertTrue(result[CKey.IS_NEW_SCREEN])
        self.assertIsNone(result[EKey.ACTION])
        self.assertFalse(result[EKey.CONTENT_EXHAUSTED])

    async def test_ground_fails_fast_on_unusable_capture(self) -> None:
        # An empty screenshot must end the run with a clear reason, not wedge it.
        blank = Mock(image=b"", xml_content=None, activity="unknown")
        context = Mock(
            is_cancelled=False,
            workflow_id="wf",
            exploration_graph=_graph_mock(),
            hitl=_quiet_hitl(),
        )
        context.perception = Mock(
            perceive=AsyncMock(return_value=blank),
            build_state=Mock(return_value=_screen_state()),
        )
        context.agent_state = Mock(update_screen=Mock(return_value=True))

        result = await ExplorationNodeProvider(context=context, vision=Mock()).ground({})

        self.assertTrue(result[CKey.IS_COMPLETE])
        self.assertEqual(result[CKey.COMPLETION_REASON], CompletionReason.PERCEPTION_FAILED)

    async def test_ground_fails_fast_when_perception_raises(self) -> None:
        context = Mock(
            is_cancelled=False,
            workflow_id="wf",
            exploration_graph=_graph_mock(),
            hitl=_quiet_hitl(),
            agent_state=Mock(step_count=0),
        )
        context.perception = Mock(perceive=AsyncMock(side_effect=DeviceError("snapshot timed out")))

        result = await ExplorationNodeProvider(context=context, vision=Mock()).ground({})

        self.assertTrue(result[CKey.IS_COMPLETE])
        self.assertEqual(result[CKey.COMPLETION_REASON], CompletionReason.PERCEPTION_FAILED)


class TestScanNode(unittest.IsolatedAsyncioTestCase):
    """Scan picks a novel action, honours exhaustion, and applies the depth floor."""

    @staticmethod
    def __context(graph: Mock, *, focus: str | None = None) -> Mock:
        return Mock(
            is_cancelled=False,
            intent="Explore application",
            focus=focus,
            exploration_graph=graph,
            metrics=Mock(record=Mock()),
        )

    async def test_returns_novel_action(self) -> None:
        action = _action("Home")
        vision = Mock(scan=AsyncMock(return_value=_analysis(action=action)))
        context = self.__context(_graph_mock())
        state = {CKey.CAPTURE: Mock(), CKey.SCREEN_STATE: _screen_state()}

        result = await ExplorationNodeProvider(context=context, vision=vision).scan(state)

        vision.scan.assert_awaited_once()
        self.assertIs(result[EKey.ACTION], action)
        self.assertFalse(result[EKey.CONTENT_EXHAUSTED])

    async def test_runs_screen_detector_once_per_screen(self) -> None:
        context = self.__context(_graph_mock())
        context.telemetry = Mock(info=AsyncMock())
        defect = Defect.from_signal(
            signal=DefectSignal.OVERLAP_CLIPPING,
            source=DefectSource.POST_RUN,
            summary="Title overlaps the cart icon",
            evidence=DefectEvidence(screen="s"),
        )
        screen_detector = Mock(inspect_screen=AsyncMock(return_value=[defect]))
        vision = Mock(scan=AsyncMock(return_value=_analysis(action=_action())))
        provider = ExplorationNodeProvider(
            context=context, vision=vision, screen_detector=screen_detector, dfs=DfsState()
        )
        capture = Mock(image=b"PNG", width=1000, height=2000, xml_content=None, screenshot_uri=None)
        state = {CKey.CAPTURE: capture, CKey.SCREEN_STATE: _screen_state("s")}

        await provider.scan(state)
        await provider.scan(state)

        screen_detector.inspect_screen.assert_awaited_once()
        events = [call.args[0] for call in context.telemetry.info.await_args_list]
        self.assertIn(DEFECT_DETECTED_EVENT, events)

    async def test_honours_exhaustion_past_the_depth_floor(self) -> None:
        vision = Mock(
            scan=AsyncMock(return_value=_analysis(action=_action(), content_exhausted=True))
        )
        context = self.__context(_graph_mock())
        # A path deep enough that the depth-floor veto does not apply.
        deep_path: List[Tuple[str, Action]] = [(f"h{i}", _action()) for i in range(4)]
        dfs = DfsState(current_path=deep_path)
        state = {CKey.CAPTURE: Mock(), CKey.SCREEN_STATE: _screen_state()}

        provider = ExplorationNodeProvider(context=context, vision=vision, dfs=dfs)
        result = await provider.scan(state)

        self.assertTrue(result[EKey.CONTENT_EXHAUSTED])
        self.assertIsNone(result[EKey.ACTION])
        self.assertEqual(dfs.phase, BFSPhase.BACKTRACK)

    async def test_depth_floor_vetoes_premature_exhaustion(self) -> None:
        vision = Mock(
            scan=AsyncMock(return_value=_analysis(action=_action(), content_exhausted=True))
        )
        context = self.__context(_graph_mock())
        dfs = DfsState()  # depth 0 -> below the floor
        state = {CKey.CAPTURE: Mock(), CKey.SCREEN_STATE: _screen_state("abc")}

        provider = ExplorationNodeProvider(context=context, vision=vision, dfs=dfs)
        result = await provider.scan(state)

        self.assertFalse(result[EKey.CONTENT_EXHAUSTED])
        self.assertEqual(dfs.phase, BFSPhase.SCAN)
        self.assertEqual(dfs.exhaustion_retries["abc"], 1)

    async def test_threads_windowed_recent_action_feedback(self) -> None:
        action = _action("Home")
        vision = Mock(scan=AsyncMock(return_value=_analysis(action=action)))
        graph = _graph_mock()
        context = self.__context(graph)
        priors = [
            StepResult(
                step=Step(action=_action(name), screen_hash="pre", step_number=index),
                success=True,
                duration=1,
                screen_changed=(name != "c"),
                pre_hash="pre",
                post_hash="post",
            )
            for index, name in enumerate(["a", "b", "c", "d"])
        ]
        state = {
            CKey.CAPTURE: Mock(),
            CKey.SCREEN_STATE: _screen_state(),
            EKey.STEP_RESULTS: priors,
        }

        await ExplorationNodeProvider(context=context, vision=vision).scan(state)

        _args, kwargs = graph.build_exploration_context.call_args
        recent = kwargs["recent_actions"]
        # Only the most recent window is surfaced, oldest ("a") dropped.
        self.assertEqual([outcome.target for outcome in recent], ["b", "c", "d"])
        self.assertFalse(recent[1].screen_changed)

    async def test_scan_passes_fully_scanned_set_to_context(self) -> None:
        vision = Mock(scan=AsyncMock(return_value=_analysis(action=_action("Home"))))
        graph = _graph_mock()
        context = self.__context(graph)
        dfs = DfsState(fully_scanned={"h1", "h2"})
        state = {CKey.CAPTURE: Mock(), CKey.SCREEN_STATE: _screen_state()}

        await ExplorationNodeProvider(context=context, vision=vision, dfs=dfs).scan(state)

        _args, kwargs = graph.build_exploration_context.call_args
        self.assertEqual(kwargs["fully_scanned"], {"h1", "h2"})

    async def test_focused_scan_passes_focus_and_records_relevance(self) -> None:
        analysis = _analysis(action=_action("Cart"), focus_relevance=FocusRelevance.ON_FOCUS)
        vision = Mock(scan=AsyncMock(return_value=analysis))
        graph = _graph_mock()
        context = self.__context(graph, focus="checkout flow")
        state = {CKey.CAPTURE: Mock(), CKey.SCREEN_STATE: _screen_state("abc")}

        await ExplorationNodeProvider(context=context, vision=vision).scan(state)

        _args, kwargs = graph.build_exploration_context.call_args
        self.assertEqual(kwargs["focus"], "checkout flow")
        graph.record_relevance.assert_awaited_once_with(
            visual_hash="abc", relevance=FocusRelevance.ON_FOCUS
        )

    async def test_unfocused_scan_skips_relevance_recording(self) -> None:
        # Even if the model returns a relevance, a broad-coverage run records nothing.
        analysis = _analysis(action=_action("Home"), focus_relevance=FocusRelevance.OFF_FOCUS)
        vision = Mock(scan=AsyncMock(return_value=analysis))
        graph = _graph_mock()
        context = self.__context(graph, focus=None)
        state = {CKey.CAPTURE: Mock(), CKey.SCREEN_STATE: _screen_state()}

        await ExplorationNodeProvider(context=context, vision=vision).scan(state)

        _args, kwargs = graph.build_exploration_context.call_args
        self.assertIsNone(kwargs["focus"])
        graph.record_relevance.assert_not_awaited()

    async def test_scan_stores_rich_description_on_the_node(self) -> None:
        # The describe_screen markdown is stored per fingerprint, not per activity.
        analysis = _analysis(action=_action("Home"))
        analysis.metadata["rich_description"] = "## Purpose\nHome screen"
        vision = Mock(scan=AsyncMock(return_value=analysis))
        graph = _graph_mock()
        context = self.__context(graph, focus=None)
        state = {CKey.CAPTURE: Mock(), CKey.SCREEN_STATE: _screen_state("abc")}

        await ExplorationNodeProvider(context=context, vision=vision).scan(state)

        graph.update_rich_description.assert_awaited_once_with(
            visual_hash="abc", rich_description="## Purpose\nHome screen"
        )

    async def test_scan_records_screen_category(self) -> None:
        # Category is recorded on every run, focus or not, for the per-screen docs.
        analysis = _analysis(action=_action("Pay"), category=ScreenCategory.PAYMENT)
        vision = Mock(scan=AsyncMock(return_value=analysis))
        graph = _graph_mock()
        context = self.__context(graph, focus=None)
        state = {CKey.CAPTURE: Mock(), CKey.SCREEN_STATE: _screen_state("abc")}

        await ExplorationNodeProvider(context=context, vision=vision).scan(state)

        graph.record_category.assert_awaited_once_with(
            visual_hash="abc", category=ScreenCategory.PAYMENT
        )

    async def test_scan_without_category_skips_category_recording(self) -> None:
        analysis = _analysis(action=_action("Home"), category=None)
        vision = Mock(scan=AsyncMock(return_value=analysis))
        graph = _graph_mock()
        context = self.__context(graph, focus=None)
        state = {CKey.CAPTURE: Mock(), CKey.SCREEN_STATE: _screen_state()}

        await ExplorationNodeProvider(context=context, vision=vision).scan(state)

        graph.record_category.assert_not_awaited()

    async def test_traversal_guard_reprompts_away_from_a_sensitive_action(self) -> None:
        # The model first picks a payment action; the guard re-prompts and the
        # benign re-pick is what the step executes.
        vision = Mock(
            scan=AsyncMock(
                side_effect=[
                    _analysis(action=_action("Proceed to Pay")),
                    _analysis(action=_action("Restaurant card")),
                ]
            )
        )
        graph = _graph_mock()
        context = self.__context(graph, focus=None)
        state = {CKey.CAPTURE: Mock(), CKey.SCREEN_STATE: _screen_state("abc")}

        result = await ExplorationNodeProvider(
            context=context, vision=vision, traversal_guard=TraversalGuard()
        ).scan(state)

        self.assertEqual(result[EKey.ACTION].natural_language_target, "Restaurant card")
        self.assertGreaterEqual(vision.scan.await_count, 2)

    async def test_without_a_guard_a_sensitive_action_passes_through(self) -> None:
        analysis = _analysis(action=_action("Proceed to Pay"))
        vision = Mock(scan=AsyncMock(return_value=analysis))
        graph = _graph_mock()
        context = self.__context(graph, focus=None)
        state = {CKey.CAPTURE: Mock(), CKey.SCREEN_STATE: _screen_state("abc")}

        result = await ExplorationNodeProvider(context=context, vision=vision).scan(state)

        self.assertEqual(result[EKey.ACTION].natural_language_target, "Proceed to Pay")

    async def test_persists_exhaustion_past_the_depth_floor(self) -> None:
        graph = _graph_mock()
        vision = Mock(
            scan=AsyncMock(return_value=_analysis(action=_action(), content_exhausted=True))
        )
        context = self.__context(graph)
        deep_path: List[Tuple[str, Action]] = [(f"h{i}", _action()) for i in range(4)]
        dfs = DfsState(current_path=deep_path)
        state = {CKey.CAPTURE: Mock(), CKey.SCREEN_STATE: _screen_state("scr")}

        await ExplorationNodeProvider(context=context, vision=vision, dfs=dfs).scan(state)

        graph.mark_exhausted.assert_awaited_once_with(visual_hash="scr")

    async def test_dedup_guard_reprompts_when_action_already_tried(self) -> None:
        tried = _action("Home")
        fresh = _action("Search")
        vision = Mock(
            scan=AsyncMock(side_effect=[_analysis(action=tried), _analysis(action=fresh)])
        )
        graph = _graph_mock(
            get_tried_actions=Mock(
                return_value=[
                    TriedAction(action_type="tap", target="Home", destination_hash="dest")
                ]
            )
        )
        context = self.__context(graph)
        state = {CKey.CAPTURE: Mock(), CKey.SCREEN_STATE: _screen_state()}

        result = await ExplorationNodeProvider(context=context, vision=vision).scan(state)

        self.assertEqual(vision.scan.await_count, 2)
        self.assertIs(result[EKey.ACTION], fresh)


class TestRecordNode(unittest.IsolatedAsyncioTestCase):
    """Record persists the step, records the transition, and advances the DFS."""

    async def test_records_transition_and_descends(self) -> None:
        graph = _graph_mock(resolve_hash=Mock(side_effect=lambda value: value))
        context = Mock(
            is_cancelled=False,
            max_steps=100,
            workflow_id="wf",
            package_name="com.app",
            device=Mock(get_current_package=AsyncMock(return_value="com.app")),
            exploration_graph=graph,
            memory=Mock(store_experience=AsyncMock()),
            history=Mock(enqueue_save_step=Mock()),
            agent_state=Mock(record_step=Mock(), step_count=1),
            telemetry=Mock(info=AsyncMock()),
        )
        context.perception = Mock(
            perceive=AsyncMock(return_value=Mock(screenshot_uri=None)),
            build_state=Mock(return_value=_screen_state("post")),
        )
        action = _action()
        step = Step(action=action, screen_hash="pre", step_number=1)
        state: Dict[str, Any] = {
            EKey.ACTION: action,
            CKey.SCREEN_STATE: _screen_state("pre"),
            CKey.STEP_RESULT: StepResult(
                step=step,
                success=True,
                duration=10,
                screen_changed=True,
                pre_hash="pre",
                post_hash="0",
            ),
        }
        dfs = DfsState(phase=BFSPhase.SCAN)

        provider = ExplorationNodeProvider(context=context, vision=Mock(), dfs=dfs)
        result = await provider.record(state)

        context.agent_state.record_step.assert_called_once()
        graph.record_transition.assert_awaited_once()
        context.memory.store_experience.assert_awaited_once()
        # pre != post and a forward tap -> descend, pushing the parent onto the path.
        self.assertEqual(dfs.current_path, [("pre", action)])
        self.assertEqual(result[EKey.BFS_PHASE], BFSPhase.SCAN.value)
        self.assertFalse(result[CKey.IS_COMPLETE])
        # The step published a progress snapshot for live observers.
        progress_call = context.telemetry.info.await_args
        self.assertEqual(progress_call.args[0], EXPLORATION_PROGRESS_EVENT)
        self.assertEqual(progress_call.kwargs["step"], 1)

    async def test_recorded_step_resets_the_no_progress_watchdog(self) -> None:
        graph = _graph_mock(resolve_hash=Mock(side_effect=lambda value: value))
        context = Mock(
            is_cancelled=False,
            max_steps=100,
            workflow_id="wf",
            package_name="com.app",
            device=Mock(get_current_package=AsyncMock(return_value="com.app")),
            exploration_graph=graph,
            memory=Mock(store_experience=AsyncMock()),
            history=Mock(enqueue_save_step=Mock()),
            agent_state=Mock(record_step=Mock(), step_count=1),
            telemetry=Mock(info=AsyncMock()),
        )
        context.perception = Mock(
            perceive=AsyncMock(return_value=Mock(screenshot_uri=None)),
            build_state=Mock(return_value=_screen_state("post")),
        )
        action = _action()
        step = Step(action=action, screen_hash="pre", step_number=1)
        state: Dict[str, Any] = {
            EKey.ACTION: action,
            CKey.SCREEN_STATE: _screen_state("pre"),
            CKey.STEP_RESULT: StepResult(
                step=step,
                success=True,
                duration=10,
                screen_changed=True,
                pre_hash="pre",
                post_hash="0",
            ),
        }
        dfs = DfsState(phase=BFSPhase.SCAN, stalled_routes=7)

        await ExplorationNodeProvider(context=context, vision=Mock(), dfs=dfs).record(state)

        self.assertEqual(dfs.stalled_routes, 0)

    @staticmethod
    def __record_context(*, has_screen: bool) -> Mock:
        context = Mock(
            is_cancelled=False,
            max_steps=100,
            workflow_id="wf",
            package_name="com.app",
            device=Mock(get_current_package=AsyncMock(return_value="com.app")),
            exploration_graph=_graph_mock(
                resolve_hash=Mock(side_effect=lambda value: value),
                has_screen=Mock(return_value=has_screen),
            ),
            memory=Mock(store_experience=AsyncMock()),
            history=Mock(enqueue_save_step=Mock()),
            agent_state=Mock(record_step=Mock(), step_count=1),
            telemetry=Mock(info=AsyncMock()),
        )
        context.perception = Mock(
            perceive=AsyncMock(return_value=Mock(screenshot_uri=None)),
            build_state=Mock(return_value=_screen_state("post")),
        )
        return context

    @staticmethod
    def __record_state() -> Dict[str, Any]:
        action = _action()
        step = Step(action=action, screen_hash="pre", step_number=1)
        return {
            EKey.ACTION: action,
            CKey.SCREEN_STATE: _screen_state("pre"),
            CKey.STEP_RESULT: StepResult(
                step=step,
                success=True,
                duration=10,
                screen_changed=True,
                pre_hash="pre",
                post_hash="0",
            ),
        }

    async def test_plateau_completes_after_too_many_steps_without_a_new_screen(self) -> None:
        # has_screen=True: the post screen is already known, so this step finds nothing new.
        context = self.__record_context(has_screen=True)
        dfs = DfsState(phase=BFSPhase.SCAN, steps_since_new_screen=MAX_STEPS_WITHOUT_NEW_SCREEN)

        result = await ExplorationNodeProvider(context=context, vision=Mock(), dfs=dfs).record(
            self.__record_state()
        )

        self.assertTrue(result[CKey.IS_COMPLETE])
        self.assertEqual(result[CKey.COMPLETION_REASON], CompletionReason.COVERAGE_PLATEAU)

    async def test_new_screen_resets_the_plateau_counter(self) -> None:
        # has_screen=False: the post screen is new, so the plateau counter resets.
        context = self.__record_context(has_screen=False)
        dfs = DfsState(phase=BFSPhase.SCAN, steps_since_new_screen=10)

        await ExplorationNodeProvider(context=context, vision=Mock(), dfs=dfs).record(
            self.__record_state()
        )

        self.assertEqual(dfs.steps_since_new_screen, 0)

    async def test_dead_tap_emits_a_functional_defect(self) -> None:
        graph = _graph_mock(resolve_hash=Mock(side_effect=lambda value: value))
        context = Mock(
            is_cancelled=False,
            max_steps=100,
            workflow_id="wf",
            package_name="com.app",
            device=Mock(get_current_package=AsyncMock(return_value="com.app")),
            exploration_graph=graph,
            memory=Mock(store_experience=AsyncMock()),
            history=Mock(enqueue_save_step=Mock()),
            agent_state=Mock(record_step=Mock(), step_count=1),
            telemetry=Mock(info=AsyncMock()),
        )
        # The post-action capture lands on the SAME screen: the predicted
        # transition never happened, so the tap was inert.
        context.perception = Mock(
            perceive=AsyncMock(return_value=Mock(screenshot_uri=None)),
            build_state=Mock(return_value=_screen_state("pre")),
        )
        action = Action(
            action_type=ActionType.TAP,
            rationale="r",
            natural_language_target="Buy",
            expected_outcome=ExpectedOutcome.NEW_SCREEN,
        )
        step = Step(action=action, screen_hash="pre", step_number=1)
        state: Dict[str, Any] = {
            EKey.ACTION: action,
            CKey.SCREEN_STATE: _screen_state("pre"),
            CKey.STEP_RESULT: StepResult(
                step=step,
                success=True,
                duration=10,
                screen_changed=True,
                pre_hash="pre",
                post_hash="0",
            ),
        }

        await ExplorationNodeProvider(
            context=context, vision=Mock(), dfs=DfsState(phase=BFSPhase.SCAN)
        ).record(state)

        defect_signals = [
            call.kwargs["signal"]
            for call in context.telemetry.info.await_args_list
            if call.args and call.args[0] == DEFECT_DETECTED_EVENT
        ]
        self.assertEqual(defect_signals, [DefectSignal.DEAD_TAP.value])


class TestNavigateNode(unittest.IsolatedAsyncioTestCase):
    """Navigate synthesises BACK presses and completes when recovery is empty."""

    async def test_backtrack_presses_hardware_back(self) -> None:
        context = Mock(
            is_cancelled=False,
            workflow_id="wf",
            package_name="com.app",
            configuration=Mock(),
            exploration_graph=_graph_mock(),
            metrics=Mock(record=Mock()),
            action_executor=Mock(
                act=AsyncMock(return_value=ExecutionResult(success=True, duration=5))
            ),
            agent_state=Mock(step_count=0),
        )
        dfs = DfsState(phase=BFSPhase.BACKTRACK, current_path=[("h0", _action())])
        state = {CKey.CAPTURE: Mock(), CKey.SCREEN_STATE: _screen_state()}

        with patch("fathom.strategies.graph.exploration.nodes.stability_wait", new=AsyncMock()):
            result = await ExplorationNodeProvider(
                context=context, vision=Mock(), dfs=dfs
            ).navigate(state)

        context.action_executor.act.assert_awaited_once()
        self.assertEqual(result[CKey.STEP_RESULT].step.action.action_type, ActionType.BACK)

    async def test_backtrack_at_root_with_no_orphans_completes(self) -> None:
        context = Mock(is_cancelled=False, exploration_graph=_graph_mock(nodes={}))
        dfs = DfsState(phase=BFSPhase.BACKTRACK, current_path=[])
        state: Dict[str, Any] = {CKey.CAPTURE: Mock()}

        result = await ExplorationNodeProvider(context=context, vision=Mock(), dfs=dfs).navigate(
            state
        )

        self.assertTrue(result[CKey.IS_COMPLETE])


class TestBfsRouteNode(unittest.IsolatedAsyncioTestCase):
    """bfs_route seeds the root and publishes the current phase."""

    async def test_establishes_root_and_publishes_phase(self) -> None:
        context = Mock(exploration_graph=_graph_mock())
        dfs = DfsState()
        state = {CKey.SCREEN_STATE: _screen_state("root")}

        result = await ExplorationNodeProvider(context=context, vision=Mock(), dfs=dfs).bfs_route(
            state
        )

        self.assertEqual(dfs.root_hash, "root")
        self.assertEqual(result[EKey.BFS_PHASE], BFSPhase.SCAN.value)

    async def test_seeds_fully_scanned_and_backtracks_exhausted_root(self) -> None:
        context = Mock(exploration_graph=_graph_mock(exhausted_hashes=Mock(return_value={"root"})))
        dfs = DfsState()
        state = {CKey.SCREEN_STATE: _screen_state("root")}

        result = await ExplorationNodeProvider(context=context, vision=Mock(), dfs=dfs).bfs_route(
            state
        )

        self.assertIn("root", dfs.fully_scanned)
        self.assertEqual(dfs.phase, BFSPhase.BACKTRACK)
        self.assertEqual(result[EKey.BFS_PHASE], BFSPhase.BACKTRACK.value)

    async def test_seeds_frontier_but_scans_unexhausted_root(self) -> None:
        context = Mock(exploration_graph=_graph_mock(exhausted_hashes=Mock(return_value={"other"})))
        dfs = DfsState()
        state = {CKey.SCREEN_STATE: _screen_state("root")}

        await ExplorationNodeProvider(context=context, vision=Mock(), dfs=dfs).bfs_route(state)

        self.assertEqual(dfs.fully_scanned, {"other"})
        self.assertEqual(dfs.phase, BFSPhase.SCAN)

    async def test_ends_stuck_when_routing_makes_no_progress(self) -> None:
        # The phase machine wedged: routing cycled the bound times without a step.
        context = Mock(exploration_graph=_graph_mock())
        dfs = DfsState(root_hash="root", stalled_routes=MAX_ROUTES_WITHOUT_PROGRESS)
        state = {CKey.SCREEN_STATE: _screen_state("root")}

        result = await ExplorationNodeProvider(context=context, vision=Mock(), dfs=dfs).bfs_route(
            state
        )

        self.assertTrue(result[CKey.IS_COMPLETE])
        self.assertEqual(result[CKey.COMPLETION_REASON], CompletionReason.STUCK)

    async def test_counts_routes_but_continues_below_the_stall_bound(self) -> None:
        context = Mock(exploration_graph=_graph_mock())
        dfs = DfsState(root_hash="root", stalled_routes=2)
        state = {CKey.SCREEN_STATE: _screen_state("root")}

        result = await ExplorationNodeProvider(context=context, vision=Mock(), dfs=dfs).bfs_route(
            state
        )

        self.assertEqual(dfs.stalled_routes, 3)
        self.assertFalse(result.get(CKey.IS_COMPLETE, False))
        self.assertEqual(result[EKey.BFS_PHASE], BFSPhase.SCAN.value)


class TestRouters(unittest.TestCase):
    """The four conditional-edge routers steer by phase and completion."""

    @staticmethod
    def __provider(dfs: DfsState, *, cancelled: bool = False, can_continue: bool = True) -> Any:
        context = Mock(
            is_cancelled=cancelled,
            exploration_graph=_graph_mock(),
            agent_state=Mock(can_continue=can_continue),
        )
        return ExplorationNodeProvider(context=context, vision=Mock(), dfs=dfs)

    def test_after_ground(self) -> None:
        provider = self.__provider(DfsState())
        self.assertEqual(provider.after_ground({CKey.CAPTURE: Mock()}), NodeName.BFS_ROUTE)
        self.assertEqual(provider.after_ground({CKey.CAPTURE: None}), NodeName.END)

    def test_after_bfs_route_dispatches_by_phase(self) -> None:
        provider = self.__provider(DfsState())
        self.assertEqual(
            provider.after_bfs_route({EKey.BFS_PHASE: BFSPhase.SCAN.value}), NodeName.SCAN
        )
        self.assertEqual(
            provider.after_bfs_route({EKey.BFS_PHASE: BFSPhase.BACKTRACK.value}), NodeName.NAVIGATE
        )

    def test_after_bfs_route_advance_ends_when_recovery_empty(self) -> None:
        provider = self.__provider(DfsState(phase=BFSPhase.ADVANCE))
        self.assertEqual(
            provider.after_bfs_route({EKey.BFS_PHASE: BFSPhase.ADVANCE.value}), NodeName.END
        )

    def test_after_bfs_route_ends_when_complete(self) -> None:
        # The no-progress watchdog completes from bfs_route; the router must end.
        provider = self.__provider(DfsState())
        self.assertEqual(
            provider.after_bfs_route({EKey.BFS_PHASE: BFSPhase.SCAN.value, CKey.IS_COMPLETE: True}),
            NodeName.END,
        )

    def test_after_scan_ends_when_complete(self) -> None:
        provider = self.__provider(DfsState())
        self.assertEqual(
            provider.after_scan({EKey.ACTION: _action(), CKey.IS_COMPLETE: True}),
            NodeName.END,
        )

    def test_after_scan(self) -> None:
        provider = self.__provider(DfsState())
        self.assertEqual(
            provider.after_scan({EKey.ACTION: _action(), EKey.CONTENT_EXHAUSTED: False}),
            NodeName.EXECUTE,
        )
        self.assertEqual(provider.after_scan({EKey.CONTENT_EXHAUSTED: True}), NodeName.BFS_ROUTE)
        self.assertEqual(
            provider.after_scan({EKey.ACTION: None, EKey.CONTENT_EXHAUSTED: False}),
            NodeName.BFS_ROUTE,
        )

    def test_after_record(self) -> None:
        provider = self.__provider(DfsState())
        self.assertEqual(provider.after_record({}), NodeName.GROUND)
        self.assertEqual(provider.after_record({CKey.IS_COMPLETE: True}), NodeName.END)

        stuck = self.__provider(DfsState(), can_continue=False)
        self.assertEqual(stuck.after_record({}), NodeName.END)


class TestInterrupts(unittest.IsolatedAsyncioTestCase):
    """Ground honours external cancel and pause signals."""

    @staticmethod
    def __context(hitl: Mock) -> Mock:
        context = Mock(
            is_cancelled=False, workflow_id="wf", exploration_graph=_graph_mock(), hitl=hitl
        )
        context.perception = Mock(
            perceive=AsyncMock(return_value=Mock(model_copy=Mock(return_value=Mock()))),
            build_state=Mock(return_value=_screen_state()),
        )
        context.agent_state = Mock(update_screen=Mock(return_value=True))
        return context

    async def test_cancel_signal_completes_the_run(self) -> None:
        hitl = _quiet_hitl()
        hitl.check_signal = AsyncMock(return_value="CANCELLED")
        context = self.__context(hitl)

        result = await ExplorationNodeProvider(context=context, vision=Mock()).ground({})

        self.assertTrue(result[CKey.IS_COMPLETE])
        context.perception.perceive.assert_not_awaited()

    async def test_pause_waits_for_resume_then_proceeds(self) -> None:
        hitl = _quiet_hitl()
        hitl.is_pause_requested = AsyncMock(return_value=True)
        context = self.__context(hitl)

        result = await ExplorationNodeProvider(context=context, vision=Mock()).ground({})

        hitl.wait_for_resume.assert_awaited_once()
        self.assertTrue(result[CKey.IS_NEW_SCREEN])


class TestPackageScope(unittest.IsolatedAsyncioTestCase):
    """Record keeps the walk inside the target package."""

    async def test_unrecoverable_drift_terminates(self) -> None:
        context = Mock(
            is_cancelled=False,
            workflow_id="wf",
            package_name="com.app",
            configuration=Mock(),
            device=Mock(get_current_package=AsyncMock(return_value="com.other"), back=AsyncMock()),
            agent_state=Mock(record_step=Mock()),
            telemetry=Mock(info=AsyncMock()),
        )
        step = Step(action=_action(), screen_hash="pre", step_number=1)
        state: Dict[str, Any] = {
            CKey.STEP_RESULT: StepResult(
                step=step,
                success=True,
                duration=1,
                screen_changed=True,
                pre_hash="pre",
                post_hash="0",
            )
        }

        with patch("fathom.strategies.graph.exploration.nodes.stability_wait", new=AsyncMock()):
            result = await ExplorationNodeProvider(context=context, vision=Mock()).record(state)

        self.assertTrue(result[CKey.IS_COMPLETE])
        self.assertIn("Left target package", result[CKey.COMPLETION_REASON])
        self.assertEqual(context.device.back.await_count, 3)
        defect_signals = [
            call.kwargs["signal"]
            for call in context.telemetry.info.await_args_list
            if call.args and call.args[0] == DEFECT_DETECTED_EVENT
        ]
        self.assertIn(DefectSignal.LEFT_PACKAGE.value, defect_signals)

    async def test_transient_drift_recovers_and_continues(self) -> None:
        graph = _graph_mock(resolve_hash=Mock(side_effect=lambda value: value))
        context = Mock(
            is_cancelled=False,
            max_steps=100,
            workflow_id="wf",
            package_name="com.app",
            configuration=Mock(),
            device=Mock(
                # Drifted on the first check, back inside the app after one BACK.
                get_current_package=AsyncMock(side_effect=["com.other", "com.app"]),
                back=AsyncMock(),
            ),
            exploration_graph=graph,
            memory=Mock(store_experience=AsyncMock()),
            history=Mock(enqueue_save_step=Mock()),
            agent_state=Mock(record_step=Mock(), step_count=1),
            telemetry=Mock(info=AsyncMock()),
        )
        context.perception = Mock(
            perceive=AsyncMock(return_value=Mock(screenshot_uri=None)),
            build_state=Mock(return_value=_screen_state("post")),
        )
        step = Step(action=_action(), screen_hash="pre", step_number=1)
        state: Dict[str, Any] = {
            EKey.ACTION: _action(),
            CKey.SCREEN_STATE: _screen_state("pre"),
            CKey.STEP_RESULT: StepResult(
                step=step,
                success=True,
                duration=1,
                screen_changed=True,
                pre_hash="pre",
                post_hash="0",
            ),
        }
        dfs = DfsState(phase=BFSPhase.SCAN, current_path=[("root", _action())])

        with patch("fathom.strategies.graph.exploration.nodes.stability_wait", new=AsyncMock()):
            result = await ExplorationNodeProvider(context=context, vision=Mock(), dfs=dfs).record(
                state
            )

        context.device.back.assert_awaited_once()
        self.assertFalse(result[CKey.IS_COMPLETE])
        graph.record_transition.assert_awaited_once()

    async def test_indeterminate_package_read_does_not_abort(self) -> None:
        # A transient unparseable focus (e.g. mCurrentFocus=null mid-launch) makes
        # get_current_package raise DeviceError; the scope check must swallow it,
        # attempt no recovery, and let the run keep exploring.
        graph = _graph_mock(resolve_hash=Mock(side_effect=lambda value: value))
        context = Mock(
            is_cancelled=False,
            max_steps=100,
            workflow_id="wf",
            package_name="com.app",
            configuration=Mock(),
            device=Mock(
                get_current_package=AsyncMock(side_effect=DeviceError("focus null")),
                back=AsyncMock(),
            ),
            exploration_graph=graph,
            memory=Mock(store_experience=AsyncMock()),
            history=Mock(enqueue_save_step=Mock()),
            agent_state=Mock(record_step=Mock(), step_count=1),
            telemetry=Mock(info=AsyncMock()),
        )
        context.perception = Mock(
            perceive=AsyncMock(return_value=Mock(screenshot_uri=None)),
            build_state=Mock(return_value=_screen_state("post")),
        )
        step = Step(action=_action(), screen_hash="pre", step_number=1)
        state: Dict[str, Any] = {
            EKey.ACTION: _action(),
            CKey.SCREEN_STATE: _screen_state("pre"),
            CKey.STEP_RESULT: StepResult(
                step=step,
                success=True,
                duration=1,
                screen_changed=True,
                pre_hash="pre",
                post_hash="0",
            ),
        }
        dfs = DfsState(phase=BFSPhase.SCAN, current_path=[("root", _action())])

        with patch("fathom.strategies.graph.exploration.nodes.stability_wait", new=AsyncMock()):
            result = await ExplorationNodeProvider(context=context, vision=Mock(), dfs=dfs).record(
                state
            )

        # No drift was confirmed, so no recovery BACK and the run continues.
        context.device.back.assert_not_awaited()
        self.assertFalse(result[CKey.IS_COMPLETE])
        graph.record_transition.assert_awaited_once()


class TestInspectScreen(unittest.IsolatedAsyncioTestCase):
    """
    Verifies WebView-aware screen defect inspection and its verification tagging.
    """

    @staticmethod
    def __empty_state_detector() -> Mock:
        defect = Defect.from_signal(
            signal=DefectSignal.EMPTY_STATE,
            source=DefectSource.POST_RUN,
            summary="blank",
            evidence=DefectEvidence(screen="fp"),
        )
        return Mock(inspect_screen=AsyncMock(return_value=[defect]))

    @staticmethod
    def __context(*, capture: Any) -> Mock:
        context = Mock(
            workflow_id="wf",
            agent_state=Mock(step_count=1),
            telemetry=Mock(info=AsyncMock()),
            perception=Mock(
                perceive=AsyncMock(return_value=capture),
                build_state=Mock(return_value=_screen_state()),
            ),
        )
        context.configuration.engine.stability_wait = 0.0
        return context

    async def __recorded_defect(self, *, capture: Any) -> Defect:
        context = self.__context(capture=capture)
        defects = Mock(record=AsyncMock())
        provider = ExplorationNodeProvider(
            context=context,
            vision=Mock(),
            screen_detector=self.__empty_state_detector(),
            defects=defects,
        )
        await provider._ExplorationNodeProvider__inspect_screen(
            fingerprint="fp", screen_state=_screen_state(), capture=capture
        )
        defects.record.assert_awaited_once()
        return defects.record.await_args.kwargs["defect"]

    async def test_webview_empty_state_is_held_for_review(self) -> None:
        """
        A blank reported on a WebView surface is recorded as needs-review.
        """

        webview_xml = (
            '<hierarchy><node class="android.webkit.WebView" '
            'bounds="[0,0][1000,1800]"/></hierarchy>'
        )
        capture = Mock(
            image=b"PNG", width=1000, height=2000, xml_content=webview_xml, screenshot_uri="u"
        )
        defect = await self.__recorded_defect(capture=capture)
        self.assertEqual(defect.verification, DefectVerification.NEEDS_REVIEW)

    async def test_native_empty_state_is_recorded_as_confirmed(self) -> None:
        """
        A blank on a native screen leads the report as confirmed.
        """

        capture = Mock(image=b"PNG", width=1000, height=2000, xml_content=None, screenshot_uri="u")
        defect = await self.__recorded_defect(capture=capture)
        self.assertEqual(defect.verification, DefectVerification.CONFIRMED)


if __name__ == "__main__":
    unittest.main()
