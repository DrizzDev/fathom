from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast
from unittest.mock import MagicMock, patch

from fathom.constants.graph import NodeName
from fathom.constants.runtime import DEFAULT_VERIFICATION_REJECTION_LIMIT
from fathom.constants.state import (
    TERMINAL_COMPLETION_REASONS,
    CommonStateKey,
    CompletionReason,
    IntentStateKey,
)
from fathom.core.agent.state import AgentState
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.screens import ScreenState
from fathom.schemas.state import VerificationLoopState
from fathom.strategies.graph.intent.builder import IntentGraphBuilder
from fathom.strategies.graph.intent.nodes.factory import IntentGraphFactory
from fathom.strategies.graph.state import IntentGraphState

SRC_ROOT = Path(__file__).resolve().parents[5] / "src" / "fathom"
BUILDER_FILE = SRC_ROOT / "strategies" / "graph" / "intent" / "builder.py"


class _RouterReturnExtractor(ast.NodeVisitor):
    """
    Extracts NodeName destinations returned by IntentGraphBuilder routers.
    """

    def __init__(self) -> None:
        self.router_returns: Dict[str, Set[str]] = {}
        self.__current_router = ""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Visit private route-after methods and collect their return values.
        """

        if node.name.startswith("_IntentGraphBuilder__route_after_") or node.name.startswith(
            "__route_after_"
        ):
            short_name = node.name.split("__route_after_")[-1]
            self.__current_router = short_name
            self.router_returns.setdefault(short_name, set())
            self.generic_visit(node)
            self.__current_router = ""
            return

        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        """
        Record returned NodeName members while inside a router.
        """

        if not self.__current_router:
            return
        value = node.value
        if (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == "NodeName"
        ):
            self.router_returns[self.__current_router].add(value.attr)
        self.generic_visit(node)


class _BuilderGraph:
    """
    Parses IntentGraphBuilder source into graph topology facts.
    """

    @staticmethod
    def __tree() -> ast.Module:
        """
        Return the parsed builder module AST.
        """

        return ast.parse(BUILDER_FILE.read_text(encoding="utf-8"))

    @staticmethod
    def registered_nodes() -> Set[str]:
        """
        Return NodeName members registered through add_node.
        """

        tree = _BuilderGraph.__tree()
        names: Set[str] = set()
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if isinstance(func, ast.Attribute) and func.attr == "add_node" and call.args:
                target = call.args[0]
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "NodeName"
                ):
                    names.add(target.attr)
        return names

    @staticmethod
    def static_edges() -> Set[Tuple[str, str]]:
        """
        Return source-destination pairs declared through add_edge.
        """

        tree = _BuilderGraph.__tree()
        edges: Set[Tuple[str, str]] = set()
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if isinstance(func, ast.Attribute) and func.attr == "add_edge" and len(call.args) == 2:
                source, destination = call.args
                if (
                    isinstance(source, ast.Attribute)
                    and isinstance(source.value, ast.Name)
                    and source.value.id == "NodeName"
                    and isinstance(destination, ast.Attribute)
                    and isinstance(destination.value, ast.Name)
                    and destination.value.id == "NodeName"
                ):
                    edges.add((source.attr, destination.attr))
        return edges

    @staticmethod
    def conditional_destinations() -> Dict[str, Set[str]]:
        """
        Return destination maps declared through add_conditional_edges.
        """

        tree = _BuilderGraph.__tree()
        destinations_by_source: Dict[str, Set[str]] = {}
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if (
                not isinstance(func, ast.Attribute)
                or func.attr != "add_conditional_edges"
                or len(call.args) < 3
            ):
                continue
            source = call.args[0]
            destinations = call.args[2]
            if not (
                isinstance(source, ast.Attribute)
                and isinstance(source.value, ast.Name)
                and source.value.id == "NodeName"
                and isinstance(destinations, ast.Dict)
            ):
                continue

            collected: Set[str] = set()
            for value in destinations.values:
                if (
                    isinstance(value, ast.Attribute)
                    and isinstance(value.value, ast.Name)
                    and value.value.id == "NodeName"
                ):
                    collected.add(value.attr)
            destinations_by_source[source.attr] = collected
        return destinations_by_source

    @staticmethod
    def router_returns() -> Dict[str, Set[str]]:
        """
        Return NodeName members each route-after method can return.
        """

        extractor = _RouterReturnExtractor()
        extractor.visit(_BuilderGraph.__tree())
        return extractor.router_returns

    @staticmethod
    def entry_point() -> str:
        """
        Return the NodeName member used as the graph entry point.
        """

        tree = _BuilderGraph.__tree()
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if isinstance(func, ast.Attribute) and func.attr == "set_entry_point" and call.args:
                target = call.args[0]
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "NodeName"
                ):
                    return target.attr
        raise AssertionError("builder.py declares no entry point")


class TestIntentGraphBuilderRoutes(unittest.TestCase):
    """
    Covers runtime router decisions on IntentGraphBuilder.
    """

    def test_cancelled_analyze_result_routes_to_end(self) -> None:
        """
        ANALYZE cancellation is terminal and must not enter VERIFY.
        """

        builder = IntentGraphBuilder(
            context=SimpleNamespace(is_cancelled=False),  # type: ignore[arg-type]
        )

        route = builder._IntentGraphBuilder__route_after_analyze(  # type: ignore[attr-defined]
            {
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.COMPLETION_REASON: CompletionReason.CANCELLED.value,
            }
        )

        self.assertEqual(route, NodeName.END)

    def test_terminal_completion_after_analyze_routes_to_end(self) -> None:
        """
        Every terminal completion reason emitted by ANALYZE must route to END, not VERIFY. The reviewer caught this when RETRY_BUDGET_EXHAUSTED was missing from the router's fatal set.
        """

        builder = IntentGraphBuilder(
            context=SimpleNamespace(is_cancelled=False),  # type: ignore[arg-type]
        )

        for reason in TERMINAL_COMPLETION_REASONS:
            with self.subTest(reason=reason):
                route = builder._IntentGraphBuilder__route_after_analyze(  # type: ignore[attr-defined]
                    {
                        CommonStateKey.IS_COMPLETE: True,
                        CommonStateKey.COMPLETION_REASON: reason,
                    },
                )
                self.assertEqual(route, NodeName.END)


class TestRouteAfterExecute(unittest.TestCase):
    """Pins routing decisions for the new conditional edge after EXECUTE."""

    @staticmethod
    def __builder(*, is_cancelled: bool = False) -> IntentGraphBuilder:
        """Build a routing-only IntentGraphBuilder bound to a minimal context."""

        return IntentGraphBuilder(
            context=SimpleNamespace(is_cancelled=is_cancelled),  # type: ignore[arg-type]
        )

    def test_cancellation_routes_to_end(self) -> None:
        """A cancelled context overrides any other state and routes to END."""

        route = self.__builder(is_cancelled=True)._IntentGraphBuilder__route_after_execute(  # type: ignore[attr-defined]
            {IntentStateKey.SHOULD_RETRY: True},
        )

        self.assertEqual(route, NodeName.END)

    def test_terminal_completion_routes_to_end(self) -> None:
        """A terminal completion reason routes to END."""

        for reason in TERMINAL_COMPLETION_REASONS:
            with self.subTest(reason=reason):
                route = self.__builder()._IntentGraphBuilder__route_after_execute(  # type: ignore[attr-defined]
                    {
                        CommonStateKey.IS_COMPLETE: True,
                        CommonStateKey.COMPLETION_REASON: reason,
                    },
                )
                self.assertEqual(route, NodeName.END)

    def test_non_terminal_completion_routes_to_verify(self) -> None:
        """A non-terminal completion (e.g. success) routes to VERIFY."""

        route = self.__builder()._IntentGraphBuilder__route_after_execute(  # type: ignore[attr-defined]
            {
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.COMPLETION_REASON: CompletionReason.SUCCESS.value,
            },
        )

        self.assertEqual(route, NodeName.VERIFY)

    def test_should_retry_routes_to_ground(self) -> None:
        """SHOULD_RETRY from EXECUTE (HITL unavailable) routes back to GROUND."""

        route = self.__builder()._IntentGraphBuilder__route_after_execute(  # type: ignore[attr-defined]
            {IntentStateKey.SHOULD_RETRY: True},
        )

        self.assertEqual(route, NodeName.GROUND)

    def test_default_path_routes_to_observe(self) -> None:
        """Successful execution with no termination/retry signal routes to OBSERVE."""

        route = self.__builder()._IntentGraphBuilder__route_after_execute(  # type: ignore[attr-defined]
            {},
        )

        self.assertEqual(route, NodeName.OBSERVE)


class TestRouteAfterRecord(unittest.TestCase):
    """
    Pins routing decisions after RECORD.
    """

    @staticmethod
    def __builder(*, is_cancelled: bool = False) -> IntentGraphBuilder:
        """
        Build a routing-only IntentGraphBuilder bound to a minimal context.
        """

        return IntentGraphBuilder(
            context=SimpleNamespace(is_cancelled=is_cancelled),  # type: ignore[arg-type]
        )

    def test_terminal_completion_routes_to_end(self) -> None:
        """
        All shared terminal completion reasons route to END.
        """

        for reason in TERMINAL_COMPLETION_REASONS:
            with self.subTest(reason=reason):
                route = self.__builder()._IntentGraphBuilder__route_after_record(  # type: ignore[attr-defined]
                    {
                        CommonStateKey.IS_COMPLETE: True,
                        CommonStateKey.COMPLETION_REASON: reason,
                    },
                )
                self.assertEqual(route, NodeName.END)

    def test_non_terminal_completion_routes_to_verify(self) -> None:
        """
        Successful completion still routes to VERIFY.
        """

        route = self.__builder()._IntentGraphBuilder__route_after_record(  # type: ignore[attr-defined]
            {
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.COMPLETION_REASON: CompletionReason.SUCCESS.value,
            },
        )

        self.assertEqual(route, NodeName.VERIFY)


class TestIntentGraphBuilderDestinations:
    """
    Verifies router return values match builder conditional-edge maps.
    """

    NAME_TO_ROUTER = {
        "GROUND": "ground",
        "ANALYZE": "analyze",
        "SUPERVISE": "supervise",
        "VERIFY": "verify",
        "RECORD": "record",
    }

    def test_every_router_return_is_an_advertised_destination(self) -> None:
        """
        Router return values must be present in their destination map.
        """

        destinations = _BuilderGraph.conditional_destinations()
        returns = _BuilderGraph.router_returns()
        problems: Dict[str, Set[str]] = {}
        for source_name, advertised in destinations.items():
            router = self.NAME_TO_ROUTER.get(source_name)
            if router is None:
                continue
            stray = returns.get(router, set()) - advertised
            if stray:
                problems[source_name] = stray
        assert problems == {}, (
            f"Routers return NodeName values missing from destination maps: {problems}"
        )

    def test_every_advertised_destination_is_actually_returned(self) -> None:
        """
        Destination-map entries must be reachable from their router.
        """

        destinations = _BuilderGraph.conditional_destinations()
        returns = _BuilderGraph.router_returns()
        problems: Dict[str, Set[str]] = {}
        for source_name, advertised in destinations.items():
            router = self.NAME_TO_ROUTER.get(source_name)
            if router is None:
                continue
            unreachable = advertised - returns.get(router, set())
            if unreachable:
                problems[source_name] = unreachable
        assert problems == {}, (
            f"Conditional-edge maps declare destinations routers never return: {problems}"
        )


class TestIntentGraphBuilderRegistration:
    """
    Verifies IntentGraphBuilder registers and connects known graph nodes.
    """

    def test_every_edge_target_is_registered(self) -> None:
        """
        Static and conditional edge targets must be registered graph nodes.
        """

        registered = _BuilderGraph.registered_nodes() | {"END"}
        static_edge_targets = {destination for _, destination in _BuilderGraph.static_edges()}
        conditional_targets = {
            destination
            for destinations in _BuilderGraph.conditional_destinations().values()
            for destination in destinations
        }
        missing = (static_edge_targets | conditional_targets) - registered
        assert missing == set(), (
            f"NodeName values appear as edge targets but were never registered: {sorted(missing)}"
        )

    def test_entry_point_is_registered(self) -> None:
        """
        The graph entry point must be a registered node.
        """

        entry = _BuilderGraph.entry_point()
        registered = _BuilderGraph.registered_nodes()
        assert entry in registered, f"Entry point {entry!r} is not registered: {sorted(registered)}"

    def test_every_registered_node_is_reachable_from_entry(self) -> None:
        """
        Every registered node must be reachable from the graph entry point.
        """

        registered = _BuilderGraph.registered_nodes()
        adjacency: Dict[str, Set[str]] = {name: set() for name in registered}
        for source, destination in _BuilderGraph.static_edges():
            adjacency.setdefault(source, set()).add(destination)
        for source, destinations in _BuilderGraph.conditional_destinations().items():
            adjacency.setdefault(source, set()).update(destinations)

        entry = _BuilderGraph.entry_point()
        seen: Set[str] = set()
        frontier: List[str] = [entry]
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(adjacency.get(current, set()))

        unreachable = registered - seen
        assert unreachable == set(), (
            f"Registered nodes are unreachable from {entry}: {sorted(unreachable)}"
        )

    def test_every_router_source_is_registered(self) -> None:
        """
        Conditional-edge source nodes must be registered graph nodes.
        """

        destinations = _BuilderGraph.conditional_destinations()
        registered = _BuilderGraph.registered_nodes()
        offenders = set(destinations.keys()) - registered
        assert offenders == set(), (
            "add_conditional_edges names sources never registered with add_node: "
            f"{sorted(offenders)}"
        )


class TestIntentGraphBuilderNodeNames:
    """
    Verifies builder node identifiers use NodeName consistently.
    """

    def test_no_node_identifier_is_a_raw_string_literal(self) -> None:
        """
        Builder graph wiring must use NodeName members instead of raw strings.
        """

        tree = ast.parse(BUILDER_FILE.read_text(encoding="utf-8"))
        offenders: List[Tuple[int, str]] = []
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in {
                "add_node",
                "add_edge",
                "add_conditional_edges",
                "set_entry_point",
            }:
                continue
            for arg in call.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    offenders.append((call.lineno, arg.value))
        assert offenders == [], (
            f"builder.py uses raw-string node identifiers instead of NodeName: {offenders}"
        )

    def test_every_routed_destination_resolves_to_a_known_node_name(self) -> None:
        """
        Builder graph wiring must reference existing NodeName members.
        """

        destinations = _BuilderGraph.conditional_destinations()
        static_edges = _BuilderGraph.static_edges()
        registered = _BuilderGraph.registered_nodes()
        all_names = (
            registered
            | {destination for _, destination in static_edges}
            | {source for source, _ in static_edges}
            | set(destinations.keys())
            | {destination for group in destinations.values() for destination in group}
        )
        valid = {member.name for member in NodeName}
        unknown = all_names - valid
        assert unknown == set(), (
            f"builder.py references NodeName attributes that do not exist: {sorted(unknown)}"
        )


class TestIntentGraphBuilderVerifyLoop:
    """
    Covers compiled builder behavior for repeated VERIFY rejections.
    """

    def test_stuck_completion_routes_to_end_without_verify(self) -> None:
        """
        STUCK is a terminal completion reason and must never enter VERIFY.
        """

        context = MagicMock(name="GraphContext")
        context.is_cancelled = False
        context.agent_state.is_complete = False
        builder = IntentGraphBuilder(context=context)

        route = builder._IntentGraphBuilder__route_after_analyze(  # type: ignore[attr-defined]
            {
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.COMPLETION_REASON: CompletionReason.STUCK.value,
                IntentStateKey.SHOULD_RETRY: False,
                IntentStateKey.PLANNED_STEP: None,
            }
        )

        assert route == NodeName.END

    def test_frozen_verify_loop_terminates_after_same_epoch_rejections(self) -> None:
        """
        The compiled graph must stop when VERIFY rejects the same epoch repeatedly.
        """

        agent_state = AgentState(
            intent="tap sign in",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
            max_steps=10,
        )
        agent_state._AgentState__step_count = 5

        context = MagicMock(name="GraphContext")
        context.agent_state = agent_state
        context.is_cancelled = False
        context.max_steps = 10

        verify_calls = {"count": 0}
        screen = ScreenState(
            activity="save-account",
            activity_hash="a" * 16,
            visual_hash="1" * 16,
            timestamp=1,
        )

        def ground(_state: IntentGraphState) -> Dict[str, object]:
            return {}

        def analyze(_state: IntentGraphState) -> Dict[str, object]:
            return {
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.COMPLETION_REASON: "planner thinks done",
                IntentStateKey.SHOULD_RETRY: False,
            }

        def verify(_state: IntentGraphState) -> Dict[str, object]:
            verify_calls["count"] += 1

            loop_state = agent_state.record_verify_rejection(
                screen=screen,
                activity="save-account",
            )

            if loop_state.consecutive_rejections >= DEFAULT_VERIFICATION_REJECTION_LIMIT:
                agent_state.mark_complete(reason=CompletionReason.STUCK.value)
                return {
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.STUCK.value,
                }

            agent_state.reset_completion()
            return {
                CommonStateKey.IS_COMPLETE: False,
                IntentStateKey.SHOULD_RETRY: True,
            }

        def passthrough(_state: IntentGraphState) -> Dict[str, object]:
            return {}

        nodes: Dict[str, Callable[[IntentGraphState], Dict[str, object]]] = {
            NodeName.GROUND: ground,
            NodeName.ANALYZE: analyze,
            NodeName.SUPERVISE: passthrough,
            NodeName.EXECUTE: passthrough,
            NodeName.OBSERVE: passthrough,
            NodeName.RECORD: passthrough,
            NodeName.VERIFY: verify,
        }

        with patch.object(IntentGraphFactory, "build", return_value=nodes):
            graph = IntentGraphBuilder(context=context).build()
            result = graph.invoke({})

        assert result[CommonStateKey.IS_COMPLETE] is True
        assert result[CommonStateKey.COMPLETION_REASON] == CompletionReason.STUCK.value
        assert verify_calls["count"] == DEFAULT_VERIFICATION_REJECTION_LIMIT
        assert agent_state.verification_loop == VerificationLoopState(
            recorded_step_count=5,
            activity="save-account",
            screen=screen,
            consecutive_rejections=DEFAULT_VERIFICATION_REJECTION_LIMIT,
        )


class TestIntentGraphBuilderRouteCharacterization(unittest.TestCase):
    """
    Characterizes the intent-graph routing decisions across the Phase-1 GROUND fix.

    These pin the observable route each conditional edge returns for a given completion / retry
    / cancellation state. Every router — GROUND included, since the Phase-1 fix — reads the
    returned graph-state dict rather than the mutable ``context.agent_state``. The destinations
    asserted here are the parity contract: they are identical to the pre-fix decision table, only
    the source GROUND consults changed from shared context to durable returned state.
    """

    __TERMINAL = CompletionReason.STUCK.value
    __NON_TERMINAL = CompletionReason.SUCCESS.value
    __STEP = object()

    @staticmethod
    def __route(
        *,
        router: str,
        state: Optional[Dict[Any, Any]] = None,
        is_cancelled: bool = False,
    ) -> str:
        """
        Invoke one conditional-edge router against a synthetic context and graph state.
        """

        builder = IntentGraphBuilder(
            context=SimpleNamespace(is_cancelled=is_cancelled),  # type: ignore[arg-type]
        )
        method = getattr(builder, f"_IntentGraphBuilder__{router}")
        return cast("str", method(state if state is not None else {}))

    @classmethod
    def __state(
        cls,
        *,
        is_complete: bool = False,
        reason: Optional[str] = None,
        should_retry: bool = False,
        has_step: bool = False,
    ) -> Dict[Any, Any]:
        """
        Build a graph-state dict with the completion / retry / planned-step keys the routers read.
        """

        state: Dict[Any, Any] = {
            CommonStateKey.IS_COMPLETE: is_complete,
            IntentStateKey.SHOULD_RETRY: should_retry,
            IntentStateKey.PLANNED_STEP: cls.__STEP if has_step else None,
        }
        if reason is not None:
            state[CommonStateKey.COMPLETION_REASON] = reason
        return state

    def test_ground_incomplete_routes_to_analyze(self) -> None:
        """
        GROUND with an incomplete run advances to ANALYZE.
        """

        self.assertEqual(self.__route(router="route_after_ground"), NodeName.ANALYZE)

    def test_ground_complete_terminal_routes_to_end(self) -> None:
        """
        GROUND completed with a terminal reason ends the run.
        """

        state = self.__state(is_complete=True, reason=self.__TERMINAL)
        self.assertEqual(self.__route(router="route_after_ground", state=state), NodeName.END)

    def test_ground_complete_non_terminal_routes_to_analyze(self) -> None:
        """
        GROUND completed with a non-terminal reason still advances to ANALYZE.
        """

        state = self.__state(is_complete=True, reason=self.__NON_TERMINAL)
        self.assertEqual(self.__route(router="route_after_ground", state=state), NodeName.ANALYZE)

    def test_analyze_cancelled_routes_to_end(self) -> None:
        """
        ANALYZE ends immediately when the run is cancelled.
        """

        self.assertEqual(
            self.__route(router="route_after_analyze", state=self.__state(), is_cancelled=True),
            NodeName.END,
        )

    def test_analyze_complete_terminal_routes_to_end(self) -> None:
        """
        ANALYZE completed with a terminal reason ends the run.
        """

        state = self.__state(is_complete=True, reason=self.__TERMINAL)
        self.assertEqual(self.__route(router="route_after_analyze", state=state), NodeName.END)

    def test_analyze_complete_non_terminal_routes_to_verify(self) -> None:
        """
        ANALYZE completed with a non-terminal reason routes to VERIFY.
        """

        state = self.__state(is_complete=True, reason=self.__NON_TERMINAL)
        self.assertEqual(self.__route(router="route_after_analyze", state=state), NodeName.VERIFY)

    def test_analyze_should_retry_routes_to_ground(self) -> None:
        """
        ANALYZE with a soft retry re-grounds.
        """

        state = self.__state(should_retry=True, has_step=True)
        self.assertEqual(self.__route(router="route_after_analyze", state=state), NodeName.GROUND)

    def test_analyze_missing_step_routes_to_ground(self) -> None:
        """
        ANALYZE with no planned step re-grounds.
        """

        state = self.__state(has_step=False)
        self.assertEqual(self.__route(router="route_after_analyze", state=state), NodeName.GROUND)

    def test_analyze_with_step_routes_to_supervise(self) -> None:
        """
        ANALYZE with a planned step advances to SUPERVISE.
        """

        state = self.__state(has_step=True)
        self.assertEqual(
            self.__route(router="route_after_analyze", state=state), NodeName.SUPERVISE
        )

    def test_supervise_cancelled_routes_to_end(self) -> None:
        """
        SUPERVISE ends when cancelled.
        """

        self.assertEqual(
            self.__route(router="route_after_supervise", state=self.__state(), is_cancelled=True),
            NodeName.END,
        )

    def test_supervise_should_retry_routes_to_ground(self) -> None:
        """
        SUPERVISE re-grounds when it asks for a retry.
        """

        state = self.__state(should_retry=True)
        self.assertEqual(self.__route(router="route_after_supervise", state=state), NodeName.GROUND)

    def test_supervise_default_routes_to_execute(self) -> None:
        """
        SUPERVISE advances to EXECUTE by default.
        """

        self.assertEqual(
            self.__route(router="route_after_supervise", state=self.__state()), NodeName.EXECUTE
        )

    def test_execute_cancelled_routes_to_end(self) -> None:
        """
        EXECUTE ends when cancelled.
        """

        self.assertEqual(
            self.__route(router="route_after_execute", state=self.__state(), is_cancelled=True),
            NodeName.END,
        )

    def test_execute_complete_terminal_routes_to_end(self) -> None:
        """
        EXECUTE completed with a terminal reason ends the run.
        """

        state = self.__state(is_complete=True, reason=self.__TERMINAL)
        self.assertEqual(self.__route(router="route_after_execute", state=state), NodeName.END)

    def test_execute_complete_non_terminal_routes_to_verify(self) -> None:
        """
        EXECUTE completed with a non-terminal reason routes to VERIFY.
        """

        state = self.__state(is_complete=True, reason=self.__NON_TERMINAL)
        self.assertEqual(self.__route(router="route_after_execute", state=state), NodeName.VERIFY)

    def test_execute_should_retry_routes_to_ground(self) -> None:
        """
        EXECUTE re-grounds on a soft retry.
        """

        state = self.__state(should_retry=True)
        self.assertEqual(self.__route(router="route_after_execute", state=state), NodeName.GROUND)

    def test_execute_default_routes_to_observe(self) -> None:
        """
        EXECUTE advances to OBSERVE by default.
        """

        self.assertEqual(
            self.__route(router="route_after_execute", state=self.__state()), NodeName.OBSERVE
        )

    def test_verify_cancelled_routes_to_end(self) -> None:
        """
        VERIFY ends when cancelled.
        """

        self.assertEqual(
            self.__route(router="route_after_verify", state=self.__state(), is_cancelled=True),
            NodeName.END,
        )

    def test_verify_complete_routes_to_end(self) -> None:
        """
        VERIFY ends when the run is complete, terminal or not.
        """

        state = self.__state(is_complete=True, reason=self.__NON_TERMINAL)
        self.assertEqual(self.__route(router="route_after_verify", state=state), NodeName.END)

    def test_verify_incomplete_routes_to_ground(self) -> None:
        """
        VERIFY re-grounds when verification has not completed the run.
        """

        self.assertEqual(
            self.__route(router="route_after_verify", state=self.__state()), NodeName.GROUND
        )

    def test_record_complete_terminal_routes_to_end(self) -> None:
        """
        RECORD completed with a terminal reason ends the run.
        """

        state = self.__state(is_complete=True, reason=self.__TERMINAL)
        self.assertEqual(self.__route(router="route_after_record", state=state), NodeName.END)

    def test_record_complete_non_terminal_routes_to_verify(self) -> None:
        """
        RECORD completed with a non-terminal reason routes to VERIFY.
        """

        state = self.__state(is_complete=True, reason=self.__NON_TERMINAL)
        self.assertEqual(self.__route(router="route_after_record", state=state), NodeName.VERIFY)

    def test_record_incomplete_routes_to_ground(self) -> None:
        """
        RECORD re-grounds when the run is not complete.
        """

        self.assertEqual(
            self.__route(router="route_after_record", state=self.__state()), NodeName.GROUND
        )
