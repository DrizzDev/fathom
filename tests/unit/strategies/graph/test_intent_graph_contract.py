"""
Structural contract tests for the Intent LangGraph subsystem.

Per-node unit tests call each node directly with hand-built state dicts,
which bypasses the LangGraph channel layer entirely. They cannot catch
the bug class that lives at the graph seam:

  - state keys written by node A but not declared on the TypedDict
    (LangGraph silently drops them; node B reads ``None``);
  - routers returning destination names that aren't registered in the
    conditional-edges dict (LangGraph raises at compile or routes to a
    ghost edge);
  - nodes added to the workflow but unreachable from the entry point.

This module covers that gap with a mix of static AST audits over the
intent-graph source files and one live compiled-graph channel test.
The intent is for these tests to surface every structural defect in
the graph wiring without depending on real adapters.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

import pytest

if TYPE_CHECKING:
    from types import ModuleType

SRC_ROOT = Path(__file__).resolve().parents[4] / "src" / "fathom"
NODES_DIR = SRC_ROOT / "strategies" / "graph" / "intent" / "nodes"
BUILDER_FILE = SRC_ROOT / "strategies" / "graph" / "intent" / "builder.py"


def __load_module_directly(*, dotted_name: str, file_path: Path) -> ModuleType:
    """
    Load one module by file path, bypassing parent ``__init__.py`` chains.

    The Fathom package's eager ``strategies/__init__.py`` re-exports adapters
    that pull in optional cloud SDKs. Loading the lean graph modules through
    that chain forces an irrelevant adapter import that may fail in stripped
    test environments. We load these leaf modules directly so the contract
    suite does not depend on every optional adapter being installed.
    """

    if dotted_name in sys.modules:
        return sys.modules[dotted_name]
    spec = importlib.util.spec_from_file_location(dotted_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {dotted_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = module
    spec.loader.exec_module(module)
    return module


__state_module = __load_module_directly(
    dotted_name="fathom.strategies.graph.state",
    file_path=SRC_ROOT / "strategies" / "graph" / "state.py",
)
__graph_constants = __load_module_directly(
    dotted_name="fathom.constants.graph",
    file_path=SRC_ROOT / "constants" / "graph.py",
)
__state_constants = __load_module_directly(
    dotted_name="fathom.constants.state",
    file_path=SRC_ROOT / "constants" / "state.py",
)

IntentGraphState = __state_module.IntentGraphState
NodeName = __graph_constants.NodeName
IntentStateKey = __state_constants.IntentStateKey
CommonStateKey = __state_constants.CommonStateKey


class _NodeFiles:
    """
    Lazy accessor for the parsed AST of every intent-graph node file.

    Tests share parsed trees so the AST is built once per session even
    though each test asks for its own slice.
    """

    @staticmethod
    def paths() -> List[Path]:
        """
        Return every node source file, excluding factory/provider
        plumbing that does not return state patches.
        """

        exclude = {"__init__.py", "factory.py", "provider.py", "effect.py", "observer.py"}
        return sorted(path for path in NODES_DIR.glob("*.py") if path.name not in exclude)

    @staticmethod
    def trees() -> Dict[Path, ast.Module]:
        """
        Parse every node file once and cache.
        """

        return {path: ast.parse(path.read_text(encoding="utf-8")) for path in _NodeFiles.paths()}


class _KeyExtractor(ast.NodeVisitor):
    """
    Collect every ``IntentStateKey.X`` / ``CommonStateKey.X`` attribute
    reference inside a parsed AST.

    Catches both reads (``state.get(IntentStateKey.X)``) and writes
    (``{IntentStateKey.X: value}``) — every textual reference to the
    enum member.
    """

    ENUM_NAMES = {"IntentStateKey", "CommonStateKey"}

    def __init__(self) -> None:
        self.keys: Set[str] = set()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in self.ENUM_NAMES:
            self.keys.add(node.attr)
        self.generic_visit(node)


class _ReturnedKeyExtractor(ast.NodeVisitor):
    """
    Collect every state-key reference that appears *inside a return*.

    Used to assert that every dict key a node writes through the
    LangGraph channel is declared on :class:`IntentGraphState`.
    """

    ENUM_NAMES = {"IntentStateKey", "CommonStateKey"}

    def __init__(self) -> None:
        self.returned_keys: Set[str] = set()
        self.string_returned_keys: Set[str] = set()
        self.__inside_return = 0

    def visit_Return(self, node: ast.Return) -> None:
        self.__inside_return += 1
        try:
            self.generic_visit(node)
        finally:
            self.__inside_return -= 1

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            self.__inside_return
            and isinstance(node.value, ast.Name)
            and node.value.id in self.ENUM_NAMES
        ):
            self.returned_keys.add(node.attr)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        if self.__inside_return:
            for key in node.keys:
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and (key.value.isupper() and "_" in key.value or key.value.isupper())
                ):
                    self.string_returned_keys.add(key.value)
        self.generic_visit(node)


class _RouterReturnExtractor(ast.NodeVisitor):
    """
    For each router function on :class:`IntentGraphBuilder`, collect
    the set of :class:`NodeName` destinations it can return.

    The set must be a subset of the destination map passed to
    :meth:`workflow.add_conditional_edges` for that source node.
    """

    def __init__(self) -> None:
        self.router_returns: Dict[str, Set[str]] = {}
        self.__current_router: str = ""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name.startswith("_IntentGraphBuilder__route_after_") or node.name.startswith(
            "__route_after_"
        ):
            short = node.name.split("__route_after_")[-1]
            self.__current_router = short
            self.router_returns.setdefault(short, set())
            self.generic_visit(node)
            self.__current_router = ""
        else:
            self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
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
    Parse :mod:`fathom.strategies.graph.intent.builder` into the data
    every topology test needs: registered nodes, edges, conditional-
    edge destination maps, and router routing tables.
    """

    @staticmethod
    def __tree() -> ast.Module:
        return ast.parse(BUILDER_FILE.read_text(encoding="utf-8"))

    @staticmethod
    def registered_nodes() -> Set[str]:
        """
        Names passed to ``workflow.add_node(NAME, ...)``.
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
        ``(src, dst)`` pairs from ``workflow.add_edge`` calls.
        """

        tree = _BuilderGraph.__tree()
        edges: Set[Tuple[str, str]] = set()
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if isinstance(func, ast.Attribute) and func.attr == "add_edge" and len(call.args) == 2:
                source, dest = call.args
                if (
                    isinstance(source, ast.Attribute)
                    and isinstance(source.value, ast.Name)
                    and source.value.id == "NodeName"
                    and isinstance(dest, ast.Attribute)
                    and isinstance(dest.value, ast.Name)
                    and dest.value.id == "NodeName"
                ):
                    edges.add((source.attr, dest.attr))
        return edges

    @staticmethod
    def conditional_destinations() -> Dict[str, Set[str]]:
        """
        Map ``source_node_name -> {destination NodeName.X values}`` for
        every ``add_conditional_edges`` call.
        """

        tree = _BuilderGraph.__tree()
        out: Dict[str, Set[str]] = {}
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
            out[source.attr] = collected
        return out

    @staticmethod
    def router_returns() -> Dict[str, Set[str]]:
        """
        Map ``router_short_name -> {NodeName.X values it can return}``.
        """

        tree = _BuilderGraph.__tree()
        extractor = _RouterReturnExtractor()
        extractor.visit(tree)
        return extractor.router_returns

    @staticmethod
    def entry_point() -> str:
        """
        The :meth:`workflow.set_entry_point` target NodeName.
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


class TestStateSchemaCompleteness:
    """
    Every enum-typed state key referenced anywhere in an intent node
    must be declared on :class:`IntentGraphState`, or LangGraph will
    silently drop the channel update.
    """

    @staticmethod
    def __referenced_keys() -> Set[str]:
        extractor = _KeyExtractor()
        for tree in _NodeFiles.trees().values():
            extractor.visit(tree)
        return extractor.keys

    @staticmethod
    def __declared_keys() -> Set[str]:
        return set(IntentGraphState.__annotations__.keys())

    def test_every_referenced_enum_key_is_declared_on_typed_dict(self) -> None:
        """
        AST-extract every IntentStateKey.X / CommonStateKey.X reference
        from every node file. Each must appear in the TypedDict, or
        the writer→reader channel will silently drop the value.
        """

        referenced = self.__referenced_keys()
        declared = self.__declared_keys()
        missing = referenced - declared
        assert missing == set(), (
            "These enum keys are referenced by intent-graph nodes but missing from "
            f"IntentGraphState.__annotations__: {sorted(missing)}. "
            "Add them to src/fathom/strategies/graph/state.py — otherwise LangGraph "
            "drops them silently at the channel boundary."
        )

    def test_every_typed_dict_key_corresponds_to_an_enum_member(self) -> None:
        """
        Inverse check: a TypedDict key with no enum counterpart is
        almost always a typo or a leftover after an enum rename.
        """

        declared = self.__declared_keys()
        enum_values = {member.value for member in IntentStateKey} | {
            member.value for member in CommonStateKey
        }
        orphans = declared - enum_values
        assert orphans == set(), (
            "These IntentGraphState keys do not match any enum member — likely a "
            f"typo or stale field: {sorted(orphans)}"
        )

    def test_no_node_returns_raw_string_state_keys(self) -> None:
        """
        Every node return must route through the enum. A raw string
        literal in a return-dict is unreviewable and silently drops if
        the spelling drifts from the TypedDict.
        """

        offenders: Dict[str, Set[str]] = {}
        for path, tree in _NodeFiles.trees().items():
            extractor = _ReturnedKeyExtractor()
            extractor.visit(tree)
            stray = extractor.string_returned_keys
            if stray:
                offenders[path.name] = stray
        assert offenders == {}, (
            f"These nodes return raw string state keys instead of going through the "
            f"IntentStateKey/CommonStateKey enums: {offenders}"
        )


class TestNodeReturnContract:
    """
    Every key a node writes through ``return`` must be declared on
    :class:`IntentGraphState`. This is the specific check that would
    have caught the EXECUTION_CONTEXT / EXECUTION_BLOCKED / ACTION_OUTCOME
    silent-drop bugs.
    """

    def test_every_returned_state_key_is_declared(self) -> None:
        declared = set(IntentGraphState.__annotations__.keys())
        offenders: Dict[str, Set[str]] = {}
        for path, tree in _NodeFiles.trees().items():
            extractor = _ReturnedKeyExtractor()
            extractor.visit(tree)
            stray = extractor.returned_keys - declared
            if stray:
                offenders[path.name] = stray
        assert offenders == {}, (
            f"These nodes write state keys that are not declared on IntentGraphState "
            f"(silent channel drop): {offenders}"
        )


class TestRoutingDestinations:
    """
    Each router function may only return :class:`NodeName` values that
    are in the destination map of its ``add_conditional_edges`` call.
    """

    NAME_TO_ROUTER = {
        "GROUND": "ground",
        "ANALYZE": "analyze",
        "SUPERVISE": "supervise",
        "VERIFY": "verify",
        "RECORD": "record",
    }

    def test_every_router_return_is_an_advertised_destination(self) -> None:
        destinations = _BuilderGraph.conditional_destinations()
        returns = _BuilderGraph.router_returns()
        problems: Dict[str, Set[str]] = {}
        for source_name, advertised in destinations.items():
            router = self.NAME_TO_ROUTER.get(source_name)
            if router is None:
                continue
            actual = returns.get(router, set())
            stray = actual - advertised
            if stray:
                problems[source_name] = stray
        assert problems == {}, (
            f"These routers return NodeName values that are not in the conditional-"
            f"edges destination map for that source: {problems}"
        )

    def test_every_advertised_destination_is_actually_returned(self) -> None:
        """
        An advertised destination that the router never returns is
        unreachable wiring — a code-smell that something was removed
        from the router without cleaning up the map.
        """

        destinations = _BuilderGraph.conditional_destinations()
        returns = _BuilderGraph.router_returns()
        problems: Dict[str, Set[str]] = {}
        for source_name, advertised in destinations.items():
            router = self.NAME_TO_ROUTER.get(source_name)
            if router is None:
                continue
            actual = returns.get(router, set())
            unreachable = advertised - actual
            if unreachable:
                problems[source_name] = unreachable
        assert problems == {}, (
            f"These conditional-edges declare destinations the router never returns: {problems}"
        )


class TestNodeRegistration:
    """
    Every NodeName referenced as an edge target must be registered via
    ``add_node`` (except :attr:`NodeName.END`, which LangGraph treats
    as the implicit terminal).
    """

    def test_every_edge_target_is_registered(self) -> None:
        registered = _BuilderGraph.registered_nodes() | {"END"}
        static_edge_targets = {dst for _, dst in _BuilderGraph.static_edges()}
        conditional_targets = {
            dst
            for destinations in _BuilderGraph.conditional_destinations().values()
            for dst in destinations
        }
        missing = (static_edge_targets | conditional_targets) - registered
        assert missing == set(), (
            "These NodeName values appear as edge targets but were never added to the "
            f"workflow via add_node: {sorted(missing)}"
        )

    def test_entry_point_is_registered(self) -> None:
        entry = _BuilderGraph.entry_point()
        registered = _BuilderGraph.registered_nodes()
        assert entry in registered, (
            f"Entry point {entry!r} is not in the registered-node set {sorted(registered)}"
        )

    def test_every_registered_node_is_reachable_from_entry(self) -> None:
        """
        Walk forward from the entry point through static + conditional
        edges. Every registered node must be visited.
        """

        registered = _BuilderGraph.registered_nodes()
        adjacency: Dict[str, Set[str]] = {name: set() for name in registered}
        for src, dst in _BuilderGraph.static_edges():
            adjacency.setdefault(src, set()).add(dst)
        for src, destinations in _BuilderGraph.conditional_destinations().items():
            adjacency.setdefault(src, set()).update(destinations)

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
            f"These nodes are registered with add_node but not reachable from the entry "
            f"point {entry}: {sorted(unreachable)}"
        )

    def test_every_router_short_name_matches_a_registered_source(self) -> None:
        """
        Sanity: every router we identified has a corresponding
        conditional-edges call for an actually-registered source node.
        """

        destinations = _BuilderGraph.conditional_destinations()
        registered = _BuilderGraph.registered_nodes()
        offenders = set(destinations.keys()) - registered
        assert offenders == set(), (
            f"add_conditional_edges names sources that were never added with add_node: "
            f"{sorted(offenders)}"
        )


class TestStateChannelRoundtrip:
    """
    Live compiled-graph test that proves the writer→channel→reader path
    actually preserves every declared key. This is the dynamic counterpart
    to :class:`TestStateSchemaCompleteness` — it would catch a regression
    where the TypedDict declares a key but LangGraph's channel pipeline
    drops it for some reason (custom reducer, frozen dataclass, etc.).
    """

    @staticmethod
    def __compile_two_node_graph(payload: Dict[str, object]) -> object:
        """
        Build a minimal ``writer -> reader`` graph that writes ``payload``
        on the first step and captures the merged state on the second.
        """

        from langgraph.graph import StateGraph

        captured: Dict[str, object] = {}

        def writer(_state: IntentGraphState) -> Dict[str, object]:
            return dict(payload)

        def reader(state: IntentGraphState) -> Dict[str, object]:
            captured.update(state)
            return {}

        workflow = StateGraph(IntentGraphState)
        workflow.add_node("writer", writer)
        workflow.add_node("reader", reader)
        workflow.set_entry_point("writer")
        workflow.add_edge("writer", "reader")
        workflow.set_finish_point("reader")
        workflow.compile().invoke({})
        return captured

    @pytest.mark.parametrize("key", sorted(IntentGraphState.__annotations__.keys()))
    def test_declared_key_survives_writer_to_reader_channel(self, key: str) -> None:
        """
        Round-trip a sentinel value through the LangGraph channel for
        every declared key. The captured reader-side state must contain
        the sentinel — otherwise the channel is silently dropping the
        declared key.
        """

        sentinel = object()
        captured = self.__compile_two_node_graph(payload={key: sentinel})
        assert key in captured, (
            f"Declared TypedDict key {key!r} did not survive a writer->reader channel "
            f"update; LangGraph dropped the value. Inspect IntentGraphState's reducer "
            f"or value type."
        )
        assert captured[key] is sentinel, (
            f"Channel mutated the sentinel for {key!r}: got {captured[key]!r}"
        )


class TestEnumNodeNameAlignment:
    """
    The :class:`NodeName` enum is the single source of truth for node
    identifiers. Routers, builders, and tests must use it consistently.
    """

    def test_no_node_identifier_is_a_raw_string_literal(self) -> None:
        """
        ``add_node("ground", ...)`` would compile but silently bypass
        the enum, defeating refactor safety. Catch raw-string sources
        in builder.py.
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
            f"builder.py uses raw-string node identifiers in place of NodeName members: {offenders}"
        )

    def test_every_routed_destination_resolves_to_a_known_node_name(self) -> None:
        """
        Each NodeName attribute referenced by a router or edge call must
        actually exist on :class:`NodeName`. Catches stale references
        after an enum rename.
        """

        destinations = _BuilderGraph.conditional_destinations()
        static_edges = _BuilderGraph.static_edges()
        registered = _BuilderGraph.registered_nodes()
        all_names = (
            registered
            | {dst for _, dst in static_edges}
            | {src for src, _ in static_edges}
            | set(destinations.keys())
            | {dst for destinations_set in destinations.values() for dst in destinations_set}
        )
        valid = {member.name for member in NodeName}
        unknown = all_names - valid
        assert unknown == set(), (
            f"builder.py references NodeName attributes that do not exist on the enum: "
            f"{sorted(unknown)}"
        )
