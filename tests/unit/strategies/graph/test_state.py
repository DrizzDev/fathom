"""
Contract tests for :mod:`fathom.strategies.graph.state`.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Set

import pytest

if TYPE_CHECKING:
    from types import ModuleType

SRC_ROOT = Path(__file__).resolve().parents[4] / "src" / "fathom"
NODES_DIR = SRC_ROOT / "strategies" / "graph" / "intent" / "nodes"


def __load_module_directly(*, dotted_name: str, file_path: Path) -> ModuleType:
    """
    Load one module by file path without importing optional adapter chains.
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
__state_constants = __load_module_directly(
    dotted_name="fathom.constants.state",
    file_path=SRC_ROOT / "constants" / "state.py",
)

IntentGraphState = __state_module.IntentGraphState
IntentStateKey = __state_constants.IntentStateKey
CommonStateKey = __state_constants.CommonStateKey


class _NodeFiles:
    """
    Provides parsed ASTs for intent node source files.
    """

    @staticmethod
    def paths() -> List[Path]:
        """
        Return node source files that can write graph state patches.
        """

        exclude = {"__init__.py", "factory.py", "provider.py", "effect.py", "observer.py"}
        return sorted(path for path in NODES_DIR.glob("*.py") if path.name not in exclude)

    @staticmethod
    def trees() -> Dict[Path, ast.Module]:
        """
        Parse node source files once for contract inspection.
        """

        return {path: ast.parse(path.read_text(encoding="utf-8")) for path in _NodeFiles.paths()}


class _KeyExtractor(ast.NodeVisitor):
    """
    Collect state enum references inside parsed node source.
    """

    ENUM_NAMES = {"IntentStateKey", "CommonStateKey"}

    def __init__(self) -> None:
        self.keys: Set[str] = set()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """
        Record enum-member attributes used by node code.
        """

        if isinstance(node.value, ast.Name) and node.value.id in self.ENUM_NAMES:
            self.keys.add(node.attr)
        self.generic_visit(node)


class _ReturnedKeyExtractor(ast.NodeVisitor):
    """
    Collect graph-state keys written by node return dictionaries.
    """

    ENUM_NAMES = {"IntentStateKey", "CommonStateKey"}

    def __init__(self) -> None:
        self.returned_keys: Set[str] = set()
        self.string_returned_keys: Set[str] = set()
        self.__inside_return = 0

    def visit_Return(self, node: ast.Return) -> None:
        """
        Visit return expressions while marking return scope.
        """

        self.__inside_return += 1
        try:
            self.generic_visit(node)
        finally:
            self.__inside_return -= 1

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """
        Record enum-state keys returned by graph nodes.
        """

        if (
            self.__inside_return
            and isinstance(node.value, ast.Name)
            and node.value.id in self.ENUM_NAMES
        ):
            self.returned_keys.add(node.attr)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        """
        Record raw string keys returned by graph nodes.
        """

        if self.__inside_return:
            for key in node.keys:
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and (key.value.isupper() and "_" in key.value or key.value.isupper())
                ):
                    self.string_returned_keys.add(key.value)
        self.generic_visit(node)


class TestStateSchemaCompleteness:
    """
    Verifies every referenced graph-state enum key exists on IntentGraphState.
    """

    @staticmethod
    def __referenced_keys() -> Set[str]:
        """
        Return every state enum member referenced by intent nodes.
        """

        extractor = _KeyExtractor()
        for tree in _NodeFiles.trees().values():
            extractor.visit(tree)
        return extractor.keys

    @staticmethod
    def __declared_keys() -> Set[str]:
        """
        Return every channel declared on IntentGraphState.
        """

        return set(IntentGraphState.__annotations__.keys())

    def test_every_referenced_enum_key_is_declared_on_typed_dict(self) -> None:
        """
        Referenced state enum keys must be declared as LangGraph channels.
        """

        referenced = self.__referenced_keys()
        declared = self.__declared_keys()
        missing = referenced - declared
        assert missing == set(), (
            "These enum keys are referenced by intent-graph nodes but missing from "
            f"IntentGraphState.__annotations__: {sorted(missing)}."
        )

    def test_every_typed_dict_key_corresponds_to_an_enum_member(self) -> None:
        """
        Declared graph-state keys must correspond to a known state enum member.
        """

        declared = self.__declared_keys()
        enum_values = {member.value for member in IntentStateKey} | {
            member.value for member in CommonStateKey
        }
        orphans = declared - enum_values
        assert orphans == set(), (
            f"These IntentGraphState keys do not match any enum member: {sorted(orphans)}"
        )

    def test_no_node_returns_raw_string_state_keys(self) -> None:
        """
        Node return dictionaries must use state enums instead of raw strings.
        """

        offenders: Dict[str, Set[str]] = {}
        for path, tree in _NodeFiles.trees().items():
            extractor = _ReturnedKeyExtractor()
            extractor.visit(tree)
            if extractor.string_returned_keys:
                offenders[path.name] = extractor.string_returned_keys
        assert offenders == {}, (
            f"These nodes return raw string state keys instead of enums: {offenders}"
        )


class TestNodeReturnContract:
    """
    Verifies graph nodes only write channels declared on IntentGraphState.
    """

    def test_every_returned_state_key_is_declared(self) -> None:
        """
        Returned state enum keys must be declared as LangGraph channels.
        """

        declared = set(IntentGraphState.__annotations__.keys())
        offenders: Dict[str, Set[str]] = {}
        for path, tree in _NodeFiles.trees().items():
            extractor = _ReturnedKeyExtractor()
            extractor.visit(tree)
            stray = extractor.returned_keys - declared
            if stray:
                offenders[path.name] = stray
        assert offenders == {}, (
            f"These nodes write state keys not declared on IntentGraphState: {offenders}"
        )


class TestStateChannelRoundtrip:
    """
    Verifies declared IntentGraphState channels survive LangGraph merging.
    """

    @staticmethod
    def __compile_two_node_graph(*, payload: Dict[str, object]) -> Dict[str, object]:
        """
        Build a writer-reader graph and return the reader-side state.
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
        Each declared graph-state key must survive a writer-to-reader update.
        """

        sentinel = object()
        captured = self.__compile_two_node_graph(payload={key: sentinel})
        assert key in captured, f"Declared TypedDict key {key!r} was dropped."
        assert captured[key] is sentinel, (
            f"Channel mutated the sentinel for {key!r}: got {captured[key]!r}"
        )
