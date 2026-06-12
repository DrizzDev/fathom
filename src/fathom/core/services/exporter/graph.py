"""
Renders an exploration snapshot into serialisable diagram formats.

The formatters are pure: they depend only on the domain snapshot model, never
on the live knowledge graph. New formats are added by implementing
:class:`GraphFormatter` and registering it with :class:`GraphExporter`.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, Mapping, Optional, Tuple

from fathom.constants.exploration import MAX_SCREEN_LABEL_LENGTH, GraphFormat
from fathom.core.exceptions import GraphExportError
from fathom.schemas.exploration import ExplorationSnapshot, ExploredScreen, ScreenTransition

_DOT_HEADER: Tuple[str, ...] = (
    "digraph exploration {",
    "  rankdir=LR;",
    '  bgcolor="#0d1117";',
    '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9,',
    '        fillcolor="#1f2937", fontcolor="#f0f6fc", color="#30363d"];',
    '  edge [fontname="Helvetica", fontsize=7, fontcolor="#8b949e", color="#58a6ff"];',
    "",
)


class GraphLabeler:
    """
    Derives human-readable labels for screens and transitions.
    """

    __CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])")
    __ACTIVITY_SUFFIX = re.compile(r"(?i)activity$")

    def screen(self, *, screen: ExploredScreen) -> str:
        """
        Returns a short label combining the screen's name and visit count.
        """

        return f"{self.__name(screen=screen)} (visits: {screen.visits})"

    def transition(self, *, transition: ScreenTransition) -> str:
        """
        Returns a label combining the transition's action, target, and count.
        """

        label = (
            f"{transition.action}: {transition.target}"
            if transition.target
            else (transition.action)
        )
        if transition.count > 1:
            label += f" (x{transition.count})"
        return label

    def friendly_activity(self, *, activity: str) -> str:
        """
        Converts an Android activity identifier into a readable name.
        """

        short = activity.split("/", 1)[1] if "/" in activity else activity
        short = short.rsplit(".", 1)[-1]
        trimmed = self.__ACTIVITY_SUFFIX.sub("", short).strip()
        spaced = self.__CAMEL_BOUNDARY.sub(" ", trimmed).strip()
        return spaced or "Screen"

    def __name(self, *, screen: ExploredScreen) -> str:
        """
        Picks the most descriptive available name for a screen.
        """

        if screen.description and screen.description.strip():
            return self.__truncate(text=screen.description.strip())
        return self.friendly_activity(activity=screen.activity)

    @staticmethod
    def __truncate(*, text: str) -> str:
        """
        Caps a label at the configured maximum length.
        """

        if len(text) <= MAX_SCREEN_LABEL_LENGTH:
            return text
        return text[: MAX_SCREEN_LABEL_LENGTH - 3].rstrip() + "..."


class GraphFormatter(ABC):
    """
    Renders an exploration snapshot into one serialised diagram format.
    """

    @abstractmethod
    def render(self, *, snapshot: ExplorationSnapshot) -> str:
        """
        Renders the snapshot to its target format.
        """


class JsonGraphFormatter(GraphFormatter):
    """
    Serialises the snapshot to an indented JSON document.
    """

    def __init__(self, *, indent: int = 2) -> None:
        self.__indent = indent

    def render(self, *, snapshot: ExplorationSnapshot) -> str:
        """
        Returns the snapshot as formatted JSON.
        """

        return snapshot.model_dump_json(indent=self.__indent)


class DotGraphFormatter(GraphFormatter):
    """
    Renders the snapshot as a GraphViz DOT document.
    """

    def __init__(self, *, labeler: Optional[GraphLabeler] = None) -> None:
        self.__labeler = labeler or GraphLabeler()

    def render(self, *, snapshot: ExplorationSnapshot) -> str:
        """
        Returns the snapshot as DOT directed-graph source.
        """

        lines = list(_DOT_HEADER)
        for screen in snapshot.screens:
            label = self.__escape(self.__labeler.screen(screen=screen))
            lines.append(f'  "{screen.hash}" [label="{label}"];')
        lines.append("")
        for transition in snapshot.transitions:
            label = self.__escape(self.__labeler.transition(transition=transition))
            lines.append(
                f'  "{transition.source}" -> "{transition.destination}" [label="{label}"];'
            )
        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def __escape(text: str) -> str:
        """
        Escapes double quotes for DOT string literals.
        """

        return text.replace('"', '\\"')


class MermaidGraphFormatter(GraphFormatter):
    """
    Renders the snapshot as a Mermaid flowchart.
    """

    def __init__(self, *, labeler: Optional[GraphLabeler] = None) -> None:
        self.__labeler = labeler or GraphLabeler()

    def render(self, *, snapshot: ExplorationSnapshot) -> str:
        """
        Returns the snapshot as a Mermaid graph definition.
        """

        lines = ["graph LR"]
        node_ids: Dict[str, str] = {}
        for index, screen in enumerate(snapshot.screens):
            node_id = f"N{index}"
            node_ids[screen.hash] = node_id
            label = self.__escape(self.__labeler.screen(screen=screen))
            lines.append(f'  {node_id}["{label}"]')

        for transition in snapshot.transitions:
            source = node_ids.get(transition.source)
            destination = node_ids.get(transition.destination)
            if source is None or destination is None:
                continue
            label = self.__escape(self.__labeler.transition(transition=transition))
            lines.append(f'  {source} -->|"{label}"| {destination}')
        return "\n".join(lines)

    @staticmethod
    def __escape(text: str) -> str:
        """
        Replaces double quotes, which would close a Mermaid label.
        """

        return text.replace('"', "'")


class GraphExporter:
    """
    Renders an exploration snapshot to a requested diagram format.
    """

    def __init__(
        self, *, formatters: Optional[Mapping[GraphFormat, GraphFormatter]] = None
    ) -> None:
        self.__formatters: Dict[GraphFormat, GraphFormatter] = (
            dict(formatters) if formatters is not None else self.__defaults()
        )

    def render(self, *, snapshot: ExplorationSnapshot, graph_format: GraphFormat) -> str:
        """
        Renders the snapshot in the requested format.
        """

        formatter = self.__formatters.get(graph_format)
        if formatter is None:
            raise GraphExportError(f"No formatter registered for graph format {graph_format.value}")
        return formatter.render(snapshot=snapshot)

    @property
    def formats(self) -> Tuple[GraphFormat, ...]:
        """
        The graph formats this exporter can render.
        """

        return tuple(self.__formatters)

    @staticmethod
    def __defaults() -> Dict[GraphFormat, GraphFormatter]:
        """
        Builds the default formatter registry.
        """

        return {
            GraphFormat.JSON: JsonGraphFormatter(),
            GraphFormat.DOT: DotGraphFormatter(),
            GraphFormat.MERMAID: MermaidGraphFormatter(),
        }
