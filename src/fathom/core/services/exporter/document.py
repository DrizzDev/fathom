"""
Builds and renders one image-free Markdown document per logical screen.

Screens are grouped by structural identity: the crawler captures a screen many
times (a form before and after typing, a list with different items), each a
distinct graph node, but they share a text-free structure hash. Nodes that share
an activity and structure hash collapse into one document, so a screen is
documented once with its content variations counted as fingerprints, while a
structurally different screen (an OTP step, a different section) stays separate.
The builder is pure: it reads the knowledge graph and the run's defects and
returns typed documents, leaving rendering and file I/O to the renderer.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from fathom.constants.document import Relation, SectionHeading
from fathom.constants.exploration import MAX_SCREEN_LABEL_LENGTH
from fathom.constants.screen import ZERO_HASH, ScreenCategory
from fathom.core.services.exporter.element import ElementText
from fathom.core.services.exporter.graph import GraphLabeler
from fathom.infrastructure.memory.knowledge_graph import GraphNode, KnowledgeGraph
from fathom.schemas.content import ScreenContent
from fathom.schemas.defect import Defect
from fathom.schemas.document import (
    DocumentIndex,
    LinkSemantics,
    ScreenDocument,
    ScreenFlow,
    ScreenLink,
)
from fathom.schemas.report import ReportMetadata

_LinkKey = Tuple[str, Optional[str], str]
_SECTION_HEADING = re.compile(r"^##\s+(?P<heading>.+?)\s*$")


class ScreenDocumentExporter:
    """
    Groups screen nodes by structural identity and builds one document each.
    """

    __NON_SLUG = re.compile(r"[^a-z0-9]+")

    def __init__(self, *, labeler: Optional[GraphLabeler] = None) -> None:
        self.__labeler = labeler or GraphLabeler()

    def build(
        self, *, graph: KnowledgeGraph, defects: List[Defect], metadata: ReportMetadata
    ) -> DocumentIndex:
        """
        Builds one document per logical screen and the index over them.
        """

        groups = self.__group(graph=graph)
        ordered = self.__order(groups=groups)
        representatives = {key: self.__representative(nodes=nodes) for key, nodes in groups.items()}
        titles = {key: self.__title(node=representatives[key]) for key in groups}
        slugs = self.__slugs(ordered=ordered, titles=titles)
        membership = self.__membership(groups=groups)
        inbound, outbound = self.__flows(graph=graph, membership=membership, titles=titles)
        defects_by_screen = self.__defects_by_screen(defects=defects)

        documents = [
            self.__document(
                nodes=groups[key],
                representative=representatives[key],
                slug=slugs[key],
                title=titles[key],
                inbound=inbound.get(key, {}),
                outbound=outbound.get(key, {}),
                defects_by_screen=defects_by_screen,
            )
            for key in ordered
        ]
        return DocumentIndex(metadata=metadata, documents=documents)

    @classmethod
    def __group(cls, *, graph: KnowledgeGraph) -> Dict[str, List[GraphNode]]:
        """
        Buckets documentable screens by their structural-identity key.
        """

        groups: Dict[str, List[GraphNode]] = {}
        for node in graph.nodes.values():
            if not cls.__is_documentable(node=node):
                continue
            groups.setdefault(cls.__group_key(node=node), []).append(node)
        return groups

    @staticmethod
    def __group_key(*, node: GraphNode) -> str:
        """
        Keys a screen by activity and structure hash, falling back to its own hash.

        A node with no usable structure hash (no interactive layout) forms its own
        group so unrelated contentless screens never merge together.
        """

        if node.structure_hash and node.structure_hash != ZERO_HASH:
            return f"{KnowledgeGraph.normalize_activity(node.activity)}|{node.structure_hash}"
        return node.visual_hash

    @staticmethod
    def __is_documentable(*, node: GraphNode) -> bool:
        """
        Whether a screen carries content worth a document.

        Transient or failed captures (a loading frame, an analysis fallback) reach
        the graph with no meaningful description and no narrative; documenting them
        only produces near-duplicate noise, so they are skipped.
        """

        if KnowledgeGraph.has_meaningful_description(node.description):
            return True
        return bool(node.rich_description and node.rich_description.strip())

    @staticmethod
    def __order(*, groups: Dict[str, List[GraphNode]]) -> List[str]:
        """
        Orders logical screens by total visits, breaking ties on the key.
        """

        def rank(key: str) -> Tuple[int, str]:
            visits = sum(node.visit_count for node in groups[key])
            return (-visits, key)

        return sorted(groups, key=rank)

    @staticmethod
    def __representative(*, nodes: List[GraphNode]) -> GraphNode:
        """
        Picks the node that best describes the logical screen.
        """

        def rank(node: GraphNode) -> Tuple[int, int, int, str]:
            described = node.description and KnowledgeGraph.has_meaningful_description(
                node.description
            )
            description_length = len(node.description) if described and node.description else 0
            return (
                description_length,
                len(node.rich_description or ""),
                node.visit_count,
                node.visual_hash,
            )

        return max(nodes, key=rank)

    def __title(self, *, node: GraphNode) -> str:
        """
        Titles a screen from its description, else from activity and category.
        """

        if node.description and KnowledgeGraph.has_meaningful_description(node.description):
            return self.__truncate(text=node.description.strip())
        base = self.__labeler.friendly_activity(activity=node.activity)
        if node.category is ScreenCategory.OTHER:
            return base
        return f"{base} ({node.category.value.title()})"

    @staticmethod
    def __truncate(*, text: str) -> str:
        """
        Caps a title at the shared maximum screen-label length.
        """

        if len(text) <= MAX_SCREEN_LABEL_LENGTH:
            return text
        return text[: MAX_SCREEN_LABEL_LENGTH - 3].rstrip() + "..."

    def __slugs(self, *, ordered: List[str], titles: Dict[str, str]) -> Dict[str, str]:
        """
        Assigns each screen a filename-safe slug, de-colliding by numeric suffix.
        """

        slugs: Dict[str, str] = {}
        seen: Dict[str, int] = {}
        for key in ordered:
            base = self.__slugify(text=titles[key])
            count = seen.get(base, 0)
            seen[base] = count + 1
            slugs[key] = base if count == 0 else f"{base}-{count + 1}"
        return slugs

    @classmethod
    def __slugify(cls, *, text: str) -> str:
        """
        Reduces a title to lowercase alphanumeric words joined by hyphens.
        """

        slug = cls.__NON_SLUG.sub("-", text.lower()).strip("-")
        return slug or "screen"

    @staticmethod
    def __membership(*, groups: Dict[str, List[GraphNode]]) -> Dict[str, str]:
        """
        Maps each canonical screen hash to the logical screen it belongs to.
        """

        membership: Dict[str, str] = {}
        for key, nodes in groups.items():
            for node in nodes:
                membership[node.visual_hash] = key
        return membership

    def __flows(
        self, *, graph: KnowledgeGraph, membership: Dict[str, str], titles: Dict[str, str]
    ) -> Tuple[Dict[str, Dict[_LinkKey, ScreenLink]], Dict[str, Dict[_LinkKey, ScreenLink]]]:
        """
        Resolves cross-screen transitions into deduped inbound and outbound links.
        """

        inbound: Dict[str, Dict[_LinkKey, ScreenLink]] = {}
        outbound: Dict[str, Dict[_LinkKey, ScreenLink]] = {}

        for source_hash, edges in graph.edges.items():
            source_key = membership.get(source_hash)
            if source_key is None:
                continue
            for edge in edges:
                destination_key = membership.get(edge.destination_hash)
                if destination_key is None or destination_key == source_key:
                    continue
                element = (
                    ElementText.visible(target=edge.action_target) if edge.action_target else None
                )
                self.__merge_link(
                    bucket=outbound.setdefault(source_key, {}),
                    action=edge.action_type,
                    element=element,
                    screen=titles[destination_key],
                    count=edge.count,
                    value=edge.value,
                    semantics=edge.semantics,
                )
                self.__merge_link(
                    bucket=inbound.setdefault(destination_key, {}),
                    action=edge.action_type,
                    element=element,
                    screen=titles[source_key],
                    count=edge.count,
                    value=edge.value,
                    semantics=edge.semantics,
                )

        return inbound, outbound

    @staticmethod
    def __merge_link(
        *,
        bucket: Dict[_LinkKey, ScreenLink],
        action: str,
        element: Optional[str],
        screen: str,
        count: int,
        value: Optional[str],
        semantics: Optional[LinkSemantics],
    ) -> None:
        """
        Adds a link to a bucket, folding repeats of the same edge into one count.
        """

        dedup: _LinkKey = (action, element, screen)
        existing = bucket.get(dedup)
        if existing is None:
            bucket[dedup] = ScreenLink(
                action=action,
                element=element,
                screen=screen,
                count=count,
                value=value,
                semantics=semantics,
            )
        else:
            existing.count += count
            if existing.value is None and value is not None:
                existing.value = value
            if existing.semantics is None and semantics is not None:
                existing.semantics = semantics

    @staticmethod
    def __defects_by_screen(*, defects: List[Defect]) -> Dict[str, List[Defect]]:
        """
        Buckets defects by the canonical screen hash they were found on.
        """

        by_screen: Dict[str, List[Defect]] = {}
        for defect in defects:
            by_screen.setdefault(defect.evidence.screen, []).append(defect)
        return by_screen

    def __document(
        self,
        *,
        nodes: List[GraphNode],
        representative: GraphNode,
        slug: str,
        title: str,
        inbound: Dict[_LinkKey, ScreenLink],
        outbound: Dict[_LinkKey, ScreenLink],
        defects_by_screen: Dict[str, List[Defect]],
    ) -> ScreenDocument:
        """
        Assembles one logical screen's document from its grouped fingerprints.
        """

        collected: List[Defect] = []
        for node in nodes:
            collected.extend(defects_by_screen.get(node.visual_hash, []))
        collected.sort(key=lambda defect: (defect.severity.rank, defect.signal.value))

        has_description = representative.description and KnowledgeGraph.has_meaningful_description(
            representative.description
        )
        description = (
            representative.description.strip()
            if has_description and representative.description
            else ""
        )
        content = self.__content(nodes=nodes)

        return ScreenDocument(
            slug=slug,
            title=title,
            category=representative.category,
            activity=representative.activity,
            purpose=content.purpose or description,
            narrative=description,
            elements=content.elements,
            actions=content.actions,
            flow=ScreenFlow(
                inbound=self.__sorted_links(bucket=inbound),
                outbound=self.__sorted_links(bucket=outbound),
            ),
            defects=collected,
            visits=sum(node.visit_count for node in nodes),
            fingerprints=len(nodes),
        )

    @classmethod
    def __content(cls, *, nodes: List[GraphNode]) -> ScreenContent:
        """
        Returns the richest structured content across the grouped fingerprints.

        Prefers content captured directly from describe_screen; for screens persisted
        before structured content existed, decomposes the richest legacy rich-description
        blob so their documents still carry elements and actions.
        """

        structured = [node.content for node in nodes if node.content is not None]
        if structured:
            return max(
                structured,
                key=lambda content: (
                    len(content.elements) + len(content.actions),
                    len(content.purpose),
                ),
            )
        blobs = [
            node.rich_description.strip()
            for node in nodes
            if node.rich_description and node.rich_description.strip()
        ]
        return cls.__decompose(text=max(blobs, key=len)) if blobs else ScreenContent()

    @classmethod
    def __decompose(cls, *, text: str) -> ScreenContent:
        """
        Recovers structured content from a legacy rich-description markdown blob.
        """

        sections = cls.__sections(text=text)
        return ScreenContent(
            purpose=sections.get(SectionHeading.PURPOSE.value, ""),
            elements=cls.__lines(text=sections.get(SectionHeading.ELEMENTS.value, "")),
            actions=cls.__lines(text=sections.get(SectionHeading.ACTIONS.value, "")),
        )

    @staticmethod
    def __sections(*, text: str) -> Dict[str, str]:
        """
        Maps each second-level heading in markdown text to its body.
        """

        sections: Dict[str, str] = {}
        heading: Optional[str] = None
        body: List[str] = []
        for line in text.splitlines():
            match = _SECTION_HEADING.match(line)
            if match:
                if heading is not None:
                    sections[heading] = "\n".join(body).strip()
                heading = match.group("heading")
                body = []
            elif heading is not None:
                body.append(line)
        if heading is not None:
            sections[heading] = "\n".join(body).strip()
        return sections

    @staticmethod
    def __lines(*, text: str) -> List[str]:
        """
        Splits a section body into trimmed, non-empty entries.
        """

        return [line.strip() for line in text.splitlines() if line.strip()]

    @staticmethod
    def __sorted_links(*, bucket: Dict[_LinkKey, ScreenLink]) -> List[ScreenLink]:
        """
        Orders links by frequency, breaking ties deterministically.
        """

        return sorted(
            bucket.values(),
            key=lambda link: (-link.count, link.screen, link.action, link.element or ""),
        )


class ScreenDocumentRenderer:
    """
    Renders screen documents and their index as Markdown.
    """

    def render_index(self, *, index: DocumentIndex) -> str:
        """
        Renders the index that links to every per-screen document.
        """

        sections = [
            self.__index_header(index=index),
            self.__index_table(documents=index.documents),
        ]
        return "\n\n".join(section for section in sections if section)

    def render_screen(self, *, document: ScreenDocument) -> str:
        """
        Renders a single screen as a self-contained, image-free document.
        """

        sections = [
            self.__screen_header(document=document),
            self.__section(heading=SectionHeading.PURPOSE.value, body=document.purpose),
            self.__screen_section(document=document),
            self.__list_section(heading=SectionHeading.ELEMENTS.value, items=document.elements),
            self.__list_section(heading=SectionHeading.ACTIONS.value, items=document.actions),
            self.__links(
                heading=SectionHeading.REACHED_FROM.value,
                links=document.flow.inbound,
                relation=Relation.INBOUND.value,
            ),
            self.__links(
                heading=SectionHeading.LEADS_TO.value,
                links=document.flow.outbound,
                relation=Relation.OUTBOUND.value,
            ),
            self.__defects(defects=document.defects),
        ]
        return "\n\n".join(section for section in sections if section)

    @staticmethod
    def __screen_section(*, document: ScreenDocument) -> str:
        """
        Renders the screen-summary section.

        Elements and actions render beneath this heading and the consumer extracts
        them from within it, so the heading stays present whenever the screen has
        elements or actions even when it carries no prose summary of its own.
        """

        body = document.narrative.strip()
        if body:
            return f"## {SectionHeading.SCREEN.value}\n{body}"
        if document.elements or document.actions:
            return f"## {SectionHeading.SCREEN.value}"
        return ""

    @staticmethod
    def __list_section(*, heading: str, items: List[str]) -> str:
        """
        Renders a per-line list section, or nothing when there are no items.
        """

        if not items:
            return ""
        return "\n".join([f"## {heading}", "", *items])

    @staticmethod
    def __index_header(*, index: DocumentIndex) -> str:
        """
        Renders the documentation title and run metadata.
        """

        metadata = index.metadata
        return "\n".join(
            [
                "# Screen Documentation",
                "",
                f"- Workflow: {metadata.workflow}",
                f"- Package: {metadata.package}",
                f"- Generated: {metadata.generated_at}",
                f"- Screens: {len(index.documents)}",
            ]
        )

    @staticmethod
    def __index_table(*, documents: List[ScreenDocument]) -> str:
        """
        Renders the table of screens linking to each document.
        """

        if not documents:
            return "No screens were documented."

        rows = [
            f"| [{document.title}]({document.slug}.md) | {document.category.value} | "
            f"{document.visits} | {len(document.defects)} |"
            for document in documents
        ]
        return "\n".join(
            [
                "## Screens",
                "",
                "| Screen | Category | Visits | Defects |",
                "| --- | --- | --- | --- |",
                *rows,
            ]
        )

    @staticmethod
    def __screen_header(*, document: ScreenDocument) -> str:
        """
        Renders the screen title and its identifying metadata.
        """

        lines = [
            f"# {document.title}",
            "",
            f"- Category: {document.category.value}",
            f"- Activity: `{document.activity}`",
            f"- Visits: {document.visits}",
        ]
        if document.fingerprints > 1:
            lines.append(f"- Variants: {document.fingerprints} captures collapsed into this screen")
        return "\n".join(lines)

    @staticmethod
    def __section(*, heading: str, body: str) -> str:
        """
        Renders a prose section, or nothing when the body is empty.
        """

        if not body.strip():
            return ""
        return f"## {heading}\n{body.strip()}"

    @staticmethod
    def __links(*, heading: str, links: List[ScreenLink], relation: str) -> str:
        """
        Renders an inbound or outbound navigation list.
        """

        if not links:
            return ""

        rendered = []
        for link in links:
            label = link.action
            if link.element:
                label += f' "{link.element}"'
            suffix = f" (x{link.count})" if link.count > 1 else ""
            rendered.append(f"- {label} {relation} {link.screen}{suffix}")
        return "\n".join([f"## {heading}", "", *rendered])

    @staticmethod
    def __defects(*, defects: List[Defect]) -> str:
        """
        Renders the per-screen defect table, most severe first.
        """

        if not defects:
            return ""

        rows = [
            f"| {defect.severity.value} | {defect.signal.value} | {defect.occurrence} | "
            f"{defect.summary} |"
            for defect in defects
        ]
        return "\n".join(
            [
                f"## {SectionHeading.DEFECTS.value}",
                "",
                "| Severity | Signal | Count | Summary |",
                "| --- | --- | --- | --- |",
                *rows,
            ]
        )
