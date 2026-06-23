"""
Builds and renders one image-free Markdown document per screen.

A screen is one canonical node in the knowledge graph (the graph already merges
revisits and near-duplicate captures), so each distinct screen gets exactly one
document and the output never duplicates files for the same screen. The builder is
pure: it reads the knowledge graph and the run's defects and returns typed
documents, leaving rendering and file I/O to the renderer and the artifact writer.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from fathom.constants.exploration import MAX_SCREEN_LABEL_LENGTH
from fathom.constants.screen import ScreenCategory
from fathom.core.services.exporter.graph import GraphLabeler
from fathom.infrastructure.memory.knowledge_graph import GraphNode, KnowledgeGraph
from fathom.schemas.defect import Defect
from fathom.schemas.document import DocumentIndex, ScreenDocument, ScreenFlow, ScreenLink
from fathom.schemas.report import ReportMetadata

_LinkKey = Tuple[str, Optional[str], str]


class ScreenDocumentExporter:
    """
    Builds one document per canonical screen node from the knowledge graph.
    """

    __NON_SLUG = re.compile(r"[^a-z0-9]+")

    def __init__(self, *, labeler: Optional[GraphLabeler] = None) -> None:
        self.__labeler = labeler or GraphLabeler()

    def build(
        self, *, graph: KnowledgeGraph, defects: List[Defect], metadata: ReportMetadata
    ) -> DocumentIndex:
        """
        Builds one document per screen and the index over them.
        """

        nodes = self.__ordered(graph=graph)
        titles = {node.visual_hash: self.__title(node=node) for node in nodes}
        slugs = self.__slugs(nodes=nodes, titles=titles)
        inbound, outbound = self.__flows(graph=graph, titles=titles)
        defects_by_screen = self.__defects_by_screen(defects=defects)

        documents = [
            self.__document(
                node=node,
                slug=slugs[node.visual_hash],
                title=titles[node.visual_hash],
                inbound=inbound.get(node.visual_hash, {}),
                outbound=outbound.get(node.visual_hash, {}),
                defects=defects_by_screen.get(node.visual_hash, []),
            )
            for node in nodes
        ]
        return DocumentIndex(metadata=metadata, documents=documents)

    @classmethod
    def __ordered(cls, *, graph: KnowledgeGraph) -> List[GraphNode]:
        """
        Documentable screens ordered by visit count, ties broken on the hash.
        """

        documentable = [node for node in graph.nodes.values() if cls.__is_documentable(node=node)]
        return sorted(documentable, key=lambda node: (-node.visit_count, node.visual_hash))

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

    def __slugs(self, *, nodes: List[GraphNode], titles: Dict[str, str]) -> Dict[str, str]:
        """
        Assigns each screen a filename-safe slug, de-colliding by numeric suffix.
        """

        slugs: Dict[str, str] = {}
        seen: Dict[str, int] = {}
        for node in nodes:
            base = self.__slugify(text=titles[node.visual_hash])
            count = seen.get(base, 0)
            seen[base] = count + 1
            slugs[node.visual_hash] = base if count == 0 else f"{base}-{count + 1}"
        return slugs

    @classmethod
    def __slugify(cls, *, text: str) -> str:
        """
        Reduces a title to lowercase alphanumeric words joined by hyphens.
        """

        slug = cls.__NON_SLUG.sub("-", text.lower()).strip("-")
        return slug or "screen"

    def __flows(
        self, *, graph: KnowledgeGraph, titles: Dict[str, str]
    ) -> Tuple[Dict[str, Dict[_LinkKey, ScreenLink]], Dict[str, Dict[_LinkKey, ScreenLink]]]:
        """
        Resolves cross-screen transitions into deduped inbound and outbound links.
        """

        inbound: Dict[str, Dict[_LinkKey, ScreenLink]] = {}
        outbound: Dict[str, Dict[_LinkKey, ScreenLink]] = {}

        for source_hash, edges in graph.edges.items():
            if source_hash not in titles:
                continue
            for edge in edges:
                destination_hash = edge.destination_hash
                if destination_hash not in titles or destination_hash == source_hash:
                    continue
                element = edge.action_target or None
                self.__merge_link(
                    bucket=outbound.setdefault(source_hash, {}),
                    action=edge.action_type,
                    element=element,
                    screen=titles[destination_hash],
                    count=edge.count,
                )
                self.__merge_link(
                    bucket=inbound.setdefault(destination_hash, {}),
                    action=edge.action_type,
                    element=element,
                    screen=titles[source_hash],
                    count=edge.count,
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
    ) -> None:
        """
        Adds a link to a bucket, folding repeats of the same edge into one count.
        """

        dedup: _LinkKey = (action, element, screen)
        existing = bucket.get(dedup)
        if existing is None:
            bucket[dedup] = ScreenLink(action=action, element=element, screen=screen, count=count)
        else:
            existing.count += count

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
        node: GraphNode,
        slug: str,
        title: str,
        inbound: Dict[_LinkKey, ScreenLink],
        outbound: Dict[_LinkKey, ScreenLink],
        defects: List[Defect],
    ) -> ScreenDocument:
        """
        Assembles a single screen's document from its node.
        """

        ordered = sorted(defects, key=lambda defect: (defect.severity.rank, defect.signal.value))
        has_purpose = node.description and KnowledgeGraph.has_meaningful_description(
            node.description
        )
        purpose = node.description.strip() if has_purpose and node.description else ""
        narrative = node.rich_description.strip() if node.rich_description else ""

        return ScreenDocument(
            slug=slug,
            title=title,
            category=node.category,
            activity=node.activity,
            purpose=purpose,
            narrative=narrative,
            flow=ScreenFlow(
                inbound=self.__sorted_links(bucket=inbound),
                outbound=self.__sorted_links(bucket=outbound),
            ),
            defects=ordered,
            visits=node.visit_count,
            fingerprints=1,
        )

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
            self.__section(heading="Purpose", body=document.purpose),
            self.__section(heading="Screen", body=document.narrative),
            self.__links(heading="Reached From", links=document.flow.inbound, relation="from"),
            self.__links(heading="Leads To", links=document.flow.outbound, relation="to"),
            self.__defects(defects=document.defects),
        ]
        return "\n\n".join(section for section in sections if section)

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
                "## Defects",
                "",
                "| Severity | Signal | Count | Summary |",
                "| --- | --- | --- | --- |",
                *rows,
            ]
        )
