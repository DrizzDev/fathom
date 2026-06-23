"""
Builds and renders one image-free Markdown document per logical screen.

A logical screen is one normalized activity plus one category. Every fingerprint
the crawl captured for that screen collapses into a single document, so the output
never duplicates files for the same screen; the extra captures are counted as
variants. The builder is pure: it reads the knowledge graph and the run's defects
and returns typed documents, leaving rendering and file I/O to the renderer and the
artifact writer.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from fathom.constants.screen import ScreenCategory
from fathom.core.services.exporter.graph import GraphLabeler
from fathom.infrastructure.memory.knowledge_graph import GraphNode, KnowledgeGraph
from fathom.schemas.defect import Defect
from fathom.schemas.document import DocumentIndex, ScreenDocument, ScreenFlow, ScreenLink
from fathom.schemas.report import ReportMetadata

_GroupKey = Tuple[str, ScreenCategory]
_LinkKey = Tuple[str, Optional[str], str]


class ScreenDocumentExporter:
    """
    Groups the knowledge graph into logical screens and builds their documents.
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
        titles = {key: self.__title(activity=key[0], category=key[1]) for key in groups}
        slugs = self.__slugs(ordered=ordered, titles=titles)
        membership = self.__membership(groups=groups)
        inbound, outbound = self.__flows(graph=graph, membership=membership, titles=titles)
        defects_by_screen = self.__defects_by_screen(defects=defects)

        documents = [
            self.__document(
                key=key,
                nodes=groups[key],
                slug=slugs[key],
                title=titles[key],
                inbound=inbound.get(key, {}),
                outbound=outbound.get(key, {}),
                defects_by_screen=defects_by_screen,
                graph=graph,
            )
            for key in ordered
        ]
        return DocumentIndex(metadata=metadata, documents=documents)

    @staticmethod
    def __group(*, graph: KnowledgeGraph) -> Dict[_GroupKey, List[GraphNode]]:
        """
        Buckets canonical screen nodes by normalized activity and category.
        """

        groups: Dict[_GroupKey, List[GraphNode]] = {}
        for node in graph.nodes.values():
            key = (KnowledgeGraph.normalize_activity(node.activity), node.category)
            groups.setdefault(key, []).append(node)
        return groups

    @staticmethod
    def __order(*, groups: Dict[_GroupKey, List[GraphNode]]) -> List[_GroupKey]:
        """
        Orders logical screens by total visits, breaking ties deterministically.
        """

        def rank(key: _GroupKey) -> Tuple[int, str, str]:
            visits = sum(node.visit_count for node in groups[key])
            return (-visits, key[0], key[1].value)

        return sorted(groups, key=rank)

    def __title(self, *, activity: str, category: ScreenCategory) -> str:
        """
        Builds a stable human-readable title from the activity and category.
        """

        base = self.__labeler.friendly_activity(activity=activity)
        if category is ScreenCategory.OTHER:
            return base
        return f"{base} ({category.value.title()})"

    def __slugs(
        self, *, ordered: List[_GroupKey], titles: Dict[_GroupKey, str]
    ) -> Dict[_GroupKey, str]:
        """
        Assigns each screen a filename-safe slug, de-colliding by numeric suffix.
        """

        slugs: Dict[_GroupKey, str] = {}
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
    def __membership(*, groups: Dict[_GroupKey, List[GraphNode]]) -> Dict[str, _GroupKey]:
        """
        Maps each canonical screen hash to the logical screen it belongs to.
        """

        membership: Dict[str, _GroupKey] = {}
        for key, nodes in groups.items():
            for node in nodes:
                membership[node.visual_hash] = key
        return membership

    def __flows(
        self,
        *,
        graph: KnowledgeGraph,
        membership: Dict[str, _GroupKey],
        titles: Dict[_GroupKey, str],
    ) -> Tuple[
        Dict[_GroupKey, Dict[_LinkKey, ScreenLink]], Dict[_GroupKey, Dict[_LinkKey, ScreenLink]]
    ]:
        """
        Resolves cross-screen transitions into deduped inbound and outbound links.
        """

        inbound: Dict[_GroupKey, Dict[_LinkKey, ScreenLink]] = {}
        outbound: Dict[_GroupKey, Dict[_LinkKey, ScreenLink]] = {}

        for source_hash, edges in graph.edges.items():
            source_key = membership.get(source_hash)
            if source_key is None:
                continue
            for edge in edges:
                destination_key = membership.get(edge.destination_hash)
                if destination_key is None or destination_key == source_key:
                    continue
                element = edge.action_target or None
                self.__merge_link(
                    bucket=outbound.setdefault(source_key, {}),
                    action=edge.action_type,
                    element=element,
                    screen=titles[destination_key],
                    count=edge.count,
                )
                self.__merge_link(
                    bucket=inbound.setdefault(destination_key, {}),
                    action=edge.action_type,
                    element=element,
                    screen=titles[source_key],
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
        key: _GroupKey,
        nodes: List[GraphNode],
        slug: str,
        title: str,
        inbound: Dict[_LinkKey, ScreenLink],
        outbound: Dict[_LinkKey, ScreenLink],
        defects_by_screen: Dict[str, List[Defect]],
        graph: KnowledgeGraph,
    ) -> ScreenDocument:
        """
        Assembles a single logical screen's document from its fingerprints.
        """

        collected: List[Defect] = []
        for node in nodes:
            collected.extend(defects_by_screen.get(node.visual_hash, []))
        collected.sort(key=lambda defect: (defect.severity.rank, defect.signal.value))

        return ScreenDocument(
            slug=slug,
            title=title,
            category=key[1],
            activity=nodes[0].activity,
            purpose=self.__purpose(nodes=nodes),
            narrative=self.__narrative(nodes=nodes, graph=graph),
            flow=ScreenFlow(
                inbound=self.__sorted_links(bucket=inbound),
                outbound=self.__sorted_links(bucket=outbound),
            ),
            defects=collected,
            visits=sum(node.visit_count for node in nodes),
            fingerprints=len(nodes),
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

    @staticmethod
    def __purpose(*, nodes: List[GraphNode]) -> str:
        """
        Picks the most descriptive short description across the fingerprints.
        """

        candidates = [
            node.description.strip()
            for node in nodes
            if node.description and KnowledgeGraph.has_meaningful_description(node.description)
        ]
        return max(candidates, key=len) if candidates else ""

    @staticmethod
    def __narrative(*, nodes: List[GraphNode], graph: KnowledgeGraph) -> str:
        """
        Returns the screen's prose, falling back to the activity-level description.

        Rich descriptions accumulate on one node per activity, so a logical screen
        whose own fingerprints carry none borrows the activity's accumulated text.
        """

        own = [
            node.rich_description.strip()
            for node in nodes
            if node.rich_description and node.rich_description.strip()
        ]
        if own:
            return max(own, key=len)

        normalized = KnowledgeGraph.normalize_activity(nodes[0].activity)
        for node in graph.nodes.values():
            if (
                KnowledgeGraph.normalize_activity(node.activity) == normalized
                and node.rich_description
                and node.rich_description.strip()
            ):
                return node.rich_description.strip()
        return ""


class ScreenDocumentRenderer:
    """
    Renders logical-screen documents and their index as Markdown.
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
        Renders a single logical screen as a self-contained, image-free document.
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
        Renders the table of logical screens linking to each document.
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
