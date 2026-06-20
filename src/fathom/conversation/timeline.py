from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from pydantic import JsonValue

from fathom.constants.collaboration import Label
from fathom.constants.conversation import EntryKind, Visibility, VisibilityRank
from fathom.conversation.cursor import CompositeTimelineCursor, OpaqueCursor
from fathom.schemas.conversation import EntryView, ThreadView, TimelineQuery, TimelineView
from fathom.schemas.interaction import Artifact, Context, Event, Message, SortOrder, Thread

if TYPE_CHECKING:
    from datetime import datetime

VISIBILITY_RANKS: Dict[Visibility, VisibilityRank] = {
    Visibility.USER: VisibilityRank.USER,
    Visibility.DEBUG: VisibilityRank.DEBUG,
    Visibility.AUDIT: VisibilityRank.AUDIT,
    Visibility.HIDDEN: VisibilityRank.HIDDEN,
}

# Sentinel labels for the four ledger kinds. Used internally during the
# consume-emit walk so per-kind cursor positions stay distinct. Single
# underscore avoids Python's class-scope name-mangling on `__name`.
_MESSAGES = "messages"
_EVENTS = "events"
_ARTIFACTS = "artifacts"
_CONTEXTS = "contexts"


class TimelineComposer:
    """
    Pure composer that walks merged per-kind candidates, consumes hidden /
    filtered rows, emits visible rows up to the global page limit, and
    derives a composite next cursor based on what was *consumed* (not just
    what was emitted) so that pages of all-hidden rows still advance.
    """

    def build(
        self,
        *,
        thread: Thread,
        messages: Tuple[Message, ...],
        events: Tuple[Event, ...],
        artifacts: Tuple[Artifact, ...],
        contexts: Tuple[Context, ...],
        query: TimelineQuery,
        inbound: CompositeTimelineCursor,
        has_more: Dict[str, bool],
        total: int,
    ) -> TimelineView:
        """
        Build a filtered, ordered, globally-limited timeline view.

        Walk semantics: hidden / filtered-out candidates are CONSUMED (advance
        the per-kind cursor) so they are not re-fetched. Visible candidates
        are EMITTED until the page is full; we stop BEFORE consuming the
        next visible entry so it surfaces on the next page.
        """

        candidates = self.__candidates(
            events=events,
            order=query.order,
            contexts=contexts,
            messages=messages,
            artifacts=artifacts,
        )

        stopped_early = False
        emitted: List[EntryView] = []
        last_consumed: Dict[str, Tuple[datetime, str]] = {}

        for kind_label, created, identifier, view in candidates:
            visible = self.__visible(entry=view, mode=query.mode) and self.__kind_visible(
                entry=view, kinds=query.kinds
            )
            if visible:
                if len(emitted) >= query.limit:
                    stopped_early = True
                    break

                emitted.append(view)
                last_consumed[kind_label] = (created, identifier)
            else:
                last_consumed[kind_label] = (created, identifier)

        next_cursor = self.__derive_next_cursor(
            inbound=inbound,
            has_more=has_more,
            last_consumed=last_consumed,
            stopped_early=stopped_early,
        )

        return TimelineView(
            total=total,
            next=next_cursor,
            entries=tuple(emitted),
            thread=self.__thread(thread=thread),
        )

    def __candidates(
        self,
        *,
        order: SortOrder,
        events: Tuple[Event, ...],
        contexts: Tuple[Context, ...],
        messages: Tuple[Message, ...],
        artifacts: Tuple[Artifact, ...],
    ) -> List[Tuple[str, datetime, str, EntryView]]:
        """
        Build the merge-sorted candidate stream tagged by source kind.

        Sort direction matches the per-kind query order so the walk emits in the same order the storage layer returned,
        keeping cursor semantics consistent (DESC = newest first; ASC = oldest first).
        """

        candidates: List[Tuple[str, datetime, str, EntryView]] = []
        for message in messages:
            candidates.append(
                (_MESSAGES, message.created, message.identity.id, self.__message(message=message))
            )
        for event in events:
            candidates.append(
                (_EVENTS, event.created, event.identity.id, self.__event(event=event))
            )
        for artifact in artifacts:
            candidates.append(
                (
                    _ARTIFACTS,
                    artifact.created,
                    artifact.identity.id,
                    self.__artifact(artifact=artifact),
                )
            )
        for context in contexts:
            candidates.append(
                (_CONTEXTS, context.created, context.identity.id, self.__context(context=context))
            )
        reverse = order is SortOrder.DESC
        candidates.sort(key=lambda candidate: (candidate[1], candidate[2]), reverse=reverse)

        return candidates

    def __derive_next_cursor(
        self,
        *,
        stopped_early: bool,
        has_more: Dict[str, bool],
        inbound: CompositeTimelineCursor,
        last_consumed: Dict[str, Tuple[datetime, str]],
    ) -> str | None:
        """
        Compose the per-kind cursor envelope for the next page.

        If nothing was consumed, the walk did not stop early, and no kind has
        more rows in storage, the page exhausted the data set — return None.
        """

        no_more = not any(has_more.values())
        if no_more and not stopped_early:
            return None

        composite = CompositeTimelineCursor(
            messages=self.__sub_cursor(
                kind=_MESSAGES,
                fallback=inbound.messages,
                last_consumed=last_consumed,
                has_more=has_more.get(_MESSAGES, False),
            ),
            events=self.__sub_cursor(
                kind=_EVENTS,
                fallback=inbound.events,
                last_consumed=last_consumed,
                has_more=has_more.get(_EVENTS, False),
            ),
            artifacts=self.__sub_cursor(
                kind=_ARTIFACTS,
                fallback=inbound.artifacts,
                last_consumed=last_consumed,
                has_more=has_more.get(_ARTIFACTS, False),
            ),
            contexts=self.__sub_cursor(
                kind=_CONTEXTS,
                fallback=inbound.contexts,
                last_consumed=last_consumed,
                has_more=has_more.get(_CONTEXTS, False),
            ),
        )

        if composite.is_empty():
            return None

        return composite.encode()

    def __sub_cursor(
        self,
        *,
        kind: str,
        has_more: bool,
        fallback: Optional[str],
        last_consumed: Dict[str, Tuple[datetime, str]],
    ) -> str | None:
        """
        Build one per-kind sub-cursor: last consumed if any, else preserve
        the inbound position when more data exists, else None.
        """

        if kind in last_consumed:
            created, identifier = last_consumed[kind]
            return OpaqueCursor(created=created, identifier=identifier).encode()

        if has_more:
            return fallback

        return None

    def message_entry(self, *, message: Message) -> EntryView:
        """
        Convert one message into a timeline entry.
        """

        return self.__message(message=message)

    def artifact_entry(self, *, artifact: Artifact) -> EntryView:
        """
        Convert one artifact reference into a timeline entry.
        """

        return self.__artifact(artifact=artifact)

    def context_entry(self, *, context: Context) -> EntryView:
        """
        Convert one context recipe into a timeline entry.
        """

        return self.__context(context=context)

    def __thread(self, *, thread: Thread) -> ThreadView:
        """
        Convert a ledger thread into a client thread view.
        """

        return ThreadView(
            title=thread.title,
            state=thread.state,
            digest=thread.digest,
            id=thread.identity.id,
            created=thread.timing.created,
            updated=thread.timing.updated,
        )

    def __message(self, *, message: Message) -> EntryView:
        """
        Convert one message into a timeline entry.
        """

        visibility = self.__label_visibility(labels=message.content.labels)

        return EntryView(
            task=message.task,
            actor=message.author,
            visibility=visibility,
            id=message.identity.id,
            kind=EntryKind.MESSAGE,
            created=message.created,
            sequence=message.sequence,
            payload={
                "kind": message.kind.value,
                "body": message.content.body,
                "audience": message.audience.value,
                "labels": [label.value for label in message.content.labels],
            },
        )

    def __event(self, *, event: Event) -> EntryView:
        """
        Convert one lifecycle event into a timeline entry.
        """

        return EntryView(
            task=event.task,
            actor=event.actor,
            id=event.identity.id,
            kind=EntryKind.EVENT,
            created=event.created,
            sequence=event.sequence,
            visibility=Visibility.DEBUG,
            payload={
                "kind": event.kind.value,
                "source": event.source.value,
                "payload": event.payload.entries,
            },
        )

    def __artifact(self, *, artifact: Artifact) -> EntryView:
        """
        Convert one artifact reference into a timeline entry.
        """

        visibility = self.__label_visibility(labels=artifact.labels)

        return EntryView(
            sequence=None,
            task=artifact.task,
            visibility=visibility,
            id=artifact.identity.id,
            kind=EntryKind.ARTIFACT,
            actor=artifact.producer,
            created=artifact.created,
            payload={
                "uri": artifact.uri,
                "mime": artifact.mime,
                "size": artifact.size,
                "kind": artifact.kind.value,
                "retention": artifact.retention,
                "backend": artifact.backend.value,
                "labels": [label.value for label in artifact.labels],
            },
        )

    def __context(self, *, context: Context) -> EntryView:
        """
        Convert one context recipe into an audit-only timeline entry.
        """

        return EntryView(
            sequence=None,
            task=context.task,
            id=context.identity.id,
            kind=EntryKind.CONTEXT,
            actor=context.consumer,
            created=context.created,
            visibility=Visibility.AUDIT,
            payload=self.__context_payload(context=context),
        )

    def __context_payload(self, *, context: Context) -> JsonValue:
        """
        Build a JSON-safe context payload without copying raw prompt text.
        """

        return {
            "hash": context.hash,
            "model": context.model,
            "builder": context.builder,
            "provider": context.provider,
            "purpose": context.purpose.value,
            "events": list(context.references.events),
            "messages": list(context.references.messages),
            "artifacts": list(context.references.artifacts),
        }

    def __visible(self, *, entry: EntryView, mode: Visibility) -> bool:
        """
        Decide whether an entry is visible in the requested mode.
        """

        mode_rank = VISIBILITY_RANKS.get(mode)
        entry_rank = VISIBILITY_RANKS.get(entry.visibility)

        if entry_rank is None or mode_rank is None:
            return False

        return entry_rank <= mode_rank

    def __kind_visible(self, *, entry: EntryView, kinds: Tuple[EntryKind, ...]) -> bool:
        """
        Decide whether an entry kind is part of the requested timeline slice.
        """

        if not kinds:
            return True

        return entry.kind in kinds

    def __label_visibility(self, *, labels: Tuple[Label, ...]) -> Visibility:
        """
        Resolve render visibility from policy labels.
        """

        if Label.DISPLAY_HIDDEN in labels:
            return Visibility.HIDDEN

        if Label.DISPLAY_AUDIT in labels:
            return Visibility.AUDIT

        if Label.DISPLAY_DEBUG in labels:
            return Visibility.DEBUG

        return Visibility.USER
