from __future__ import annotations

from typing import TYPE_CHECKING, Callable, ClassVar, Dict, List, Optional, Tuple

from pydantic import BaseModel, JsonValue

from fathom.constants.collaboration import Label, MessageKind
from fathom.constants.conversation import EntryKind, TimelineSource, Visibility, VisibilityRank
from fathom.conversation.cursor import CompositeTimelineCursor, OpaqueCursor
from fathom.schemas.conversation import (
    EntryView,
    ThreadView,
    TimelineQuery,
    TimelineView,
)
from fathom.schemas.conversation.wire import (
    WireAnswerBody,
    WireProgressBody,
    WireQuestionBody,
    WireRequestBody,
    WireResultBody,
)
from fathom.schemas.interaction import Artifact, Context, Event, Message, SortOrder

if TYPE_CHECKING:
    from datetime import datetime


class TimelineComposer:
    """
    Pure composer that walks merged per-kind candidates, consumes hidden / filtered rows, emits visible rows up to the global page limit,
    and derives a composite next cursor based on what was *consumed* (not just what was emitted) so that pages of all-hidden rows still advance.
    """

    __COMPACT_PROJECTORS: ClassVar[Dict[MessageKind, Callable[..., Optional[BaseModel]]]] = {
        MessageKind.ANSWER: WireAnswerBody.project,
        MessageKind.RESULT: WireResultBody.project,
        MessageKind.REQUEST: WireRequestBody.project,
        MessageKind.PROGRESS: WireProgressBody.project,
        MessageKind.QUESTION: WireQuestionBody.project,
    }

    def __init__(self) -> None:
        """
        Initialize the pure timeline entry composer.
        """

    def build(
        self,
        *,
        total: int,
        thread: ThreadView,
        query: TimelineQuery,
        events: Tuple[Event, ...],
        has_more: Dict[str, bool],
        contexts: Tuple[Context, ...],
        messages: Tuple[Message, ...],
        artifacts: Tuple[Artifact, ...],
        inbound: CompositeTimelineCursor,
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
            mode=query.mode,
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
            thread=thread,
            next=next_cursor,
            entries=tuple(emitted),
        )

    def __candidates(
        self,
        *,
        mode: Visibility,
        order: SortOrder,
        events: Tuple[Event, ...],
        contexts: Tuple[Context, ...],
        messages: Tuple[Message, ...],
        artifacts: Tuple[Artifact, ...],
    ) -> List[Tuple[str, datetime, str, EntryView]]:
        """
        Build the merge-sorted candidate stream tagged by source kind.
        """

        candidates: List[Tuple[str, datetime, str, EntryView]] = []
        for message in messages:
            candidates.append(
                (
                    TimelineSource.MESSAGES.value,
                    message.created,
                    message.identity.id,
                    self.__message(message=message, mode=mode),
                )
            )
        for event in events:
            candidates.append(
                (
                    TimelineSource.EVENTS.value,
                    event.created,
                    event.identity.id,
                    self.__event(event=event),
                )
            )
        for artifact in artifacts:
            candidates.append(
                (
                    TimelineSource.ARTIFACTS.value,
                    artifact.created,
                    artifact.identity.id,
                    self.__artifact(artifact=artifact),
                )
            )
        for context in contexts:
            candidates.append(
                (
                    TimelineSource.CONTEXTS.value,
                    context.created,
                    context.identity.id,
                    self.__context(context=context),
                )
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
                fallback=inbound.messages,
                last_consumed=last_consumed,
                kind=TimelineSource.MESSAGES.value,
                has_more=has_more.get(TimelineSource.MESSAGES.value, False),
            ),
            events=self.__sub_cursor(
                fallback=inbound.events,
                last_consumed=last_consumed,
                kind=TimelineSource.EVENTS.value,
                has_more=has_more.get(TimelineSource.EVENTS.value, False),
            ),
            artifacts=self.__sub_cursor(
                fallback=inbound.artifacts,
                last_consumed=last_consumed,
                kind=TimelineSource.ARTIFACTS.value,
                has_more=has_more.get(TimelineSource.ARTIFACTS.value, False),
            ),
            contexts=self.__sub_cursor(
                fallback=inbound.contexts,
                last_consumed=last_consumed,
                kind=TimelineSource.CONTEXTS.value,
                has_more=has_more.get(TimelineSource.CONTEXTS.value, False),
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
        Build one per-kind sub-cursor: last consumed if any, else preserve the inbound position.
        """

        if kind in last_consumed:
            created, identifier = last_consumed[kind]
            return OpaqueCursor(created=created, identifier=identifier).encode()

        return fallback

    def message_entry(self, *, message: Message) -> EntryView:
        """
        Convert one message into a timeline entry with the full audit-shape body.
        """

        return self.__message(message=message, mode=Visibility.AUDIT)

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

    def __message(self, *, message: Message, mode: Visibility) -> EntryView:
        """
        Convert one message into a timeline entry with a mode-appropriate body.
        """

        visibility = self.__label_visibility(labels=message.content.labels)
        payload: Dict[str, JsonValue] = {
            "kind": message.kind.value,
            "body": self.__project_body(
                mode=mode,
                kind=message.kind,
                body=message.content.body,
            ),
        }
        if mode is not Visibility.USER:
            payload["audience"] = message.audience.value
            payload["labels"] = [label.value for label in message.content.labels]

        return EntryView(
            payload=payload,
            task=message.task,
            actor=message.author,
            visibility=visibility,
            id=message.identity.id,
            kind=EntryKind.MESSAGE,
            created=message.created,
            sequence=message.sequence,
        )

    @classmethod
    def __project_body(
        cls,
        *,
        body: JsonValue,
        mode: Visibility,
        kind: MessageKind,
    ) -> JsonValue:
        """
        Project one stored body into the shape requested by mode.
        """

        if mode is not Visibility.USER:
            return body

        if not isinstance(body, dict):
            return body

        if (projector := cls.__COMPACT_PROJECTORS.get(kind)) is None:
            return body

        if (projected := projector(body=body)) is None:
            return body

        return projected.model_dump(mode="json", exclude_none=True)

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

        mode_rank = self.__visibility_rank(visibility=mode)
        entry_rank = self.__visibility_rank(visibility=entry.visibility)

        if entry_rank is None or mode_rank is None:
            return False

        return entry_rank <= mode_rank

    @staticmethod
    def __visibility_rank(*, visibility: Visibility) -> Optional[VisibilityRank]:
        """
        Convert a timeline visibility value to its filtering rank.
        """

        if visibility is Visibility.USER:
            return VisibilityRank.USER

        if visibility is Visibility.DEBUG:
            return VisibilityRank.DEBUG

        if visibility is Visibility.AUDIT:
            return VisibilityRank.AUDIT

        if visibility is Visibility.HIDDEN:
            return VisibilityRank.HIDDEN

        return None

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
