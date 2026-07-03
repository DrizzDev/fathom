from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from pydantic import JsonValue

from fathom.constants.collaboration import (
    Audience,
    MessageKind,
    ScriptFormat,
    ScriptStatus,
    TaskKind,
    TaskState,
    ThreadState,
)
from fathom.constants.conversation import RunState
from fathom.core.exceptions import InteractionError
from fathom.core.services.conversation.summary import ConversationSummaryProjector
from fathom.schemas import conversation as ConversationEntities


class TestConversationSummaryProjector(unittest.TestCase):
    """
    Verify Fathom-owned conversation summary projection.
    """

    def test_projects_one_completed_run(self) -> None:
        """
        Project one completed run from request, progress, result, and script rows.
        """

        anchor = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)
        task = "11111111-1111-4111-8111-111111111111"
        request = self.__message(
            identifier="22222222-2222-4222-8222-222222222222",
            task=task,
            kind=MessageKind.REQUEST,
            created=anchor,
            body={"intent": "search shoes", "package": "com.browser"},
        )
        progress = self.__message(
            identifier="33333333-3333-4333-8333-333333333333",
            task=task,
            kind=MessageKind.PROGRESS,
            created=anchor + timedelta(seconds=1),
            body={
                "step": 1,
                "status": "completed",
                "action": {
                    "type": "tap",
                    "target": "search",
                    "rationale": "Search field is visible.",
                    "confidence": 0.91,
                },
                "analysis": "Tap the search field.",
                "rationale": "Search field is visible.",
                "observation": {
                    "summary": "Search field is focused.",
                    "evidence": "Keyboard is visible.",
                    "screen": "screen-1",
                    "changed": True,
                },
                "summary": "Tapped",
            },
        )
        result = self.__message(
            identifier="44444444-4444-4444-8444-444444444444",
            task=task,
            kind=MessageKind.RESULT,
            created=anchor + timedelta(seconds=2),
            body={"status": "success", "summary": "Done"},
        )
        script = ConversationEntities.ScriptView(
            id="55555555-5555-4555-8555-555555555555",
            task=task,
            title="Script",
            format=ScriptFormat.TEXT_PLAIN,
            status=ScriptStatus.ACTIVE,
            revision=1,
            checksum="abc",
            size=12,
            content="tap search",
            created_by="actor",
            updated_by="actor",
            created=anchor,
            updated=anchor + timedelta(seconds=3),
        )
        tree = ConversationEntities.TaskTreeView(
            thread=ConversationEntities.ThreadView(
                id="66666666-6666-4666-8666-666666666666",
                title="Summary",
                state=ThreadState.ACTIVE,
                digest=None,
                created=anchor,
                updated=anchor,
            ),
            total=1,
            roots=(
                ConversationEntities.TaskNodeView(
                    id=task,
                    root=task,
                    parent=None,
                    execution=ConversationEntities.ExecutionReference(
                        id="77777777-7777-4777-8777-777777777777",
                    ),
                    kind=TaskKind.FATHOM,
                    state=TaskState.SUCCEEDED,
                    objective="search shoes",
                    summary=None,
                    assignee="actor",
                    created=anchor,
                    started=anchor,
                    ended=anchor + timedelta(seconds=2),
                    children=(),
                ),
            ),
        )
        projector = ConversationSummaryProjector()

        runs = projector.runs(
            scripts=(script,),
            messages=(request, progress, result),
            task_states=projector.task_states(tree=tree),
            task_executions=projector.task_executions(tree=tree),
        )
        overview = projector.overview(runs=runs)

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].task, task)
        self.assertEqual(runs[0].state, RunState.SUCCEEDED)
        self.assertEqual(runs[0].intent.text, "search shoes")
        self.assertEqual(runs[0].outcome.summary, "Done")
        self.assertEqual(runs[0].milestones[0].summary, "Tapped")
        self.assertEqual(runs[0].milestones[0].status, "completed")
        action = runs[0].milestones[0].action
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.target, "search")
        self.assertIsNone(runs[0].milestones[0].analysis)
        observation = runs[0].milestones[0].observation
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.summary, "Search field is focused.")
        self.assertIsNotNone(runs[0].script)
        self.assertEqual(runs[0].script.id, script.id)
        self.assertEqual(overview.status, RunState.SUCCEEDED)
        self.assertEqual(overview.activity, script.updated)

    def test_missing_task_execution_raises_interaction_error(self) -> None:
        """
        Missing execution mappings fail explicitly instead of leaking KeyError.
        """

        anchor = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)
        task = "11111111-1111-4111-8111-111111111111"
        request = self.__message(
            identifier="22222222-2222-4222-8222-222222222222",
            task=task,
            kind=MessageKind.REQUEST,
            created=anchor,
            body={"intent": "search shoes", "package": "com.browser"},
        )
        projector = ConversationSummaryProjector()

        with self.assertRaisesRegex(
            InteractionError,
            "Run task has no execution mapping.",
        ):
            projector.runs(
                scripts=(),
                messages=(request,),
                task_states={},
                task_executions={},
            )

    def __message(
        self,
        *,
        identifier: str,
        task: str,
        kind: MessageKind,
        created: datetime,
        body: JsonValue,
    ) -> ConversationEntities.MessageView:
        """
        Build one message view for projection tests.
        """

        return ConversationEntities.MessageView(
            id=identifier,
            task=task,
            author="actor",
            reply=None,
            kind=kind,
            audience=Audience.THREAD,
            body=body,
            sequence=1,
            labels=(),
            created=created,
        )
