from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from fathom.constants.collaboration import MessageKind, TaskState
from fathom.constants.conversation import RunState
from fathom.core.exceptions import InteractionError
from fathom.schemas import conversation as ConversationSchemas
from fathom.schemas import interaction as InteractionSchemas


class SummarySource(Protocol):
    """
    Service surface required to load records for one conversation summary.
    """

    async def get(
        self, *, query: ConversationSchemas.ConversationThreadQuery
    ) -> ConversationSchemas.ThreadView:
        """
        Load one visible conversation.
        """

        ...

    async def tasks(
        self, *, query: ConversationSchemas.TaskTreeQuery
    ) -> ConversationSchemas.TaskTreeView:
        """
        Load the visible task tree.
        """

        ...

    async def summary_messages(
        self, *, query: InteractionSchemas.SummaryMessagesQuery
    ) -> Tuple[ConversationSchemas.MessageView, ...]:
        """
        Load bounded summary message rows.
        """

        ...

    async def summary_scripts(
        self, *, query: InteractionSchemas.SummaryScriptsQuery
    ) -> Tuple[ConversationSchemas.ScriptView, ...]:
        """
        Load bounded summary script rows.
        """

        ...

    async def messages(
        self, *, query: ConversationSchemas.MessageListQuery
    ) -> ConversationSchemas.MessagePage:
        """
        Load one page of visible messages.
        """

        ...

    async def artifacts(
        self, *, query: ConversationSchemas.ArtifactListQuery
    ) -> ConversationSchemas.ArtifactPage:
        """
        Load one page of visible artifacts.
        """

        ...


class RunProjection:
    """
    Groups source records that belong to one run-root task.
    """

    def __init__(self, *, request: ConversationSchemas.MessageView) -> None:
        """
        Store the request message that defines the run.
        """

        self.source_request = request
        self.progress: List[ConversationSchemas.MessageView] = []
        self.latest_script: Optional[ConversationSchemas.ScriptView] = None
        self.latest_result: Optional[ConversationSchemas.MessageView] = None

    def keep_request(self, *, message: ConversationSchemas.MessageView) -> None:
        """
        Keep the deterministic earliest request for duplicate request rows.
        """

        if (message.created, message.id) < (
            self.source_request.created,
            self.source_request.id,
        ):
            self.source_request = message

    def milestone(self, *, message: ConversationSchemas.MessageView) -> None:
        """
        Add one progress message to the run projection.
        """

        self.progress.append(message)

    def keep_result(self, *, message: ConversationSchemas.MessageView) -> None:
        """
        Keep the deterministic latest result message.
        """

        if self.latest_result is None or (message.created, message.id) > (
            self.latest_result.created,
            self.latest_result.id,
        ):
            self.latest_result = message

    def keep_script(self, *, script: ConversationSchemas.ScriptView) -> None:
        """
        Keep the deterministic latest script row.
        """

        if self.latest_script is None or (script.updated, script.id) > (
            self.latest_script.updated,
            self.latest_script.id,
        ):
            self.latest_script = script


class ConversationSummaryProjector:
    """
    Projects conversation source records into the summary response.
    """

    def task_states(self, *, tree: ConversationSchemas.TaskTreeView) -> Dict[str, TaskState]:
        """
        Flatten the task tree into a task-state lookup.
        """

        states: Dict[str, TaskState] = {}
        self.__collect_task_states(nodes=tree.roots, states=states)

        return states

    def task_executions(self, *, tree: ConversationSchemas.TaskTreeView) -> Dict[str, str]:
        """
        Flatten the task tree into a task-execution lookup.
        """

        executions: Dict[str, str] = {}
        self.__collect_task_executions(nodes=tree.roots, executions=executions)

        return executions

    def runs(
        self,
        *,
        task_executions: Dict[str, str],
        task_states: Dict[str, TaskState],
        scripts: Sequence[ConversationSchemas.ScriptView],
        messages: Sequence[ConversationSchemas.MessageView],
    ) -> Tuple[ConversationSchemas.RunOverview, ...]:
        """
        Project per-run overviews from summary source rows.
        """

        projections: Dict[str, RunProjection] = {}

        for message in messages:
            if message.kind is not MessageKind.REQUEST or message.task is None:
                continue

            if (projection := projections.get(message.task)) is None:
                projections[message.task] = RunProjection(request=message)
                continue

            projection.keep_request(message=message)

        for message in messages:
            if message.task is None or (projection := projections.get(message.task)) is None:
                continue

            if message.kind is MessageKind.PROGRESS:
                projection.milestone(message=message)

            elif message.kind is MessageKind.RESULT:
                projection.keep_result(message=message)

        for script in scripts:
            if script.task is None or (projection := projections.get(script.task)) is None:
                continue

            projection.keep_script(script=script)

        overviews = [
            self.__run(
                task=task,
                projection=projection,
                task_states=task_states,
                task_executions=task_executions,
            )
            for task, projection in projections.items()
        ]
        overviews.sort(key=lambda overview: overview.started, reverse=True)
        return tuple(overviews)

    def overview(
        self, *, runs: Tuple[ConversationSchemas.RunOverview, ...]
    ) -> ConversationSchemas.OverviewBlock:
        """
        Derive the conversation header from per-run summaries.
        """

        if not runs:
            return ConversationSchemas.OverviewBlock()

        latest = runs[0]
        return ConversationSchemas.OverviewBlock(
            status=latest.state,
            activity=max(run.updated for run in runs),
        )

    def __collect_task_states(
        self,
        *,
        states: Dict[str, TaskState],
        nodes: Sequence[ConversationSchemas.TaskNodeView],
    ) -> None:
        """
        Add task node states to the supplied lookup.
        """

        for node in nodes:
            states[node.id] = node.state
            self.__collect_task_states(nodes=node.children, states=states)

    def __collect_task_executions(
        self,
        *,
        executions: Dict[str, str],
        nodes: Sequence[ConversationSchemas.TaskNodeView],
    ) -> None:
        """
        Add task node execution ids to the supplied lookup.
        """

        for node in nodes:
            executions[node.id] = node.execution.id
            self.__collect_task_executions(nodes=node.children, executions=executions)

    def __run(
        self,
        *,
        task: str,
        projection: RunProjection,
        task_executions: Dict[str, str],
        task_states: Dict[str, TaskState],
    ) -> ConversationSchemas.RunOverview:
        """
        Project one run overview from one grouped source bundle.
        """

        result = projection.latest_result
        script = projection.latest_script
        request = projection.source_request

        started = request.created
        completed = result.created if result is not None else None

        latest_script = script.updated if script is not None else None
        latest_milestone = max(
            (message.created for message in projection.progress),
            default=None,
        )
        updated = max(
            timestamp
            for timestamp in (started, completed, latest_milestone, latest_script)
            if timestamp is not None
        )
        execution = task_executions.get(task)
        if execution is None:
            raise InteractionError("Run task has no execution mapping.")

        return ConversationSchemas.RunOverview(
            task=task,
            started=started,
            updated=updated,
            completed=completed,
            script=self.__script(script=script),
            intent=self.__intent(request=request),
            outcome=self.__outcome(result=result),
            workflow=self.__workflow_reference(request=request),
            milestones=self.__milestones(progress=projection.progress),
            execution=ConversationSchemas.ExecutionReference(id=execution),
            state=self.__state(task=task, result=result, task_states=task_states),
        )

    @staticmethod
    def __workflow_reference(
        *, request: Optional[ConversationSchemas.MessageView]
    ) -> Optional[ConversationSchemas.WorkflowReference]:
        """
        Extract the workflow reference from the request message metadata, when present.
        """

        if request is None:
            return None

        body = request.body
        if not isinstance(body, dict):
            return None

        workflow = body.get("workflow")
        if isinstance(workflow, str) and workflow:
            return ConversationSchemas.WorkflowReference(id=workflow)

        return None

    def __state(
        self,
        *,
        task: str,
        task_states: Dict[str, TaskState],
        result: Optional[ConversationSchemas.MessageView],
    ) -> RunState:
        """
        Map authoritative task state onto the client-facing summary state.
        """

        state = task_states.get(task)

        if state is TaskState.SUCCEEDED:
            return RunState.SUCCEEDED

        if state in (TaskState.FAILED, TaskState.EXPIRED):
            return RunState.FAILED

        if state is TaskState.CANCELLED:
            return RunState.CANCELLED

        if state in (TaskState.QUEUED, TaskState.RUNNING, TaskState.BLOCKED, TaskState.WAITING):
            return RunState.RUNNING

        if result is None:
            return RunState.RUNNING

        return RunState.UNKNOWN

    def __intent(
        self, *, request: ConversationSchemas.MessageView
    ) -> ConversationSchemas.IntentOverview:
        """
        Build the intent overview from one request message.
        """

        body = ConversationSchemas.IntentBody.model_validate(request.body or {})
        packages = (
            ConversationSchemas.IntentPackages(
                target=body.package,
                initial=body.starting_package,
            )
            if (body.package or body.starting_package)
            else None
        )

        return ConversationSchemas.IntentOverview(
            text=body.intent,
            packages=packages,
            recorded=request.created,
        )

    def __outcome(
        self, *, result: Optional[ConversationSchemas.MessageView]
    ) -> ConversationSchemas.OutcomeOverview:
        """
        Build the outcome overview from one result message.
        """

        if result is None:
            return ConversationSchemas.OutcomeOverview()

        body = ConversationSchemas.ResultBody.model_validate(result.body or {})

        return ConversationSchemas.OutcomeOverview(
            status=body.status,
            detail=body.detail,
            recorded=result.created,
            summary=body.summary or body.reason,
        )

    def __milestones(
        self, *, progress: Sequence[ConversationSchemas.MessageView]
    ) -> Tuple[ConversationSchemas.MilestoneOverview, ...]:
        """
        Build milestones from progress messages, newest first.
        """

        milestones: List[ConversationSchemas.MilestoneOverview] = []

        for message in sorted(
            progress,
            key=lambda entry: (
                entry.created,
                self.__progress_step(message=entry),
                entry.id,
            ),
            reverse=True,
        ):
            body = ConversationSchemas.ProgressBody.model_validate(message.body or {})

            if body.step is None or body.step < 1:
                continue

            milestones.append(
                ConversationSchemas.MilestoneOverview(
                    step=body.step,
                    status=body.status,
                    action=self.__action(raw=body.action),
                    summary=body.summary,
                    recorded=message.created,
                    observation=self.__observation(raw=body.observation),
                )
            )

        return tuple(milestones)

    @staticmethod
    def __action(*, raw: Optional[Dict[str, Any]]) -> Optional[ConversationSchemas.ActionOverview]:
        """
        Build the trimmed action projection from the stored raw dict.
        """

        if not isinstance(raw, dict):
            return None

        return ConversationSchemas.ActionOverview(
            type=raw.get("type") if isinstance(raw.get("type"), str) else None,
            target=raw.get("target") if isinstance(raw.get("target"), str) else None,
            rationale=raw.get("rationale") if isinstance(raw.get("rationale"), str) else None,
        )

    @staticmethod
    def __observation(
        *, raw: Optional[Dict[str, Any]]
    ) -> Optional[ConversationSchemas.ObservationOverview]:
        """
        Build the trimmed observation projection from the stored raw dict.
        """

        if not isinstance(raw, dict):
            return None

        return ConversationSchemas.ObservationOverview(
            summary=raw.get("summary") if isinstance(raw.get("summary"), str) else None,
            evidence=raw.get("evidence") if isinstance(raw.get("evidence"), str) else None,
        )

    def __progress_step(self, *, message: ConversationSchemas.MessageView) -> int:
        """
        Return the progress step used for deterministic milestone ordering.
        """

        body = ConversationSchemas.ProgressBody.model_validate(message.body or {})
        if body.step is None:
            return 0

        return body.step

    def __script(
        self, *, script: Optional[ConversationSchemas.ScriptView]
    ) -> Optional[ConversationSchemas.ScriptOverview]:
        """
        Build the script overview pointer for one run.
        """

        if script is None:
            return None

        return ConversationSchemas.ScriptOverview(
            id=script.id,
            size=script.size,
            title=script.title,
            updated=script.updated,
            revision=script.revision,
        )


class SummaryLoader:
    """
    Loads and projects all source rows for the conversation summary endpoint.
    """

    def __init__(self, *, projector: Optional[ConversationSummaryProjector] = None) -> None:
        """
        Store the summary projector dependency.
        """

        self.__projector = projector or ConversationSummaryProjector()

    async def load(
        self,
        *,
        source: SummarySource,
        query: ConversationSchemas.SummaryQuery,
    ) -> ConversationSchemas.SummaryView:
        """
        Return one projected conversation summary.
        """

        conversation, task_tree, messages_total, artifacts_total = await asyncio.gather(
            source.get(
                query=ConversationSchemas.ConversationThreadQuery(
                    tenant=query.tenant,
                    thread=query.thread,
                    operator=query.operator,
                )
            ),
            source.tasks(
                query=ConversationSchemas.TaskTreeQuery(
                    tenant=query.tenant,
                    thread=query.thread,
                    operator=query.operator,
                )
            ),
            source.messages(
                query=ConversationSchemas.MessageListQuery(
                    limit=1,
                    tenant=query.tenant,
                    thread=query.thread,
                    operator=query.operator,
                )
            ),
            source.artifacts(
                query=ConversationSchemas.ArtifactListQuery(
                    limit=1,
                    tenant=query.tenant,
                    thread=query.thread,
                    operator=query.operator,
                )
            ),
        )

        messages, scripts = await asyncio.gather(
            source.summary_messages(
                query=InteractionSchemas.SummaryMessagesQuery(
                    tenant=query.tenant,
                    thread=query.thread,
                    operator=query.operator,
                    kinds=(
                        MessageKind.RESULT,
                        MessageKind.REQUEST,
                        MessageKind.PROGRESS,
                    ),
                )
            ),
            source.summary_scripts(
                query=InteractionSchemas.SummaryScriptsQuery(
                    tenant=query.tenant,
                    thread=query.thread,
                    operator=query.operator,
                )
            ),
        )

        runs = self.__projector.runs(
            scripts=scripts,
            messages=messages,
            task_states=self.__projector.task_states(tree=task_tree),
            task_executions=self.__projector.task_executions(tree=task_tree),
        )

        return ConversationSchemas.SummaryView(
            runs=runs,
            thread=conversation,
            overview=self.__projector.overview(runs=runs),
            counts=ConversationSchemas.OverviewCounts(
                runs=len(runs),
                scripts=len(scripts),
                messages=messages_total.total,
                artifacts=artifacts_total.total,
            ),
        )
