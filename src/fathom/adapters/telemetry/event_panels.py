from __future__ import annotations

from typing import Any, Mapping, Optional

from rich.panel import Panel

from fathom.constants.events import FathomEvent


def render_event_panel(
    *,
    event_type: Any,
    message: str,
    context: Mapping[str, Any],
    level: str = "info",
) -> Optional[Panel]:
    """
    Map a telemetry event to a ``rich.Panel`` for operator display.

    Returns ``None`` for events that don't warrant a dedicated panel —
    callers apply their own plain-text fallback (e.g. ConsoleTelemetryAdapter
    prints a colored status line for workflow-lifecycle events).

    Shared between ``ConsoleTelemetryAdapter`` and the demo TUI so panel
    styling cannot drift between the two surfaces.
    """

    if event_type == FathomEvent.INTENT_CLASSIFIED:
        return __intent_classified(context=context)

    if event_type == FathomEvent.DECOMPOSITION_COMPLETE:
        return __decomposition(context=context)

    if event_type == FathomEvent.SUB_GOAL_STARTED:
        return __sub_goal_started(context=context)

    if event_type == FathomEvent.SUB_GOAL_COMPLETED:
        return __sub_goal_completed(context=context)

    if event_type == FathomEvent.HITL_REQUESTED:
        return __hitl_requested(message=message, context=context)

    if event_type == FathomEvent.REASONING:
        return __reasoning(message=message, context=context)

    if event_type == FathomEvent.PLANNED_ACTION:
        return __planned_action(message=message, context=context)

    if event_type == FathomEvent.STEP_COMPLETED:
        return __step_completed(context=context)

    if level == "error":
        return Panel.fit(
            f"[bold red]{message}[/bold red]",
            border_style="red",
        )

    return None


def __intent_classified(*, context: Mapping[str, Any]) -> Panel:
    should_decompose = bool(context.get("should_decompose", True))
    verdict = (
        "[bold magenta]Complex[/bold magenta] task — will be decomposed."
        if should_decompose
        else "[bold green]Simple[/bold green] task — running end-to-end."
    )
    return Panel.fit(
        f"[bold #a88fd8]🧠 Understanding your request[/bold #a88fd8]\n{verdict}",
        border_style="#a88fd8",
    )


def __decomposition(*, context: Mapping[str, Any]) -> Optional[Panel]:
    raw = context.get("sub_goals") or []
    sub_goals = [str(item) for item in raw if item]
    if not sub_goals:
        return None
    numbered = "\n".join(
        f"[bold cyan]{index}[/bold cyan] · {description}"
        for index, description in enumerate(sub_goals, start=1)
    )
    noun = "step" if len(sub_goals) == 1 else "steps"
    return Panel.fit(
        f"[bold #6b3fd4]📋 Plan ({len(sub_goals)} {noun})[/bold #6b3fd4]\n{numbered}",
        border_style="#6b3fd4",
    )


def __sub_goal_started(*, context: Mapping[str, Any]) -> Panel:
    human_index, total, description = __human_subgoal_fields(context=context)
    return Panel.fit(
        f"[bold cyan]🎯 Starting sub-goal {human_index}/{total}[/bold cyan]\n{description}",
        border_style="cyan",
    )


def __sub_goal_completed(*, context: Mapping[str, Any]) -> Panel:
    human_index, total, description = __human_subgoal_fields(context=context)
    return Panel.fit(
        f"[bold green]✓ Completed sub-goal {human_index}/{total}[/bold green]\n{description}",
        border_style="green",
    )


def __hitl_requested(*, message: str, context: Mapping[str, Any]) -> Panel:
    prompt = context.get("prompt") or message
    return Panel.fit(
        f"[bold yellow]⏸  Awaiting your input[/bold yellow]\n{prompt}",
        border_style="yellow",
    )


def __reasoning(*, message: str, context: Mapping[str, Any]) -> Panel:
    step_number = context.get("step", "?")
    reasoning = context.get("reasoning") or message
    return Panel.fit(
        f"[bold #a88fd8]💭 Step {step_number} · Reasoning[/bold #a88fd8]\n{reasoning}",
        border_style="#a88fd8",
    )


def __planned_action(*, message: str, context: Mapping[str, Any]) -> Panel:
    step_number = context.get("step", "?")
    action_description = context.get("action_description") or message
    target = context.get("target")
    confidence = context.get("confidence")
    body_parts = [f"[bold white]{action_description}[/bold white]"]
    if target:
        body_parts.append(f"[dim]target:[/dim] {target}")
    if isinstance(confidence, (int, float)):
        body_parts.append(f"[dim]confidence:[/dim] {float(confidence):.2f}")
    body = "\n".join(body_parts)
    return Panel.fit(
        f"[bold #6b3fd4]▶ Step {step_number} · Action[/bold #6b3fd4]\n{body}",
        border_style="#6b3fd4",
    )


def __step_completed(*, context: Mapping[str, Any]) -> Panel:
    step_number = context.get("step", "?")
    success = bool(context.get("success"))
    action_description = context.get("action_description") or "unknown"
    observation = context.get("observation") or "—"
    border = "green" if success else "red"
    icon = "✓" if success else "✗"
    title_color = "green" if success else "red"
    return Panel.fit(
        f"[bold {title_color}]{icon} Step {step_number} · "
        f"{'Success' if success else 'Failed'}[/bold {title_color}]\n"
        f"[dim]action:[/dim] {action_description}\n"
        f"[dim]observation:[/dim] {observation}",
        border_style=border,
    )


def __human_subgoal_fields(*, context: Mapping[str, Any]) -> tuple[Any, Any, str]:
    index = context.get("index", 0)
    total = context.get("total", 1)
    description = context.get("description") or "(unnamed)"
    human_index = int(index) + 1 if isinstance(index, int) else "?"
    return human_index, total, description
