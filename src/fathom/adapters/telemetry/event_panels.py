from __future__ import annotations

from typing import Any, Mapping, Optional

from rich.panel import Panel

from fathom.constants.events import ExplorationEvent


def render_event_panel(
    *,
    event_type: Any,
    message: str,
    context: Mapping[str, Any],
    level: str = "info",
) -> Optional[Panel]:
    """
    Map an exploration event to a ``rich.Panel`` for the TUI body.

    Returns ``None`` for events that only update header counters and
    don't deserve their own body panel (e.g. ``LLM_CALL_COMPLETED`` at
    non-verbose level).
    """

    if event_type == ExplorationEvent.WORKFLOW_STARTED:
        return __workflow_started(context=context)

    if event_type == ExplorationEvent.SCREEN_DISCOVERED:
        return __screen_discovered(context=context)

    if event_type == ExplorationEvent.SCREEN_REVISITED:
        return __screen_revisited(context=context)

    if event_type == ExplorationEvent.ACTION_PLANNED:
        return __action_planned(context=context)

    if event_type == ExplorationEvent.ACTION_EXECUTED:
        return __action_executed(context=context)

    if event_type == ExplorationEvent.PHASE_TRANSITION:
        return __phase_transition(context=context)

    if event_type == ExplorationEvent.NAVIGATION_STARTED:
        return __navigation_started(context=context)

    if event_type == ExplorationEvent.BACKTRACK:
        return __backtrack(context=context)

    if event_type == ExplorationEvent.WORKFLOW_COMPLETED:
        return __workflow_completed(context=context)

    if event_type == ExplorationEvent.WORKFLOW_CANCELLED:
        return __workflow_cancelled(context=context)

    if event_type == ExplorationEvent.ERROR or level == "error":
        return Panel.fit(
            f"[bold red]{message or context.get('error') or 'Error'}[/bold red]",
            border_style="red",
        )

    return None


def __workflow_started(*, context: Mapping[str, Any]) -> Panel:
    package = context.get("package") or "(auto-detect)"
    max_steps = context.get("max_steps", "—")
    focus = context.get("focus")
    body = (
        f"[bold #a88fd8]🚀 Exploration starting[/bold #a88fd8]\n"
        f"[dim]Package:[/dim] {package}    "
        f"[dim]Max steps:[/dim] {max_steps}"
    )
    if focus:
        body += f"\n[dim]Focus:[/dim] {focus}"
    return Panel.fit(body, border_style="#a88fd8")


def __screen_discovered(*, context: Mapping[str, Any]) -> Panel:
    activity = context.get("activity") or context.get("screen_hash") or "unknown"
    unique = context.get("unique_screens")
    coverage = context.get("coverage")
    suffix_parts = []
    if unique is not None:
        suffix_parts.append(f"#{unique} mapped")
    if isinstance(coverage, (int, float)):
        suffix_parts.append(f"{coverage:.0f}% coverage")
    suffix = "  ·  ".join(suffix_parts)
    body = f"[bold green]🆕 New screen[/bold green]  {activity}"
    if suffix:
        body += f"\n[dim]{suffix}[/dim]"
    return Panel.fit(body, border_style="green")


def __screen_revisited(*, context: Mapping[str, Any]) -> Optional[Panel]:
    activity = context.get("activity") or context.get("screen_hash")
    if not activity:
        return None
    return Panel.fit(
        f"[dim]↺ Revisited[/dim] {activity}",
        border_style="grey50",
    )


def __action_planned(*, context: Mapping[str, Any]) -> Panel:
    action_type = context.get("action_type") or "action"
    target = context.get("target") or context.get("element") or ""
    reasoning = context.get("reasoning")
    head = f"[bold blue]📍 {action_type}[/bold blue]"
    if target:
        head += f"  [cyan]{target}[/cyan]"
    body = head
    if reasoning:
        snippet = str(reasoning)
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"
        body += f"\n[dim]{snippet}[/dim]"
    return Panel.fit(body, border_style="blue")


def __action_executed(*, context: Mapping[str, Any]) -> Optional[Panel]:
    success = bool(context.get("success", True))
    if success:
        return None
    detail = context.get("error") or context.get("action") or "action failed"
    return Panel.fit(
        f"[bold red]✗ Action failed[/bold red]  [dim]{detail}[/dim]",
        border_style="red",
    )


def __phase_transition(*, context: Mapping[str, Any]) -> Optional[Panel]:
    src = context.get("from")
    dst = context.get("to")
    if not dst:
        return None
    if src and src == dst:
        return None
    arrow = f"{src} → {dst}" if src else dst
    return Panel.fit(
        f"[bold magenta]➜ phase[/bold magenta]  {arrow}",
        border_style="magenta",
    )


def __navigation_started(*, context: Mapping[str, Any]) -> Panel:
    steps = context.get("steps")
    target = context.get("target") or "destination"
    suffix = f" ({steps} hop{'s' if steps != 1 else ''})" if steps else ""
    return Panel.fit(
        f"[bold cyan]🧭 Navigating[/bold cyan]  {target}{suffix}",
        border_style="cyan",
    )


def __backtrack(*, context: Mapping[str, Any]) -> Panel:
    via = context.get("action") or "—"
    return Panel.fit(
        f"[bold yellow]⤺ Backtrack[/bold yellow]  via {via}",
        border_style="yellow",
    )


def __workflow_completed(*, context: Mapping[str, Any]) -> Panel:
    success = bool(context.get("success", True))
    reason = context.get("completion_reason") or ("done" if success else "stopped")
    unique = context.get("unique_screens")
    coverage = context.get("coverage")
    color = "green" if success else "red"
    icon = "✓" if success else "✗"
    head = f"[bold {color}]{icon} Workflow {'completed' if success else 'failed'}[/bold {color}]"
    body = f"{head}\n[dim]{reason}[/dim]"
    stats: list[str] = []
    if unique is not None:
        stats.append(f"{unique} unique screens")
    if isinstance(coverage, (int, float)):
        stats.append(f"{coverage:.1f}% coverage")
    if stats:
        body += f"\n[dim]{'  ·  '.join(stats)}[/dim]"
    return Panel.fit(body, border_style=color)


def __workflow_cancelled(*, context: Mapping[str, Any]) -> Panel:
    reason = context.get("completion_reason") or "cancelled by user"
    return Panel.fit(
        f"[bold yellow]⏹ Workflow cancelled[/bold yellow]\n[dim]{reason}[/dim]",
        border_style="yellow",
    )
