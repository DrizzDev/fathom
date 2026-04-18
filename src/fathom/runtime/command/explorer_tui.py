"""
Fixed-window Textual TUI for the ``fathom explore --tui`` command.

Layout:

- **Header** (fixed): package + step counter + elapsed + screens + tokens.
- **Body** (scrollable): per-event panels rendered via ``render_event_panel``.
- **Footer** (fixed): current BFS phase + status icon + key hints.

The agent runs on a worker thread (``__run_workflow_in_thread``) so the
synchronous blocking calls inside the device adapters don't freeze the
Textual main loop. Telemetry from the agent is marshaled back to the
main thread via ``call_from_thread``.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from logging import getLogger
from typing import Any, Awaitable, Callable, Coroutine, Dict, Optional, cast

from rich.panel import Panel
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import RichLog, Static

from fathom.adapters.telemetry.event_panels import render_event_panel
from fathom.constants.events import ExplorationEvent

logger = getLogger(__name__)


class _StatusBar(Static):  # type: ignore[misc]
    """
    Reactive header showing the current run state.
    """

    package: reactive[str] = reactive("")
    step: reactive[int] = reactive(0)
    max_steps: reactive[int] = reactive(0)
    elapsed: reactive[float] = reactive(0.0)
    unique_screens: reactive[int] = reactive(0)
    coverage: reactive[float] = reactive(0.0)
    tokens_prompt: reactive[int] = reactive(0)
    tokens_completion: reactive[int] = reactive(0)
    tokens_cached: reactive[int] = reactive(0)
    status_icon: reactive[str] = reactive("⠋")

    def render(self) -> Panel:
        prompt_k = self.tokens_prompt / 1000
        completion_k = self.tokens_completion / 1000
        cached_k = self.tokens_cached / 1000
        max_str = f"/{self.max_steps}" if self.max_steps else ""
        title = (
            f"{self.status_icon}  Fathom · explorer   "
            f"Step {self.step}{max_str}   ⏱ {self.elapsed:0.1f}s   "
            f"🗺 {self.unique_screens} screens ({self.coverage:0.0f}%)   "
            f"tokens ↑{prompt_k:0.1f}k ↓{completion_k:0.1f}k · cached {cached_k:0.1f}k"
        )
        package = self.package or "(auto-detect)"
        return Panel(
            f"[dim]Package:[/dim] {package}",
            title=title,
            border_style="#6b3fd4",
            padding=(0, 1),
        )


class _FooterBar(Static):  # type: ignore[misc]
    """
    Reactive footer showing the active BFS phase + status hint.
    """

    phase: reactive[str] = reactive("scan")
    status: reactive[str] = reactive("")
    hint: reactive[str] = reactive("q to quit · ↑/↓ / PageUp / PageDown to scroll")

    def render(self) -> Panel:
        status_line = self.status or "—"
        body = (
            f"[bold cyan]🌀[/bold cyan]  phase: [bold]{self.phase}[/bold]   "
            f"[dim]{status_line}[/dim]\n[dim]{self.hint}[/dim]"
        )
        return Panel(body, border_style="#6b3fd4", padding=(0, 1))


class ExplorerApp(App[int]):  # type: ignore[misc]
    """
    Textual app that runs a Fathom exploration workflow inside a pinned
    header / scrollable body / pinned footer layout.
    """

    TITLE = "Fathom · explorer"

    CSS = """
    Screen {
        background: #0d0a1f;
    }

    #status {
        height: 5;
        padding: 0;
    }

    #body {
        height: 1fr;
        border: solid #6b3fd4;
        background: #0d0a1f;
    }

    #footer {
        height: 5;
        padding: 0;
    }
    """

    BINDINGS = [
        Binding("q", "request_cancel", "Cancel"),
        Binding("ctrl+c", "request_cancel", "Cancel"),
    ]

    def __init__(
        self,
        *,
        package: str,
        max_steps: int,
        workflow: Callable[[], Awaitable[bool]],
        on_cancel: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Build the app with a pre-wired workflow coroutine factory.

        ``workflow`` is a zero-arg coroutine-returning callable that the
        app kicks off on mount. It must return a bool (``True`` on
        success, ``False`` on failure). ``on_cancel`` is invoked when
        the user hits ``q`` / ``ctrl+c``; the workflow itself is
        responsible for noticing cancellation and returning.
        """

        super().__init__()
        self.__package = package
        self.__max_steps = max_steps
        self.__workflow = workflow
        self.__on_cancel = on_cancel
        self.__started_at: Optional[float] = None
        self.__cancel_requested = False
        self.__worker_thread: Optional[threading.Thread] = None
        self.exit_code: int = 1
        self.__state: Dict[str, Any] = {
            "step": 0,
            "tokens": {"prompt": 0, "completion": 0, "cached": 0},
            "phase": "scan",
            "status": "starting…",
            "status_icon": "⠋",
            "unique_screens": 0,
            "coverage": 0.0,
        }

    # ------------------------------------------------------------------
    # Textual lifecycle
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield _StatusBar(id="status")
        yield RichLog(
            id="body",
            highlight=True,
            markup=True,
            wrap=True,
            auto_scroll=True,
        )
        yield _FooterBar(id="footer")

    async def on_mount(self) -> None:
        self.__started_at = time.time()
        status = self.query_one("#status", _StatusBar)
        status.package = self.__package
        status.max_steps = self.__max_steps
        self.set_interval(0.25, self.__tick)

        # Run the agent on a *daemon* thread so it can't outlive the
        # process. Textual's @work(thread=True) uses a non-daemon
        # ThreadPoolExecutor, so on force-quit the agent would keep
        # running LLM calls in the background until its asyncio.run
        # returned. Daemon threads die when the main thread exits.
        self.__worker_thread = threading.Thread(
            target=self.__run_workflow,
            name="explorer_workflow",
            daemon=True,
        )
        self.__worker_thread.start()

    async def on_unmount(self) -> None:
        """
        Final safety net: whatever the exit path, signal cancellation
        so the worker thread's agent stops doing device / LLM work.
        """

        if self.__on_cancel is not None:
            with contextlib.suppress(Exception):
                self.__on_cancel()

    def __tick(self) -> None:
        if self.__started_at is None:
            return
        status = self.query_one("#status", _StatusBar)
        status.elapsed = time.time() - self.__started_at
        status.step = int(self.__state["step"])
        status.status_icon = str(self.__state["status_icon"])
        status.unique_screens = int(self.__state["unique_screens"])
        status.coverage = float(self.__state["coverage"])
        tokens = self.__state["tokens"]
        status.tokens_prompt = int(tokens["prompt"])
        status.tokens_completion = int(tokens["completion"])
        status.tokens_cached = int(tokens["cached"])

        footer = self.query_one("#footer", _FooterBar)
        footer.phase = str(self.__state["phase"])
        footer.status = str(self.__state["status"])

    def __run_workflow(self) -> None:
        """Worker-thread entry: drive the workflow coroutine to completion."""

        try:
            coro = cast("Coroutine[Any, Any, bool]", self.__workflow())
            success: bool = asyncio.run(coro)
            self.exit_code = 0 if success else 1
            self.__thread_safe_call(self.__render_workflow_finish, success=success)
        except Exception as exception:  # noqa: BLE001
            logger.exception("Explorer workflow crashed")
            self.exit_code = 1
            self.__thread_safe_call(self.__render_workflow_error, error=str(exception))

    def __render_workflow_finish(self, *, success: bool) -> None:
        self.__state["status_icon"] = "✓" if success else "✗"
        self.__state["status"] = "completed" if success else "failed"
        self.__schedule_exit()

    def __render_workflow_error(self, *, error: str) -> None:
        self.__state["status_icon"] = "✗"
        self.__state["status"] = "error"
        with contextlib.suppress(Exception):
            body = self.query_one("#body", RichLog)
            body.write(
                Panel.fit(
                    f"[bold red]Workflow error:[/bold red] {error}",
                    border_style="red",
                )
            )
        self.__schedule_exit()

    def __schedule_exit(self) -> None:
        """Close the Textual app after the workflow finishes."""

        with contextlib.suppress(Exception):
            # Slight delay so the final status panel paints before teardown.
            self.set_timer(0.4, lambda: self.exit(self.exit_code))

    def __thread_safe_call(
        self,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if bool(getattr(self, "is_running", False)):
            with contextlib.suppress(Exception):
                self.call_from_thread(callback, *args, **kwargs)
                return
        callback(*args, **kwargs)

    # ------------------------------------------------------------------
    # Bindings
    # ------------------------------------------------------------------

    def action_request_cancel(self) -> None:
        """
        Two-stage quit.

        First press signals ``runner.cancel()`` so the agent's finalizers
        (exploration report, KG export) still run, and updates the
        footer to show the cancel is in flight. Second press force-exits
        the TUI. The worker thread is a daemon so it dies when the
        process exits regardless of what it's currently doing.
        """

        # Always signal cancellation on every press — the first press
        # kicks it off, later presses are idempotent and safe.
        if self.__on_cancel is not None:
            with contextlib.suppress(Exception):
                self.__on_cancel()

        if self.__cancel_requested:
            self.exit_code = 1
            self.exit(self.exit_code)
            return

        self.__cancel_requested = True
        self.__state["status"] = "cancelling… (press q again to force quit)"
        self.__state["status_icon"] = "⏹"

        # Safety net: if the workflow doesn't wind down within 30s,
        # force-exit so the user isn't stranded inside a frozen TUI.
        with contextlib.suppress(Exception):
            self.set_timer(30.0, lambda: self.exit(1))

    # ------------------------------------------------------------------
    # Event ingress (called from agent worker thread)
    # ------------------------------------------------------------------

    def record_event(self, event_type: ExplorationEvent, context: Dict[str, Any]) -> None:
        """Thread-safe entry point: route an exploration event into the UI."""

        self.__thread_safe_call(self.__record_event_impl, event_type=event_type, context=context)

    def __record_event_impl(
        self,
        *,
        event_type: ExplorationEvent,
        context: Dict[str, Any],
    ) -> None:
        message = str(context.get("message", ""))
        level = "error" if event_type == ExplorationEvent.ERROR else "info"

        panel = render_event_panel(
            event_type=event_type,
            message=message,
            context=context,
            level=level,
        )
        if panel is not None:
            with contextlib.suppress(Exception):
                body = self.query_one("#body", RichLog)
                body.write(panel)

        self.__update_state_from_event(event_type=event_type, context=context)

    def __update_state_from_event(
        self,
        *,
        event_type: ExplorationEvent,
        context: Dict[str, Any],
    ) -> None:
        if event_type == ExplorationEvent.STEP_COMPLETED:
            step = context.get("step")
            if isinstance(step, int):
                self.__state["step"] = step
            phase = context.get("phase")
            if phase:
                self.__state["phase"] = str(phase)

        elif event_type == ExplorationEvent.LLM_CALL_COMPLETED:
            tokens = self.__state["tokens"]
            for key, ctx_key in (
                ("prompt", "prompt_tokens"),
                ("completion", "completion_tokens"),
                ("cached", "cached_tokens"),
            ):
                raw = context.get(ctx_key, 0) or 0
                try:
                    tokens[key] += int(raw)
                except (TypeError, ValueError):
                    continue

        elif event_type in (ExplorationEvent.SCREEN_DISCOVERED, ExplorationEvent.SCREEN_REVISITED):
            unique = context.get("unique_screens")
            if isinstance(unique, int):
                self.__state["unique_screens"] = unique
            coverage = context.get("coverage")
            if isinstance(coverage, (int, float)):
                self.__state["coverage"] = float(coverage)

        elif event_type == ExplorationEvent.PHASE_TRANSITION:
            dst = context.get("to")
            if dst:
                self.__state["phase"] = str(dst)

        elif event_type == ExplorationEvent.NAVIGATION_STARTED:
            target = context.get("target") or "destination"
            self.__state["status"] = f"navigating → {target}"

        elif event_type == ExplorationEvent.BACKTRACK:
            self.__state["status"] = "backtracking"

        elif event_type == ExplorationEvent.ACTION_PLANNED:
            target = context.get("target") or context.get("action_type", "")
            self.__state["status"] = f"planned: {target}"

        elif event_type == ExplorationEvent.ACTION_EXECUTED:
            success = bool(context.get("success", True))
            self.__state["status"] = "action ok" if success else "action failed"

        elif event_type == ExplorationEvent.WORKFLOW_STARTED:
            self.__state["status"] = "exploring…"
            self.__state["status_icon"] = "▶"
            package = context.get("package")
            if package:
                self.__package = str(package)
                with contextlib.suppress(Exception):
                    self.query_one("#status", _StatusBar).package = self.__package

        elif event_type == ExplorationEvent.WORKFLOW_COMPLETED:
            success = bool(context.get("success", True))
            self.__state["status_icon"] = "✓" if success else "✗"
            reason = context.get("completion_reason")
            self.__state["status"] = (
                str(reason) if reason else ("completed" if success else "failed")
            )

        elif event_type == ExplorationEvent.WORKFLOW_CANCELLED:
            self.__state["status_icon"] = "⏹"
            self.__state["status"] = "cancelled"

    # ------------------------------------------------------------------
    # Test hooks
    # ------------------------------------------------------------------

    def _state_snapshot(self) -> Dict[str, Any]:
        snapshot = dict(self.__state)
        snapshot["tokens"] = dict(snapshot["tokens"])
        return snapshot
