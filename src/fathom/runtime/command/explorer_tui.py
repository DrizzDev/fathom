"""
Fixed-window Textual TUI for the ``fathom explore --tui`` command.

Layout:

- **Header** (fixed): package + step counter + elapsed + screens + tokens.
- **Body** (scrollable): streamed activity lines from the run.
- **Footer** (fixed): current DFS phase + status + key hints.

The agent runs on a daemon worker thread (``__run_workflow``) so the blocking
calls inside the device adapters never freeze the Textual main loop. Progress
snapshots and activity lines are marshaled back to the main thread via
``call_from_thread``.
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

from fathom.schemas.exploration import ExplorationProgress

logger = getLogger(__name__)

# ASCII run-state markers shown in the header (no emojis, per house standards).
STATUS_STARTING = "..."
STATUS_RUNNING = "*"
STATUS_DONE = "OK"
STATUS_FAILED = "FAIL"
STATUS_CANCELLING = "STOP"

# Seconds before a final status panel paints and the app tears down.
EXIT_PAINT_DELAY = 0.4
# Seconds after a cancel request before the TUI force-exits a stuck run.
CANCEL_FORCE_TIMEOUT = 30.0
# Header/footer refresh cadence.
REFRESH_INTERVAL = 0.25


class _StatusBar(Static):  # type: ignore[misc, unused-ignore]
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
    status_icon: reactive[str] = reactive(STATUS_STARTING)

    def render(self) -> Panel:
        prompt_k = self.tokens_prompt / 1000
        completion_k = self.tokens_completion / 1000
        cached_k = self.tokens_cached / 1000
        max_str = f"/{self.max_steps}" if self.max_steps else ""
        title = (
            f"[{self.status_icon}]  Fathom explorer   "
            f"Step {self.step}{max_str}   {self.elapsed:0.1f}s   "
            f"{self.unique_screens} screens ({self.coverage:0.0f}%)   "
            f"tokens up {prompt_k:0.1f}k down {completion_k:0.1f}k cached {cached_k:0.1f}k"
        )
        package = self.package or "(auto-detect)"
        return Panel(
            f"[dim]Package:[/dim] {package}",
            title=title,
            border_style="#6b3fd4",
            padding=(0, 1),
        )


class _FooterBar(Static):  # type: ignore[misc, unused-ignore]
    """
    Reactive footer showing the active DFS phase and a status hint.
    """

    phase: reactive[str] = reactive("scan")
    status: reactive[str] = reactive("")
    hint: reactive[str] = reactive("q to quit - up/down / PageUp / PageDown to scroll")

    def render(self) -> Panel:
        status_line = self.status or "-"
        body = (
            f"phase: [bold]{self.phase}[/bold]   [dim]{status_line}[/dim]\n[dim]{self.hint}[/dim]"
        )
        return Panel(body, border_style="#6b3fd4", padding=(0, 1))


class ExplorerApp(App[int]):  # type: ignore[misc, unused-ignore]
    """
    Textual app that runs a Fathom exploration workflow inside a pinned
    header / scrollable body / pinned footer layout.
    """

    TITLE = "Fathom - explorer"

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

        ``workflow`` is a zero-arg coroutine-returning callable kicked off on
        mount; it returns ``True`` on success. ``on_cancel`` is invoked when the
        user hits ``q`` / ``ctrl+c``; the workflow is responsible for noticing
        cancellation and returning.
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
            "status": "starting...",
            "status_icon": STATUS_STARTING,
            "unique_screens": 0,
            "coverage": 0.0,
        }

    # ── Textual lifecycle ────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield _StatusBar(id="status")
        yield RichLog(id="body", highlight=True, markup=True, wrap=True, auto_scroll=True)
        yield _FooterBar(id="footer")

    async def on_mount(self) -> None:
        self.__started_at = time.time()
        status = self.query_one("#status", _StatusBar)
        status.package = self.__package
        status.max_steps = self.__max_steps
        self.set_interval(REFRESH_INTERVAL, self.__tick)

        # Daemon thread so the agent can't outlive the process on a force-quit.
        self.__worker_thread = threading.Thread(
            target=self.__run_workflow, name="explorer_workflow", daemon=True
        )
        self.__worker_thread.start()

    async def on_unmount(self) -> None:
        """
        Final safety net: signal cancellation so the worker thread stops doing
        device / LLM work whatever the exit path.
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
        """
        Worker-thread entry: drive the workflow coroutine to completion.
        """

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
        self.__state["status_icon"] = STATUS_DONE if success else STATUS_FAILED
        self.__state["status"] = "completed" if success else "failed"
        self.__schedule_exit()

    def __render_workflow_error(self, *, error: str) -> None:
        self.__state["status_icon"] = STATUS_FAILED
        self.__state["status"] = "error"
        with contextlib.suppress(Exception):
            body = self.query_one("#body", RichLog)
            body.write(
                Panel.fit(f"[bold red]Workflow error:[/bold red] {error}", border_style="red")
            )
        self.__schedule_exit()

    def __schedule_exit(self) -> None:
        """
        Close the Textual app after the workflow finishes.
        """

        with contextlib.suppress(Exception):
            self.set_timer(EXIT_PAINT_DELAY, lambda: self.exit(self.exit_code))

    def __thread_safe_call(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        if bool(getattr(self, "is_running", False)):
            with contextlib.suppress(Exception):
                self.call_from_thread(callback, *args, **kwargs)
                return
        callback(*args, **kwargs)

    # ── Bindings ─────────────────────────────────────────────────────────────

    def action_request_cancel(self) -> None:
        """
        Two-stage quit: the first press signals ``on_cancel`` so finalizers
        (report, KG export) still run; a second press force-exits the TUI.
        """

        if self.__on_cancel is not None:
            with contextlib.suppress(Exception):
                self.__on_cancel()

        if self.__cancel_requested:
            self.exit_code = 1
            self.exit(self.exit_code)
            return

        self.__cancel_requested = True
        self.__state["status"] = "cancelling... (press q again to force quit)"
        self.__state["status_icon"] = STATUS_CANCELLING

        with contextlib.suppress(Exception):
            self.set_timer(CANCEL_FORCE_TIMEOUT, lambda: self.exit(1))

    # ── Progress + activity ingress (called from the worker thread) ───────────

    def update_progress(self, progress: ExplorationProgress) -> None:
        """
        Thread-safe entry point: apply a progress snapshot to the header/footer.
        """

        self.__thread_safe_call(self.__apply_progress, progress=progress)

    def append_activity(self, message: str, *, level: str = "info") -> None:
        """
        Thread-safe entry point: write one activity line into the body.
        """

        self.__thread_safe_call(self.__append_activity, message=message, level=level)

    def __apply_progress(self, *, progress: ExplorationProgress) -> None:
        self.__state["step"] = progress.step
        self.__state["phase"] = progress.phase.value
        self.__state["unique_screens"] = progress.unique_screens
        self.__state["coverage"] = progress.coverage
        self.__state["tokens"] = {
            "prompt": progress.tokens.prompt,
            "completion": progress.tokens.completion,
            "cached": progress.tokens.cached,
        }
        if progress.status:
            self.__state["status"] = progress.status
        if self.__state["status_icon"] in (STATUS_STARTING, STATUS_RUNNING):
            self.__state["status_icon"] = STATUS_RUNNING

    def __append_activity(self, *, message: str, level: str) -> None:
        if not message.strip():
            return
        style = "red" if level == "error" else "white"
        with contextlib.suppress(Exception):
            body = self.query_one("#body", RichLog)
            body.write(f"[{style}]{message}[/{style}]")

    # ── Test hooks ────────────────────────────────────────────────────────────

    def _state_snapshot(self) -> Dict[str, Any]:
        snapshot = dict(self.__state)
        snapshot["tokens"] = dict(snapshot["tokens"])
        return snapshot
