"""
Fixed-window Textual TUI for the ``fathom demo`` command.

Launches a full-screen app with three regions:

- **Header** (fixed): intent + step counter + elapsed + token totals.
- **Body** (scrollable): cue panels for FathomEvents, plus any
  ``rich``-rendered content produced by the workflow.
- **Footer** (fixed): current sub-goal + status icon.

Only ``fathom demo`` uses this path. ``fathom run`` keeps the plain
scrolling-console pipeline. The Drizz wizard is a separate path via
``questionary`` and lands here via the ``demo`` subcommand dispatch.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from logging import getLogger
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Coroutine, Dict, Optional, cast

if TYPE_CHECKING:
    import concurrent.futures

from rich.panel import Panel
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Input, Label, RichLog, Static

from fathom.adapters.telemetry.event_panels import render_event_panel
from fathom.constants.events import FathomEvent

logger = getLogger(__name__)


class _StatusBar(Static):  # type: ignore[misc]
    """
    Reactive header showing the current run state.

    Reactive attributes auto-trigger ``render`` whenever they change
    so the header stays in sync with the TUI state without manual
    refreshes from the workflow side.
    """

    step: reactive[int] = reactive(0)
    elapsed: reactive[float] = reactive(0.0)
    tokens_prompt: reactive[int] = reactive(0)
    tokens_completion: reactive[int] = reactive(0)
    tokens_cached: reactive[int] = reactive(0)
    status_icon: reactive[str] = reactive("⠋")
    intent: reactive[str] = reactive("")

    def render(self) -> Panel:
        prompt_k = self.tokens_prompt / 1000
        completion_k = self.tokens_completion / 1000
        cached_k = self.tokens_cached / 1000
        title = (
            f"{self.status_icon}  Fathom · demo   "
            f"Step {self.step}   ⏱ {self.elapsed:0.1f}s   "
            f"tokens ↑{prompt_k:0.1f}k ↓{completion_k:0.1f}k · cached {cached_k:0.1f}k"
        )
        intent_display = self.intent or "—"
        if len(intent_display) > 200:
            intent_display = intent_display[:200] + "…"
        return Panel(
            f"[dim]Intent:[/dim] {intent_display}",
            title=title,
            border_style="#6b3fd4",
            padding=(0, 1),
        )


class _FooterBar(Static):  # type: ignore[misc]
    """
    Reactive footer showing the active sub-goal + hint text.
    """

    sub_goal: reactive[str] = reactive("")
    hint: reactive[str] = reactive("q to quit · ↑/↓ / PageUp / PageDown to scroll")

    def render(self) -> Panel:
        body = f"[bold cyan]🎯[/bold cyan]  {self.sub_goal or '—'}\n[dim]{self.hint}[/dim]"
        return Panel(body, border_style="#6b3fd4", padding=(0, 1))


class _HitlAskScreen(ModalScreen[str]):  # type: ignore[misc]
    """
    Modal collecting a single-line HITL response.

    Bound to a ``concurrent.futures.Future`` supplied by
    ``TuiSignalAdapter.ask`` — when the user presses Enter, the
    modal sets the future result so the agent's worker thread can
    resume.
    """

    CSS = """
    _HitlAskScreen {
        align: center middle;
    }
    _HitlAskScreen > Vertical {
        background: #1e1a3f;
        border: thick #a88fd8;
        padding: 1 2;
        width: 70;
        height: auto;
    }
    _HitlAskScreen Label {
        padding: 0 0 1 0;
        color: #e8e4ff;
    }
    _HitlAskScreen Input {
        border: solid #6b3fd4;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        *,
        prompt: str,
        future: "concurrent.futures.Future[str]",
    ) -> None:
        """
        Build a modal for a single HITL question.
        """

        super().__init__()
        self.__prompt = prompt
        self.__future = future

    def compose(self) -> ComposeResult:
        """
        Assemble a prompt label + input field inside a padded panel.
        """

        with Vertical():
            yield Label(f"[bold yellow]⏸  Agent needs your input[/]\n{self.__prompt}")
            yield Input(
                placeholder="Type your answer and press Enter…",
                id="hitl_input",
            )

    def on_mount(self) -> None:
        """
        Focus the input so the user can start typing immediately.
        """

        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """
        Deliver the user's text to the agent and dismiss.
        """

        if not self.__future.done():
            self.__future.set_result(event.value)
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        """
        Escape key → deliver an empty response and dismiss.

        Gives the agent a structured "no answer" rather than blocking
        indefinitely if the user abandons the prompt.
        """

        if not self.__future.done():
            self.__future.set_result("")
        self.dismiss("")


class DemoApp(App[int]):  # type: ignore[misc]
    """
    Textual app that runs a Fathom intent workflow inside a pinned
    header / scrollable body / pinned footer layout.
    """

    TITLE = "Fathom · demo"

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
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def __init__(
        self,
        *,
        intent: str,
        workflow: Callable[[], Awaitable[bool]],
    ) -> None:
        """
        Build the app with a pre-wired workflow coroutine factory.

        ``workflow`` is a zero-arg coroutine-returning callable that
        the app kicks off on mount. It must return a bool
        (``True`` on success, ``False`` on failure). Wiring (runner,
        signal adapter, telemetry) is the caller's responsibility so
        this module stays UI-only.
        """

        super().__init__()
        self.__intent = intent
        self.__workflow = workflow
        self.__started_at: Optional[float] = None
        self.exit_code: int = 1
        self.__state: Dict[str, Any] = {
            "step": 0,
            "tokens": {"prompt": 0, "completion": 0, "cached": 0},
            "sub_goal": "",
            "status_icon": "⠋",
        }

    # ------------------------------------------------------------------
    # Textual lifecycle
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        """
        Assemble the layout: status → body → footer.
        """

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
        """
        Seed the header with the intent, start the ticker, and kick
        off the workflow on a dedicated worker thread.
        """

        self.__started_at = time.time()
        status = self.query_one("#status", _StatusBar)
        status.intent = self.__intent
        self.set_interval(0.25, self.__tick)
        self.__run_workflow_in_thread()

    def __tick(self) -> None:
        """
        Refresh elapsed time on the header and mirror aggregate state
        into the reactive widgets. Called at ~4 Hz.
        """

        if self.__started_at is None:
            return
        status = self.query_one("#status", _StatusBar)
        status.elapsed = time.time() - self.__started_at
        status.step = int(self.__state["step"])
        status.status_icon = str(self.__state["status_icon"])
        tokens = self.__state["tokens"]
        status.tokens_prompt = int(tokens["prompt"])
        status.tokens_completion = int(tokens["completion"])
        status.tokens_cached = int(tokens["cached"])

        footer = self.query_one("#footer", _FooterBar)
        footer.sub_goal = str(self.__state["sub_goal"])

    @work(thread=True, exclusive=True, name="demo_workflow")  # type: ignore[untyped-decorator]
    def __run_workflow_in_thread(self) -> None:
        """
        Drive the supplied workflow coroutine from a worker thread.

        The agent pipeline contains several synchronous blocking calls
        (e.g. ``subprocess.run`` inside device adapters, PIL image
        decode). Running the workflow on Textual's main asyncio loop
        would freeze the UI whenever one of those calls fires. By
        giving the workflow its own thread with its own asyncio loop,
        we let those blocking calls stall the worker thread without
        affecting Textual's rendering loop.

        UI updates produced by telemetry events are marshaled back to
        the main thread via ``call_from_thread`` (see ``record_event``
        and ``show_panel``).
        """

        try:
            coro = cast("Coroutine[Any, Any, bool]", self.__workflow())
            success: bool = asyncio.run(coro)
            self.exit_code = 0 if success else 1
            self.__thread_safe_call(self.__render_workflow_finish, success=success)
        except Exception as exception:
            logger.exception("Demo workflow crashed")
            self.exit_code = 1
            self.__thread_safe_call(self.__render_workflow_error, error=str(exception))

    def __render_workflow_finish(self, *, success: bool) -> None:
        """
        Main-thread continuation that paints the final success/fail
        panel and updates the header icon after the workflow ends.
        """

        self.__state["status_icon"] = "✓" if success else "✗"
        with contextlib.suppress(Exception):
            body = self.query_one("#body", RichLog)
            body.write(
                Panel.fit(
                    f"[bold {'green' if success else 'red'}]"
                    f"{'✓ Workflow completed successfully' if success else '✗ Workflow failed'}"
                    "[/]",
                    border_style="green" if success else "red",
                )
            )

    def __render_workflow_error(self, *, error: str) -> None:
        """
        Main-thread continuation that paints a red error panel when
        the worker thread raised.
        """

        self.__state["status_icon"] = "✗"
        with contextlib.suppress(Exception):
            body = self.query_one("#body", RichLog)
            body.write(
                Panel.fit(
                    f"[bold red]Workflow error:[/bold red] {error}",
                    border_style="red",
                )
            )

    def __thread_safe_call(
        self,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Invoke ``callback`` on the main Textual thread when the app is
        running, or run it directly when the app is not running (e.g.
        in unit tests that exercise ``record_event`` / ``show_panel``
        without mounting the full app).
        """

        if bool(getattr(self, "is_running", False)):
            # ``return`` is not suppressed by ``contextlib.suppress`` —
            # it unwinds normally when the marshal succeeds. Only
            # exceptions from ``call_from_thread`` fall through to the
            # direct invocation below.
            with contextlib.suppress(Exception):
                self.call_from_thread(callback, *args, **kwargs)
                return
        callback(*args, **kwargs)

    # ------------------------------------------------------------------
    # Event ingress from TuiTelemetryAdapter
    # ------------------------------------------------------------------

    def show_panel(self, panel: Panel) -> None:
        """
        Thread-safely push a rich Panel into the scrollable body.

        Safe to call from a worker thread: marshals to the main
        Textual thread via ``call_from_thread`` when the app is
        running, or runs directly (e.g. in unit tests) otherwise.
        """

        self.__thread_safe_call(self.__show_panel_impl, panel=panel)

    def push_hitl_ask(
        self,
        prompt: str,
        future: "concurrent.futures.Future[str]",
    ) -> None:
        """
        Push an HITL ask modal bound to ``future``.

        Must run on the main Textual thread — ``TuiSignalAdapter``
        invokes this via ``call_from_thread`` from the agent's
        worker thread.
        """

        screen = _HitlAskScreen(prompt=prompt, future=future)
        self.push_screen(screen)

    def __show_panel_impl(self, *, panel: Panel) -> None:
        """
        Main-thread-only continuation that actually writes to the
        RichLog widget.
        """

        with contextlib.suppress(Exception):
            body = self.query_one("#body", RichLog)
            body.write(panel)

    def record_event(
        self,
        *,
        level: str,
        message: str,
        context: Dict[str, Any],
    ) -> None:
        """
        Thread-safe entry point for ``TuiTelemetryAdapter`` — routes a
        telemetry event into body panels + aggregate state.
        """

        self.__thread_safe_call(
            self.__record_event_impl,
            level=level,
            message=message,
            context=context,
        )

    def __record_event_impl(
        self,
        *,
        level: str,
        message: str,
        context: Dict[str, Any],
    ) -> None:
        """
        Main-thread-only continuation that renders the event panel and
        updates aggregate state.
        """

        event_type = context.get("type")

        panel = self.__panel_for_event(
            level=level, message=message, event_type=event_type, context=context
        )
        if panel is not None:
            # App not yet mounted or tearing down → swallow lookup
            # / write failures without blocking state updates.
            with contextlib.suppress(Exception):
                body = self.query_one("#body", RichLog)
                body.write(panel)

        self.__update_state_from_event(event_type=event_type, context=context)

    def __panel_for_event(
        self,
        *,
        level: str,
        message: str,
        event_type: Any,
        context: Dict[str, Any],
    ) -> Optional[Panel]:
        """
        Map a telemetry event to a ``rich.Panel`` for the body scroll.

        Delegates to the shared ``render_event_panel`` so TUI and console
        styling cannot drift.
        """

        return render_event_panel(
            event_type=event_type,
            message=message,
            context=context,
            level=level,
        )

    def __update_state_from_event(
        self,
        *,
        event_type: Any,
        context: Dict[str, Any],
    ) -> None:
        """
        Update the aggregate state dict that the ticker reads.
        """

        if event_type == FathomEvent.STEP_COMPLETED:
            step = context.get("step")
            if isinstance(step, int):
                self.__state["step"] = step

        elif event_type == FathomEvent.LLM_CALL_COMPLETED:
            metrics = context.get("metrics") or {}
            tokens = self.__state["tokens"]
            for key, metric_key in (
                ("prompt", "prompt_tokens"),
                ("completion", "completion_tokens"),
                ("cached", "cached_tokens"),
            ):
                raw = metrics.get(metric_key, 0) or 0
                try:
                    tokens[key] += int(raw)
                except (TypeError, ValueError):
                    continue

        elif event_type == FathomEvent.SUB_GOAL_STARTED:
            description = context.get("description")
            if description:
                self.__state["sub_goal"] = str(description)

        elif event_type == FathomEvent.WORKFLOW_COMPLETED:
            self.__state["status_icon"] = "✓" if bool(context.get("success", True)) else "✗"

        elif event_type == FathomEvent.WORKFLOW_CANCELLED:
            self.__state["status_icon"] = "✗"

        elif event_type == FathomEvent.HITL_REQUESTED:
            self.__state["status_icon"] = "⏸"

        elif event_type == FathomEvent.HITL_RECEIVED:
            self.__state["status_icon"] = "▶"

    # ------------------------------------------------------------------
    # Test hooks
    # ------------------------------------------------------------------

    def _state_snapshot(self) -> Dict[str, Any]:
        """
        Return a shallow copy of the aggregate state for tests.
        """

        snapshot = dict(self.__state)
        snapshot["tokens"] = dict(snapshot["tokens"])
        return snapshot
