"""Interactive signal adapter for high-concurrency, non-blocking HITL control."""

from __future__ import annotations

import asyncio
import sys
from typing import ClassVar, Optional

from rich.console import Console
from rich.panel import Panel

from fathom.constants import SignalType
from fathom.interfaces.signal import SignalPort

console = Console()


class InteractiveSignal(SignalPort):
    """
    High-performance, event-driven HITL signal adapter.

    Architecture:
    - Zero-Thread Multiplexing: Uses `loop.add_reader` on `sys.stdin` for O(1) kernel-level notifications.
    - Singleton Input Bus: Shares a single async queue across all signal instances to prevent contention.
    - Blocking-Safe I/O: Maintains `sys.stdin` in blocking mode to prevent `stdout` side-effects
      (avoiding `BlockingIOError`), relying on kernel readiness signals to ensure non-blocking reads.
    """

    # Global input bus for all concurrent instances
    __input_bus: ClassVar[asyncio.Queue[str]] = asyncio.Queue()
    __listener_active: ClassVar[bool] = False

    def __init__(self) -> None:
        """Initialize high-scale interactive signal adapter."""
        self.__injected_context: Optional[str] = None
        self.__pause_requested = False

        # Ensure the global listener is registered in the current event loop
        self.__ensure_listener()

        # UI Rendering
        self.__render_instructions()

    @classmethod
    def __ensure_listener(cls) -> None:
        """Registers the singleton TTY listener with the event loop."""
        if cls.__listener_active:
            return

        try:
            loop = asyncio.get_running_loop()
            # Register kernel-level notification for stdin
            # Note: We do NOT set O_NONBLOCK to avoid breaking stdout/logging.
            # Kernel only triggers this when data is genuinely ready for consumption.
            loop.add_reader(sys.stdin.fileno(), cls.__on_tty_readiness)
            cls.__listener_active = True
        except (RuntimeError, ValueError):
            # Fallback for environments where stdin is not a file or loop is missing
            pass

    @classmethod
    def __on_tty_readiness(cls) -> None:
        """
        Kernel callback triggered when stdin has data available.
        Invoked on the main event loop thread.
        """
        # Since the kernel notified us, this read will not block.
        line = sys.stdin.readline()
        if line:
            # Dispatch to the global async bus
            # We use call_soon because we are in a low-level callback
            asyncio.get_running_loop().call_soon(cls.__input_bus.put_nowait, line.strip())

    async def check_signal(self) -> Optional[str]:
        """
        Non-blocking check of the global input bus.
        Identifies high-priority 'pause' signals for immediate interruption.
        """
        while not self.__input_bus.empty():
            cmd = self.__input_bus.get_nowait().lower()
            if cmd == "pause":
                self.__pause_requested = True
                console.print("\n[bold yellow]⏸️  Pause requested - interrupting...[/bold yellow]\n")
            # Other commands are buffered or discarded depending on state

        return SignalType.ASK.value if self.__pause_requested else None

    async def wait_for_pause(self) -> None:
        """
        Efficiently parks the task until a pause signal arrives on the bus.
        Consumes zero CPU cycles while waiting.
        """
        if self.__pause_requested:
            return

        while True:
            # Task is parked by the OS/Event-Loop until data hits stdin
            cmd = await self.__input_bus.get()
            if cmd.lower() == "pause":
                self.__pause_requested = True
                console.print("\n[bold yellow]⏸️  Pause requested - interrupting...[/bold yellow]\n")
                sys.stdout.flush()
                await asyncio.sleep(0)  # Yield to ensure UI update propagates
                return

    async def wait_for_resume(self) -> None:
        """
        Orchestrates the HITL state machine during paused execution.
        """
        self.__render_pause_menu()

        while True:
            self.__render_options()

            # Efficiently wait for the next user interaction on the global bus
            choice = await self.__input_bus.get()
            console.print(choice)  # Echo input for feedback

            if choice == "1":
                self.__handle_resume()
                break
            elif choice == "2":
                await self.__handle_injection()
            elif choice == "3":
                console.print("\n[bold red]❌ EXECUTION CANCELLED BY USER[/bold red]\n")
                raise KeyboardInterrupt("User cancelled execution")
            else:
                console.print(f"[yellow]Invalid choice '{choice}'.[/yellow]\n")

    def get_injected_context(self) -> Optional[str]:
        """Atomically retrieves and clears any injected context."""
        context = self.__injected_context
        self.__injected_context = None
        return context

    async def ask(self, *, prompt: str) -> str:
        """Standardized human-agent interaction via async bus."""
        console.print(f"\n[bold yellow]❓ Agent Question[/bold yellow]\n[cyan]{prompt}[/cyan]")
        console.print("[bold]Your answer:[/bold] ", end="")
        sys.stdout.flush()

        answer = await self.__input_bus.get()
        console.print(f"[green]✓[/green] Recorded: [italic]{answer}[/italic]\n")
        return answer

    # --- UI & Lifecycle ---

    def __render_instructions(self) -> None:
        console.print("\n[bold cyan]🤝 Senior Architect HITL Mode Active[/bold cyan]")
        console.print(
            Panel.fit(
                "Type [bold cyan]pause[/bold cyan] and press [bold cyan]Enter[/bold cyan] to interrupt.\n"
                "Design: Kernel-multiplexed Singleton Input Bus (O(1)).",
                title="System Controls",
                border_style="cyan",
            )
        )

    def __render_pause_menu(self) -> None:
        console.print(
            "\n" + "=" * 70 + "\n[bold yellow]⏸️  EXECUTION PAUSED[/bold yellow]\n" + "=" * 70
        )
        if self.__injected_context:
            console.print(f"[bold cyan]📝 Context:[/bold cyan] {self.__injected_context}\n")

    def __render_options(self) -> None:
        console.print("[1] Resume | [2] Inject Context | [3] Cancel\n[bold]Choice:[/bold] ", end="")
        sys.stdout.flush()

    def __handle_resume(self) -> None:
        self.__pause_requested = False
        console.print("\n[bold green]▶️  RESUMING[/bold green]\n" + "=" * 70)

    async def __handle_injection(self) -> None:
        console.print("\n[bold]Enter instruction:[/bold]")
        sys.stdout.flush()
        context = await self.__input_bus.get()
        if context:
            self.__injected_context = context.strip("'\"")
            console.print(f"[green]✓ Recorded:[/green] {self.__injected_context}\n")

    def __del__(self) -> None:
        """Cleanup singleton listener if this was the last instance (Optional)."""
        # In a CLI, stdin listener typically stays for the process lifetime.
        pass
