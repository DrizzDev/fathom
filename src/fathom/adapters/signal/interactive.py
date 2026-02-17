"""Interactive signal adapter for human-in-the-loop control."""

from __future__ import annotations

import asyncio
import queue
import sys
import threading
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from fathom.constants import SignalType
from fathom.interfaces.signal import SignalPort

console = Console()


class InteractiveSignal(SignalPort):
    """Interactive signal with robust threaded input handling."""

    def __init__(self) -> None:
        """Initialize interactive signal adapter."""
        self.__paused = False
        self.__injected_context: Optional[str] = None
        self.__resume_event = asyncio.Event()
        self.__pause_requested = False
        self.__command_queue: queue.Queue[str] = queue.Queue()
        self.__stop_listener = False

        # Start command listener thread
        # This thread is the EXCLUSIVE consumer of sys.stdin to avoid race conditions
        self.__listener_thread = threading.Thread(
            target=self.__listen_for_commands, daemon=True, name="HITLCommandListener"
        )
        self.__listener_thread.start()

        # Show instructions
        console.print("\n[bold cyan]🤝 Interactive HITL Mode Enabled[/bold cyan]")
        console.print("[dim]• Agent will ask questions when uncertain (confidence < 50%)[/dim]")
        console.print("[dim]• Type 'pause' and press Enter to pause IMMEDIATELY[/dim]")
        console.print("[dim]• Press Ctrl+C to cancel execution[/dim]\n")

        console.print(
            Panel.fit(
                "[bold yellow]To Pause Manually:[/bold yellow]\n"
                "1. Type: [bold cyan]pause[/bold cyan]\n"
                "2. Press: [bold cyan]Enter[/bold cyan]\n"
                "3. Agent pauses immediately (even during LLM calls)",
                title="Manual Pause Instructions",
                border_style="cyan",
            )
        )
        console.print()

    def __listen_for_commands(self) -> None:
        """Background thread that continuously reads stdin."""
        while not self.__stop_listener:
            try:
                # Blocking read (thread-safe)
                if sys.stdin:
                    line = sys.stdin.readline()
                    if line:
                        self.__command_queue.put(line.strip())
            except Exception:
                break

    async def check_signal(self) -> Optional[str]:
        """Check for control signal. Flushes queue looking for 'pause'."""
        try:
            # Drain queue to check for 'pause' command
            # We only care about 'pause' when running. Other input is ignored/discarded.
            while not self.__command_queue.empty():
                cmd = self.__command_queue.get_nowait().lower()
                if cmd == "pause":
                    self.__pause_requested = True
                    console.print(
                        "\n[bold yellow]⏸️  Pause requested - pausing immediately...[/bold yellow]\n"
                    )
        except queue.Empty:
            pass

        if self.__pause_requested:
            return SignalType.ASK.value

        return None

    async def __get_input_async(self) -> str:
        """Async wrapper to poll the command queue."""
        while True:
            try:
                # Non-blocking check
                return self.__command_queue.get_nowait()
            except queue.Empty:
                # Small sleep to yield to event loop
                await asyncio.sleep(0.1)

    async def wait_for_resume(self) -> None:
        """Block until RESUME signal received. Handles interactive menu."""
        console.print("\n" + "=" * 70)
        console.print("[bold yellow]⏸️  EXECUTION PAUSED[/bold yellow]")
        console.print("=" * 70 + "\n")

        # Show current context if any
        if self.__injected_context:
            console.print("[bold cyan]📝 Current Context:[/bold cyan]")
            console.print(f"[dim]{self.__injected_context}[/dim]\n")

        while True:
            console.print(
                Panel.fit(
                    "[cyan]Options:[/cyan]\n"
                    "  [bold]1.[/bold] Resume execution\n"
                    "  [bold]2.[/bold] Inject additional context\n"
                    "  [bold]3.[/bold] Cancel execution",
                    title="HITL Control",
                    border_style="yellow",
                )
            )

            console.print("\n[bold]Your choice (1/2/3):[/bold] ", end="")
            sys.stdout.flush()

            # Wait for input from queue
            choice = await self.__get_input_async()
            console.print(choice)  # Echo input

            if choice not in ["1", "2", "3"]:
                if not choice:
                    choice = "1"  # Default
                else:
                    console.print(
                        f"[yellow]Invalid choice '{choice}'. Please enter 1, 2, or 3.[/yellow]\n"
                    )
                    continue

            console.print(f"[green]→ You chose: {choice}[/green]")

            if choice == "1":
                # Resume
                self.__paused = False
                self.__pause_requested = False
                self.__resume_event.set()

                console.print("\n" + "=" * 70)
                console.print("[bold green]▶️  RESUMING EXECUTION[/bold green]")
                if self.__injected_context:
                    console.print(
                        f"[bold cyan]📝 With Context:[/bold cyan] [italic]{self.__injected_context}[/italic]"
                    )
                console.print("=" * 70 + "\n")
                break

            elif choice == "2":
                # Inject context
                console.print("\n" + "-" * 70)
                console.print("[bold cyan]💡 INJECT ADDITIONAL CONTEXT[/bold cyan]")
                console.print("-" * 70)
                console.print("[bold]Enter your instruction:[/bold]")
                sys.stdout.flush()

                context = await self.__get_input_async()
                console.print(context)  # Echo

                if context and (
                    (context.startswith("'") and context.endswith("'"))
                    or (context.startswith('"') and context.endswith('"'))
                ):
                    context = context[1:-1]

                if context:
                    self.__injected_context = context
                    console.print("\n[green]✓ Instruction Recorded[/green]")
                    console.print(f"[bold cyan]New:[/bold cyan] {context}")
                    console.print("[dim]This will be sent to the LLM with priority.[/dim]\n")
                else:
                    console.print("[yellow]⚠ No instruction provided[/yellow]\n")

            elif choice == "3":
                console.print("\n[bold red]❌ EXECUTION CANCELLED BY USER[/bold red]\n")
                raise KeyboardInterrupt("User cancelled execution")

    async def ask(self, *, prompt: str) -> str:
        """Request human input with prompt."""
        console.print("\n[bold yellow]❓ Agent Question[/bold yellow]")
        console.print(
            Panel.fit(
                f"[cyan]{prompt}[/cyan]", title="Agent needs your help", border_style="yellow"
            )
        )
        console.print("[bold]Your answer:[/bold] ", end="")
        sys.stdout.flush()

        answer = await self.__get_input_async()
        console.print(f"[green]✓[/green] Answer recorded: [italic]{answer}[/italic]\n")
        return answer

    def is_pause_requested(self) -> bool:
        """Check if pause is requested (for immediate cancellation)."""
        return self.__pause_requested

    def pause(self) -> None:
        """Pause execution."""
        self.__paused = True
        self.__pause_requested = True
        self.__resume_event.clear()

    def get_injected_context(self) -> Optional[str]:
        """Get injected context and clear it."""
        context = self.__injected_context
        self.__injected_context = None
        return context

    def has_injected_context(self) -> bool:
        """Check if there's injected context available."""
        return self.__injected_context is not None

    def __del__(self) -> None:
        """Cleanup listener thread."""
        self.__stop_listener = True
