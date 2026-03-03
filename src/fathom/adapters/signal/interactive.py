from __future__ import annotations

import asyncio
import logging
import sys
from typing import ClassVar, Optional

from rich.console import Console
from rich.panel import Panel

from fathom.constants import SignalType
from fathom.interfaces.signal import SignalPort

console = Console()
logger = logging.getLogger(__name__)


class SharedInputBus:
    """
    Broadcast-capable input bus for stdin.
    Allows multiple concurrent listeners to receive and filter the same input stream.
    """

    def __init__(self) -> None:
        self.__condition = asyncio.Condition()
        self.__latest_input: Optional[str] = None

    async def put(self, value: str) -> None:
        """
        Broadcast a new value to all waiting listeners.
        """

        async with self.__condition:
            self.__latest_input = value
            self.__condition.notify_all()

    async def get(self) -> str:
        """
        Wait for the next broadcasted value.
        """

        async with self.__condition:
            await self.__condition.wait()
            return self.__latest_input or ""


class InteractiveSignal(SignalPort):
    """
    High-performance, event-driven HITL signal adapter.

    Architecture:
    - Zero-Thread Multiplexing: Uses `loop.add_reader` on `sys.stdin` for O(1) kernel-level notifications.
    - Singleton Input Bus: Broadcasts input to all active listeners (preventing theft).
    """

    # Global input bus for all concurrent instances
    __listener_active: ClassVar[bool] = False
    __input_bus: ClassVar[SharedInputBus] = SharedInputBus()

    def __init__(self) -> None:
        """
        Initialize high-scale interactive signal adapter.
        """

        self.__pause_requested = False
        self.__injected_context: Optional[str] = None

        # Ensure the global listener is registered in the current event loop
        self.__ensure_listener()

        # UI Rendering
        self.__render_instructions()

    @classmethod
    def __ensure_listener(cls) -> None:
        """
        Registers the singleton TTY listener with the event loop.
        """

        if cls.__listener_active:
            return

        try:
            loop = asyncio.get_running_loop()
            loop.add_reader(sys.stdin.fileno(), cls.__on_tty_readiness)
            cls.__listener_active = True
        except (RuntimeError, ValueError):
            pass

    @classmethod
    def __on_tty_readiness(cls) -> None:
        """
        Kernel callback triggered when stdin has data available.
        """

        line = sys.stdin.readline()

        # EOF detection
        if line == "":
            try:
                loop = asyncio.get_running_loop()
                loop.remove_reader(sys.stdin.fileno())
                cls.__listener_active = False
            except Exception:  # nosec
                pass
            return

        if line is not None:
            content = line.rstrip("\n").strip()
            # Push to the broadcast bus
            asyncio.get_running_loop().create_task(cls.__input_bus.put(content))

    async def check_signal(self) -> Optional[str]:
        """
        Checks if pause is currently requested.
        """

        return SignalType.ASK.value if self.__pause_requested else None

    async def wait_for_pause(self) -> None:
        """
        Efficiently parks the task until a pause signal arrives.
        """

        if self.__pause_requested:
            return

        logger.debug("Waiting for pause command...")

        while True:
            cmd = await self.__input_bus.get()
            if cmd.lower() == "pause":
                self.__pause_requested = True
                console.print("\n[bold yellow]⏸️  Pause requested - interrupting...[/bold yellow]\n")
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

            if not choice:
                continue

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

    async def get_injected_context(self) -> Optional[str]:
        """
        DEPRECATED: Use peek_next_context and consume_context.
        Atomically retrieves and clears any injected context.
        """

        context = self.__injected_context
        self.__injected_context = None
        return context

    async def peek_next_context(self) -> Optional[str]:
        """
        Peek at the current injected context.
        """

        return self.__injected_context

    async def consume_context(self) -> None:
        """
        Clear the current injected context.
        """

        self.__injected_context = None

    async def is_pause_requested(self) -> bool:
        """
        Check if pause is requested.
        """

        return self.__pause_requested

    async def has_injected_context(self) -> bool:
        """
        Check if there is injected context available.
        """

        return self.__injected_context is not None

    async def ask(self, *, prompt: str) -> str:
        """
        Standardized human-agent interaction via broadcast bus.
        """

        console.print(f"\n[bold yellow]❓ Agent Question[/bold yellow]\n[cyan]{prompt}[/cyan]")
        console.print("[bold]Your answer:[/bold] ", end="")
        sys.stdout.flush()

        try:
            while True:
                answer = await asyncio.wait_for(self.__input_bus.get(), timeout=60.0)
                if answer and answer.lower() != "pause":
                    console.print(f"[green]✓[/green] Recorded: [italic]{answer}[/italic]\n")
                    return answer
        except asyncio.TimeoutError:
            timeout_msg = (
                "SYSTEM: User did not respond within 60s. Proceed autonomously if possible. "
                "Analyze the current screen and determine if you can continue toward the goal "
                "or if you must mark the task as failed."
            )
            console.print(f"\n[bold yellow]⏳ Timeout: {timeout_msg}[/bold yellow]\n")
            return timeout_msg

    def __render_instructions(self) -> None:
        """
        Log Instructions
        """

        console.print("\n[bold cyan]HITL Mode Active[/bold cyan]")
        console.print(
            Panel.fit(
                "Type [bold cyan]pause[/bold cyan] and press [bold cyan]Enter[/bold cyan] to interrupt.\n"
                "Design: Kernel-multiplexed Singleton Input Bus (O(1)).",
                title="System Controls",
                border_style="cyan",
            )
        )

    def __render_pause_menu(self) -> None:
        """
        Render Menu (For Pause)
        """

        console.print(
            "\n" + "=" * 70 + "\n[bold yellow]⏸️  EXECUTION PAUSED[/bold yellow]\n" + "=" * 70
        )
        if self.__injected_context:
            console.print(f"[bold cyan]📝 Context:[/bold cyan] {self.__injected_context}\n")

    def __render_options(self) -> None:
        """
        Renders HITL Options
        """

        console.print("[1] Resume | [2] Inject Context | [3] Cancel\n[bold]Choice:[/bold] ", end="")
        sys.stdout.flush()

    def __handle_resume(self) -> None:
        """
        Handler For Resume Event
        """

        logger.info(f"InteractiveSignal: Clearing pause flag (was: {self.__pause_requested})")
        self.__pause_requested = False
        console.print("\n[bold green]▶️  RESUMING[/bold green]\n" + "=" * 70)

    async def __handle_injection(self) -> None:
        """
        Handler For Injection Event
        """

        console.print("\n[bold]Enter instruction:[/bold]")
        sys.stdout.flush()

        if context := await self.__input_bus.get():
            self.__injected_context = context.strip("'\"")
            console.print(f"[green]✓ Recorded:[/green] {self.__injected_context}\n")

    def __del__(self) -> None:
        """
        Cleanup singleton listener if this was the last instance (Optional).
        """

        # In a CLI, stdin listener typically stays for the process lifetime.
        pass
