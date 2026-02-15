"""Interactive signal adapter for human-in-the-loop control."""

from __future__ import annotations

import asyncio
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from fathom.adapters.signal.file_watcher import FileWatcher
from fathom.constants import SignalType
from fathom.interfaces.signal import SignalPort

console = Console()


class InteractiveSignal(SignalPort):
    """
    Interactive signal adapter for human-in-the-loop control.
    
    Provides real-time interaction capabilities:
    - ASK: Agent asks user for clarification when uncertain
    - INJECT: Inject additional context into execution
    - RESUME: Resume execution after providing context
    - MANUAL PAUSE: User can pause execution at any time using files
    
    This is production-grade HITL with:
    1. Automatic pause when agent is uncertain (confidence < 50%)
    2. Manual pause using file-based control (.fathom_pause)
    """

    def __init__(self) -> None:
        """Initialize interactive signal adapter."""
        self.__paused = False
        self.__injected_context: Optional[str] = None
        self.__pending_question: Optional[str] = None
        self.__user_answer: Optional[str] = None
        self.__resume_event = asyncio.Event()
        
        # File-based pause control
        self.__file_watcher = FileWatcher()
        self.__file_watcher.start()
        
        # Show instructions
        console.print("[bold cyan]🤝 Interactive HITL Mode Enabled[/bold cyan]")
        console.print("[dim]• Agent will ask questions when uncertain (confidence < 50%)[/dim]")
        console.print("[dim]• You can pause execution at ANY time using files[/dim]\n")
        console.print(Panel.fit(
            self.__file_watcher.get_instructions(),
            title="Manual Pause Instructions",
            border_style="cyan"
        ))

    async def check_signal(self) -> Optional[str]:
        """
        Check for control signal.
        
        Returns:
            Signal type: ASK or None
        """
        # Check file-based pause request
        if self.__file_watcher.is_pause_requested():
            self.__paused = True
            return SignalType.ASK.value
        
        # Check if there's a pending question from agent
        if self.__pending_question:
            return SignalType.ASK.value
        
        return None

    async def wait_for_resume(self) -> None:
        """
        Block until RESUME signal received.
        
        This is called when execution is paused. It displays an interactive
        menu allowing the user to:
        1. Resume execution
        2. Inject additional context
        3. Cancel execution
        
        Also handles file-based pause/resume.
        """
        console.print("\n[bold yellow]⏸️  Execution Paused[/bold yellow]")
        
        # Check if paused by file watcher
        if self.__file_watcher.is_pause_requested():
            console.print("[dim]Paused by file (.fathom_pause detected)[/dim]")
            console.print("[dim]Waiting for resume signal...[/dim]\n")
            
            # Wait for file-based resume
            while self.__file_watcher.is_pause_requested():
                # Check for injected context from file
                if self.__file_watcher.has_injected_context():
                    context = self.__file_watcher.get_injected_context()
                    if context:
                        self.__injected_context = context
                        console.print(f"[green]✓[/green] Context injected from file: [italic]{context}[/italic]\n")
                
                await asyncio.sleep(0.5)
            
            # Clear pause request
            self.__file_watcher.clear_pause_request()
            self.__paused = False
            console.print("[bold green]▶️  Resuming execution...[/bold green]\n")
            return
        
        # Interactive menu for agent-initiated pause
        console.print("[dim]The agent is waiting for your input...[/dim]\n")
        
        while True:
            console.print(Panel.fit(
                "[cyan]Options:[/cyan]\n"
                "  [bold]1.[/bold] Resume execution\n"
                "  [bold]2.[/bold] Inject additional context\n"
                "  [bold]3.[/bold] Cancel execution",
                title="HITL Control",
                border_style="yellow"
            ))
            
            choice = Prompt.ask(
                "Choose an option",
                choices=["1", "2", "3"],
                default="1"
            )
            
            if choice == "1":
                # Resume execution
                self.__paused = False
                self.__resume_event.set()
                console.print("[bold green]▶️  Resuming execution...[/bold green]\n")
                break
            
            elif choice == "2":
                # Inject context
                console.print("\n[bold cyan]💡 Inject Additional Context[/bold cyan]")
                console.print("[dim]Provide additional information to help the agent make better decisions.[/dim]")
                console.print("[dim]Examples:[/dim]")
                console.print("[dim]  - 'The login button is at the bottom of the screen'[/dim]")
                console.print("[dim]  - 'Use test@example.com as the email'[/dim]")
                console.print("[dim]  - 'Skip the tutorial screens'[/dim]\n")
                
                context = Prompt.ask("Enter context (or press Enter to skip)")
                
                if context.strip():
                    self.__injected_context = context.strip()
                    console.print(f"[green]✓[/green] Context injected: [italic]{context}[/italic]\n")
                else:
                    console.print("[yellow]No context provided[/yellow]\n")
            
            elif choice == "3":
                # Cancel execution
                console.print("[bold red]❌ Execution cancelled by user[/bold red]")
                raise KeyboardInterrupt("User cancelled execution")

    async def request_input(self, *, prompt: str) -> str:
        """
        Request human input with prompt.
        
        This is called when the agent needs clarification or is stuck.
        
        Args:
            prompt: Question or clarification request from agent
        
        Returns:
            User's answer
        """
        console.print(f"\n[bold yellow]❓ Agent Question[/bold yellow]")
        console.print(Panel.fit(
            f"[cyan]{prompt}[/cyan]",
            title="Agent needs your help",
            border_style="yellow"
        ))
        
        answer = Prompt.ask("Your answer")
        
        console.print(f"[green]✓[/green] Answer recorded: [italic]{answer}[/italic]\n")
        
        return answer

    def pause(self) -> None:
        """Pause execution (called externally or by user)."""
        self.__paused = True
        self.__resume_event.clear()

    def get_injected_context(self) -> Optional[str]:
        """
        Get injected context and clear it.
        
        Returns:
            Injected context string, or None if no context
        """
        # Check file watcher first
        if self.__file_watcher.has_injected_context():
            file_context = self.__file_watcher.get_injected_context()
            if file_context:
                return file_context
        
        # Then check interactive context
        context = self.__injected_context
        self.__injected_context = None  # Clear after reading
        return context

    def has_injected_context(self) -> bool:
        """Check if there's injected context available."""
        return (
            self.__injected_context is not None 
            or self.__file_watcher.has_injected_context()
        )
    
    def __del__(self) -> None:
        """Cleanup file watcher on deletion."""
        try:
            self.__file_watcher.stop()
        except Exception:
            pass
