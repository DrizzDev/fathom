"""File-based pause control for manual pause functionality."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional


class FileWatcher:
    """
    File-based pause control that watches for pause/resume files.
    
    This approach avoids terminal I/O conflicts by using the filesystem
    for communication between user and agent.
    
    Usage:
    - User creates .fathom_pause file to pause
    - User creates .fathom_context file with context to inject
    - User deletes .fathom_pause file to resume
    """
    
    def __init__(self, watch_dir: Optional[str] = None) -> None:
        """
        Initialize file watcher.
        
        Args:
            watch_dir: Directory to watch for control files (defaults to current directory)
        """
        self.__watch_dir = Path(watch_dir) if watch_dir else Path.cwd()
        self.__pause_file = self.__watch_dir / ".fathom_pause"
        self.__context_file = self.__watch_dir / ".fathom_context"
        self.__resume_file = self.__watch_dir / ".fathom_resume"
        
        self.__pause_requested = False
        self.__injected_context: Optional[str] = None
        self.__stop_requested = False
        self.__watcher_thread: Optional[threading.Thread] = None
        
    def start(self) -> None:
        """Start watching for control files in background thread."""
        self.__watcher_thread = threading.Thread(
            target=self.__watch_loop,
            daemon=True,
            name="FileWatcher"
        )
        self.__watcher_thread.start()
    
    def stop(self) -> None:
        """Stop the file watcher."""
        self.__stop_requested = True
        if self.__watcher_thread:
            self.__watcher_thread.join(timeout=1.0)
        
        # Cleanup control files
        self.__cleanup_files()
    
    def is_pause_requested(self) -> bool:
        """Check if pause was requested."""
        return self.__pause_requested
    
    def clear_pause_request(self) -> None:
        """Clear the pause request flag."""
        self.__pause_requested = False
    
    def get_injected_context(self) -> Optional[str]:
        """Get and clear injected context."""
        context = self.__injected_context
        self.__injected_context = None
        return context
    
    def has_injected_context(self) -> bool:
        """Check if there's injected context available."""
        return self.__injected_context is not None
    
    def __watch_loop(self) -> None:
        """Background loop watching for control files."""
        while not self.__stop_requested:
            try:
                # Check for pause file
                if self.__pause_file.exists() and not self.__pause_requested:
                    self.__pause_requested = True
                
                # Check for context file
                if self.__context_file.exists() and not self.__injected_context:
                    try:
                        context = self.__context_file.read_text().strip()
                        if context:
                            self.__injected_context = context
                        # Delete context file after reading
                        self.__context_file.unlink()
                    except Exception:
                        pass
                
                # Check for resume file
                if self.__resume_file.exists():
                    # Delete pause file to resume
                    if self.__pause_file.exists():
                        self.__pause_file.unlink()
                    # Delete resume file
                    self.__resume_file.unlink()
                    self.__pause_requested = False
                
                # Sleep briefly
                time.sleep(0.1)
                
            except Exception:
                # Silently continue on errors
                pass
    
    def __cleanup_files(self) -> None:
        """Cleanup control files."""
        try:
            if self.__pause_file.exists():
                self.__pause_file.unlink()
            if self.__context_file.exists():
                self.__context_file.unlink()
            if self.__resume_file.exists():
                self.__resume_file.unlink()
        except Exception:
            pass
    
    def get_instructions(self) -> str:
        """Get user instructions for manual pause."""
        return f"""
Manual Pause Instructions:
--------------------------
To pause execution:
  touch {self.__pause_file}

To inject context (while paused):
  echo "Your context here" > {self.__context_file}

To resume execution:
  touch {self.__resume_file}
  (or delete {self.__pause_file})

Example:
  # Pause
  touch .fathom_pause
  
  # Inject context
  echo "Open ChatGPT app and ask it to research opencrawler" > .fathom_context
  
  # Resume
  touch .fathom_resume
"""
