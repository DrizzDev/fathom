from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from governance.errors import GovernanceError


class SourceIndex(ABC):
    """
    Enumerates the first-party Python modules a fitness run must govern.
    """

    @abstractmethod
    def paths(self, *, repo: Path) -> List[Path]:
        """
        Return the absolute paths of every governed module under the repository.
        """


class GitIndex(SourceIndex):
    """
    Discovers first-party modules from git: tracked plus untracked-non-ignored files.

    This defers the source-versus-external decision to the repository's own ignore rules
    rather than a directory-name allowlist, and fails closed when git cannot enumerate.
    """

    __COMMAND: List[str] = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        "*.py",
    ]

    def paths(self, *, repo: Path) -> List[Path]:
        """
        Return every first-party Python module under the repository as reported by git.
        """

        try:
            # Fixed git argv, no shell and no user-controlled input; git is resolved from PATH.
            completed = subprocess.run(  # nosec B603 B607
                [self.__COMMAND[0], "-C", str(repo), *self.__COMMAND[1:]],
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as exception:
            raise GovernanceError(f"git file discovery failed in {repo}: {exception}") from exception

        relatives = sorted({entry for entry in completed.stdout.split("\0") if entry})
        return [repo / relative for relative in relatives]
