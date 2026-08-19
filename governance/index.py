from __future__ import annotations

import subprocess  # nosec B404 - governance shells out to git to scope the source index
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
            raise GovernanceError(
                f"git file discovery failed in {repo}: {exception}"
            ) from exception

        relatives = sorted({entry for entry in completed.stdout.split("\0") if entry})
        # ``git ls-files --cached`` reports tracked paths, including files deleted from the
        # working tree. A tracked deletion is absent from the current source tree, so it is
        # excluded at discovery here. A path that exists now but disappears before the checker
        # reads it stays a fail-closed read error at parse time, never a silent skip.
        return [path for relative in relatives if (path := repo / relative).is_file()]
