from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Set

from governance.errors import GovernanceError
from governance.index import GitIndex


class GitIndexTest(unittest.TestCase):
    """
    Pins git-based first-party discovery: scope follows the repository's ignore rules, not
    directory-name heuristics, and fails closed outside a git repository.
    """

    @staticmethod
    def __init_repo(*, root: Path) -> None:
        """
        Initialize an empty git repository at the given root.
        """

        subprocess.run(["git", "init", "-q", str(root)], check=True)

    @staticmethod
    def __write(*, root: Path, relative: str, code: str = "value = 1\n") -> None:
        """
        Write a module at a repository-relative path.
        """

        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)

    def __relatives(self, *, root: Path) -> Set[str]:
        """
        Return the repository-relative paths the index reports.
        """

        return {str(path.relative_to(root)) for path in GitIndex().paths(repo=root)}

    def test_includes_root_level_and_reserved_name_directories(self) -> None:
        """
        Root-level modules and directories named like reserved patterns are first-party by default.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.__init_repo(root=root)
            self.__write(root=root, relative="example.py")
            self.__write(root=root, relative="tools/probe.py")
            self.__write(root=root, relative=".internal/probe.py")
            self.__write(root=root, relative="src/fathom/build/probe.py")

            relatives = self.__relatives(root=root)

            self.assertIn("example.py", relatives)
            self.assertIn("tools/probe.py", relatives)
            self.assertIn(".internal/probe.py", relatives)
            self.assertIn("src/fathom/build/probe.py", relatives)

    def test_excludes_gitignored_directories(self) -> None:
        """
        A gitignored directory is excluded because git's own ignore rules drive discovery.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.__init_repo(root=root)
            (root / ".gitignore").write_text(".venv/\n")
            self.__write(root=root, relative=".venv/lib/vendor.py")
            self.__write(root=root, relative="src/fathom/keep.py")

            relatives = self.__relatives(root=root)

            self.assertIn("src/fathom/keep.py", relatives)
            self.assertNotIn(".venv/lib/vendor.py", relatives)

    def test_includes_force_tracked_file_in_ignored_directory(self) -> None:
        """
        A file force-tracked into an otherwise-ignored directory is still first-party.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.__init_repo(root=root)
            (root / ".gitignore").write_text("secret/\n")
            self.__write(root=root, relative="secret/tracked.py")
            subprocess.run(["git", "-C", str(root), "add", "-f", "secret/tracked.py"], check=True)

            self.assertIn("secret/tracked.py", self.__relatives(root=root))

    def test_reports_unusual_filenames_verbatim(self) -> None:
        """
        A non-ASCII filename is reported literally, not in git's quoted-path escaping.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.__init_repo(root=root)
            self.__write(root=root, relative="src/fathom/naïve.py")

            self.assertIn("src/fathom/naïve.py", self.__relatives(root=root))

    def test_excludes_tracked_file_deleted_from_working_tree(self) -> None:
        """
        A file tracked in the index but deleted from disk is absent from discovery, never opened.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.__init_repo(root=root)
            self.__write(root=root, relative="src/fathom/keep.py")
            self.__write(root=root, relative="src/fathom/gone.py")
            subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
            (root / "src/fathom/gone.py").unlink()

            relatives = self.__relatives(root=root)

            self.assertIn("src/fathom/gone.py", self.__git_ls_files(root=root))
            self.assertIn("src/fathom/keep.py", relatives)
            self.assertNotIn("src/fathom/gone.py", relatives)

    def test_includes_new_untracked_existing_file(self) -> None:
        """
        A new, untracked, on-disk first-party module is still discovered for scanning.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.__init_repo(root=root)
            self.__write(root=root, relative="src/fathom/fresh.py")

            self.assertIn("src/fathom/fresh.py", self.__relatives(root=root))

    def test_fails_closed_outside_a_git_repository(self) -> None:
        """
        Discovery raises rather than scanning nothing when git cannot enumerate.
        """

        with tempfile.TemporaryDirectory() as directory, self.assertRaises(GovernanceError):
            GitIndex().paths(repo=Path(directory))

    @staticmethod
    def __git_ls_files(*, root: Path) -> Set[str]:
        """
        Return the raw tracked-plus-untracked paths git reports, before existence filtering.
        """

        completed = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                "*.py",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return {entry for entry in completed.stdout.splitlines() if entry}
