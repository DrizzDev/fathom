from __future__ import annotations

import sys

from fathom.runtime.command.application import CommandApplication


class CommandLine:
    """
    Entry facade for Fathom command execution.
    """

    @classmethod
    def run(cls) -> int:
        """
        Run command application.
        """

        return CommandApplication().run()


def main() -> int:
    """
    Module entrypoint.
    """

    return CommandLine.run()


if __name__ == "__main__":
    sys.exit(main())
