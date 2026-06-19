from __future__ import annotations

import json
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fathom.base.logger import BaseLogger


class BaseLoggerFileHandlerJsonTest(unittest.TestCase):
    """
    Pins that :meth:`BaseLogger.attach_file_handler` writes JSON regardless of
    the console renderer (ANSI / structlog dev mode must not leak into the file).
    """

    def setUp(self) -> None:
        """
        Each test gets a clean temporary directory for the file handler target.
        """

        self.__tempdir = TemporaryDirectory()
        self.__log_path = Path(self.__tempdir.name) / "run.log"

    def tearDown(self) -> None:
        """
        Detach any FileHandler created during the test and clean up the temp dir.
        """

        root = logging.getLogger()
        for handler in list(root.handlers):
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(
                self.__log_path.resolve()
            ):
                handler.close()
                root.removeHandler(handler)
        self.__tempdir.cleanup()

    def test_attach_file_handler_emits_json_lines(self) -> None:
        """
        Every line in the attached log file must parse as JSON with the
        ``event`` and ``level`` keys structlog injects via the shared chain.
        """

        BaseLogger.configure()
        BaseLogger.attach_file_handler(path=self.__log_path)

        logger = logging.getLogger("fathom.test.logger")
        logger.error(
            "abort detector misclassified response",
            extra={"event": "abort.detector.parse_failed", "confidence": 0.42},
        )

        for handler in logging.getLogger().handlers:
            handler.flush()

        lines = [line for line in self.__log_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertTrue(lines, "expected at least one JSON line in the run log")

        record = json.loads(lines[-1])
        self.assertEqual(record["event"], "abort.detector.parse_failed")
        self.assertEqual(record["level"], "error")
        self.assertEqual(record["confidence"], 0.42)

    def test_attach_file_handler_strips_console_color_codes(self) -> None:
        """
        Even when the console renderer would emit ANSI escapes, the file
        formatter must produce plain JSON without ``\\x1b[`` sequences.
        """

        BaseLogger.configure()
        BaseLogger.attach_file_handler(path=self.__log_path)

        logging.getLogger("fathom.test.logger").info(
            "captured",
            extra={"event": "test.captured"},
        )

        for handler in logging.getLogger().handlers:
            handler.flush()

        contents = self.__log_path.read_text(encoding="utf-8")
        self.assertNotIn("\x1b[", contents)
