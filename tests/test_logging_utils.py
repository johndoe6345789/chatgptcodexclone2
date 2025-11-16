import unittest
from pathlib import Path

from codex_clone import logging_utils


class TestLoggingUtils(unittest.TestCase):
    def test_log_line_creates_file_and_writes(self) -> None:
        # Ensure we start from a clean file if it exists.
        log_path = Path(logging_utils.__file__).resolve().parent.parent / "codex.log"
        if log_path.exists():
            log_path.unlink()

        logging_utils.log_line("test line 123")
        self.assertTrue(log_path.exists(), "codex.log should be created")

        data = log_path.read_text(encoding="utf-8")
        self.assertIn("test line 123", data)
        self.assertIn("[", data)  # timestamp prefix
        self.assertIn("]", data)


if __name__ == "__main__":
    unittest.main()
