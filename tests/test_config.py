import os
import unittest
from unittest.mock import patch

from codex_clone.config import load_config


class TestConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = load_config()
        self.assertEqual(cfg.base_url, "http://localhost:1234")
        self.assertEqual(cfg.model, "local-coder")
        self.assertAlmostEqual(cfg.temperature, 0.2)
        self.assertEqual(cfg.max_tokens, 2048)
        self.assertIsNone(cfg.api_key)

    def test_env_overrides(self) -> None:
        env = {
            "CODEX_BASE_URL": "http://127.0.0.1:9999",
            "CODEX_MODEL": "my-model",
            "CODEX_TEMPERATURE": "0.55",
            "CODEX_MAX_TOKENS": "4096",
            "CODEX_API_KEY": "secret-key",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = load_config()
        self.assertEqual(cfg.base_url, "http://127.0.0.1:9999")
        self.assertEqual(cfg.model, "my-model")
        self.assertAlmostEqual(cfg.temperature, 0.55)
        self.assertEqual(cfg.max_tokens, 4096)
        self.assertEqual(cfg.api_key, "secret-key")


if __name__ == "__main__":
    unittest.main()
