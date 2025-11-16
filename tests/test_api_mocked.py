import json
import unittest
from unittest.mock import patch, MagicMock

from codex_clone.api import send_chat, CodexError
from codex_clone.config import Config


class TestApiMocked(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config(
            base_url="http://localhost:1234",
            api_key=None,
            model="local-coder",
            system_prompt="",
            temperature=0.1,
            max_tokens=128,
        )

    @patch("codex_clone.api.urllib.request.urlopen")
    def test_send_chat_success(self, mock_urlopen) -> None:
        messages = [{"role": "user", "content": "hi"}]
        body = json.dumps(
            {
                "choices": [
                    {"message": {"content": "hello back"}},
                ]
            }
        ).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_resp
        mock_ctx.__exit__.return_value = False
        mock_urlopen.return_value = mock_ctx

        reply = send_chat(messages, self.config)
        self.assertEqual(reply, "hello back")

    @patch("codex_clone.api.urllib.request.urlopen")
    def test_send_chat_error(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = OSError("connection refused")
        with self.assertRaises(CodexError):
            send_chat([{"role": "user", "content": "hi"}], self.config)


if __name__ == "__main__":
    unittest.main()
