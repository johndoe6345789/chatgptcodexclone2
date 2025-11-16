import unittest
import queue

from codex_portable import SocketBackendClient  # type: ignore[import]


class DummyQueue(queue.Queue):
    def put(self, item, block=True, timeout=None):  # type: ignore[override]
        super().put(item, block=block, timeout=timeout)


class TestSocketBackendClientBasic(unittest.TestCase):
    def test_req_id_increments(self) -> None:
        backend_log = DummyQueue()
        chat = DummyQueue()
        client = SocketBackendClient(backend_log, chat)

        # Accessing protected member only for testing ID behaviour.
        first = client._next_req_id()
        second = client._next_req_id()
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("req-"))
        self.assertTrue(second.startswith("req-"))


if __name__ == "__main__":
    unittest.main()
