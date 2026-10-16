#!/usr/bin/env python3
"""The Nemotron lane (NVIDIA's Bolt on vLLM, proxy v6.40): Bolt's chat template reads
enable_thinking and medium_effort, and no reasoning effort. Measured on the reference box
(2026-10-15, vLLM nightly 77154ca6): a top-level reasoning_effort "none" sent as it came left the
answer inside the reasoning, its content null; enable_thinking false answered "391" in 4 tokens.
pi and the OpenAI SDKs send the field, so the proxy says it in the template's words: none and
minimal turn the reasoning off, low and medium ask the shorter one, high and above the default.
A request that names enable_thinking keeps its choice, and every other lane, route or doubt goes
as it came (the Kolibri lane's template takes "none" itself)."""
import http.server
import io
import json
import os
import sys
import threading
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_proxy_vllm_lane import KEY, FakeVllm, _keep_env, load_proxy  # noqa: E402

BOLT = "nemotron-3.5-bolt"
CHAT = "/v1/chat/completions"


class TheTranslation(unittest.TestCase):
    """nemotron_effort alone: what each level becomes, and what it leaves alone."""

    @classmethod
    def setUpClass(cls):
        _keep_env(cls)
        os.environ["UPSTREAM"] = "http://127.0.0.1:9"
        cls.mod = load_proxy("kproxy_nemotron_effort_pure")

    def setUp(self):
        self.mod._SERVED.update(names=(BOLT,), engine="vllm", ts=time.time())

    def go(self, body, path=CHAT):
        raw = json.dumps(body).encode() if not isinstance(body, bytes) else body
        out, level = self.mod.nemotron_effort(raw, path)
        return (json.loads(out) if out != raw else None), level

    def test_each_level_becomes_what_bolts_template_reads(self):
        want = {"none": {"enable_thinking": False}, "minimal": {"enable_thinking": False},
                "low": {"enable_thinking": True, "medium_effort": True},
                "medium": {"enable_thinking": True, "medium_effort": True},
                "high": {"enable_thinking": True}, "xhigh": {"enable_thinking": True}, "max": {"enable_thinking": True}}
        for level, kw in want.items():
            for where in ("top", "kwargs"):
                with self.subTest(level=level, where=where):
                    body = {"model": BOLT, "messages": [{"role": "user", "content": "hi"}]}
                    if where == "top":
                        body["reasoning_effort"] = level
                    else:
                        body["chat_template_kwargs"] = {"reasoning_effort": level}
                    sent, got = self.go(body)
                    self.assertEqual(got, level)
                    self.assertEqual(sent["chat_template_kwargs"], kw)
                    self.assertNotIn("reasoning_effort", sent)

    def test_the_kwargs_level_wins_over_the_top_level_one_and_both_go(self):
        sent, got = self.go({"model": BOLT, "reasoning_effort": "high", "messages": [],
                             "chat_template_kwargs": {"reasoning_effort": "none", "keep": 1}})
        self.assertEqual(got, "none")
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": False, "keep": 1})
        self.assertNotIn("reasoning_effort", sent)

    def test_a_request_that_names_enable_thinking_keeps_its_choice(self):
        sent, got = self.go({"model": BOLT, "reasoning_effort": "none", "messages": [],
                             "chat_template_kwargs": {"enable_thinking": True, "medium_effort": True}})
        self.assertEqual(got, "none")
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": True, "medium_effort": True})
        self.assertNotIn("reasoning_effort", sent)      # the field vLLM mishandles goes all the same

    def test_another_lane_route_level_or_shape_goes_as_it_came(self):
        body = {"model": BOLT, "reasoning_effort": "none", "messages": []}
        for names, engine in (((BOLT,), "sglang"), (("kolibri-1",), "vllm"), (("qwen3.8-27b",), "sglang"), ((), ""),
                              (("nemotron",), "vllm"), (("nvidia-llama",), "vllm")):
            with self.subTest(names=names, engine=engine):
                self.mod._SERVED.update(names=names, engine=engine)
                self.assertEqual(self.go(body), (None, None))
        self.mod._SERVED.update(names=(BOLT,), engine="vllm")
        self.assertEqual(self.go(body, path="/v1/completions"), (None, None))
        self.assertEqual(self.go({"model": BOLT, "messages": []}), (None, None))
        self.assertEqual(self.go({"model": BOLT, "reasoning_effort": "turbo", "messages": []}), (None, None))
        self.assertEqual(self.go({"model": BOLT, "reasoning_effort": "none", "chat_template_kwargs": "x"}), (None, None))
        self.assertEqual(self.go(b'{"reasoning_effort": "none", not json'), (None, None))
        self.assertEqual(self.go(b'["reasoning_effort"]'), (None, None))


class ThroughTheProxy(unittest.TestCase):
    """A fake vLLM serving nemotron-3.5-bolt behind the real proxy."""

    @classmethod
    def setUpClass(cls):
        _keep_env(cls)
        cls.engine = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeVllm)
        cls.engine.daemon_threads = True
        threading.Thread(target=cls.engine.serve_forever, daemon=True).start()
        os.environ.update(UPSTREAM=f"http://127.0.0.1:{cls.engine.server_port}", KEEPALIVE_S="0.4",
                          CLIENT_IO_S="5", PROMPT_CEILING_TOKENS="0", FLASH_PROMPT_CEILING_TOKENS="0")
        cls.mod = load_proxy("kproxy_nemotron_effort")
        cls.mod._api_key = lambda: KEY
        cls.lines = []
        cls.mod.log = lambda msg: cls.lines.append(f"[proxy] {msg}")
        cls.err = redirect_stderr(io.StringIO())
        cls.err.__enter__()
        cls.proxy = cls.mod.Server(("127.0.0.1", 0), cls.mod.H)
        threading.Thread(target=cls.proxy.serve_forever, daemon=True).start()
        cls.port = cls.proxy.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.err.__exit__(None, None, None)
        cls.proxy.shutdown()
        cls.proxy.server_close()
        cls.engine.shutdown()
        cls.engine.server_close()
        FakeVllm.served, FakeVllm.owner = "kolibri-1", "vllm"

    def post(self, body):
        import socket
        raw = json.dumps(body).encode()
        s = socket.create_connection(("127.0.0.1", self.port), timeout=20)
        s.sendall(f"POST {CHAT} HTTP/1.1\r\nHost: p\r\nAuthorization: Bearer {KEY}\r\n"
                  f"Content-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n".encode() + raw)
        s.settimeout(20)
        got = b""
        while True:
            c = s.recv(65536)
            if not c:
                break
            got += c
        s.close()
        return got

    def test_effort_none_reaches_bolt_as_thinking_off_and_kolibri_as_it_came(self):
        for served, want in ((BOLT, {"chat_template_kwargs": {"enable_thinking": False}}),
                             ("kolibri-1", {"reasoning_effort": "none"})):
            with self.subTest(served=served):
                FakeVllm.served, FakeVllm.owner = served, "vllm"
                with FakeVllm.lock:
                    FakeVllm.bodies.clear()
                self.mod.invalidate_pool()
                self.mod.served_models()
                got = self.post({"model": served, "stream": True, "reasoning_effort": "none",
                                 "messages": [{"role": "user", "content": "What is 17 * 23?"}]})
                self.assertTrue(got.startswith(b"HTTP/1.1 200"), got[:200])
                sent = FakeVllm.bodies[-1]
                self.assertEqual({k: sent.get(k) for k in want}, want)
                if served == BOLT:
                    self.assertNotIn("reasoning_effort", sent)
        self.assertTrue(any("not what the Nemotron template reads" in ln for ln in self.lines), self.lines[-5:])

    def test_the_journal_says_what_each_level_became(self):
        # it said "enable_thinking and medium_effort" whatever the level (found in review, 2026-10-16)
        FakeVllm.served, FakeVllm.owner = BOLT, "vllm"
        self.mod.invalidate_pool()
        self.mod.served_models()
        want = {"minimal": "enable_thinking false", "low": "enable_thinking and medium_effort",
                "xhigh": "enable_thinking alone"}
        for level in want:
            got = self.post({"model": BOLT, "reasoning_effort": level, "max_tokens": 8,
                             "messages": [{"role": "user", "content": "hi"}]})
            self.assertTrue(got.startswith(b"HTTP/1.1 200"), got[:200])
        for level, how in want.items():
            with self.subTest(level=level):
                said = [ln for ln in self.lines if f"reasoning_effort {level} is not what the Nemotron" in ln]
                self.assertEqual(len(said), 1, self.lines[-8:])
                self.assertIn(f"relayed as {how}, unless", said[0])


if __name__ == "__main__":
    unittest.main()
