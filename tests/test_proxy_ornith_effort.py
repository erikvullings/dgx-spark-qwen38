#!/usr/bin/env python3
"""The Ornith 1.5 122B lane (Ornith AI's model on vLLM, proxy v6.45): Ornith's chat template reads
enable_thinking and no reasoning effort, so a "none" handed to it as it came left the reasoning on.
The box maps the levels onto it (2026-10-21): none and minimal turn the reasoning off, every other
level keeps it whole. none and minimal win over the request's own enable_thinking, as on the Qwen
lanes (omp sends enable_thinking true beside its level), a level that keeps it leaves the request's
own choice, and every other lane, route or doubt goes as it came."""
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

ORNITH = "ornith-1.5-122b"
CHAT = "/v1/chat/completions"


class TheTranslation(unittest.TestCase):
    """ornith_effort alone: what each level becomes, and what it leaves alone."""

    @classmethod
    def setUpClass(cls):
        _keep_env(cls)
        os.environ["UPSTREAM"] = "http://127.0.0.1:9"
        cls.mod = load_proxy("kproxy_ornith_effort_pure")

    def setUp(self):
        self.mod._SERVED.update(names=(ORNITH,), engine="vllm", ts=time.time())

    def go(self, body, path=CHAT):
        raw = json.dumps(body).encode() if not isinstance(body, bytes) else body
        out, level = self.mod.ornith_effort(raw, path)
        return (json.loads(out) if out != raw else None), level

    def test_none_and_minimal_turn_it_off_and_every_other_level_keeps_it(self):
        want = {"none": False, "minimal": False, "low": True, "medium": True, "high": True, "xhigh": True, "max": True}
        for level, on in want.items():
            for where in ("top", "kwargs"):
                with self.subTest(level=level, where=where):
                    body = {"model": ORNITH, "messages": [{"role": "user", "content": "hi"}]}
                    if where == "top":
                        body["reasoning_effort"] = level
                    else:
                        body["chat_template_kwargs"] = {"reasoning_effort": level}
                    sent, got = self.go(body)
                    self.assertEqual(got, level)
                    self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": on})
                    self.assertNotIn("reasoning_effort", sent)

    def test_the_kwargs_level_wins_over_the_top_level_one_and_both_go(self):
        sent, got = self.go({"model": ORNITH, "reasoning_effort": "high", "messages": [],
                             "chat_template_kwargs": {"reasoning_effort": "none", "keep": 1}})
        self.assertEqual(got, "none")
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": False, "keep": 1})
        self.assertNotIn("reasoning_effort", sent)

    def test_none_and_minimal_win_over_the_requests_own_enable_thinking(self):
        # omp, as pi-gen.py writes its config, sends enable_thinking true beside its level: its minimal
        # (none for this lane) reasoned in full (found in review, 2026-10-22)
        for level in ("none", "minimal"):
            for where in ("top", "kwargs"):
                with self.subTest(level=level, where=where):
                    body = {"model": ORNITH, "messages": [], "chat_template_kwargs": {"enable_thinking": True}}
                    if where == "top":
                        body["reasoning_effort"] = level
                    else:
                        body["chat_template_kwargs"]["reasoning_effort"] = level
                    sent, got = self.go(body)
                    self.assertEqual(got, level)
                    self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": False})
                    self.assertNotIn("reasoning_effort", sent)

    def test_with_a_level_that_keeps_it_the_requests_own_choice_stays(self):
        sent, got = self.go({"model": ORNITH, "reasoning_effort": "high", "messages": [],
                             "chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(got, "high")
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("reasoning_effort", sent)

    def test_another_lane_route_level_or_shape_goes_as_it_came(self):
        body = {"model": ORNITH, "reasoning_effort": "none", "messages": []}
        for names, engine in (((ORNITH,), "sglang"), (("kolibri-1",), "vllm"), (("nemotron-3.5-bolt",), "vllm"),
                              (("qwen3.8-flash-next-fp4",), "vllm"), ((), ""), (("ornith",), "vllm")):
            with self.subTest(names=names, engine=engine):
                self.mod._SERVED.update(names=names, engine=engine)
                out, level = self.mod.ornith_effort(json.dumps(body).encode(), CHAT)
                self.assertIsNone(level)
                self.assertEqual(json.loads(out), body)
        self.mod._SERVED.update(names=(ORNITH,), engine="vllm")
        self.assertEqual(self.go(body, path="/v1/completions"), (None, None))
        self.assertEqual(self.go({"model": ORNITH, "messages": []}), (None, None))
        self.assertEqual(self.go({"model": ORNITH, "reasoning_effort": "turbo", "messages": []}), (None, None))
        self.assertEqual(self.go({"model": ORNITH, "reasoning_effort": "none", "chat_template_kwargs": "x"}), (None, None))
        self.assertEqual(self.go(b'{"reasoning_effort": "none", not json'), (None, None))
        self.assertEqual(self.go(b'["reasoning_effort"]'), (None, None))


class ThroughTheProxy(unittest.TestCase):
    """A fake vLLM serving ornith-1.5-122b behind the real proxy."""

    @classmethod
    def setUpClass(cls):
        _keep_env(cls)
        cls.engine = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeVllm)
        cls.engine.daemon_threads = True
        threading.Thread(target=cls.engine.serve_forever, daemon=True).start()
        os.environ.update(UPSTREAM=f"http://127.0.0.1:{cls.engine.server_port}", KEEPALIVE_S="0.4",
                          CLIENT_IO_S="5", PROMPT_CEILING_TOKENS="0", FLASH_PROMPT_CEILING_TOKENS="0")
        cls.mod = load_proxy("kproxy_ornith_effort")
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

    def serve(self, served):
        FakeVllm.served, FakeVllm.owner = served, "vllm"
        with FakeVllm.lock:
            FakeVllm.bodies.clear()
        self.mod.invalidate_pool()
        self.mod.served_models()

    def test_effort_none_reaches_ornith_as_thinking_off(self):
        self.serve(ORNITH)
        got = self.post({"model": ORNITH, "stream": True, "reasoning_effort": "none",
                         "messages": [{"role": "user", "content": "What is 17 * 23?"}]})
        self.assertTrue(got.startswith(b"HTTP/1.1 200"), got[:200])
        sent = FakeVllm.bodies[-1]
        self.assertEqual(sent.get("chat_template_kwargs"), {"enable_thinking": False})
        self.assertNotIn("reasoning_effort", sent)
        self.assertTrue(any("not what the Ornith template reads: relayed as enable_thinking false" in ln
                            for ln in self.lines), self.lines[-5:])

    def test_a_level_above_none_reaches_it_with_its_reasoning_on(self):
        self.serve(ORNITH)
        got = self.post({"model": ORNITH, "reasoning_effort": "low", "max_tokens": 8,
                         "messages": [{"role": "user", "content": "hi"}]})
        self.assertTrue(got.startswith(b"HTTP/1.1 200"), got[:200])
        sent = FakeVllm.bodies[-1]
        self.assertEqual(sent.get("chat_template_kwargs"), {"enable_thinking": True})
        self.assertNotIn("reasoning_effort", sent)
        said = [ln for ln in self.lines if "reasoning_effort low is not what the Ornith template reads" in ln]
        self.assertEqual(len(said), 1, self.lines[-8:])
        self.assertIn("relayed as enable_thinking true, unless", said[0])

    def test_the_nemotron_lane_keeps_its_own_translation(self):
        self.serve("nemotron-3.5-bolt")
        got = self.post({"model": "nemotron-3.5-bolt", "reasoning_effort": "low", "max_tokens": 8,
                         "messages": [{"role": "user", "content": "hi"}]})
        self.assertTrue(got.startswith(b"HTTP/1.1 200"), got[:200])
        self.assertEqual(FakeVllm.bodies[-1].get("chat_template_kwargs"), {"enable_thinking": True, "medium_effort": True})


if __name__ == "__main__":
    unittest.main()
