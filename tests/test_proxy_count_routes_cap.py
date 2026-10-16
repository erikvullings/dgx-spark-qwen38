"""The engine's counting routes have a body ceiling of their own (2026-10-16).

/tokenize and /detokenize generate nothing, so the oversize guard never measured them, and only
MAX_BODY_BYTES (256 MiB) bounded what they carried to the engine's tokenizer (found in review,
2026-10-16). Past COUNT_BODY_MAX_BYTES (16 MiB) they are refused with a 413 before a byte of the
body is read. /v1/messages/count_tokens, which an Anthropic client sends its whole conversation
to, images included, keeps MAX_BODY_BYTES.

2026-10-17: SGLang v0.5.21 serves the same two handlers at /v1/tokenize and /v1/detokenize, which
the ceiling missed, and the size guard handed a nominated prompt to the engine's /tokenize whole,
up to MAX_BODY_BYTES, the very load that ceiling keeps from it (found in review, 2026-10-17): a
prompt whose text alone is past the ceiling is refused as too long, uncounted.
"""
import http.server
import importlib.util
import json
import os
import socket
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path


def _keep_env(cls):
    """Put os.environ back as this class found it (tests/test_suite_isolation.py holds it)."""
    saved = dict(os.environ)
    cls.addClassCleanup(lambda: (os.environ.clear(), os.environ.update(saved)))


def _no_key():
    """The serving key of this harness's proxy: none, whatever HOME holds."""
    raise FileNotFoundError("no serving key in this harness")


HERE = Path(__file__).resolve()
SPEC = importlib.util.spec_from_file_location("kproxy_count_cap", HERE.parents[1] / "keepalive-proxy.py")
OVER = 17 * 1024 * 1024


class Engine(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        Engine.seen.append((self.path, n))
        out = json.dumps({"count": 1, "input_tokens": 1, "tokens": [1]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


class CountRoutesCap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _keep_env(cls)
        cls.engine = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Engine)
        cls.engine.daemon_threads = True
        threading.Thread(target=cls.engine.serve_forever, daemon=True).start()
        os.environ["UPSTREAM"] = f"http://127.0.0.1:{cls.engine.server_port}"
        for k in ("QWEN38_CLIENT_KEYS_FILE", "QWEN38_UPSTREAM_API_KEY", "PROXY_HOLD_UNITS",
                  "MAX_BODY_BYTES", "COUNT_BODY_MAX_BYTES", "BODY_BUDGET_BYTES"):
            os.environ.pop(k, None)
        cls.mod = importlib.util.module_from_spec(SPEC)
        sys.argv = ["keepalive-proxy.py"]
        SPEC.loader.exec_module(cls.mod)
        cls.mod._api_key = _no_key
        cls.proxy = cls.mod.Server(("127.0.0.1", 0), cls.mod.H)
        threading.Thread(target=cls.proxy.serve_forever, daemon=True).start()
        cls.port = cls.proxy.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.proxy.shutdown()
        cls.proxy.server_close()
        cls.engine.shutdown()
        cls.engine.server_close()

    def setUp(self):
        Engine.seen.clear()

    def announce(self, path, length):
        """Send the head of a request whose body never comes: what the proxy answers then
        was decided before it read a byte."""
        s = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        try:
            s.sendall(f"POST {path} HTTP/1.1\r\nHost: p\r\nContent-Type: application/json\r\n"
                      f"Content-Length: {length}\r\n\r\n".encode())
            s.settimeout(3)
            try:
                return s.recv(4096).split(b"\r\n", 1)[0]
            except socket.timeout:
                return b"(no answer: the proxy waits for the body)"
        finally:
            s.close()

    def post(self, path, data):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    def post_full(self, path, data, key):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_the_v1_spellings_of_the_counting_routes_have_the_same_ceiling(self):
        for path in ("/v1/tokenize", "/v1/detokenize"):
            with self.subTest(path=path):
                self.assertTrue(self.announce(path, OVER).startswith(b"HTTP/1.1 413"))
        self.assertEqual(Engine.seen, [])

    def test_a_prompt_past_the_counting_ceiling_is_refused_uncounted(self):
        # a measured pool the body's size nominates it against, and no wedge
        self.mod._POOL.update(tokens=1_000_000, window=1000, ts=time.time())
        self.addCleanup(self.mod._POOL.update, tokens=None, window=None, ts=0.0)
        # a key the proxy counts with: without one, no count is ever asked for
        real = self.mod._api_key
        self.mod._api_key = lambda: "count-key"
        self.addCleanup(setattr, self.mod, "_api_key", real)
        doc = {"model": "m", "messages": [{"role": "user", "content": "x" * OVER}]}
        status, body = self.post_full("/v1/chat/completions", json.dumps(doc).encode(), "count-key")
        self.assertEqual(status, 400, body[:300])
        self.assertEqual(json.loads(body)["error"]["code"], "context_length_exceeded")
        self.assertEqual(Engine.seen, [], "the prompt was handed to the engine")

    def test_system_one_refuses_past_the_counting_ceiling_uncounted(self):
        self.mod._POOL.update(tokens=1_000_000, window=1000, ts=time.time())
        self.addCleanup(self.mod._POOL.update, tokens=None, window=None, ts=0.0)
        real = self.mod._api_key
        self.mod._api_key = lambda: "count-key"
        self.addCleanup(setattr, self.mod, "_api_key", real)
        plan = {"prefix": "x" * OVER, "branches": [(0, "Is it?", None, None, None)]}
        with self.assertRaises(self.mod.SystemOneUsage) as refused:
            self.mod.systemone_guard(plan)
        self.assertIn("the prompt is too long", str(refused.exception))
        self.assertEqual(Engine.seen, [], "the prompt was handed to the engine")

    def test_a_counting_route_past_its_ceiling_is_413_before_its_body(self):
        for path in ("/tokenize", "/detokenize"):
            with self.subTest(path=path):
                self.assertTrue(self.announce(path, OVER).startswith(b"HTTP/1.1 413"))
        self.assertEqual(Engine.seen, [])

    def test_an_ordinary_count_still_goes(self):
        data = json.dumps({"prompt": "hello world"}).encode()
        self.assertEqual(self.post("/tokenize", data), 200)
        self.assertEqual(Engine.seen, [("/tokenize", len(data))])

    def test_the_anthropic_count_keeps_the_body_cap(self):
        """Claude Code counts its whole conversation there, screenshots included."""
        doc = {"model": "m", "messages": [{"role": "user", "content": "x" * OVER}]}
        self.assertEqual(self.post("/v1/messages/count_tokens", json.dumps(doc).encode()), 200)
        self.assertEqual([p for p, _ in Engine.seen], ["/v1/messages/count_tokens"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
