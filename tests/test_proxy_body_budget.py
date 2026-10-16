"""A budget for the bodies in flight across every request (2026-10-16).

MAX_BODY_BYTES caps one body at 256 MiB, and nothing capped them together: each is read whole
and parsed again by the guards, and four legitimate bodies near the cap reach the unit's
MemoryMax=1G at once, where the kernel kills the proxy and every stream in flight with it (found
in review, 2026-10-16). Past BODY_BUDGET_BYTES a body is not read: 503 with Retry-After.

This harness sets the budget to 1 MiB and holds a first body of 700 kB in the fake engine.
"""
import http.server
import importlib.util
import json
import os
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
SPEC = importlib.util.spec_from_file_location("kproxy_budget", HERE.parents[1] / "keepalive-proxy.py")


class Engine(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    hold_next = False
    holding = threading.Event()
    release = threading.Event()
    received = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        out = b'{"detail":"Not Found"}'
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        Engine.received.append(n)
        if Engine.hold_next:
            Engine.hold_next = False
            Engine.holding.set()
            Engine.release.wait(20)
        out = json.dumps({"id": "chatcmpl-x", "object": "chat.completion", "choices": []}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def body_of(n_bytes):
    filler = "word " * (n_bytes // 5)
    return json.dumps({"model": "m", "stream": False,
                       "messages": [{"role": "user", "content": filler}]}).encode()


class BodyBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _keep_env(cls)
        cls.engine = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Engine)
        cls.engine.daemon_threads = True
        threading.Thread(target=cls.engine.serve_forever, daemon=True).start()
        os.environ["UPSTREAM"] = f"http://127.0.0.1:{cls.engine.server_port}"
        os.environ["BODY_BUDGET_BYTES"] = str(1024 * 1024)
        for k in ("QWEN38_CLIENT_KEYS_FILE", "QWEN38_UPSTREAM_API_KEY", "PROXY_HOLD_UNITS"):
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
        Engine.release.set()
        cls.proxy.shutdown()
        cls.proxy.server_close()
        cls.engine.shutdown()
        cls.engine.server_close()

    def setUp(self):
        # a pool far larger than these bodies, measured, and no wedge: the size guard lets them by
        self.mod._POOL.update(tokens=10_000_000, window=1000, ts=time.time())
        Engine.received.clear()
        Engine.holding.clear()
        Engine.release.clear()

    def post(self, data, out=None):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                got = (r.status, dict(r.headers), r.read())
        except urllib.error.HTTPError as e:
            got = (e.code, dict(e.headers), e.read())
        if out is not None:
            out.append(got)
        return got

    def released(self):
        """Whether every body's budget is back: a request gives it back after its answer is
        written, so the client can read the answer first."""
        deadline = time.time() + 5
        while getattr(self.mod, "_BODY_BUDGET", {"used": 0})["used"] and time.time() < deadline:
            time.sleep(0.02)
        return getattr(self.mod, "_BODY_BUDGET", {"used": 0})["used"] == 0

    def test_a_body_past_the_budget_in_flight_waits_for_a_retry(self):
        first = []
        Engine.hold_next = True
        t = threading.Thread(target=self.post, args=(body_of(700_000), first), daemon=True)
        t.start()
        self.assertTrue(Engine.holding.wait(15), "the first body never reached the engine")
        try:
            status, headers, body = self.post(body_of(700_000))
            self.assertEqual(status, 503, body[:200])
            self.assertEqual(headers.get("Retry-After"), "5")
            self.assertIn("retry it unchanged", json.loads(body)["error"]["message"])
            self.assertEqual(len(Engine.received), 1, "the second body reached the engine")
            status, _, body = self.post(body_of(100_000))     # what fits still goes meanwhile
            self.assertEqual(status, 200, body[:200])
        finally:
            Engine.release.set()
            t.join(20)
        self.assertEqual(first[0][0], 200)
        self.assertTrue(self.released(), "a finished request kept its share of the budget")
        status, _, body = self.post(body_of(700_000))         # and the budget comes back
        self.assertEqual(status, 200, body[:200])
        self.assertTrue(self.released())


if __name__ == "__main__":
    unittest.main(verbosity=2)
