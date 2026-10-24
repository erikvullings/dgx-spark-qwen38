"""A budget for the bodies in flight across every request (2026-10-16).

MAX_BODY_BYTES caps one body at 256 MiB, and nothing capped them together: each is read whole
and parsed again by the guards, and four legitimate bodies near the cap reached the unit's
MemoryMax of 1G at once, where the kernel kills the proxy and every stream in flight with it
(found in review, 2026-10-16; 2G since 2026-10-18). Past BODY_BUDGET_BYTES a body is not read:
503 with Retry-After.

This harness sets the budget to 1 MiB and holds a first body of 700 kB in the fake engine.

The budget was taken whole on the Content-Length, before a byte was read, and kept to the end: a
key holder announcing large bodies and sending nothing held it all, and every other body, the
box's own clients' included, got a 503 (found in review, 2026-10-17). A body at or under
BODY_SMALL_BYTES (64 KiB in this harness) is now always read and holds none of it, and a larger
one takes its bytes as they arrive.
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
        os.environ["BODY_SMALL_BYTES"] = str(64 * 1024)
        os.environ["BODY_WAIT_S"] = "5"
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
        deadline = time.time() + 30    # a wait for a state, not a pace: a slow runner gets there too
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


    def announce(self, length):
        """The head of a request whose body does not come (yet)."""
        s = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        s.sendall(f"POST /v1/chat/completions HTTP/1.1\r\nHost: p\r\nContent-Type: application/json\r\n"
                  f"Content-Length: {length}\r\n\r\n".encode())
        self.addCleanup(s.close)
        return s

    def answer(self, s):
        s.settimeout(20)
        got = b""
        try:
            while True:
                c = s.recv(65536)
                if not c:
                    break
                got += c
        except ConnectionResetError:
            pass
        return got.split(b"\r\n", 1)[0]

    def network_peer(self, stall_s=1.0):
        """Loopback stands in for a peer on the network, with a short stall delay, both put back."""
        self.addCleanup(setattr, self.mod, "_counted_peer", self.mod._counted_peer)
        self.addCleanup(setattr, self.mod, "BODY_STALL_S", getattr(self.mod, "BODY_STALL_S", None))
        self.mod._counted_peer = lambda addr: True
        self.mod.BODY_STALL_S = stall_s

    def raw_post(self, path, body, pieces=1, gap_s=0.0, announce=None):
        """POST body to path on a raw socket, in pieces gap_s apart; the status line, or b"" when
        the proxy ended the connection without an answer, and the seconds it took."""
        s = socket.create_connection(("127.0.0.1", self.port), timeout=20)
        self.addCleanup(s.close)
        t0 = time.time()
        s.sendall(b"POST %s HTTP/1.1\r\nHost: p\r\nContent-Type: application/json\r\n"
                  b"Content-Length: %d\r\nConnection: close\r\n\r\n" % (path.encode(), announce or len(body)))
        step = -(-len(body) // pieces) if body else 0
        try:
            for i in range(0, len(body), step or 1):
                if i:
                    time.sleep(gap_s)
                s.sendall(body[i:i + step])
        except OSError:
            pass
        got = b""
        try:
            while True:
                c = s.recv(4096)
                if not c:
                    break
                got += c
        except (ConnectionResetError, socket.timeout):
            pass
        return got.split(b"\r\n", 1)[0], time.time() - t0

    def test_a_body_on_an_open_route_is_refused_unread(self):
        """/health, /ready and /metrics take no key and no body (the engine serves them on GET): a
        body there was read whole, and up to BODY_SMALL_BYTES it took nothing from the budget
        (D04 of the 2026-10-24 security review). Announced and never sent, it is refused at once."""
        for path in ("/health", "/ready", "/metrics"):
            with self.subTest(path=path):
                status, took = self.raw_post(path, b"", announce=4 * 1024 * 1024)
                self.assertTrue(status.startswith(b"HTTP/1.1 413"), status)
                self.assertLess(took, 5)
        self.assertEqual(Engine.received, [])
        self.assertTrue(self.released())

    def test_a_network_body_that_stops_is_let_go_and_its_room_goes_back(self):
        """Read under CLIENT_IO_S, a body that stopped held its thread and its room in the budget
        for 900 s a byte (D04). A network peer's now gets BODY_STALL_S without a byte."""
        self.network_peer(stall_s=1.0)
        status, took = self.raw_post("/v1/chat/completions", body_of(200_000)[:100_000],
                                     announce=len(body_of(200_000)))
        self.assertEqual(status, b"")
        self.assertLess(took, 8)
        self.assertTrue(self.released(), "a body that stopped kept its share of the budget")
        self.assertEqual(Engine.received, [])

    def test_a_slow_body_that_keeps_coming_is_read(self):
        """No byte for BODY_STALL_S, not BODY_STALL_S in all: a large body on a slow link passes."""
        self.network_peer(stall_s=1.0)
        status, took = self.raw_post("/v1/chat/completions", body_of(200_000), pieces=6, gap_s=0.4)
        self.assertTrue(status.startswith(b"HTTP/1.1 200"), status)
        self.assertGreater(took, 1.5)                       # longer in all than the delay

    def test_loopback_keeps_the_normal_delay_in_its_body(self):
        self.addCleanup(setattr, self.mod, "BODY_STALL_S", getattr(self.mod, "BODY_STALL_S", None))
        self.mod.BODY_STALL_S = 1.0
        status, took = self.raw_post("/v1/chat/completions", body_of(200_000), pieces=2, gap_s=2.0)
        self.assertTrue(status.startswith(b"HTTP/1.1 200"), status)

    def test_bytes_announced_and_never_sent_keep_no_body_out(self):
        s = self.announce(1_040_000)
        deadline = time.time() + 30    # a wait for a state, not a pace: a slow runner gets there too
        while not self.mod._BODY_BUDGET["used"] and time.time() < deadline:
            time.sleep(0.02)
        self.assertTrue(self.mod._BODY_BUDGET["used"], "the proxy never took the announced body in")
        status, _, body = self.post(body_of(300_000))
        self.assertEqual(status, 200, body[:200])
        status, _, body = self.post(body_of(30_000))
        self.assertEqual(status, 200, body[:200])
        s.close()
        self.assertTrue(self.released(), "a body that never came kept its share of the budget")

    def test_a_small_body_is_always_read(self):
        first = []
        Engine.hold_next = True
        t = threading.Thread(target=self.post, args=(body_of(1_030_000), first), daemon=True)
        t.start()
        self.assertTrue(Engine.holding.wait(15), "the first body never reached the engine")
        try:
            status, _, body = self.post(body_of(30_000))
            self.assertEqual(status, 200, body[:200])
            status, _, body = self.post(body_of(100_000))     # a larger one still waits its turn
            self.assertEqual(status, 503, body[:200])
        finally:
            Engine.release.set()
            t.join(20)
        self.assertEqual(first[0][0], 200)
        self.assertTrue(self.released())

    def test_of_two_bodies_read_at_once_past_the_budget_the_older_goes(self):
        """Taken as they arrive, two bodies can fill the budget together, and then each needs
        more: the newer one is refused and gives its room back, the older waits for it, and
        never are both refused. The harness feeds them so that they meet exactly there."""
        step, data = 64 * 1024, body_of(700_000)

        def held(n):
            deadline = time.time() + 30    # a wait for a state, not a pace: a slow runner gets there too
            while self.mod._BODY_BUDGET["used"] != n and time.time() < deadline:
                time.sleep(0.02)
            return self.mod._BODY_BUDGET["used"]

        a = self.announce(len(data))
        self.assertEqual(held(step), step)              # the older is the one read first
        b = self.announce(len(data))
        # b admitted and reading before a's bytes come: else a holds 8 steps when b's handler
        # starts, and b is refused at its admission, by a race of the test and not the budget
        self.assertEqual(held(2 * step), 2 * step)
        a.sendall(data[:7 * step])
        b.sendall(data[:7 * step])
        self.assertEqual(held(16 * step), 16 * step)    # the budget, full, and both want more
        a.sendall(data[7 * step:8 * step])
        b.sendall(data[7 * step:8 * step])
        a.sendall(data[8 * step:])
        self.assertTrue(self.answer(a).startswith(b"HTTP/1.1 200"))
        self.assertTrue(self.answer(b).startswith(b"HTTP/1.1 503"))
        self.assertTrue(self.released())


if __name__ == "__main__":
    unittest.main(verbosity=2)
