"""The network's share of the proxy's port (2026-10-16).

Every connection holds a thread and a descriptor, up to CLIENT_IO_S before its request line is
even read, and nothing counted them: any peer that reached the port, with no key at all, could
open connections until the descriptors ran out, and then no client was accepted, the cockpit's
and the local agents' included (found in review, 2026-10-16). A connection from another
machine now counts against a ceiling per address and one for the whole network, and past
either it is closed before a thread starts for it. Loopback is not counted.

No key is asked before a request's headers are in, and a connection that sent none held its
slot for CLIENT_IO_S: four addresses at their ceiling of silent connections shut the port to
every other peer on the network, with no key, for 900 s (found in review, 2026-10-17). A peer on
the network now has CLIENT_HEADER_S, in all, for a request's line and headers, a byte at a time
or not; the normal delay applies once they are in.

The harness reaches the proxy from 127.0.0.1 only, so the scenarios that need a peer on the
network count loopback in its stead (_counted_peer is swapped for the scenario); what loopback
is, and that it is never counted, is checked on its own.
"""
import http.server
import importlib.util
import os
import socket
import sys
import threading
import time
import unittest
from pathlib import Path


def _keep_env(cls):
    """Put os.environ back as this class found it (tests/test_suite_isolation.py holds it)."""
    saved = dict(os.environ)
    cls.addClassCleanup(lambda: (os.environ.clear(), os.environ.update(saved)))


def _no_key():
    """The serving key of this harness's proxy: none, whatever HOME holds."""
    raise FileNotFoundError("no serving key in this harness")


HERE = Path(__file__).resolve()
SPEC = importlib.util.spec_from_file_location("kproxy_conn_ceiling", HERE.parents[1] / "keepalive-proxy.py")


class Engine(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"OK")

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.do_GET()


class ConnectionCeiling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _keep_env(cls)
        cls.engine = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Engine)
        cls.engine.daemon_threads = True
        threading.Thread(target=cls.engine.serve_forever, daemon=True).start()
        os.environ["UPSTREAM"] = f"http://127.0.0.1:{cls.engine.server_port}"
        os.environ["CLIENT_IO_S"] = "20"           # a silent connection left behind ends soon
        for k in ("QWEN38_CLIENT_KEYS_FILE", "QWEN38_UPSTREAM_API_KEY", "PROXY_HOLD_UNITS"):
            os.environ.pop(k, None)
        cls.mod = importlib.util.module_from_spec(SPEC)
        sys.argv = ["keepalive-proxy.py"]
        SPEC.loader.exec_module(cls.mod)
        cls.mod._api_key = _no_key
        # in a tuple: a function kept as a class attribute would come back as a bound method
        cls.real = (getattr(cls.mod, "_counted_peer", None),)
        cls.header_s = getattr(cls.mod, "CLIENT_HEADER_S", None)
        cls.header_max = getattr(cls.mod, "HEADER_BYTES_MAX", None)
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
        # the connections of the scenario before end on their own threads: start from none
        deadline = time.time() + 10
        while getattr(self.proxy, "_conn_open", None) and time.time() < deadline:
            time.sleep(0.05)
        self.socks = []
        self.mod.MAX_CONN_PER_ADDRESS = 3
        self.mod.MAX_REMOTE_CONN = 5
        self.addCleanup(self._close_all)

    def _close_all(self):
        for s in self.socks:
            try:
                s.close()
            except OSError:
                pass
        self.mod._counted_peer = self.real[0]
        self.mod.CLIENT_HEADER_S = self.header_s
        self.mod.HEADER_BYTES_MAX = self.header_max

    def silent(self):
        """A connection that says nothing, the shape that held a thread for CLIENT_IO_S."""
        s = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        self.socks.append(s)
        return s

    def closed_at_once(self, s, within=3.0):
        """True when the proxy ends this connection without being asked anything."""
        s.settimeout(within)
        try:
            return s.recv(1) == b""
        except ConnectionResetError:
            return True
        except socket.timeout:
            return False

    def health(self):
        s = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        try:
            s.sendall(b"GET /health HTTP/1.1\r\nHost: p\r\nConnection: close\r\n\r\n")
            s.settimeout(5)
            got = b""
            while True:
                c = s.recv(4096)
                if not c:
                    break
                got += c
            return got.split(b"\r\n", 1)[0]
        except OSError:
            return b""
        finally:
            s.close()

    def test_loopback_is_never_counted_and_the_network_is(self):
        counted = self.real[0]
        self.assertIsNotNone(counted, "the proxy has no notion of a peer on the network")
        for addr in ("127.0.0.1", "127.0.0.53", "::1"):
            self.assertFalse(counted(addr), addr)
        for addr in ("192.168.1.20", "100.101.102.103", "10.0.0.2", "fe80::1", "not an address"):
            self.assertTrue(counted(addr), addr)

    def test_one_address_past_its_ceiling_is_closed_before_a_thread_starts(self):
        self.mod._counted_peer = lambda addr: True     # loopback stands in for a peer on the network
        held = [self.silent() for _ in range(3)]
        extra = self.silent()
        self.assertTrue(self.closed_at_once(extra), "a 4th silent connection from one address was kept")
        for s in held:
            s.settimeout(0.3)
            with self.assertRaises(socket.timeout, msg="a connection under the ceiling was closed"):
                s.recv(1)

    def test_a_slot_comes_back_when_its_connection_ends(self):
        self.mod._counted_peer = lambda addr: True
        held = [self.silent() for _ in range(3)]
        self.assertTrue(self.closed_at_once(self.silent()))
        held[0].close()
        deadline = time.time() + 10
        status = b""
        while time.time() < deadline:
            status = self.health()
            if status.startswith(b"HTTP/1.1 200"):
                break
            time.sleep(0.2)
        self.assertTrue(status.startswith(b"HTTP/1.1 200"), status)

    def test_the_network_as_a_whole_has_a_ceiling(self):
        self.mod._counted_peer = lambda addr: True
        self.mod.MAX_CONN_PER_ADDRESS = 0                # no per-address ceiling: the total decides
        for _ in range(5):
            self.silent()
        self.assertTrue(self.closed_at_once(self.silent()), "a 6th connection from the network was kept")

    def test_a_silent_peer_on_the_network_is_let_go_at_the_header_deadline(self):
        self.mod._counted_peer = lambda addr: True
        self.mod.CLIENT_HEADER_S = 1.0
        t0 = time.time()
        self.assertTrue(self.closed_at_once(self.silent(), within=6.0),
                        "a silent connection from the network outlived the header deadline")
        self.assertLess(time.time() - t0, 5.0)

    def test_a_trickle_does_not_stretch_the_header_deadline(self):
        """The deadline is for the line and the headers in all, not for each byte of them."""
        self.mod._counted_peer = lambda addr: True
        self.mod.CLIENT_HEADER_S = 1.0
        s = self.silent()
        s.sendall(b"GET /health HTTP/1.1\r\nHost: p\r\nX-Slow: ")
        t0, closed = time.time(), False
        while not closed and time.time() - t0 < 6:
            try:
                s.sendall(b"a")
                closed = self.closed_at_once(s, within=0.25)
            except OSError:
                closed = True
        self.assertTrue(closed, "a header sent a byte at a time held the connection past the deadline")
        self.assertLess(time.time() - t0, 5.0)

    def test_once_its_headers_are_in_a_request_has_the_normal_delay(self):
        self.mod._counted_peer = lambda addr: True
        self.mod.CLIENT_HEADER_S = 1.0
        s = self.silent()
        s.sendall(b"POST /x HTTP/1.1\r\nHost: p\r\nContent-Type: application/json\r\n"
                  b"Content-Length: 2\r\nConnection: close\r\n\r\n")
        time.sleep(2.0)                   # past the header deadline, far under CLIENT_IO_S
        s.sendall(b"{}")
        s.settimeout(10)
        got = b""
        while True:
            c = s.recv(4096)
            if not c:
                break
            got += c
        self.assertTrue(got.startswith(b"HTTP/1.1 200"), got[:200])

    def answer_to(self, head):
        """The status line the proxy answers a request with these headers, and whether it then
        ended the connection."""
        s = self.silent()
        s.sendall(head)
        s.settimeout(10)
        got = b""
        try:
            while True:
                c = s.recv(4096)
                if not c:
                    break
                got += c
        except (ConnectionResetError, socket.timeout):
            pass
        return got.split(b"\r\n", 1)[0]

    @staticmethod
    def padded(lines, size):
        return (b"GET /health HTTP/1.1\r\nHost: p\r\nConnection: close\r\n"
                + b"".join(b"X-Pad-%d: %s\r\n" % (i, b"a" * size) for i in range(lines)) + b"\r\n")

    def test_a_peer_past_the_header_budget_gets_431(self):
        """The standard parser bounds each header line and their number, not their sum: about
        6.4 MiB a connection before any key (D03 of the 2026-10-24 security review). Each line
        here is far under its bound, the three together are past the budget."""
        self.mod._counted_peer = lambda addr: True
        self.mod.HEADER_BYTES_MAX = 4096
        status = self.answer_to(self.padded(3, 2000))
        self.assertTrue(status.startswith(b"HTTP/1.1 431"), status)

    def test_headers_under_the_budget_pass(self):
        self.mod._counted_peer = lambda addr: True
        self.mod.HEADER_BYTES_MAX = 4096
        status = self.answer_to(self.padded(3, 1000))
        self.assertTrue(status.startswith(b"HTTP/1.1 200"), status)

    def test_loopback_has_no_header_budget(self):
        self.mod.HEADER_BYTES_MAX = 4096
        status = self.answer_to(self.padded(3, 2000))
        self.assertTrue(status.startswith(b"HTTP/1.1 200"), status)

    def test_the_budget_holds_the_largest_client_many_times(self):
        # 952 bytes: the largest request line and headers the box's eight clients sent, measured
        # through a relay on 2026-10-24 (Claude Code on /v1/messages, 21 headers)
        self.assertEqual(self.header_max, 64 * 1024)
        self.assertGreater(self.header_max, 60 * 952)

    def test_loopback_keeps_the_normal_delay_before_its_headers(self):
        self.mod.CLIENT_HEADER_S = 1.0
        self.assertFalse(self.closed_at_once(self.silent(), within=2.5),
                         "a client on the box was held to the network's header deadline")

    def test_local_clients_keep_the_port_past_any_ceiling(self):
        """The ceilings are the network's: clients on the box are not counted."""
        for _ in range(8):                               # past both ceilings of this harness
            self.silent()
        self.assertTrue(self.health().startswith(b"HTTP/1.1 200"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
