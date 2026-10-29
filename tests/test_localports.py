#!/usr/bin/env python3
"""tests/localports.py: the port a test starts a server on stays the test's, beside any other program.

ci-local.sh runs the workflow's jobs side by side on one machine, three of them over the same test files,
and eleven of those files took a port number out of a socket they then closed: anyone could be handed
the port before the server it was for bound it, a stranger could answer in its place, and a port meant
to be dead could come alive (code review, 2026-10-28). Each answer of localports.py is proven here on the
machine itself, and the last case keeps any test file from taking a port that way again. It reads the
idiom, not every way to write it: a closed server's port taken for a dead one, as test_proxy_hold did
(found in the review of the fix), is a held port now too."""
import http.server
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time
import unittest
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import localports  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
# The files that may hold the words: this module and its test (the old way, as a witness), run_suite.py,
# which reads them, its own test's planted file, and the reaper's listener, which keeps its socket open.
SPARED = {"tests/localports.py", "tests/test_localports.py", "tests/run_suite.py", "tests/test_run_suite.py",
          "tests/test_suite_reaper.py"}
SERVER = ("import http.server, sys\n"
          "class H(http.server.BaseHTTPRequestHandler):\n"
          "    def do_GET(self):\n"
          "        self.send_response(200); self.send_header('Content-Length', '4'); self.end_headers()\n"
          "        self.wfile.write(b'mine')\n"
          "    def log_message(self, *a):\n"
          "        pass\n"
          "http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()\n")


class Ok(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


def get(port):
    """What the port answers: the body, or the kind of error."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            return r.read().decode()
    except OSError as e:
        return type(e).__name__


def a_stranger_binds(port, host="127.0.0.1", reuse=False):
    s = socket.socket()
    try:
        if reuse:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


class AHeldPort(unittest.TestCase):
    def test_refuses_a_connection_and_no_one_else_can_bind_it(self):
        port = localports.held(self)
        self.assertEqual(get(port), "URLError")
        for host in ("127.0.0.1", "0.0.0.0"):
            for reuse in (False, True):
                self.assertFalse(a_stranger_binds(port, host, reuse), (host, reuse))
        with self.assertRaises(OSError):
            http.server.HTTPServer(("127.0.0.1", port), Ok)

    def test_the_kernel_never_hands_it_out(self):
        # what the other jobs do: bind port 0, with SO_REUSEPORT or not
        ports = {localports.held(self) for _ in range(20)}
        for reuseport in (False, True):
            for _ in range(1000):
                with socket.socket() as s:
                    if reuseport:
                        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                    s.bind(("127.0.0.1", 0))
                    self.assertNotIn(s.getsockname()[1], ports, f"SO_REUSEPORT {reuseport}")

    def test_a_server_comes_and_goes_on_it_and_it_stays_the_tests(self):
        port = localports.held(self)
        for _ in range(3):
            srv = localports.serve(port, http.server.ThreadingHTTPServer, Ok)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            self.assertIsInstance(srv, http.server.ThreadingHTTPServer)
            self.assertEqual([get(port) for _ in range(3)], ["ok"] * 3)
            srv.shutdown()
            srv.server_close()
            self.assertEqual(get(port), "URLError", "down again, as an engine that stopped")
            self.assertFalse(a_stranger_binds(port, reuse=True), "and still the test's")

    def test_a_second_server_on_a_port_that_listens_is_refused(self):
        # SO_REUSEPORT would have the kernel share the connections between the two
        port = localports.held(self)
        srv = localports.serve(port, http.server.ThreadingHTTPServer, Ok)
        self.addCleanup(srv.server_close)
        with self.assertRaises(OSError):
            localports.serve(port, http.server.ThreadingHTTPServer, Ok)


class ASpawnedServer(unittest.TestCase):
    def start(self, port):
        proc = subprocess.Popen([sys.executable, "-c", SERVER, str(port)], stderr=subprocess.DEVNULL)
        self.addCleanup(localports._end, proc)
        return proc

    def test_listens_on_a_port_the_kernel_gave_it(self):
        given = []
        proc, port = localports.spawn(lambda p: given.append(p) or self.start(p))
        self.assertEqual(given, [0], "no number is chosen before the bind")
        self.assertIsNone(proc.poll())
        self.assertEqual(get(port), "mine")

    def test_says_why_when_it_ends_or_never_listens(self):
        def ends(port):
            return subprocess.Popen([sys.executable, "-c", "import sys; sys.stderr.write('no such cert'); raise SystemExit(3)"],
                                    stderr=subprocess.PIPE)
        t0 = time.monotonic()
        with self.assertRaises(RuntimeError) as cm:
            localports.spawn(ends)
        self.assertIn("ended with 3 before it listened: no such cert", str(cm.exception), "its own reason, kept")
        self.assertLess(time.monotonic() - t0, 10, "said at once, not tried again")
        sleeps = []

        def silent(port):
            sleeps.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]))
            return sleeps[-1]
        with self.assertRaises(RuntimeError) as cm:
            localports.spawn(silent, within=1)
        self.assertIn("did not listen within 1 s", str(cm.exception))
        self.assertIsNotNone(sleeps[0].poll(), "and it is ended")

    def test_a_server_on_two_ports_is_refused(self):
        two = ("import socket, time\n"
               "a, b = socket.socket(), socket.socket()\n"
               "a.bind(('127.0.0.1', 0)); b.bind(('127.0.0.1', 0)); a.listen(); b.listen(); time.sleep(60)\n")
        procs = []

        def start(port):
            procs.append(subprocess.Popen([sys.executable, "-c", two]))
            self.addCleanup(localports._end, procs[-1])
            return procs[-1]
        with self.assertRaises(RuntimeError) as cm:
            localports.spawn(start)
        self.assertIn("listens on 2 ports", str(cm.exception), "which one is meant cannot be guessed")
        self.assertIsNotNone(procs[0].poll())


class NoTestFile(unittest.TestCase):
    def test_takes_a_port_number_out_of_a_socket_it_closes(self):
        files = sorted(str(p.relative_to(REPO)) for d in ("tests", "dashboard/tests") for p in (REPO / d).glob("*.py"))
        self.assertIn("tests/test_proxy_hold.py", files)
        taking = [f for f in files if f not in SPARED and "getsockname()[1]" in (REPO / f).read_text()]
        self.assertEqual(taking, [], "take it from tests/localports.py: held() for a port nothing listens on, "
                                     "serve() for a server of the test's process, spawn() for one of its own")


if __name__ == "__main__":
    unittest.main(verbosity=2)
