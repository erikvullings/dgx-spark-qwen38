#!/usr/bin/env python3
"""tests/localports.py: the port a test starts a server on stays the test's, beside any other program.

ci-local.sh runs the workflow's jobs side by side on one machine, three of them over the same test files,
and eleven of those files took a port number out of a socket they then closed: anyone could be handed
the port before the server it was for bound it, a stranger could answer in its place, and a port meant
to be dead could come alive (code review, 2026-10-28). Each answer of localports.py is proven here on the
machine itself, and the last case keeps any test file from taking a port that way again."""
import http.server
import os
import pathlib
import socket
import subprocess
import sys
import threading
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
        _, port = localports.held(self)
        self.assertEqual(get(port), "URLError")
        for host in ("127.0.0.1", "0.0.0.0"):
            for reuse in (False, True):
                self.assertFalse(a_stranger_binds(port, host, reuse), (host, reuse))
        with self.assertRaises(OSError):
            http.server.HTTPServer(("127.0.0.1", port), Ok)

    def test_the_old_way_left_it_to_anyone(self):
        # the witness: a port read out of a socket that is then closed is anyone's at once
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.assertTrue(a_stranger_binds(port))

    def test_a_server_comes_and_goes_on_it_and_it_stays_the_tests(self):
        _, port = localports.held(self)
        for _ in range(3):
            srv = localports.serve(port, http.server.ThreadingHTTPServer, Ok)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            self.assertIsInstance(srv, http.server.ThreadingHTTPServer)
            self.assertEqual([get(port) for _ in range(3)], ["ok"] * 3)
            srv.shutdown()
            srv.server_close()
            self.assertEqual(get(port), "URLError", "down again, as an engine that stopped")
            self.assertFalse(a_stranger_binds(port, reuse=True), "and still the test's")


class ASpawnedServer(unittest.TestCase):
    def start(self, port):
        proc = subprocess.Popen([sys.executable, "-c", SERVER, str(port)], stderr=subprocess.DEVNULL)
        self.addCleanup(localports._end, proc)
        return proc

    def stranger(self):
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Ok)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (srv.shutdown(), srv.server_close()))
        return srv.server_address[1]

    def test_is_kept_once_the_port_is_its_own(self):
        proc, port = localports.spawn(self.start)
        self.assertIsNone(proc.poll())
        self.assertEqual(get(port), "mine")

    def test_moves_on_when_someone_else_listens_on_the_port(self):
        taken = self.stranger()
        # the old way: start the server there, then take a connection that succeeds for it ready
        old = self.start(taken)
        socket.create_connection(("127.0.0.1", taken), timeout=5).close()
        self.assertEqual(get(taken), "ok", "the old way would have tested the stranger")
        old.wait(timeout=10)
        tries = iter([taken])
        proc, port = localports.spawn(self.start, pick=lambda: next(tries, None))
        self.assertNotEqual(port, taken)
        self.assertEqual((get(port), get(taken)), ("mine", "ok"))

    def test_a_server_behind_a_wrapper_is_its_own(self):
        def wrapped(port):
            # "; :" keeps bash from replacing itself with the server: the listener is a child of the process
            proc = subprocess.Popen(["bash", "-c", f'"{sys.executable}" -c "$0" {port}; :', SERVER],
                                    stderr=subprocess.DEVNULL, start_new_session=True)
            self.addCleanup(self.end_group, proc)
            return proc
        _, port = localports.spawn(wrapped)
        self.assertEqual(get(port), "mine")

    def end_group(self, proc):
        try:
            os.killpg(proc.pid, 15)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)

    def test_says_why_when_no_try_is_kept(self):
        def ends(port):
            return subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
        with self.assertRaises(RuntimeError) as cm:
            localports.spawn(ends, tries=2, within=5)
        self.assertIn("after 2 tries", str(cm.exception))
        self.assertIn("it ended with 3", str(cm.exception))
        taken = self.stranger()
        with self.assertRaises(RuntimeError) as cm:
            localports.spawn(self.start, tries=1, pick=lambda: taken)
        self.assertIn(f"{taken}: someone else listens there", str(cm.exception))


class NoTestFile(unittest.TestCase):
    def test_takes_a_port_number_out_of_a_socket_it_closes(self):
        files = sorted(str(p.relative_to(REPO)) for d in ("tests", "dashboard/tests") for p in (REPO / d).glob("*.py"))
        self.assertIn("tests/test_proxy_hold.py", files)
        taking = [f for f in files if f not in SPARED and "getsockname()[1]" in (REPO / f).read_text()]
        self.assertEqual(taking, [], "take it from tests/localports.py: held() for a port nothing listens on, "
                                     "serve() for a server of the test's process, spawn() for one of its own")


if __name__ == "__main__":
    unittest.main(verbosity=2)
