#!/usr/bin/env python3
"""The cockpit stops generations through the proxy (v1.40.3, proxy v6.47).

SGLang ends an aborted generation as a finished one (end_turn on /v1/messages), so the memory floor's
abort_all, sent to the engine straight, handed each client the part generated before the stop as its
whole answer: Claude Code kept a compaction summary cut in the middle of a word on the reference box
(2026-10-25 07:42). The floor and the operator's Abort all go to the proxy's POST /proxy/stop first,
which cuts each client and stops each generation, vLLM's too; the floor's stop carries its floor and
its pause, for which the proxy holds new generations back. A proxy that does not take it leaves the
way of before. Both servers below are fakes on loopback: the box's own proxy and engine are never
asked (COCKPIT_PROXY and COCKPIT_ENGINE point at them).
"""
import http.server
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
DASH = HERE.parents[1]
REPO = HERE.parents[2]
ENV_BEFORE = {}


def setUpModule():
    ENV_BEFORE.update(os.environ)


def tearDownModule():
    for name in set(os.environ) - set(ENV_BEFORE):
        del os.environ[name]
    os.environ.update(ENV_BEFORE)


class Recorder(http.server.BaseHTTPRequestHandler):
    """The proxy's stop route or the engine's abort route: what each was sent, and an answer."""
    calls = []
    lock = threading.Lock()
    proxy_takes = True           # False: a proxy older than v6.47, which relays the path and gets a 404

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        with self.lock:
            Recorder.calls.append((self.server.name, self.path, self.headers.get("Authorization"),
                                   json.loads(body or b"{}")))
        if self.server.name == "proxy" and self.path == "/proxy/stop" and self.proxy_takes:
            code, out = 200, {"stopped": 2, "abort_all": True}
        elif self.server.name == "engine" and self.path == "/abort_request":
            code, out = 200, {}
        else:
            code, out = 404, {"detail": "Not Found"}
        raw = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def serve(name):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Recorder)
    srv.daemon_threads = True
    srv.name = name
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.box = Path(tempfile.mkdtemp(prefix="cockpit-floor-stop-"))
        cls.proxy, cls.engine = serve("proxy"), serve("engine")
        os.environ.update(COCKPIT_DRY_RUN="1", COCKPIT_CONFIG_DIR=str(cls.box), COCKPIT_REPO_DIR=str(REPO),
                          COCKPIT_PORT="0", COCKPIT_AGENT_PORT="0", COCKPIT_AUTOHEAL="0",
                          COCKPIT_PROXY=f"http://127.0.0.1:{cls.proxy.server_port}",
                          COCKPIT_ENGINE=f"http://127.0.0.1:{cls.engine.server_port}")
        (cls.box / "api-key").write_text("test-key\n")
        sys.path.insert(0, str(DASH))
        spec = importlib.util.spec_from_file_location("cockpit_floor_stop_under_test", DASH / "cockpit.py")
        cls.ck = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.ck)

    @classmethod
    def tearDownClass(cls):
        for srv in (cls.proxy, cls.engine):
            srv.shutdown()
            srv.server_close()
        shutil.rmtree(cls.box, ignore_errors=True)

    def setUp(self):
        with Recorder.lock:
            Recorder.calls.clear()
        Recorder.proxy_takes = True
        self.serving(None)

    def serving(self, unit):
        with self.ck.LIFE_LOCK:
            self.ck.LIFE["states"] = {unit: "ready"} if unit else {}


class TheStop(Base):
    def test_the_floor_stops_through_the_proxy_with_its_floor_and_pause(self):
        done = self.ck.engine_abort_all(why="MemAvailable 2.9 GiB under the 3.0 GiB floor with 1 running")
        self.assertEqual(done, {"via": "proxy", "stopped": 2, "abort_all": True})
        self.assertEqual(len(Recorder.calls), 1, Recorder.calls)
        name, path, auth, body = Recorder.calls[0]
        self.assertEqual((name, path, auth), ("proxy", "/proxy/stop", "Bearer test-key"))
        self.assertEqual(body, {"why": "MemAvailable 2.9 GiB under the 3.0 GiB floor with 1 running",
                                "kind": "floor", "floor_gib": self.ck.MEM_FLOOR_GIB,
                                "hold_s": self.ck.MEM_FLOOR_COOLDOWN_S})

    def test_the_operators_abort_all_holds_nothing_back(self):
        ok, lines, data = self.ck.job_abort_all(None)
        self.assertTrue(ok)
        self.assertIn("2 client(s) cut", lines[0])
        self.assertEqual(data["via"], "proxy")
        (name, path, _, body), = Recorder.calls
        self.assertEqual((name, path), ("proxy", "/proxy/stop"))
        self.assertEqual(body, {"why": "Abort all, from the cockpit", "kind": "operator"})

    def test_a_proxy_that_does_not_take_it_leaves_the_way_of_before(self):
        Recorder.proxy_takes = False
        done = self.ck.engine_abort_all(why="floor")
        self.assertEqual(done["via"], "engine")
        self.assertIn("the proxy did not take the stop", done["missed"])
        self.assertEqual([(c[0], c[1], c[3]) for c in Recorder.calls],
                         [("proxy", "/proxy/stop", {"why": "floor", "kind": "floor",
                                                    "floor_gib": self.ck.MEM_FLOOR_GIB,
                                                    "hold_s": self.ck.MEM_FLOOR_COOLDOWN_S}),
                          ("engine", "/abort_request", {"abort_all": True})])

    def test_on_a_vllm_lane_without_the_proxy_both_reasons_are_said(self):
        Recorder.proxy_takes = False
        self.serving("qwen38-kolibri.service")
        with self.assertRaises(RuntimeError) as e:
            self.ck.engine_abort_all()
        self.assertIn("no route that aborts", str(e.exception))
        self.assertIn("the proxy did not take the stop", str(e.exception))
        self.assertEqual([c[0] for c in Recorder.calls], ["proxy"])     # never the engine


class TheFloor(Base):
    def test_the_floor_says_it_stopped_through_the_proxy(self):
        ck, saved = self.ck, {}
        for name in ("DRY_RUN", "engine_load", "mem_available_gib", "add_event", "audit"):
            saved[name] = getattr(ck, name)
        events, audits = [], []
        ck.DRY_RUN = False
        ck.engine_load = lambda timeout=3: [{"num_reqs": 1, "num_waiting_reqs": 0}]
        ck.mem_available_gib = lambda: 2.5
        ck.add_event = lambda kind, text: events.append((kind, text))
        ck.audit = audits.append
        ck.MEM_FLOOR.update(last_abort=None, aborts=0, last_try=None, last_via=None)
        try:
            out = ck.collect_engine_fast()
        finally:
            for name, value in saved.items():
                setattr(ck, name, value)
        floor = [e for e in events if e[0] == "mem_floor"]
        self.assertEqual(len(floor), 1, events)
        self.assertIn("every generation stopped through the proxy (2 cut, their clients retry)", floor[0][1])
        self.assertEqual(out["mem_floor"]["via"], "proxy")
        self.assertEqual([a for a in audits if a.get("kind") == "mem_floor"][0]["via"], "proxy")


if __name__ == "__main__":
    unittest.main()
