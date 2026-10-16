"""A held request is judged again by the engine that comes back (2026-10-16).

While the engine restarts, the proxy holds a request and sends it once the engine answers again
(role 6). The sampling guard judged the body once, before, against the vocabulary of the engine
that was there, and the request then went as it was to the engine that came back, which can be
another lane: an id the old one held and the new one does not takes the new one down
(sglang#31597; found in review, 2026-10-16). The body is now judged again after the hold,
against a vocabulary learned since the hold began.

The fake engine is a 248,320-token lane, then a stretch of 503s (what SGLang answers while it
starts), then a 128,000-token lane on the same port, as a switch to Kolibri is seen from here.
"""
import http.server
import importlib.util
import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path


def _keep_env(cls):
    """Put os.environ back as this class found it (tests/test_suite_isolation.py holds it)."""
    saved = dict(os.environ)
    cls.addClassCleanup(lambda: (os.environ.clear(), os.environ.update(saved)))


HERE = Path(__file__).resolve()
SPEC = importlib.util.spec_from_file_location("kproxy_rejudge", HERE.parents[1] / "keepalive-proxy.py")
VOCABS = {"qwen": 248320, "kolibri": 128000}


class Engine(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state = "qwen"
    relayed = []

    def log_message(self, *a):
        pass

    def _json(self, status, doc):
        out = json.dumps(doc).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        if Engine.state == "down":
            self._json(503, {"detail": "starting"})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": "m", "owned_by": "sglang"}]})
        elif self.path in ("/get_server_info", "/server_info"):
            self._json(200, {"max_total_num_tokens": 1_000_000, "max_req_input_len": 1000})
        else:
            self._json(200, {"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if Engine.state == "down":
            self._json(503, {"detail": "starting"})
            return
        if "logit_bias" in body:
            last = VOCABS[Engine.state] - 1
            self._json(400, {"object": "error", "message": f"logit_bias must has keys in [0, {last}], "
                                                           "got 999999999"})
            return
        Engine.relayed.append((Engine.state, body.get("stop_token_ids")))
        self._json(200, {"id": "chatcmpl-x", "object": "chat.completion", "choices": []})


class RejudgeAfterHold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _keep_env(cls)
        cls.engine = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Engine)
        cls.engine.daemon_threads = True
        threading.Thread(target=cls.engine.serve_forever, daemon=True).start()
        os.environ["UPSTREAM"] = f"http://127.0.0.1:{cls.engine.server_port}"
        os.environ["PROXY_HOLD_POLL_S"] = "0.2"
        os.environ["PROXY_HOLD_MAX_S"] = "30"
        for k in ("QWEN38_CLIENT_KEYS_FILE", "QWEN38_UPSTREAM_API_KEY", "PROXY_HOLD_UNITS"):
            os.environ.pop(k, None)
        cls.mod = importlib.util.module_from_spec(SPEC)
        sys.argv = ["keepalive-proxy.py"]
        SPEC.loader.exec_module(cls.mod)
        cls.mod._api_key = lambda: "test-key"
        # the unit is on its way back: what systemd says on the box, which this harness has not
        cls.mod.engine_coming = lambda: True
        cls.proxy = cls.mod.Server(("127.0.0.1", 0), cls.mod.H)
        threading.Thread(target=cls.proxy.serve_forever, daemon=True).start()
        cls.port = cls.proxy.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.proxy.shutdown()
        cls.proxy.server_close()
        cls.engine.shutdown()
        cls.engine.server_close()

    def post(self, stop_ids):
        doc = {"model": "m", "stream": False, "stop_token_ids": stop_ids,
               "messages": [{"role": "user", "content": "hi"}]}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                     data=json.dumps(doc).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer test-key"})
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_a_body_held_through_a_lane_switch_is_judged_by_the_new_lane(self):
        Engine.state, Engine.relayed[:] = "qwen", []
        self.assertEqual(self.post([5])[0], 200)                 # the guard learns 248,320 here
        self.assertEqual(self.mod.served_vocab(), 248320)
        Engine.state = "down"
        threading.Timer(1.5, lambda: setattr(Engine, "state", "kolibri")).start()
        status, body = self.post([200000])                      # Qwen holds it, Kolibri does not
        self.assertEqual(status, 400, body)
        self.assertIn("past the 128000 tokens", json.loads(body)["error"]["message"])
        self.assertEqual(Engine.relayed, [("qwen", [5])], "the new lane was handed an id it dies on")
        self.assertEqual(self.post([5])[0], 200)                 # and what it holds still goes
        self.assertEqual(Engine.relayed[-1], ("kolibri", [5]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
