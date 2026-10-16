"""Token ids while the served vocabulary is unknown (2026-10-16).

The sampling guard learns the vocabulary from the engine's refusal of an out-of-range
logit_bias, and an engine that cannot say it (busy past the probe's 8 s, paused, restarting, or
answering instead of refusing) leaves it at 0, where the guard stood down: only negative ids
were refused, and an id past the vocabulary, which takes the scheduler down for every client
(sglang#31597), was relayed on trust (found in review, 2026-10-16). An id under the smallest
vocabulary of any lane the proxy fronts (UNKNOWN_VOCAB_FLOOR, Kolibri's 128,000) is in all of
them and still goes; one at or past it now waits for a retry, a 503, until the vocabulary is
known.

The fake engine answers the probe instead of refusing it, so the vocabulary stays unknown.
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


HERE = Path(__file__).resolve()
SPEC = importlib.util.spec_from_file_location("kproxy_unknown_vocab", HERE.parents[1] / "keepalive-proxy.py")


class Engine(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
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
        if self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": "m", "owned_by": "sglang"}]})
        else:
            self._json(404, {"detail": "Not Found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if "logit_bias" not in body:
            Engine.relayed.append((self.path, body))
        # the probe included: answered, never refused, so it names no vocabulary
        self._json(200, {"id": "chatcmpl-x", "object": "chat.completion", "choices": []})


class UnknownVocabulary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _keep_env(cls)
        cls.engine = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Engine)
        cls.engine.daemon_threads = True
        threading.Thread(target=cls.engine.serve_forever, daemon=True).start()
        os.environ["UPSTREAM"] = f"http://127.0.0.1:{cls.engine.server_port}"
        for k in ("QWEN38_CLIENT_KEYS_FILE", "QWEN38_UPSTREAM_API_KEY", "PROXY_HOLD_UNITS"):
            os.environ.pop(k, None)
        cls.mod = importlib.util.module_from_spec(SPEC)
        sys.argv = ["keepalive-proxy.py"]
        SPEC.loader.exec_module(cls.mod)
        cls.mod._api_key = lambda: "test-key"
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
        Engine.relayed.clear()
        self.mod._POOL.update(tokens=10_000_000, window=1000, ts=time.time())

    def post(self, path, doc):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                     data=json.dumps({"model": "m", **doc}).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer test-key"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def test_an_id_past_every_lanes_floor_waits_while_the_vocabulary_is_unknown(self):
        for path, doc in (("/v1/chat/completions", {"messages": [], "stop_token_ids": [999999999]}),
                          ("/v1/chat/completions", {"messages": [], "stop_token_ids": [1, 128000]}),
                          ("/generate", {"input_ids": [[1, 2], [300000]]}),
                          ("/v1/completions", {"prompt": [5, 200000]})):
            with self.subTest(path=path, doc=doc):
                status, headers, body = self.post(path, doc)
                self.assertEqual(status, 503, body)
                self.assertEqual(headers.get("Retry-After"), "10")
                self.assertIn("retry it unchanged", json.loads(body)["error"]["message"])
        self.assertEqual(self.mod.served_vocab(), 0, "the harness lost its unknown vocabulary")
        self.assertEqual(Engine.relayed, [], "an id no lane may hold reached the engine")

    def test_ids_every_lane_holds_still_go(self):
        for path, doc in (("/v1/chat/completions", {"messages": [], "stop_token_ids": [1, 127999]}),
                          ("/generate", {"input_ids": [5, 6, 7]}),
                          ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})):
            with self.subTest(path=path, doc=doc):
                status, _, body = self.post(path, doc)
                self.assertEqual(status, 200, body)
        self.assertEqual(len(Engine.relayed), 3)

    def test_a_negative_id_is_still_refused_outright(self):
        status, _, body = self.post("/v1/chat/completions", {"messages": [], "stop_token_ids": [-3]})
        self.assertEqual(status, 400, body)
        self.assertEqual(Engine.relayed, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
