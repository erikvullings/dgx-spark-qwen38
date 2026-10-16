"""/v1/score through the proxy's sampling guard (2026-10-16).

The engine's scoring route takes token ids in two fields the guard never read: `query`
(Union[str, List[int]]) and `items` (Union[str, List[str], List[List[int]]]). They index the
embedding as input_ids do, and an id no vocabulary holds takes the scheduler down for every
client (the sglang#31597 family the guard already refuses on the other routes). The route was
missing from SAMPLE_GUARD_ROUTES, so a body of them went to the engine unread (found in review,
2026-10-16).

A fake engine stands in for SGLang: it answers the guard's vocabulary probe the way SGLang
refuses an out-of-range logit_bias, and records every /v1/score it is handed.
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
SPEC = importlib.util.spec_from_file_location("kproxy_score", HERE.parents[1] / "keepalive-proxy.py")
VOCAB = 1000


class Engine(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    scored = []

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
        if self.path == "/v1/chat/completions" and "logit_bias" in body:
            # SamplingParams.verify()'s own words, which the guard reads the vocabulary from
            self._json(400, {"object": "error", "message": "logit_bias must has keys in "
                                                           f"[0, {VOCAB - 1}], got 999999999"})
            return
        if self.path == "/v1/score":
            Engine.scored.append(body)
            self._json(200, {"scores": [[0.5]], "object": "scoring"})
            return
        self._json(404, {"detail": "Not Found"})


class ScoreGuard(unittest.TestCase):
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
        # the key the proxy signs its vocabulary probe with, and so the one its door asks for
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
        Engine.scored.clear()

    def score(self, doc):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/score",
                                     data=json.dumps({"model": "m", **doc}).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer test-key"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_ids_no_vocabulary_holds_are_refused_before_the_engine(self):
        for doc in ({"query": [1, 2], "items": [[3, -1]]},
                    {"query": [-7], "items": [[3]]},
                    {"query": "is it?", "items": [[4, 5], [-2]]}):
            with self.subTest(doc=doc):
                status, body = self.score(doc)
                self.assertEqual(status, 400, body)
                self.assertIn("negative token id", json.loads(body)["error"]["message"])
        self.assertEqual(Engine.scored, [], "the engine was handed a body it dies on")

    def test_ids_past_the_served_vocabulary_are_refused(self):
        for field, doc in (("query", {"query": [1, VOCAB], "items": [[3]]}),
                           ("items", {"query": [1], "items": [[2], [3, VOCAB + 5]]}),
                           ("items", {"query": [1], "items": [["999999"]]})):
            with self.subTest(doc=doc):
                status, body = self.score(doc)
                self.assertEqual(status, 400, body)
                msg = json.loads(body)["error"]["message"]
                self.assertIn(f"{field} contains", msg)
                self.assertIn(f"past the {VOCAB} tokens", msg)
        self.assertEqual(Engine.scored, [])

    def test_text_and_ids_in_range_are_relayed(self):
        """Text is text, whatever its characters: an item that spells a number is no id."""
        for doc in ({"query": "is the sky blue?", "items": ["yes", "no", "999999999"]},
                    {"query": [1, 2, VOCAB - 1], "items": [[0, 3], [4]]}):
            with self.subTest(doc=doc):
                status, body = self.score(doc)
                self.assertEqual(status, 200, body)
        self.assertEqual(len(Engine.scored), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
