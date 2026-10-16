"""Bodies the engine's count routes cannot take, where a wedge is possible (2026-10-16).

The oversize guard counts a nominated body with the engine's own tokenizer, and a shape it
could not translate for /tokenize (the native /generate fields, /v1/responses' input) was judged
by the pre-v6.34 average of 2.5 bytes a token. A token is at least one byte, and digits are one
token each, so a body of digits went through at well over twice its estimate, past the limit,
into the zone where the engine queues a prompt and never admits it (found in review,
2026-10-16). The native /generate shape is now counted (its ids count themselves, its text is a
prompt), and what still cannot be counted is judged by the bound where a wedge is possible.

The fake engine counts as a tokenizer does where it matters here: a digit is one token, a word
is one token.
"""
import http.server
import importlib.util
import json
import os
import re
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
SPEC = importlib.util.spec_from_file_location("kproxy_uncountable", HERE.parents[1] / "keepalive-proxy.py")
LIMIT = 9200                    # the usable share of a 10,000-token pool


def tokens(text):
    return len(re.findall(r"\d|[^\d\s]+", text))


class Engine(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    generated = []

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
            # the sampling guard's vocabulary probe, refused as SamplingParams.verify() does
            self._json(400, {"object": "error", "message": "logit_bias must has keys in [0, 248319], "
                                                           "got 999999999"})
            return
        if self.path == "/tokenize":
            if "prompt" not in body:
                self._json(400, {"object": "error", "message": "prompt or messages, please"})
                return
            prompt = body["prompt"]
            n = sum(tokens(p) for p in prompt) if isinstance(prompt, list) else tokens(prompt)
            self._json(200, {"tokens": [], "count": n, "max_model_len": 9990})
            return
        Engine.generated.append(self.path)
        self._json(200, {"id": "x", "object": "response", "output": []})


class UncountableShapes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _keep_env(cls)
        cls.engine = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Engine)
        cls.engine.daemon_threads = True
        threading.Thread(target=cls.engine.serve_forever, daemon=True).start()
        os.environ["UPSTREAM"] = f"http://127.0.0.1:{cls.engine.server_port}"
        for k in ("QWEN38_CLIENT_KEYS_FILE", "QWEN38_UPSTREAM_API_KEY", "PROXY_HOLD_UNITS",
                  "PROMPT_CEILING_TOKENS", "FLASH_PROMPT_CEILING_TOKENS"):
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
        Engine.generated.clear()
        # the engine's window (9,990) is past the 9,200 usable: a wedge is possible
        self.mod._POOL.update(tokens=10000, window=9990, ts=time.time())

    def post(self, path, doc):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                     data=json.dumps(doc).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer test-key"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def refused_as_too_long(self, status, body):
        self.assertEqual(status, 400, body)
        self.assertEqual(json.loads(body)["error"]["code"], "context_length_exceeded")

    def test_digits_in_a_native_generate_text_are_counted(self):
        digits = "7" * 15000                      # 15,000 tokens; the average said 6,000
        status, body = self.post("/generate", {"text": digits, "sampling_params": {"max_new_tokens": 1}})
        self.refused_as_too_long(status, body)
        self.assertEqual(Engine.generated, [])

    def test_native_ids_count_themselves(self):
        status, body = self.post("/generate", {"input_ids": [7] * 12000})
        self.refused_as_too_long(status, body)
        status, body = self.post("/generate", {"input_ids": [[7] * 3000, [8] * 3000]})
        self.assertEqual(status, 200, body)
        self.assertEqual(Engine.generated, ["/generate"])

    def test_a_shape_nobody_can_count_is_judged_by_the_bound(self):
        doc = {"model": "m", "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "7" * 15000}]}]}
        status, body = self.post("/v1/responses", doc)
        self.refused_as_too_long(status, body)
        self.assertEqual(Engine.generated, [])

    def test_prose_that_fits_is_relayed(self):
        status, body = self.post("/generate", {"text": "word " * 3000})   # 15 kB, 3,000 tokens
        self.assertEqual(status, 200, body)
        self.assertEqual(Engine.generated, ["/generate"])

    def test_where_the_engine_refuses_past_its_window_the_average_still_decides(self):
        self.mod._POOL.update(tokens=10000, window=8000, ts=time.time())
        doc = {"model": "m", "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "7" * 15000}]}]}
        status, body = self.post("/v1/responses", doc)
        self.assertEqual(status, 200, body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
