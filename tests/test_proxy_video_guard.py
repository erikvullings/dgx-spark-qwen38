"""A video through the oversize guard (2026-10-16).

Where the engine accepts a prompt past the share of its pool the guard lets through (a wedge:
queued, never admitted, until a restart), the guard counts every body that could be that long.
A video was charged a flat 4,096 tokens, where SGLang's own sizing lets one cost about 88,000,
and a body carrying a dozen video URLs is a few kilobytes, which the byte estimate never
nominated for a count at all: it went to the engine unmeasured (found in review, 2026-10-16).

A fake engine counts words for /tokenize, the way tests/test_proxy_guard.py's does, and records
the generations it is handed.
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
SPEC = importlib.util.spec_from_file_location("kproxy_video", HERE.parents[1] / "keepalive-proxy.py")


class Engine(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    generated = []
    counted = []

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
        if self.path == "/tokenize":
            Engine.counted.append(body)
            words = sum(len(p.get("text", "").split()) if isinstance(p, dict) else len(str(p).split())
                        for m in body.get("messages", [])
                        for p in (m["content"] if isinstance(m.get("content"), list) else [m.get("content", "")]))
            self._json(200, {"tokens": [], "count": words, "max_model_len": 1000000})
            return
        Engine.generated.append(body)
        self._json(200, {"id": "chatcmpl-x", "object": "chat.completion", "choices": []})


def chat(videos, words=50):
    content = [{"type": "text", "text": " ".join(["word"] * words)}]
    content += [{"type": "video_url", "video_url": {"url": f"https://example.invalid/{i}.mp4"}}
                for i in range(videos)]
    return {"model": "m", "stream": False, "messages": [{"role": "user", "content": content}]}


class VideoGuard(unittest.TestCase):
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
        Engine.counted.clear()

    def lane(self, pool, window):
        self.mod._POOL.update(tokens=pool, window=window, ts=time.time())

    def post(self, doc):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                     data=json.dumps(doc).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer test-key"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_a_small_body_of_videos_is_counted_and_refused_where_a_wedge_is_possible(self):
        self.lane(100000, 99990)                 # the engine's window is past the 92,000 usable
        doc = chat(videos=10)
        self.assertLess(len(json.dumps(doc)), 4096, "a few kilobytes")
        status, body = self.post(doc)
        self.assertEqual(status, 400, body)
        err = json.loads(body)["error"]
        self.assertEqual(err["code"], "context_length_exceeded")
        self.assertIn("the prompt is too long", err["message"])
        self.assertEqual(len(Engine.counted), 1, "the body was never counted")
        self.assertEqual(Engine.generated, [], "the engine was handed a prompt that can wedge it")

    def test_one_video_that_fits_still_goes(self):
        self.lane(100000, 99990)
        status, body = self.post(chat(videos=1))       # its ceiling and its bytes fit: no count needed
        self.assertEqual(status, 200, body)
        self.assertEqual(len(Engine.generated), 1)

    def test_where_the_engine_refuses_past_its_window_nothing_changes(self):
        """The flash lane: its window (262,138) is under the 428,823 usable, so the engine
        refuses a longer prompt itself, and a video keeps its flat budget, uncounted."""
        self.lane(466112, 262138)
        status, body = self.post(chat(videos=10))
        self.assertEqual(status, 200, body)
        self.assertEqual(Engine.counted, [])
        self.assertEqual(len(Engine.generated), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
