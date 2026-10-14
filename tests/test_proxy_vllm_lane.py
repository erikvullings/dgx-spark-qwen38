#!/usr/bin/env python3
"""The Kolibri lane serves on vLLM behind this proxy (v6.37), and vLLM is not SGLang where the
proxy had assumed it was. Each test here is one difference, measured on the reference box
against vLLM 0.29.0 serving Kolibri-1 (2026-10-14), and the fake engine below answers the way
that engine answered:
  - no /server_info and no /get_server_info (404): its pool is the kv_cache_size_tokens label
    of the cache_config_info gauge on /metrics, its window the max_model_len of /v1/models;
    without them every body past 256 KiB was held as "not measured yet" for good;
  - /v1/models says "owned_by": "vllm";
  - a model it does not serve is a 404 ("The model `x` does not exist."), where SGLang answers
    any name, so a client set up for one lane broke on this one;
  - a request whose connection closes is stopped (one running before the client left, none
    0.5 s after, no token in the next 2 s, stream and plain answer alike), and there is no
    /abort_request: the proxy drained such a request up to DRAIN_MAX_S;
  - logit_bias past the vocabulary is refused with "Vocabulary size: 128000", and only for the
    served model name (the probe named "probe", which vLLM refuses first).
And on SGLang nothing changes: a name it does not serve goes as it came."""
import http.server
import importlib.util
import io
import json
import os
import socket
import sys
import threading
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path

HERE = Path(__file__).resolve()
KEY = "test-key-vllm-lane"
METRICS = (
    '# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig\n'
    '# TYPE vllm:cache_config_info gauge\n'
    'vllm:cache_config_info{_block_size_resolved="True",block_size="16",cache_dtype="fp8",'
    'enable_prefix_caching="True",engine="0",gpu_memory_utilization="0.55",'
    'kv_cache_size_tokens="2014877",num_gpu_blocks="134815"} 1.0\n'
    'vllm:num_requests_running{engine="0",model_name="kolibri-1"} 0.0\n')


def _keep_env(cls):
    saved = dict(os.environ)
    cls.addClassCleanup(lambda: (os.environ.clear(), os.environ.update(saved)))


class FakeVllm(http.server.BaseHTTPRequestHandler):
    """vLLM 0.29.0 serving kolibri-1, as far as this proxy meets it."""
    protocol_version = "HTTP/1.1"
    served = "kolibri-1"
    owner = "vllm"
    bodies = []            # every chat body it received, parsed
    closed_at = {}         # prompt -> when the proxy closed a stream it was writing
    aborts = []            # /abort_request calls (vLLM has none: 404)
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        out = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/metrics":
            out = METRICS.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        elif path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": FakeVllm.served, "object": "model",
                                                        "owned_by": FakeVllm.owner, "max_model_len": 262144}]})
        elif path == "/server_info" and FakeVllm.owner == "sglang":
            self._json(200, {"max_total_num_tokens": 863644, "max_req_input_len": 262138})
        else:
            self._json(404, {"detail": "Not Found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if path == "/abort_request":
            with FakeVllm.lock:
                FakeVllm.aborts.append(raw)
            self._json(404, {"detail": "Not Found"})
            return
        body = json.loads(raw)
        with FakeVllm.lock:
            FakeVllm.bodies.append(body)
        if FakeVllm.owner == "vllm" and body.get("model") != FakeVllm.served:
            self._json(404, {"error": {"message": f"The model `{body.get('model')}` does not exist.",
                                       "type": "NotFoundError", "param": "model", "code": 404}})
            return
        if body.get("logit_bias"):
            self._json(400, {"error": {"message": "token_id(s) [999999999] in logit_bias contain out-of-vocab "
                                                  "token ids. Vocabulary size: 128000 (parameter=logit_bias, "
                                                  "value=[999999999])", "type": "BadRequestError", "code": 400}})
            return
        prompt = body["messages"][0]["content"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        n = 200 if prompt.startswith("slow") else 2
        try:
            for i in range(n):
                ev = {"id": "chatcmpl-0123456789abcdef", "object": "chat.completion.chunk",
                      "model": FakeVllm.served, "choices": [{"index": 0, "delta": {"content": f"t{i} "}}]}
                self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
                self.wfile.flush()
                time.sleep(0.05 if n > 2 else 0)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError:
            with FakeVllm.lock:
                FakeVllm.closed_at[prompt] = time.time()


def load_proxy(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parents[1] / "keepalive-proxy.py")
    mod = importlib.util.module_from_spec(spec)
    sys.argv = ["keepalive-proxy.py"]
    spec.loader.exec_module(mod)
    return mod


class PureParts(unittest.TestCase):
    """What needs no engine: reading /metrics, renaming a body, forgetting an engine."""

    @classmethod
    def setUpClass(cls):
        _keep_env(cls)
        os.environ["UPSTREAM"] = "http://127.0.0.1:9"
        cls.mod = load_proxy("kproxy_vllm_pure")

    def setUp(self):
        self.mod._SERVED.update(names=(), engine="", ts=0.0)
        self.mod._ENGINE_SEEN.update(names=(), engine="")

    def test_the_pool_is_the_kv_cache_size_label(self):
        self.assertEqual(self.mod.vllm_pool_facts(METRICS, [{"id": "kolibri-1", "max_model_len": 262144}]),
                         (2014877, 262144))

    def test_without_the_label_blocks_times_block_size(self):
        text = METRICS.replace('kv_cache_size_tokens="2014877",', "")
        self.assertEqual(self.mod.vllm_pool_facts(text, [])[0], 134815 * 16)

    def test_an_engine_with_neither_stays_unmeasured(self):
        self.assertEqual(self.mod.vllm_pool_facts("sglang:num_running_reqs 0\n", None), (0, 0))

    def body(self, model="qwen3.8-27b"):
        return json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}]}).encode()

    def test_vllm_gets_the_served_name(self):
        self.mod._SERVED.update(names=("kolibri-1",), engine="vllm", ts=time.time())
        out, was = self.mod.route_model_name(self.body(), "/v1/chat/completions")
        self.assertEqual(was, "qwen3.8-27b")
        self.assertEqual(json.loads(out)["model"], "kolibri-1")
        for path in ("/v1/messages", "/v1/messages/count_tokens", "/v1/completions", "/tokenize"):
            self.assertEqual(json.loads(self.mod.route_model_name(self.body(), path)[0])["model"], "kolibri-1", path)

    def test_a_served_name_a_foreign_route_or_a_doubt_go_as_they_came(self):
        self.mod._SERVED.update(names=("kolibri-1",), engine="vllm", ts=time.time())
        for body, path in ((self.body("kolibri-1"), "/v1/chat/completions"),
                           (self.body(), "/v1/systemone"),
                           (b'{"model": "x", "messages": [', "/v1/chat/completions"),
                           (b'[{"model": "x"}]', "/v1/chat/completions"),
                           (json.dumps({"model": 7}).encode(), "/v1/chat/completions")):
            with self.subTest(path=path, body=body[:30]):
                self.assertEqual(self.mod.route_model_name(body, path), (body, None))

    def test_sglang_bodies_are_never_touched(self):
        self.mod._SERVED.update(names=("qwen3.8-flash-next",), engine="sglang", ts=time.time())
        body = self.body()
        self.assertEqual(self.mod.route_model_name(body, "/v1/chat/completions"), (body, None))

    def test_another_engine_drops_the_vocabulary_and_the_rid_rule(self):
        m = self.mod
        m.note_engine(("qwen3.8-27b",), "sglang")
        m._VOCAB.update(size=248320, ts=time.time())
        m._rid_override_honoured = True
        m.note_engine(("qwen3.8-27b",), "sglang")          # the same engine, restarted
        self.assertEqual((m._VOCAB["size"], m._rid_override_honoured), (248320, True))
        with redirect_stderr(io.StringIO()):
            m.note_engine(("kolibri-1",), "vllm")              # a lane switch
        self.assertEqual((m._VOCAB["size"], m._rid_override_honoured), (0, None))

    def test_system_one_says_it_needs_a_qwen_lane(self):
        self.mod._SERVED.update(names=("kolibri-1",), engine="vllm", ts=time.time())
        self.mod.served_models = lambda: ("kolibri-1",)
        with self.assertRaises(self.mod.SystemOneUsage) as e:
            self.mod.systemone_model("jev-latest")
        self.assertIn("System One needs a Qwen lane", str(e.exception))
        self.assertIn("kolibri-1", str(e.exception))


class ThroughTheProxy(unittest.TestCase):
    """A fake vLLM behind the real proxy."""

    @classmethod
    def setUpClass(cls):
        _keep_env(cls)
        cls.engine = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeVllm)
        cls.engine.daemon_threads = True
        threading.Thread(target=cls.engine.serve_forever, daemon=True).start()
        os.environ.update(UPSTREAM=f"http://127.0.0.1:{cls.engine.server_port}", KEEPALIVE_S="0.4",
                          CLIENT_IO_S="5", PROMPT_CEILING_TOKENS="0", FLASH_PROMPT_CEILING_TOKENS="0")
        cls.mod = load_proxy("kproxy_vllm_lane")
        cls.mod._api_key = lambda: KEY
        cls.log = io.StringIO()
        cls.err = redirect_stderr(cls.log)
        cls.err.__enter__()
        cls.proxy = cls.mod.Server(("127.0.0.1", 0), cls.mod.H)
        threading.Thread(target=cls.proxy.serve_forever, daemon=True).start()
        cls.port = cls.proxy.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.err.__exit__(None, None, None)
        cls.proxy.shutdown()
        cls.proxy.server_close()
        cls.engine.shutdown()
        cls.engine.server_close()

    def setUp(self):
        FakeVllm.served, FakeVllm.owner = "kolibri-1", "vllm"
        with FakeVllm.lock:
            FakeVllm.bodies.clear(); FakeVllm.aborts.clear(); FakeVllm.closed_at.clear()
        self.mod.invalidate_pool()

    def post(self, body, read_all=True):
        raw = json.dumps(body).encode()
        s = socket.create_connection(("127.0.0.1", self.port), timeout=20)
        s.sendall(f"POST /v1/chat/completions HTTP/1.1\r\nHost: p\r\nAuthorization: Bearer {KEY}\r\n"
                  f"Content-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n".encode() + raw)
        if not read_all:
            return s
        s.settimeout(20)
        got = b""
        while True:
            c = s.recv(65536)
            if not c:
                break
            got += c
        s.close()
        return got

    def test_the_pool_and_the_window_come_from_metrics_and_models(self):
        self.assertEqual(self.mod.pool_tokens(), 2014877)
        self.assertEqual(self.mod._POOL["window"], 262144)
        self.assertEqual((self.mod._SERVED["names"], self.mod._SERVED["engine"]), (("kolibri-1",), "vllm"))

    def test_a_body_past_256_kib_is_served_not_held(self):
        got = self.post({"model": "kolibri-1", "stream": True,
                         "messages": [{"role": "user", "content": "x " * 150000}]})
        self.assertTrue(got.startswith(b"HTTP/1.1 200"), got[:120])
        self.assertIn(b"[DONE]", got)

    def test_a_client_set_up_for_another_lane_reaches_kolibri(self):
        got = self.post({"model": "qwen3.8-27b", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
        self.assertTrue(got.startswith(b"HTTP/1.1 200"), got[:200])
        self.assertEqual(FakeVllm.bodies[-1]["model"], "kolibri-1")
        self.assertIn("model 'qwen3.8-27b' is not what this lane serves", self.log.getvalue())

    def test_a_client_that_leaves_ends_the_generation_at_once_and_is_never_drained(self):
        s = self.post({"model": "kolibri-1", "stream": True,
                       "messages": [{"role": "user", "content": "slow essay"}]}, read_all=False)
        s.settimeout(10)
        s.recv(4096)                     # the stream is on
        t0 = time.time()
        s.close()
        end = time.time() + 5
        while time.time() < end and "slow essay" not in FakeVllm.closed_at:
            time.sleep(0.05)
        self.assertIn("slow essay", FakeVllm.closed_at, "the engine never saw its connection close")
        self.assertLess(FakeVllm.closed_at["slow essay"] - t0, 2.0)
        self.assertEqual(FakeVllm.aborts, [], "vLLM has no /abort_request")
        self.assertIn("closing the upstream: vLLM stops what it generates", self.log.getvalue())
        self.assertNotIn("drained an abandoned", self.log.getvalue())
        # and no SGLang advice in a vLLM lane's journal (it did, on the reference box)
        self.assertNotIn("SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES", self.log.getvalue())

    def test_the_vocabulary_is_read_in_vllms_words_with_the_served_name(self):
        self.mod.pool_tokens()                       # names known, as on any prompt
        self.assertEqual(self.mod.served_vocab(), 128000)
        self.assertTrue(any(b.get("logit_bias") and b["model"] == "kolibri-1" for b in FakeVllm.bodies))

    def test_on_sglang_a_foreign_name_goes_as_it_came(self):
        FakeVllm.served, FakeVllm.owner = "qwen3.8-27b", "sglang"
        self.mod.served_models()
        got = self.post({"model": "my-alias", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
        self.assertTrue(got.startswith(b"HTTP/1.1 200"), got[:200])
        self.assertEqual(FakeVllm.bodies[-1]["model"], "my-alias")


if __name__ == "__main__":
    unittest.main(verbosity=2)
