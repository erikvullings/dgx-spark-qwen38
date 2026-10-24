#!/usr/bin/env python3
"""A body the proxy writes again for the engine keeps its text as UTF-8 (proxy v6.46).

Seven guards parse a body and write it again (the reasoning effort routes, the Qwen, Nemotron and
Ornith effort translations, the tool schema patterns, the model name on vLLM), and the prompt
counter writes the text it sends the engine's tokenizer. json.dumps on its own escapes every
character outside ASCII: a Chinese character went from 3 bytes to 6 and an emoji from 4 to 12, so
the body the engine received grew by two to three times, the proxy held it at 5.65 and 7.98 times
its size while its guards worked on it, against 3.98 for an image as base64 (measured on the
reference box, 2026-10-24, tests/test_proxy_memory_ceiling.py), and the counter measured a prompt
against COUNT_BODY_MAX_BYTES at that inflated size. Each guard here gets a body of Chinese and emoji text that makes it rewrite, and the
rewritten body must be about the size of the one sent, parse to the same text, and carry the
guard's change. A lone surrogate, which a JSON escape allows and UTF-8 cannot carry, keeps its
escape.
"""
import importlib.util
import json
import os
import pathlib
import sys
import time
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = "请用中文回答这个问题，并且解释每一步。" * 40 + "😀🚀🔥✨" * 60


def load_proxy(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "keepalive-proxy.py")
    mod = importlib.util.module_from_spec(spec)
    sys.argv = ["keepalive-proxy.py"]
    spec.loader.exec_module(mod)
    return mod


def body(**extra):
    doc = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": TEXT}]}
    doc.update(extra)
    return json.dumps(doc, ensure_ascii=False).encode()


class TheRewrittenBodyStaysUtf8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        saved = dict(os.environ)
        cls.addClassCleanup(lambda: (os.environ.clear(), os.environ.update(saved)))
        os.environ["UPSTREAM"] = "http://127.0.0.1:9"
        os.environ["PROXY_HOLD_UNITS"] = ""
        cls.kp = load_proxy("kproxy_utf8")

    def setUp(self):
        self.kp._SERVED.update(names=(), engine="", ts=0.0)

    def check(self, sent, out, changed):
        self.assertIsNot(out, sent, "the guard did not rewrite: the case tests nothing")
        self.assertLessEqual(len(out), len(sent) * 1.05,
                             f"{len(sent)} bytes sent, {len(out)} written again: the text was escaped")
        doc = json.loads(out)
        self.assertEqual(doc["messages"][0]["content"], TEXT)
        changed(doc)

    def test_the_effort_moved_into_chat_template_kwargs(self):
        sent = body(reasoning_effort="lean")
        out, moved, _ = self.kp.route_reasoning_effort(sent, "/v1/chat/completions")
        self.assertEqual(moved, "lean")
        self.check(sent, out, lambda d: self.assertEqual(d["chat_template_kwargs"], {"reasoning_effort": "lean"}))

    def test_the_effort_dropped_beside_the_open_door(self):
        sent = body(reasoning_effort="lean", chat_template_kwargs={"reasoning_effort": "low"})
        out, _, dropped = self.kp.route_reasoning_effort(sent, "/v1/chat/completions")
        self.assertEqual(dropped, "lean")
        self.check(sent, out, lambda d: self.assertNotIn("reasoning_effort", d))

    def test_none_on_a_qwen_lane(self):
        self.kp._SERVED.update(names=("qwen3.8-27b",), engine="sglang", ts=time.time())
        sent = body(reasoning_effort="none")
        out, changed = self.kp.qwen_effort_none(sent, "/v1/chat/completions")
        self.assertTrue(changed)
        self.check(sent, out, lambda d: self.assertIs(d["chat_template_kwargs"]["enable_thinking"], False))

    def test_an_effort_on_the_nemotron_lane(self):
        self.kp._SERVED.update(names=("nemotron-3.5-bolt",), engine="vllm", ts=time.time())
        sent = body(reasoning_effort="low")
        out, level = self.kp.nemotron_effort(sent, "/v1/chat/completions")
        self.assertEqual(level, "low")
        self.check(sent, out, lambda d: self.assertIs(d["chat_template_kwargs"]["enable_thinking"], True))

    def test_an_effort_on_the_ornith_lane(self):
        self.kp._SERVED.update(names=("ornith-1.5-122b",), engine="vllm", ts=time.time())
        sent = body(reasoning_effort="none")
        out, level = self.kp.ornith_effort(sent, "/v1/chat/completions")
        self.assertEqual(level, "none")
        self.check(sent, out, lambda d: self.assertIs(d["chat_template_kwargs"]["enable_thinking"], False))

    def test_the_prompt_the_counter_sends_the_engine(self):
        # the counter's payload was escaped too, and measured against COUNT_BODY_MAX_BYTES at that size
        sent = body()
        seen = []

        class Answer:
            def read(self):
                return b'{"count": 123}'

        def urlopen(req, timeout=None):
            seen.append(req.data)
            return Answer()
        old = (self.kp._api_key, self.kp.urllib.request.urlopen, self.kp.COUNT_BODY_MAX_BYTES)
        self.addCleanup(lambda: (setattr(self.kp, "_api_key", old[0]),
                                 setattr(self.kp.urllib.request, "urlopen", old[1]),
                                 setattr(self.kp, "COUNT_BODY_MAX_BYTES", old[2])))
        self.kp._api_key = lambda: "test-key"
        self.kp.urllib.request.urlopen = urlopen
        # a ceiling the text fits in as UTF-8 and not escaped: counted, not refused as too long to count
        self.kp.COUNT_BODY_MAX_BYTES = len(sent) + 200
        self.assertEqual(self.kp.tokenize_count(sent, "/v1/chat/completions"), 123)
        self.assertLessEqual(len(seen[0]), len(sent) * 1.05,
                             f"{len(sent)} bytes sent, {len(seen[0])} sent on to count: the text was escaped")
        self.assertEqual(json.loads(seen[0])["messages"][0]["content"], TEXT)

    def test_a_tool_pattern_python_rejects(self):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {"x": {"type": "string", "pattern": "^\\p{L}+$"}}}}}]
        sent = body(tools=tools)
        out, dropped = self.kp.sanitize_tool_schemas(sent, "/v1/chat/completions")
        self.assertTrue(dropped)
        self.check(sent, out, lambda d: self.assertNotIn(
            "pattern", d["tools"][0]["function"]["parameters"]["properties"]["x"]))

    def test_the_served_name_on_vllm(self):
        self.kp._SERVED.update(names=("kolibri-1",), engine="vllm", ts=time.time())
        sent = body()
        out, was = self.kp.route_model_name(sent, "/v1/chat/completions")
        self.assertEqual(was, "qwen3.8-27b")
        self.check(sent, out, lambda d: self.assertEqual(d["model"], "kolibri-1"))

    def test_a_lone_surrogate_keeps_its_escape(self):
        sent = b'{"model": "m", "reasoning_effort": "lean", "messages": [{"role": "user", "content": "a\\ud83d b"}]}'
        out, moved, _ = self.kp.route_reasoning_effort(sent, "/v1/chat/completions")
        self.assertEqual(moved, "lean")
        self.assertEqual(json.loads(out)["messages"][0]["content"], json.loads(sent)["messages"][0]["content"])
        self.assertIn(b"\\ud83d", out)

    def test_no_guard_writes_a_body_with_the_escaping_default(self):
        src = (REPO / "keepalive-proxy.py").read_text()
        for written in ("json.dumps(j).encode()", "json.dumps(req).encode()"):
            self.assertNotIn(written, src,
                             "a body for the engine is written with json.dumps' default, which escapes every non-ASCII character")


if __name__ == "__main__":
    unittest.main()
