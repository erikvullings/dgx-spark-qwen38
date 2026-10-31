"""Offline tests for the keepalive proxy's abort contract (v6.14).

A client that gives up leaves a request the engine is still decoding unless the proxy
names it and aborts it in time. SGLang's abort_request() returns early when the rid is
no longer in TokenizerManager.rid_to_state, and the client disconnect deletes that entry
first, so an abort that arrives after the socket close is discarded in silence
(sglang #35255). The fake engine below reproduces exactly that rule: an abort for a rid
whose state is gone is recorded as ignored, not as an abort. And it sees the close when it
happens, as SGLang does, not at its own next write: a fake that noticed the close only when
a write failed, up to an event gap later, could not tell an abort sent before the close from
one sent just after it, and a proxy that closed first passed every test here (found in
review, 2026-09-24).

Three holes measured on the reference box on 2026-09-09 and closed here:
  - a client that gave up during prefill left a request the proxy could not name,
    because it learned the rid from the first SSE event (4,470 flood lines, 6 min);
  - /v1/messages returned msg_<uuid>, which names nothing the engine knows, and the
    proxy sent it to /abort_request anyway (241 flood lines);
  - the abort was fired in a thread racing the socket close (1,742 flood lines).
"""
import http.server
import importlib.util
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path


def _no_key():
    """The serving key this harness's proxy holds: none, whatever HOME holds. Its clients send
    no key, as these tests were written, and since v6.34 a proxy that knows a key refuses them
    at the door (the CI's offline step gives HOME one). The door has its own tests, in
    tests/test_proxy_door.py."""
    raise FileNotFoundError("no serving key in this harness")


def _keep_env(cls):
    """Put os.environ back as this class found it, once it is done: the variables set for
    the proxy under test (UPSTREAM and the rest) stayed set for every module after this
    one (found in review, 2026-09-24; tests/test_suite_isolation.py holds it)."""
    saved = dict(os.environ)
    cls.addClassCleanup(lambda: (os.environ.clear(), os.environ.update(saved)))


HERE = Path(__file__).resolve()
SPEC = importlib.util.spec_from_file_location("kproxy_abort", HERE.parents[1] / "keepalive-proxy.py")

PREFILL_S = 3.0          # engine silence before the first event, i.e. a long prefill
EVENT_GAP_S = 0.1


class FakeEngine(http.server.BaseHTTPRequestHandler):
    """SGLang as far as the abort contract is concerned."""
    protocol_version = "HTTP/1.1"
    live = {}            # rid -> True while the engine would still find its state
    aborted = []         # rids the engine actually aborted
    ignored = []         # aborts that arrived after the state was gone (the #35255 hole)
    seen_rid_header = []
    drained = []         # requests the engine could write to the very end
    conns = {}           # rid -> the connection its answer goes out on
    honour = True        # SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES, off by default upstream
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def _sse_head(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()

    def _register(self, rid):
        """The request's state exists before its first byte is out. In SGLang the stream's
        headers go out and generate_request runs _init_req_state right after, with no await
        in between, so an abort (handled on the same event loop) cannot land between the
        two. This fake is threaded: registering after the headers let a fast abort land in
        that gap and read as discarded, about once in a CI run (seen 2026-09-24)."""
        with self.lock:
            FakeEngine.live[rid] = True
            FakeEngine.conns[rid] = self.connection

    @staticmethod
    def _peer_closed(conn):
        """Whether the proxy has closed this request's connection. The proxy sends nothing
        more on it once the body is read, so a peek that finds the end of the stream is the
        close, however long before the next event it came. The proxy reads the abort's
        answer before it closes, so an abort that arrives while this says True was sent
        after the close."""
        if conn is None:
            return False
        try:
            return conn.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
        except BlockingIOError:
            return False
        except OSError:
            return True

    def _write_events(self, rid, ids, prefill=0.0, count=6):
        """Write SSE events until the reader goes away; report which happened. The state
        was registered by _register, before the headers."""
        try:
            if prefill:
                time.sleep(prefill)
            for i in range(count):
                rec = b"data: " + json.dumps({"id": ids, "n": i}).encode() + b"\n\n"
                self.wfile.write(rec)
                self.wfile.flush()
                time.sleep(EVENT_GAP_S)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            with self.lock:
                FakeEngine.drained.append(rid)
        except Exception:
            pass                      # the reader closed: this is the disconnect path
        finally:
            with self.lock:
                FakeEngine.live.pop(rid, None)     # _discard_pending_req_states

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        path = self.path.split("?")[0]
        if path == "/abort_request":
            rid = json.loads(body or b"{}").get("rid")
            with self.lock:
                if self._peer_closed(FakeEngine.conns.get(rid)):
                    FakeEngine.live.pop(rid, None)     # the disconnect deleted it first
                # abort_request(): "rid not in self.rid_to_state -> return"
                (FakeEngine.aborted if rid in FakeEngine.live else FakeEngine.ignored).append(rid)
                FakeEngine.live.pop(rid, None)
            out = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            return
        override = self.headers.get("x-override-rid")
        with self.lock:
            FakeEngine.seen_rid_header.append((path, override))
        if path == "/v1/messages":
            # The Anthropic route mints the id its client sees (msg_<uuid>), and its rid is the
            # header's when the engine honours overrides: it builds its request with the chat
            # route's _convert_to_internal_request, which applies them (v0.5.19, v0.5.21 and the
            # flash nightly, read 2026-10-16). This fake said "no overrides" until then.
            rid = (override if FakeEngine.honour else None) or uuid.uuid4().hex
            self._register(rid)
            self._sse_head()
            self._write_events(rid, f"msg_{uuid.uuid4().hex}", prefill=0.0)
            return
        rid = (override if FakeEngine.honour else None) or uuid.uuid4().hex
        prefill = PREFILL_S if b"slow-prefill" in body else 0.0
        self._register(rid)
        self._sse_head()
        self._write_events(rid, rid, prefill=prefill)


class _Systemd:
    """The proxy's subprocess, answering `systemctl show` as a box serving one unit does."""

    def __init__(self, exec_start):
        self.exec_start = exec_start

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def run(self, args, **kw):
        if list(args[:2]) != ["systemctl", "show"]:
            return subprocess.run(args, **kw)
        if "ExecStart" in args:
            out = f"ActiveState=active\nExecStart={self.exec_start}\n"
        elif "InvocationID" in args:
            out = "0123456789abcdef0123456789abcdef\n"
        else:
            out = "active\n"
        return subprocess.CompletedProcess(args, 0, out, "")


class Abort(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _keep_env(cls)
        cls.engine = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeEngine)
        cls.engine.daemon_threads = True
        threading.Thread(target=cls.engine.serve_forever, daemon=True).start()
        os.environ["UPSTREAM"] = f"http://127.0.0.1:{cls.engine.server_port}"
        os.environ["KEEPALIVE_S"] = "0.4"          # notice a dead client quickly
        os.environ["CLIENT_IO_S"] = "5"
        cls.mod = importlib.util.module_from_spec(SPEC)
        sys.argv = ["keepalive-proxy.py"]
        SPEC.loader.exec_module(cls.mod)
        cls.mod._api_key = _no_key
        cls.proxy = cls.mod.Server(("127.0.0.1", 0), cls.mod.H)
        threading.Thread(target=cls.proxy.serve_forever, daemon=True).start()
        cls.port = cls.proxy.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.proxy.shutdown()
        cls.engine.shutdown()

    def setUp(self):
        with FakeEngine.lock:
            FakeEngine.live.clear(); FakeEngine.aborted.clear(); FakeEngine.ignored.clear()
            FakeEngine.seen_rid_header.clear(); FakeEngine.drained.clear(); FakeEngine.conns.clear()
            FakeEngine.honour = True
        # what the engine does with the header is learned, never assumed: each test
        # starts from "not known yet", the way the proxy starts.
        self.mod._rid_override_honoured = None

    def _prove_override(self):
        """One complete answer teaches the proxy whether the engine takes its rid."""
        s = socket.create_connection(("127.0.0.1", self.port), timeout=15)
        raw = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: p\r\n"
                  b"Content-Type: application/json\r\n"
                  + f"Content-Length: {len(raw)}\r\n\r\n".encode() + raw)
        s.settimeout(15)
        got = b""
        while b"[DONE]" not in got:
            c = s.recv(4096)
            if not c:
                break
            got += c
        s.close()
        with FakeEngine.lock:
            FakeEngine.live.clear(); FakeEngine.aborted.clear(); FakeEngine.ignored.clear()
            FakeEngine.seen_rid_header.clear(); FakeEngine.drained.clear(); FakeEngine.conns.clear()

    def _serving_unit(self, argv):
        """The lane's unit, active, its ExecStart running argv (no answer from the engine yet)."""
        real = (self.mod.subprocess, self.mod.HOLD_UNITS)
        self.mod.subprocess = _Systemd(f"{{ path=/bin/bash ; argv[]={argv} ; ignore_errors=no ; "
                                       f"start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }}")
        self.mod.HOLD_UNITS = ("qwen38-sglang.service",)
        cache = getattr(self.mod, "_UNIT_OVERRIDE", None)
        if cache is not None:
            cache.update(ts=-1e9, set=None)

        def back():
            self.mod.subprocess, self.mod.HOLD_UNITS = real
            if cache is not None:
                cache.update(ts=-1e9, set=None)
        self.addCleanup(back)

    def _abandon_anthropic(self):
        self._send("/v1/messages", {"messages": [{"role": "user", "content": "hi"}]}, read_bytes=1)
        self.assertTrue(self._wait(lambda: FakeEngine.aborted or FakeEngine.ignored or FakeEngine.drained))
        return [rid for path, rid in FakeEngine.seen_rid_header if path == "/v1/messages"]

    # ---- helpers ---------------------------------------------------------------
    def _send(self, path, body, read_bytes=0, hard_close=True):
        """Post a request through the proxy, read what is asked, then vanish."""
        s = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        # FakeEngine answers every request as a stream, so each one asks for a stream:
        # since v6.25 the proxy reads the flag (a non-streamed answer is watched instead)
        raw = json.dumps({**body, "stream": True}).encode()
        s.sendall(f"POST {path} HTTP/1.1\r\nHost: p\r\nContent-Type: application/json\r\n"
                  f"Content-Length: {len(raw)}\r\n\r\n".encode() + raw)
        got = b""
        if read_bytes:
            s.settimeout(10)
            while len(got) < read_bytes:
                c = s.recv(4096)
                if not c:
                    break
                got += c
        if hard_close:
            # RST, so the proxy's next write fails now instead of buffering
            s.setsockopt(socket.SOL_SOCKET, socket.SO_GROOTER, struct.pack("ii", 1, 0))
        s.close()
        return got

    def _wait(self, pred, timeout=15):
        end = time.time() + timeout
        while time.time() < end:
            with FakeEngine.lock:
                if pred():
                    return True
            time.sleep(0.05)
        return False

    # ---- tests -----------------------------------------------------------------
    def test_proxy_names_the_request_before_it_starts(self):
        """The rid rides on the request, so it exists before the first event does."""
        self._send("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]},
                   read_bytes=1)
        self.assertTrue(self._wait(lambda: FakeEngine.seen_rid_header))
        path, override = FakeEngine.seen_rid_header[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertTrue(override and len(override) == 32, f"no rid imposed: {override!r}")

    def test_client_lost_during_prefill_is_aborted(self):
        """The 6-minute zombie of 2026-09-09: the client gave up before the engine had
        emitted anything, so the proxy had no id to abort with. Now it has one."""
        self._prove_override()
        self._send("/v1/chat/completions",
                   {"messages": [{"role": "user", "content": "slow-prefill"}]})
        self.assertTrue(self._wait(lambda: FakeEngine.aborted or FakeEngine.ignored),
                        "the engine was never told to stop")
        self.assertEqual(len(FakeEngine.aborted), 1, f"ignored={FakeEngine.ignored}")
        self.assertEqual(FakeEngine.aborted[0], FakeEngine.seen_rid_header[0][1])

    def test_abort_reaches_the_engine_before_the_socket_closes(self):
        """An abort sent after the close is discarded in silence (#35255), so the order
        is the whole point: the engine must still hold the state when it arrives."""
        self._send("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]},
                   read_bytes=1)
        self.assertTrue(self._wait(lambda: FakeEngine.aborted or FakeEngine.ignored))
        self.assertEqual(FakeEngine.ignored, [], "the abort lost the race against the close")
        self.assertEqual(len(FakeEngine.aborted), 1)

    def test_anthropic_id_is_never_sent_as_a_rid(self):
        """msg_<uuid> names nothing the engine knows: aborting with it is a no-op that
        reads like a success in the log. The answer is drained instead."""
        self._send("/v1/messages", {"messages": [{"role": "user", "content": "hi"}]},
                   read_bytes=1)
        self.assertTrue(self._wait(lambda: FakeEngine.drained or FakeEngine.aborted
                                   or FakeEngine.ignored))
        self.assertEqual(FakeEngine.aborted, [])
        self.assertEqual(FakeEngine.ignored, [])
        self.assertEqual(len(FakeEngine.drained), 1,
                         "the abandoned Anthropic answer was not read to its end")

    def test_an_abandoned_anthropic_stream_is_aborted_once_the_override_is_proven(self):
        """Claude Code's dialect was drained for want of a rid, up to DRAIN_MAX_S of decode for
        nobody (2026-10-16): the proxy now names it, and aborts it once a chat stream has shown
        that the engine takes the name."""
        self._prove_override()
        self._send("/v1/messages", {"messages": [{"role": "user", "content": "hi"}]}, read_bytes=1)
        self.assertTrue(self._wait(lambda: FakeEngine.aborted or FakeEngine.ignored or FakeEngine.drained))
        sent = [rid for path, rid in FakeEngine.seen_rid_header if path == "/v1/messages"]
        self.assertTrue(sent and sent[0] and len(sent[0]) == 32, f"no rid imposed: {sent!r}")
        self.assertEqual(FakeEngine.aborted, sent, f"drained={FakeEngine.drained}")
        self.assertEqual(FakeEngine.ignored, [])

    def test_an_abandoned_anthropic_stream_is_aborted_when_the_serving_unit_sets_the_override(self):
        """A box served by Claude Code alone never streams the chat route, so the proof never came
        and every abandoned answer was drained, after each engine restart too (found in review,
        2026-10-17). The unit serving now, starting its engine with
        SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES=1, is that proof."""
        self._serving_unit("/bin/bash -c exec /usr/bin/docker run --rm --name qwen38-sglang "
                           "-e SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES=1 lmsysorg/sglang python3 -m sglang.launch_server")
        sent = self._abandon_anthropic()
        self.assertTrue(sent and sent[0] and len(sent[0]) == 32, f"no rid imposed: {sent!r}")
        self.assertEqual(FakeEngine.aborted, sent, f"drained={FakeEngine.drained}")
        self.assertEqual(FakeEngine.ignored, [])

    def test_a_launcher_script_that_sets_the_override_is_read(self):
        """The flash lane's unit runs launch-flash.sh, which holds the engine's flags."""
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
            f.write("#!/bin/bash\n# -e SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES=0 is not ours\n"
                    "exec /usr/bin/docker run --rm --name qwen38-flash \\\n"
                    "  -e SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES=1 \\\n  lmsysorg/sglang\n")
        self.addCleanup(os.unlink, f.name)
        self._serving_unit(f"/bin/bash {f.name}")
        sent = self._abandon_anthropic()
        self.assertEqual(FakeEngine.aborted, sent, f"drained={FakeEngine.drained}")

    def test_a_unit_without_the_override_is_still_drained(self):
        """vLLM's lanes, and an engine started without it: nothing changes for them."""
        self._serving_unit("/bin/bash -c exec /usr/bin/docker run --rm --name qwen38-kolibri vllm/vllm-openai")
        self._abandon_anthropic()
        self.assertEqual(FakeEngine.aborted, [])
        self.assertEqual(FakeEngine.ignored, [])
        self.assertEqual(len(FakeEngine.drained), 1)

    def test_an_anthropic_stream_never_decides_whether_the_override_is_honoured(self):
        """Its first event carries msg_<uuid>, never the rid: read as a proof, it would say
        "ignored" and take the abort away from every route."""
        s = socket.create_connection(("127.0.0.1", self.port), timeout=15)
        raw = json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True}).encode()
        s.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: p\r\nContent-Type: application/json\r\n"
                  + f"Content-Length: {len(raw)}\r\n\r\n".encode() + raw)
        s.settimeout(15)
        got = b""
        while b"[DONE]" not in got:
            c = s.recv(4096)
            if not c:
                break
            got += c
        s.close()
        self.assertIn(b"[DONE]", got)
        self.assertIsNone(self.mod._rid_override_honoured)

    def test_engine_ignoring_the_header_is_drained_not_falsely_aborted(self):
        """SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES is off by default, so the engine answers
        with a rid of its own. A rid the proxy invented would then abort nothing while
        logging a success, which is worse than saying nothing: the answer is drained."""
        with FakeEngine.lock:
            FakeEngine.honour = False
        self._prove_override()
        self.assertIs(self.mod._rid_override_honoured, False)
        self._send("/v1/chat/completions",
                   {"messages": [{"role": "user", "content": "slow-prefill"}]})
        self.assertTrue(self._wait(lambda: FakeEngine.drained or FakeEngine.aborted
                                   or FakeEngine.ignored, timeout=20))
        self.assertEqual(FakeEngine.aborted, [])
        self.assertEqual(FakeEngine.ignored, [])
        self.assertEqual(len(FakeEngine.drained), 1, "the abandoned answer was dropped, not drained")

    def test_a_client_that_stays_is_relayed_untouched(self):
        """Nothing above may cost a normal request its stream."""
        s = socket.create_connection(("127.0.0.1", self.port), timeout=15)
        raw = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: p\r\n"
                  b"Content-Type: application/json\r\n"
                  + f"Content-Length: {len(raw)}\r\n\r\n".encode() + raw)
        s.settimeout(15)
        got = b""
        while b"[DONE]" not in got:
            c = s.recv(4096)
            if not c:
                break
            got += c
        s.close()
        self.assertIn(b"[DONE]", got)
        self.assertEqual(got.count(b'"n": 5'), 1, "the last event never reached the client")
        # The engine writes its drain down after the [DONE] is out, so the client can hold
        # the whole answer before the engine's thread has recorded it: the count read 0 once
        # in a full local run (2026-10-04). Wait for the record, as the tests above do, then
        # half a second more: an abort sent after a whole answer was relayed reaches the
        # engine after its drain, and lands in ignored, its state gone by then.
        self.assertTrue(self._wait(lambda: FakeEngine.drained or FakeEngine.aborted
                                   or FakeEngine.ignored, timeout=10))
        time.sleep(0.5)
        with FakeEngine.lock:
            self.assertEqual(FakeEngine.aborted, [])
            self.assertEqual(FakeEngine.ignored, [], "a request relayed whole was aborted after it")
            self.assertEqual(len(FakeEngine.drained), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
