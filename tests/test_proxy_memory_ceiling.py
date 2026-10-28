#!/usr/bin/env python3
"""The proxy unit's MemoryMax holds the bodies the proxy accepts at once (2026-10-18).

BODY_BUDGET_BYTES (384 MiB) bounds the bytes of the bodies in flight, and MemoryMax with
MemorySwapMax=0 the proxy's memory, so that a proxy past any real use is stopped alone. But a
body costs the proxy about four times its bytes while the guards work on it: the
reasoning_effort route parses it and writes it again. Measured in a cloud container (x86_64,
Python 3.13.16): one body of 250 MiB, under MAX_BODY_BYTES, with reasoning_effort "lean", took
the proxy from 32 to 1034 MiB, and under systemd 255 a unit capped at 1G was killed by the
kernel for it ("Memory cgroup out of memory"), with every stream in flight; the client got an
empty reply (found in a cloud review). On the reference box (aarch64, Python 3.12, 2026-10-18),
three runs each: 3.99 times a 24 MiB body and 4.01 times a 250 MiB one, the proxy going from 31
to 1033 MiB for the larger, the same for the proxy of v1.36.2.

This measures that rate on the machine that runs it, on the real proxy as a process, for three
24 MiB bodies sent as UTF-8, as clients send them: an image as base64 (ASCII, the shape of the
largest legitimate bodies), Chinese text and emoji, each on a proxy of its own (its peak only
grows), and holds the unit to the costliest: the idle proxy plus the budget at that rate fits
under MemoryMax, with a tenth to spare. Until the rewrite kept the text as UTF-8 (v6.46), text in
several bytes per character cost more, as the rewrite escaped every such character: measured on
the reference box on 2026-10-24 with these bodies, 5.65 times for Chinese and 7.98 for emoji on
v6.45, against 3.98 and 3.99 on v6.46, and 3.98 to 3.99 for base64 on both (three runs each).
"""
import http.client
import http.server
import json
import os
import pathlib
import re
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import localports  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
BODY_MIB = 24


class FakeEngine(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def reply(self, obj, code=200):
        out = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            return self.reply({"object": "list", "data": [{"id": "qwen3.8-27b", "object": "model"}]})
        return self.reply({})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        while n > 0:
            chunk = self.rfile.read(min(n, 1 << 20))
            if not chunk:
                break
            n -= len(chunk)
        self.reply({"id": "c", "object": "chat.completion", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]})


class Engine(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def peak_mib(pid):
    """The process's highest resident set so far (VmHWM), in MiB."""
    with open(f"/proc/{pid}/status") as f:
        return next(int(ln.split()[1]) for ln in f if ln.startswith("VmHWM:")) / 1024


def unit_bytes(name):
    """A size directive of the proxy's unit template, in bytes (systemd's K, M, G are 1024's)."""
    text = (REPO / "qwen38-keepalive.service.template").read_text()
    m = re.search(rf"^{name}=(\d+)([KMGT]?)$", text, re.M)
    if not m:
        raise AssertionError(f"the proxy unit has no {name}= the test can read")
    return int(m.group(1)) * 1024 ** " KMGT".index(m.group(2) or " ")


def proxy_default(name):
    """A size default of keepalive-proxy.py, as its source spells it (a product of whole numbers)."""
    text = (REPO / "keepalive-proxy.py").read_text()
    m = re.search(rf'^{name} = int\(os\.environ\.get\("{name}", str\(([\d *]+)\)\)', text, re.M)
    if not m:
        raise AssertionError(f"keepalive-proxy.py no longer spells the {name} default the way this test reads it")
    value = 1
    for factor in m.group(1).split("*"):
        value *= int(factor)
    return value


SHAPES = {
    "base64": lambda: [{"type": "text", "text": "describe"},
                       {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * (BODY_MIB << 20)}}],
    "chinese": lambda: "中" * ((BODY_MIB << 20) // 3),
    "emoji": lambda: "😀" * ((BODY_MIB << 20) // 4),
}


class TheCeilingHoldsTheBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(("127.0.0.1", 0), FakeEngine)
        threading.Thread(target=cls.engine.serve_forever, daemon=True).start()
        cls.home = pathlib.Path(tempfile.mkdtemp(prefix="kp-ceiling-"))
        (cls.home / ".config" / "qwen38").mkdir(parents=True)
        (cls.home / ".config" / "qwen38" / "api-key").write_text("k-test\n")

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()
        cls.engine.server_close()
        shutil.rmtree(cls.home, ignore_errors=True)

    def rate(self, shape):
        """The proxy's peak resident growth for one body of this shape, over the body's bytes, on a proxy
        of its own: its peak (VmHWM) only grows, so a second body would measure from the first's peak."""
        env = {"PATH": os.environ["PATH"], "HOME": str(self.home), "PROXY_BIND": "127.0.0.1",
               "UPSTREAM": f"http://127.0.0.1:{self.engine.server_address[1]}", "PROXY_HOLD_UNITS": ""}
        proxy, port = localports.spawn(lambda port: subprocess.Popen(
            [sys.executable, str(REPO / "keepalive-proxy.py"), str(port)], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        try:
            self.send(port, {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "2+2?"}]})
            idle = peak_mib(proxy.pid)
            body = json.dumps({"model": "qwen3.8-27b", "reasoning_effort": "lean",
                               "messages": [{"role": "user", "content": SHAPES[shape]()}]}, ensure_ascii=False).encode()
            self.assertIsNotNone(self.send(port, body))
            self.assertIsNone(proxy.poll(), f"the proxy died on the {shape} body")
            return idle, (peak_mib(proxy.pid) - idle) / (len(body) / 2**20)
        finally:
            proxy.terminate()
            proxy.wait(10)

    def send(self, port, doc):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
        try:
            body = doc if isinstance(doc, bytes) else json.dumps(doc).encode()
            c.request("POST", "/v1/chat/completions", body=body,
                      headers={"Content-Type": "application/json", "Authorization": "Bearer k-test"})
            r = c.getresponse()
            r.read()
            return r.status
        finally:
            c.close()

    def test_the_idle_proxy_and_the_budget_at_the_costliest_rate_fit_under_memorymax(self):
        rates = {}
        for shape in SHAPES:
            idle, rates[shape] = self.rate(shape)
            self.assertGreater(rates[shape], 1, f"the {shape} body did not reach the guards: the measurement measured nothing")
        shape, rate = max(rates.items(), key=lambda kv: kv[1])
        budget = proxy_default("BODY_BUDGET_BYTES") / 2**20
        ceiling = unit_bytes("MemoryMax") / 2**20
        need = idle + rate * max(budget, proxy_default("MAX_BODY_BYTES") / 2**20)
        self.assertLessEqual(need, 0.9 * ceiling,
                             f"rates {', '.join(f'{k} {v:.2f}' for k, v in rates.items())}: idle {idle:.0f} MiB + {rate:.1f} "
                             f"({shape}) x a budget of {budget:.0f} MiB = {need:.0f} MiB, over nine tenths of MemoryMax "
                             f"({ceiling:.0f} MiB): the kernel kills the proxy first")

    def test_the_ceiling_is_not_moved_to_swap(self):
        # with swap on the box, MemoryMax alone moves the excess to swap (measured 2026-10-08)
        self.assertEqual(unit_bytes("MemorySwapMax"), 0)

    def test_the_readers_read_what_is_there(self):
        # so a spelling that changed fails here, and not as a ceiling of zero
        self.assertEqual(proxy_default("BODY_BUDGET_BYTES"), 384 * 1024 * 1024)
        self.assertEqual(proxy_default("MAX_BODY_BYTES"), 256 * 1024 * 1024)
        self.assertGreaterEqual(unit_bytes("MemoryMax"), 1024 ** 3)


if __name__ == "__main__":
    unittest.main()
