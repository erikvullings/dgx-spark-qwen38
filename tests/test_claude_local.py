#!/usr/bin/env python3
"""claude-local runs Claude Code on the lane the box serves (discussion #59, 2026-10-23).

What it hands Claude Code is checked here end to end, against a fake proxy that serves
/v1/models and a fake `claude` that writes down its arguments and its environment: the key
only in the environment, never on the command line, the model the proxy serves, and the
context and answer budget derived from the lane's opencode limits by Claude Code's own rule
(its longest prompt is CLAUDE_CODE_MAX_CONTEXT_TOKENS - 23,000, measured against Claude Code
2.1.304 on 2026-10-23). The installer's own lines write the launcher with HOME in a temporary
directory, and leave a claude-local that is not this repo's where it is, but for the one
discussion #59's reply gave to copy there, which they keep beside and replace."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "claude-local.py"
spec = importlib.util.spec_from_file_location("claude_local", SCRIPT)
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)
KEY = "test-key-not-a-real-one"
# The launcher discussion #59's reply of 2026-10-22 gave to copy to ~/.local/bin/claude-local,
# byte for byte as it was published
FROM_59 = rb'''#!/bin/bash
# ~/.local/bin/claude-local: Claude Code on the lane the box serves
export ANTHROPIC_BASE_URL=http://127.0.0.1:30001   # from another machine: http://<box>:30001
export ANTHROPIC_AUTH_TOKEN="$(head -1 ~/.config/qwen38/api-key)"
export ANTHROPIC_API_KEY=""
# any model name Claude Code sends is answered by the lane that serves
export ANTHROPIC_MODEL=qwen3.8-27b
# how much context Claude Code may fill: a pair that fits every lane's 262K window,
# or the 27B's measured pair when it serves in its 1M mode (the default)
CTX=205000 OUT=32000
systemctl is-active --quiet qwen38-sglang.service && CTX=529000 OUT=250000
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=$CTX CLAUDE_CODE_EXTRA_BODY="{\"max_tokens\":$OUT}"
# a long prefill or a long answer is not a dead stream
export API_TIMEOUT_MS=3600000 CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000 CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS=1800000
# Unsloth's advice for local servers: a header that changes at every request defeats the prefix cache
export CLAUDE_CODE_ATTRIBUTION_HEADER=0 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
exec claude "$@"
'''


class Pair(unittest.TestCase):
    def test_the_longest_prompt_is_the_lanes_opencode_context(self):
        # FP4-Local's limits on the reference box: the prompt and the answer stay under its
        # 262,144-token window with opencode's worst step to spare
        ctx, out, _ = cl.claude_pair((205_000, 32_000), 262_144, 32_000)
        self.assertEqual((ctx, out), (228_000, 32_000))
        self.assertEqual(ctx - 23_000 + out, 237_000)
        self.assertEqual(cl.claude_pair((555_000, 185_000), 1_010_000, 32_000)[:2], (578_000, 185_000))
        self.assertEqual(cl.claude_pair((990_000, 32_000), 0, 32_000)[:2], (1_013_000, 32_000))

    def test_a_smaller_output_setting_moves_the_block_with_it(self):
        # Claude Code subtracts min(its output setting, 20,000) + 3,000
        self.assertEqual(cl.claude_pair((205_000, 32_000), 262_144, 8_000)[0], 205_000 + 11_000)

    def test_a_lane_opencode_knows_nothing_of_gets_the_pair_that_fits_every_lane(self):
        ctx, out, why = cl.claude_pair(None, 262_144, 32_000)
        self.assertEqual((ctx, out), (205_000, 32_000))
        self.assertIn("fits every lane", why)
        # and a window smaller than every lane today brings it down to what that window holds
        ctx, out, _ = cl.claude_pair(None, 131_072, 32_000)
        self.assertEqual(ctx - 23_000 + out, 131_072)


class Fake(BaseHTTPRequestHandler):
    models = {"data": [{"id": "qwen3.8-flash-next-fp4", "max_model_len": 262_144}]}
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        Fake.seen.append((self.path, self.headers.get("Authorization")))
        body = json.dumps(Fake.models).encode()
        self.send_response(200 if self.path == "/v1/models" else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp(prefix="claude-local-"))
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(self.d)]))
        self.cfg = self.d / "config"
        self.cfg.mkdir()
        (self.cfg / "api-key").write_text(KEY + "\n")
        (self.cfg / "opencode.json").write_text(json.dumps({"provider": {"flash-local": {
            "options": {"baseURL": self.base + "/v1"},
            "models": {"qwen3.8-flash-next-fp4": {"limit": {"context": 205_000, "input": 205_000, "output": 32_000}}}}}}))
        self.bin = self.d / "bin"
        self.bin.mkdir()
        self.record = self.d / "claude-saw.json"
        fake = self.bin / "claude"
        fake.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                        f"json.dump({{'argv': sys.argv, 'env': dict(os.environ)}}, open({str(self.record)!r}, 'w'))\n")
        fake.chmod(0o755)
        Fake.seen = []

    def launch(self, *args, env=None, with_claude=True):
        base_env = {"HOME": str(self.d), "QWEN38_CONFIG_DIR": str(self.cfg), "LANG": "C.UTF-8",
                    "PATH": (f"{self.bin}:" if with_claude else "") + "/usr/bin:/bin"}
        r = subprocess.run([sys.executable, "-I", str(SCRIPT), *args], capture_output=True, text=True,
                           timeout=60, env={**base_env, **(env or {})})
        saw = json.loads(self.record.read_text()) if self.record.exists() else None
        return r, saw

    def test_claude_gets_the_lane_the_key_and_the_limits(self):
        r, saw = self.launch("-p", "hello")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(saw["argv"][1:], ["-p", "hello"])
        env = saw["env"]
        self.assertEqual(env["ANTHROPIC_BASE_URL"], self.base)
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], KEY)
        self.assertEqual(env["ANTHROPIC_API_KEY"], "")
        self.assertEqual(env["ANTHROPIC_MODEL"], "qwen3.8-flash-next-fp4")
        self.assertEqual(env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "228000")
        self.assertEqual(json.loads(env["CLAUDE_CODE_EXTRA_BODY"]), {"max_tokens": 32_000})
        self.assertNotIn("CLAUDE_CODE_ATTRIBUTION_HEADER", env, "measured to change nothing (2026-10-22)")
        # the proxy was asked with the key, as a header
        self.assertEqual(Fake.seen, [("/v1/models", f"Bearer {KEY}")])

    def test_the_key_is_never_on_a_command_line(self):
        r, saw = self.launch("-p", "hello")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(any(KEY in a for a in saw["argv"]))

    def test_your_own_settings_are_kept(self):
        mine = {"CLAUDE_CODE_MAX_CONTEXT_TOKENS": "150000", "CLAUDE_CODE_EXTRA_BODY": '{"max_tokens": 8000}',
                "ANTHROPIC_MODEL": "my-name", "API_TIMEOUT_MS": "60000"}
        r, saw = self.launch(env=mine)
        self.assertEqual(r.returncode, 0, r.stderr)
        for k, v in mine.items():
            self.assertEqual(saw["env"][k], v, k)

    def test_a_lane_without_opencode_limits_gets_the_pair_that_fits_every_lane(self):
        Fake.models = {"data": [{"id": "some-new-lane"}]}
        self.addCleanup(setattr, Fake, "models", {"data": [{"id": "qwen3.8-flash-next-fp4", "max_model_len": 262_144}]})
        r, saw = self.launch()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(saw["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "205000")

    def test_the_probe_goes_to_the_box_even_with_an_http_proxy_set(self):
        # urllib followed http_proxy, and the key went out in clear to it unless no_proxy named
        # 127.0.0.1 (found in review, 2026-10-24); Claude Code is told to go direct too
        seen = []
        class Proxy(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass
            def do_GET(self):
                seen.append((self.path, self.headers.get("Authorization")))
                self.send_response(502); self.send_header("Content-Length", "0"); self.end_headers()
        proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        self.addCleanup(proxy.shutdown)
        url = f"http://127.0.0.1:{proxy.server_address[1]}"
        r, saw = self.launch(env={"http_proxy": url, "HTTP_PROXY": url, "no_proxy": "localhost"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(seen, [], "the probe went through the HTTP proxy")
        self.assertIn("127.0.0.1", saw["env"]["NO_PROXY"].split(","))
        self.assertEqual(saw["env"]["no_proxy"].split(",")[0], "localhost", "what was there is kept")

    def test_an_answer_that_is_an_error_says_its_code(self):
        Fake.models = {"error": "x"}
        self.addCleanup(setattr, Fake, "models", {"data": [{"id": "qwen3.8-flash-next-fp4", "max_model_len": 262_144}]})
        class Refusing(Fake):
            def do_GET(self):
                self.send_response(401); self.send_header("Content-Length", "0"); self.end_headers()
        srv = ThreadingHTTPServer(("127.0.0.1", 0), Refusing)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        r, saw = self.launch(env={"CLAUDE_LOCAL_URL": f"http://127.0.0.1:{srv.server_address[1]}"})
        self.assertEqual(r.returncode, 1)
        self.assertIn("answered HTTP 401: the box's key was refused", r.stderr)
        self.assertNotIn("load one from the cockpit", r.stderr)
        # a body that is not the list it should be is said, not a traceback
        r, saw = self.launch()
        self.assertEqual(r.returncode, 1)
        self.assertIn("lists no model", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_an_output_setting_that_is_not_a_number_is_said(self):
        r, saw = self.launch(env={"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32k"})
        self.assertEqual(r.returncode, 2)
        self.assertIn("CLAUDE_CODE_MAX_OUTPUT_TOKENS=32k is not a number of tokens", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertIsNone(saw)

    def test_an_extra_body_of_yours_keeps_the_lanes_answer_budget(self):
        r, saw = self.launch(env={"CLAUDE_CODE_EXTRA_BODY": '{"temperature": 0.2}'})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(saw["env"]["CLAUDE_CODE_EXTRA_BODY"]), {"temperature": 0.2, "max_tokens": 32_000})

    def test_a_claude_model_name_of_yours_is_said_to_undo_the_context(self):
        r, saw = self.launch(env={"ANTHROPIC_MODEL": "claude-sonnet-4-5"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(saw["env"]["ANTHROPIC_MODEL"], "claude-sonnet-4-5")
        self.assertIn("names a Claude model", r.stderr)

    def test_a_config_dir_another_tool_exports_is_not_taken(self):
        r, saw = self.launch(env={"CONFIG_DIR": str(self.d / "elsewhere")})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(saw["env"]["ANTHROPIC_AUTH_TOKEN"], KEY)

    def test_no_claude_says_where_to_get_it(self):
        r, saw = self.launch(with_claude=False)
        self.assertEqual(r.returncode, 127)
        self.assertIn("Claude Code is not installed", r.stderr)
        self.assertIn("https://code.claude.com/docs/en/setup", r.stderr)
        self.assertIsNone(saw)

    def test_no_lane_says_to_load_one(self):
        (self.cfg / "opencode.json").write_text(json.dumps({"provider": {"x": {"options": {"baseURL": "http://127.0.0.1:1/v1"}}}}))
        r, saw = self.launch()
        self.assertEqual(r.returncode, 1)
        self.assertIn("load one from the cockpit first", r.stderr)
        self.assertIsNone(saw)


class TheProxyItReaches(unittest.TestCase):
    def test_with_no_opencode_config_the_launchers_port(self):
        # with --no-opencode there is no opencode.json, and the default port missed a box whose
        # proxy listens elsewhere (found in review, 2026-10-24); checked in this process, so no old
        # code ever reaches the proxy on this box's port
        old = os.environ.get("QWEN38_PROXY_URL")
        self.addCleanup(lambda: os.environ.__setitem__("QWEN38_PROXY_URL", old) if old is not None
                        else os.environ.pop("QWEN38_PROXY_URL", None))
        os.environ["QWEN38_PROXY_URL"] = "http://127.0.0.1:31001"
        self.assertEqual(cl.proxy_base({}), "http://127.0.0.1:31001")
        self.assertEqual(cl.proxy_base({"provider": {"x": {"options": {"baseURL": "http://127.0.0.1:30001/v1"}}}}),
                         "http://127.0.0.1:30001")


class TheInstallerWritesIt(unittest.TestCase):
    TEXT = (REPO / "install.sh").read_text()
    START = TEXT.index('if [ -z "$SECONDARY" ]; then\n  CL_BIN=')
    BLOCK = TEXT[START:TEXT.index("\nfi\n", TEXT.index('echo "installed the claude-local launcher at', START)) + len("\nfi\n")]

    def run_block(self, existing=None):
        home = pathlib.Path(tempfile.mkdtemp(prefix="claude-local-home-"))
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(home)]))
        bindir = self.bindir = home / ".local/bin"
        bindir.mkdir(parents=True)
        if existing is not None:
            (bindir / "claude-local").write_bytes(existing)
        script = ("set -euo pipefail\n"
                  f'HOME="{home}"; CONFIG_DIR="{home}/.config/qwen38"; REPO_DIR=/the/clone; SECONDARY=""; PROXY_PORT=31001\n'
                  + self.BLOCK)
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                           env={"PATH": "/usr/bin:/bin", "HOME": str(home)})
        out = bindir / "claude-local"
        return r, out.read_bytes() if out.exists() else None

    def test_a_box_gets_the_launcher(self):
        r, text = self.run_block()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(b"dgx-spark-qwen38", text)
        self.assertIn(b'exec python3 -I "/the/clone/claude-local.py" "$@"', text)
        # its settings by names of this repo's own, the proxy's port the box's
        self.assertIn(b'export QWEN38_CONFIG_DIR="${QWEN38_CONFIG_DIR:-', text)
        self.assertIn(b'export QWEN38_PROXY_URL="${QWEN38_PROXY_URL:-http://127.0.0.1:31001}"', text)
        self.assertNotIn(b"export CONFIG_DIR", text)

    def test_a_launcher_that_cannot_be_written_does_not_stop_the_install(self):
        # set -e stopped the whole run before its units (found in review, 2026-10-24)
        home = pathlib.Path(tempfile.mkdtemp(prefix="claude-local-home-"))
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(home)]))
        (home / ".local").mkdir()
        (home / ".local/bin").write_text("a file where the folder goes\n")
        script = ("set -euo pipefail\n"
                  f'HOME="{home}"; CONFIG_DIR="{home}/.config/qwen38"; REPO_DIR=/the/clone; SECONDARY=""; PROXY_PORT=31001\n'
                  + self.BLOCK + 'echo "went on"\n')
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                           env={"PATH": "/usr/bin:/bin", "HOME": str(home)})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("NOTE: could not write the claude-local launcher", r.stdout)
        self.assertIn("went on", r.stdout)

    def test_a_foreign_claude_local_is_left_alone(self):
        foreign = b"#!/bin/sh\necho someone else's\n"
        r, text = self.run_block(existing=foreign)
        self.assertEqual(text, foreign)
        self.assertIn("an unrelated 'claude-local' exists", r.stdout)

    def test_the_launcher_59s_reply_gave_is_kept_beside_and_replaced(self):
        r, text = self.run_block(existing=FROM_59)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(b'exec python3 -I "/the/clone/claude-local.py" "$@"', text)
        self.assertEqual((self.bindir / "claude-local.from-59").read_bytes(), FROM_59)
        self.assertIn("the claude-local of discussion #59's reply is kept as", r.stdout)
        self.assertNotIn("unrelated", r.stdout)

    def test_an_edited_one_is_kept_beside_as_it_was(self):
        # a model or an address of one's own changes neither its first comment nor its last line
        edited = FROM_59.replace(b"ANTHROPIC_MODEL=qwen3.8-27b", b"ANTHROPIC_MODEL=mine").replace(
            b"http://127.0.0.1:30001", b"http://spark.local:30001")
        r, text = self.run_block(existing=edited)
        self.assertIn(b"dgx-spark-qwen38", text)
        self.assertEqual((self.bindir / "claude-local.from-59").read_bytes(), edited)

    def test_a_file_with_one_of_its_lines_only_is_left_alone(self):
        for keep in (b"# ~/.local/bin/claude-local: Claude Code on the lane the box serves\n", b'exec claude "$@"\n'):
            with self.subTest(keep):
                other = b"#!/bin/sh\n" + keep + b"echo mine\n"
                r, text = self.run_block(existing=other)
                self.assertEqual(text, other)
                self.assertFalse((self.bindir / "claude-local.from-59").exists())
                self.assertIn("an unrelated 'claude-local' exists", r.stdout)

    def test_the_uninstaller_lists_and_removes_only_ours(self):
        text = (REPO / "uninstall.sh").read_text()
        self.assertIn("""if grep -q 'dgx-spark-qwen38' "$HOME/.local/bin/claude-local" 2>/dev/null; then
  rm -f "$HOME/.local/bin/claude-local\"""", text)
        self.assertIn('"  launcher  $HOME/.local/bin/claude-local (this repo\'s Claude Code launcher)"', text)


if __name__ == "__main__":
    unittest.main()
