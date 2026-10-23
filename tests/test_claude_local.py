#!/usr/bin/env python3
"""claude-local runs Claude Code on the lane the box serves (discussion #59, 2026-10-23).

What it hands Claude Code is checked here end to end, against a fake proxy that serves
/v1/models and a fake `claude` that writes down its arguments and its environment: the key
only in the environment, never on the command line, the model the proxy serves, and the
context and answer budget derived from the lane's opencode limits by Claude Code's own rule
(its longest prompt is CLAUDE_CODE_MAX_CONTEXT_TOKENS - 23,000, measured against Claude Code
2.1.304 on 2026-10-23). The installer's own lines write the launcher with HOME in a temporary
directory, and leave a claude-local that is not this repo's where it is."""
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
        base_env = {"HOME": str(self.d), "CONFIG_DIR": str(self.cfg), "LANG": "C.UTF-8",
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


class TheInstallerWritesIt(unittest.TestCase):
    TEXT = (REPO / "install.sh").read_text()
    START = TEXT.index('if [ -z "$SECONDARY" ]; then\n  CL_BIN=')
    BLOCK = TEXT[START:TEXT.index("\nfi\n", TEXT.index('echo "installed the claude-local launcher at', START)) + len("\nfi\n")]

    def run_block(self, existing=None):
        home = pathlib.Path(tempfile.mkdtemp(prefix="claude-local-home-"))
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(home)]))
        bindir = home / ".local/bin"
        bindir.mkdir(parents=True)
        if existing is not None:
            (bindir / "claude-local").write_bytes(existing)
        script = ("set -euo pipefail\n"
                  f'HOME="{home}"; CONFIG_DIR="{home}/.config/qwen38"; REPO_DIR=/the/clone; SECONDARY=""\n' + self.BLOCK)
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                           env={"PATH": "/usr/bin:/bin", "HOME": str(home)})
        out = bindir / "claude-local"
        return r, out.read_bytes() if out.exists() else None

    def test_a_box_gets_the_launcher(self):
        r, text = self.run_block()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(b"dgx-spark-qwen38", text)
        self.assertIn(b'exec python3 -I "/the/clone/claude-local.py" "$@"', text)

    def test_a_foreign_claude_local_is_left_alone(self):
        foreign = b"#!/bin/sh\necho someone else's\n"
        r, text = self.run_block(existing=foreign)
        self.assertEqual(text, foreign)
        self.assertIn("an unrelated 'claude-local' exists", r.stdout)

    def test_the_uninstaller_lists_and_removes_only_ours(self):
        text = (REPO / "uninstall.sh").read_text()
        self.assertIn("""if grep -q 'dgx-spark-qwen38' "$HOME/.local/bin/claude-local" 2>/dev/null; then
  rm -f "$HOME/.local/bin/claude-local\"""", text)
        self.assertIn('"  launcher  $HOME/.local/bin/claude-local (this repo\'s Claude Code launcher)"', text)


if __name__ == "__main__":
    unittest.main()
