#!/usr/bin/env python3
"""./install-nemotron.sh --uninstall gives the boot back to the box's text lane, takes its opencode
provider out, and keeps the key's middleware while the Kolibri lane uses it; and the Kolibri
lane's uninstall keeps it while the Nemotron lane does (vllm-auth is the two vLLM lanes' one
folder: removed by either, it left the other's engine without its middleware, which its unit
mounts read-only and loads at every start).

The scripts' own uninstall blocks run here, whole, against a fake unit path, a fake config
directory and a fake /etc/systemd/system; sudo, systemctl and docker are stubs that write down
every call, as in test_kolibri_uninstall_chain.py.
"""
import json
import os
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]


def block(script):
    text = (REPO / script).read_text()
    start = text.index('if [ "$ACTION" = uninstall ]; then')
    return text[start:text.index("\n  exit 0\nfi\n", start) + len("\n  exit 0\nfi\n")]


NEM_BLOCK = block("install-nemotron.sh")
KOL_BLOCK = block("install-kolibri.sh")


class TheUninstall(unittest.TestCase):
    def run_block(self, lane="nemotron", files=None, at_boot=False, opencode=None,
                  units=("qwen38-flash.service", "qwen38-sglang.service")):
        """lane: whose uninstall runs; units: what the fake /etc/systemd/system holds."""
        d = pathlib.Path(tempfile.mkdtemp(prefix="nem-uninstall-"))
        sd = d / "systemd"
        sd.mkdir()
        for u in units:
            (sd / u).write_text("[Service]\n")
        self.home = d / "home"
        cfg = self.cfg = d / "cfg"
        cfg.mkdir()
        (self.home / ".config" / "opencode").mkdir(parents=True)
        (cfg / "vllm-auth").mkdir()
        (cfg / "nemotron-cache" / "vllm").mkdir(parents=True)
        (cfg / "vllm-auth" / "vllm_engine_auth.py").write_text("# the middleware\n")
        if opencode is not None:
            (cfg / "opencode.json").write_text(json.dumps(opencode, indent=2))
            (self.home / ".config" / "opencode" / "opencode.json").write_text(json.dumps(opencode, indent=2))
        for name, value in (files or {}).items():
            (cfg / name).write_text(value + "\n")
        bin_ = d / "bin"
        bin_.mkdir()
        log = self.log = d / "calls.log"
        for tool in ("sudo", "docker"):
            (bin_ / tool).write_text(f'#!/bin/bash\necho "{tool} $*" >> "{log}"\nexit 0\n')
        unit_name = f"qwen38-{lane}.service"
        (bin_ / "systemctl").write_text(
            f'#!/bin/bash\necho "systemctl $*" >> "{log}"\n'
            f'[ "$1" = is-enabled ] && [ "${{@: -1}}" = {unit_name} ] && exit {0 if at_boot else 1}\n'
            '[ "$1" = is-enabled ] && exit 1\nexit 0\n')
        for f in bin_.iterdir():
            f.chmod(0o755)
        unit = d / unit_name
        unit.write_text("[Service]\n")
        pins = (["NEMOTRON_BASE_IMAGE=vllm/vllm-openai@sha256:0",
                 "NEMOTRON_BOLT_REPO=nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
                 "NEMOTRON_BOLT_DRAFT_REPO=nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr"]
                if lane == "nemotron" else
                ["KOLIBRI_BASE_IMAGE=vllm/vllm-openai@sha256:0", "KOLIBRI_FP8_REPO=Aleph-Alpha/Kolibri-1",
                 "KOLIBRI_NVFP4_REPO=Quantinine/Kolibri-1-NVFP4A16-FP8"])
        script = "\n".join([
            "set -euo pipefail", "step(){ :; }",
            f'LANES_UNIT_DIR="{sd}"', f'. "{REPO}/lanes.sh"',
            f'HERE="{REPO}"', "ACTION=uninstall", f"UNIT={unit_name}", f'INSTALLED="{unit}"',
            f"CONTAINER=qwen38-{lane}", f'CONFIG_DIR="{cfg}"', f'HF_CACHE="{d}/hf"', *pins,
            (NEM_BLOCK if lane == "nemotron" else KOL_BLOCK).replace("/etc/systemd/system/", f"{sd}/"),
        ])
        env = dict(os.environ, PATH=f"{bin_}:/usr/bin:/bin", HOME=str(self.home))
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=30)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.out = r.stdout
        return cfg

    def enabled(self):
        return [ln.split()[-1] for ln in self.log.read_text().splitlines() if ln.startswith("sudo systemctl enable")]

    def test_at_boot_it_gives_the_boot_back_to_the_text_lane_noted(self):
        self.run_block(files={"text-lane": "qwen38-flash.service"}, at_boot=True)
        self.assertEqual(self.enabled(), ["qwen38-flash.service"])
        self.assertIn("sudo systemctl disable --now qwen38-nemotron.service", self.log.read_text())

    def test_with_nothing_noted_the_installed_text_lane_gets_the_boot(self):
        self.run_block(at_boot=True)
        self.assertEqual(self.enabled(), ["qwen38-sglang.service"])
        self.run_block(at_boot=True, units=())
        self.assertEqual(self.enabled(), [])
        self.assertIn("no engine is enabled at boot any more", self.out)

    def test_not_at_boot_it_enables_nothing(self):
        self.run_block(files={"text-lane": "qwen38-flash.service"}, at_boot=False)
        self.assertEqual(self.enabled(), [])

    def conf(self, default):
        return {"$schema": "https://opencode.ai/config.json", "model": default, "small_model": default,
                "compaction": {"auto": True, "preserve_recent_tokens": 46000},
                "provider": {p: {"options": {"baseURL": "http://127.0.0.1:30001/v1", "apiKey": "{file:/k/api-key}"},
                                 "models": {m: {"name": m, "limit": {"context": c, "output": o}}}}
                             for p, m, c, o in (("qwen38", "qwen3.8-27b", 529000, 250000),
                                                ("flashnext", "qwen3.8-flash-next", 205000, 32000),
                                                ("nemotron", "nemotron-3.5-bolt", 205000, 32000))}}

    def test_opencode_loses_the_provider_and_its_default_follows_the_lane_before(self):
        self.run_block(files={"text-lane": "qwen38-flash.service"}, opencode=self.conf("nemotron/nemotron-3.5-bolt"))
        for path in (self.cfg / "opencode.json", self.home / ".config/opencode/opencode.json"):
            left = json.loads(path.read_text())
            self.assertNotIn("nemotron", left["provider"], path)
            self.assertEqual((left["model"], left["small_model"]), ("flashnext/qwen3.8-flash-next",) * 2, path)

    def test_the_compaction_follows_the_default_put_back(self):
        self.run_block(files={"text-lane": "qwen38-sglang.service"}, opencode=self.conf("nemotron/nemotron-3.5-bolt"))
        want = int(subprocess.run([str(REPO / "oc-limits.sh"), "--preserve", "529000"],
                                  capture_output=True, text=True, check=True).stdout)
        left = json.loads((self.cfg / "opencode.json").read_text())
        self.assertEqual(left["model"], "qwen38/qwen3.8-27b")
        self.assertEqual(left["compaction"]["preserve_recent_tokens"], want)

    def test_a_default_that_was_not_nemotron_is_left_where_it_was(self):
        self.run_block(files={"text-lane": "qwen38-flash.service"}, opencode=self.conf("qwen38/qwen3.8-27b"))
        left = json.loads((self.cfg / "opencode.json").read_text())
        self.assertEqual(left["model"], "qwen38/qwen3.8-27b")
        self.assertNotIn("nemotron", left["provider"])

    def test_the_middleware_stays_while_the_other_vllm_lane_uses_it(self):
        for lane, other in (("nemotron", "qwen38-kolibri.service"), ("kolibri", "qwen38-nemotron.service")):
            with self.subTest(lane=lane):
                cfg = self.run_block(lane=lane, units=("qwen38-sglang.service", other))
                self.assertTrue((cfg / "vllm-auth" / "vllm_engine_auth.py").exists(),
                                f"the {other} engine would start without its middleware")
                self.assertIn("vllm-auth kept", self.out)
                cfg = self.run_block(lane=lane, units=("qwen38-sglang.service",))
                self.assertFalse((cfg / "vllm-auth").exists(), "alone, it goes with the lane")

    def test_its_images_container_and_caches_go_and_its_checkpoints_stay(self):
        self.run_block()
        calls = self.log.read_text()
        self.assertIn("docker rm -f qwen38-nemotron", calls)
        self.assertIn("docker images --format {{.Repository}}:{{.Tag}} qwen38-nemotron", calls)
        self.assertFalse((self.cfg / "nemotron-cache").exists())


if __name__ == "__main__":
    unittest.main()
