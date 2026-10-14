#!/usr/bin/env python3
"""./install-kolibri.sh --uninstall gives the boot back to the box's text lane, and keeps the record.

Until v1.29 a switch to a side lane (image, video, Kolibri) wrote down the lane it took the boot
from, in lane-before-<lane>, and these uninstalls followed and spliced those links. Since v1.30
there is one record, text-lane (lanes.sh): the links of a box switched before are read once,
while Kolibri is still the lane at boot, and their answer becomes the record.

The script's own uninstall block runs here, whole, against a fake unit path, a fake
config directory and a fake /etc/systemd/system (the box's own was read before, review of
2026-10-14); sudo, systemctl and docker are stubs that write down every call.
"""
import json
import os
import pathlib
import subprocess
import tempfile
import time
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "install-kolibri.sh").read_text()
START = TEXT.index('if [ "$ACTION" = uninstall ]; then')
END = TEXT.index("\n  exit 0\nfi\n", START) + len("\n  exit 0\nfi\n")
BLOCK = TEXT[START:END]


class TheUninstallKeepsTheChain(unittest.TestCase):
    def run_block(self, files, kolibri_at_boot=False, opencode=None, units=("qwen38-flash.service",
                                                                          "qwen38-sglang.service")):
        """units: the engine units the fake /etc/systemd/system holds."""
        d = pathlib.Path(tempfile.mkdtemp(prefix="kol-uninstall-"))
        sd = d / "systemd"
        sd.mkdir()
        for u in units:
            (sd / u).write_text("[Service]\n")
        self.home = d / "home"
        cfg = d / "cfg"
        cfg.mkdir()
        (self.home / ".config" / "opencode").mkdir(parents=True)
        if opencode is not None:
            (cfg / "opencode.json").write_text(json.dumps(opencode, indent=2))
            (self.home / ".config" / "opencode" / "opencode.json").write_text(json.dumps(opencode, indent=2))
        self.cfg = cfg
        for name, value in files.items():
            (cfg / name).write_text(value + "\n")
            if name.startswith("lane-before-"):
                # older than any record, as on a box: only a version before v1.30 wrote links
                # (a link newer than the record is read again, test_text_lane)
                t = time.time() - 60
                os.utime(cfg / name, (t, t))
        bin_ = d / "bin"
        bin_.mkdir()
        log = self.log = d / "calls.log"
        for tool in ("sudo", "docker"):
            (bin_ / tool).write_text(f'#!/bin/bash\necho "{tool} $*" >> "{log}"\nexit 0\n')
        # is-enabled answers for Kolibri as the case wants, and no for every other unit
        (bin_ / "systemctl").write_text(
            f'#!/bin/bash\necho "systemctl $*" >> "{log}"\n'
            f'[ "$1" = is-enabled ] && [ "${{@: -1}}" = qwen38-kolibri.service ] && exit {0 if kolibri_at_boot else 1}\n'
            '[ "$1" = is-enabled ] && exit 1\nexit 0\n')
        for f in bin_.iterdir():
            f.chmod(0o755)
        unit = d / "qwen38-kolibri.service"
        unit.write_text("[Service]\n")
        script = "\n".join([
            "set -euo pipefail",
            "step(){ :; }",
            f'LANES_UNIT_DIR="{sd}"', f'. "{REPO}/lanes.sh"',   # the library the script sources
            f'HERE="{REPO}"',
            "ACTION=uninstall",
            "UNIT=qwen38-kolibri.service",
            f'INSTALLED="{unit}"',
            "CONTAINER=qwen38-kolibri",
            f'CONFIG_DIR="{cfg}"',
            f'HF_CACHE="{d}/hf"',
            "KOLIBRI_BASE_IMAGE=vllm/vllm-openai@sha256:0",
            "KOLIBRI_FP8_REPO=Aleph-Alpha/Kolibri-1",
            "KOLIBRI_NVFP4_REPO=Quantinine/Kolibri-1-NVFP4A16-FP8",
            BLOCK.replace("/etc/systemd/system/", f"{sd}/"),
        ])
        env = dict(os.environ, PATH=f"{bin_}:/usr/bin:/bin", HOME=str(self.home))
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=30)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.out = r.stdout
        return {p.name: p.read_text().strip() for p in cfg.iterdir()
                if p.name.startswith("lane-before-") or p.name == "text-lane"}

    def test_the_links_of_a_box_switched_before_become_the_record(self):
        # the reference box on 2026-10-14: flash, then image, then Kolibri, then image again
        left = self.run_block({"lane-before-image": "qwen38-kolibri.service",
                               "lane-before-kolibri": "qwen38-flash.service"})
        self.assertEqual(left["text-lane"], "qwen38-flash.service")
        self.assertNotIn("lane-before-kolibri", left, "its own link goes with the lane")
        left = self.run_block({"lane-before-video": "qwen38-kolibri.service",
                               "lane-before-kolibri": "qwen38-sglang.service"})
        self.assertEqual(left["text-lane"], "qwen38-sglang.service")

    def test_links_that_lead_nowhere_write_no_record(self):
        # image and Kolibri naming each other, or a link with nothing behind it: no text lane
        # can be told, and nothing is written (the links of v1.29 are left as they are)
        for files in ({"lane-before-image": "qwen38-kolibri.service", "lane-before-kolibri": "qwen38-image.service"},
                      {"lane-before-video": "qwen38-kolibri.service"}):
            with self.subTest(files=files):
                left = self.run_block(files)
                self.assertNotIn("text-lane", left)

    def test_a_record_already_there_is_kept_and_the_links_are_not_read(self):
        left = self.run_block({"text-lane": "qwen38-flash.service",
                               "lane-before-kolibri": "qwen38-sglang.service"}, kolibri_at_boot=True)
        self.assertEqual(left["text-lane"], "qwen38-flash.service")
        self.assertEqual(self.enabled(), ["qwen38-flash.service"])

    def test_opencode_loses_the_provider_and_its_default_follows_the_lane_before(self):
        # left as they were, opencode offered and defaulted to a model nothing served, and its
        # small-model calls sent "none" to the Qwen template, which refuses it
        conf = {"$schema": "https://opencode.ai/config.json", "model": "kolibri/kolibri-1",
                "small_model": "kolibri/kolibri-1",
                "provider": {p: {"options": {"baseURL": "http://127.0.0.1:30001/v1",
                                             "apiKey": "{file:/k/api-key}"},
                                 "models": {m: {"name": m}}}
                             for p, m in (("qwen38", "qwen3.8-27b"), ("flashnext", "qwen3.8-flash-next"),
                                          ("kolibri", "kolibri-1"))}}
        self.run_block({"lane-before-kolibri": "qwen38-flash.service"}, opencode=conf)
        for path in (self.cfg / "opencode.json", self.home / ".config/opencode/opencode.json"):
            left = json.loads(path.read_text())
            self.assertNotIn("kolibri", left["provider"], path)
            self.assertEqual((left["model"], left["small_model"]),
                             ("flashnext/qwen3.8-flash-next",) * 2, path)
            self.assertEqual(left["provider"]["flashnext"]["models"]["qwen3.8-flash-next"]["name"],
                             "qwen3.8-flash-next", "no name changed")

    def test_a_default_that_was_not_kolibri_is_left_where_it_was(self):
        conf = {"model": "qwen38/qwen3.8-27b",
                "provider": {p: {"options": {"baseURL": "http://127.0.0.1:30001/v1",
                                             "apiKey": "{file:/k/api-key}"},
                                 "models": {m: {"name": m}}}
                             for p, m in (("qwen38", "qwen3.8-27b"), ("kolibri", "kolibri-1"))}}
        self.run_block({"lane-before-kolibri": "qwen38-flash.service"}, opencode=conf)
        left = json.loads((self.cfg / "opencode.json").read_text())
        self.assertEqual(left["model"], "qwen38/qwen3.8-27b")
        self.assertNotIn("kolibri", left["provider"])

    def enabled(self):
        return [ln.split()[-1] for ln in self.log.read_text().splitlines() if ln.startswith("sudo systemctl enable")]

    def test_kolibri_at_boot_still_gives_the_boot_back_to_its_own_lane(self):
        # the handover reads Kolibri's link before anything else touches it
        d_files = {"lane-before-kolibri": "qwen38-flash.service",
                   "lane-before-image": "qwen38-kolibri.service"}
        self.run_block(d_files, kolibri_at_boot=True)
        self.assertEqual(self.enabled(), ["qwen38-flash.service"])
        # the links are read while Kolibri is still the lane at boot, before its unit goes
        self.assertLess(BLOCK.index("text_lane_migrate"), BLOCK.index('sudo systemctl disable --now "$UNIT"'))

    def test_the_boot_goes_back_past_a_side_lane_removed_since(self):
        # flash, then image, then Kolibri; the image lane uninstalled: Kolibri's link named a unit
        # that was gone, and its uninstall left no lane at boot (found in review, 2026-10-14)
        self.run_block({"lane-before-kolibri": "qwen38-image.service",
                        "lane-before-image": "qwen38-flash.service"}, kolibri_at_boot=True)
        self.assertEqual(self.enabled(), ["qwen38-flash.service"])

    def test_with_nothing_noted_the_installed_text_lane_gets_the_boot(self):
        # a link that ends nowhere left the box with no lane at boot until v1.29; a text lane
        # installed is better than none (the 27B first), and none installed enables nothing
        self.run_block({"lane-before-kolibri": "qwen38-video.service"}, kolibri_at_boot=True)
        self.assertEqual(self.enabled(), ["qwen38-sglang.service"])
        self.run_block({"lane-before-kolibri": "qwen38-video.service"}, kolibri_at_boot=True,
                       units=("qwen38-flash.service",))
        self.assertEqual(self.enabled(), ["qwen38-flash.service"])
        self.run_block({}, kolibri_at_boot=True, units=())
        self.assertEqual(self.enabled(), [])
        self.assertIn("no engine is enabled at boot any more", self.out)

    def test_the_compaction_follows_the_default_put_back(self):
        # sized from Kolibri's window, it stayed at 38,000 tokens on a 27B whose own table keeps
        # far more (found in review, 2026-10-14)
        conf = {"model": "kolibri/kolibri-1", "compaction": {"auto": True, "preserve_recent_tokens": 38000},
                "provider": {"qwen38": {"options": {"baseURL": "http://127.0.0.1:30001/v1", "apiKey": "{file:/k/api-key}"},
                                        "models": {"qwen3.8-27b": {"name": "qwen3.8-27b",
                                                                   "limit": {"context": 529000, "output": 250000}}}},
                             "kolibri": {"options": {"baseURL": "http://127.0.0.1:30001/v1", "apiKey": "{file:/k/api-key}"},
                                         "models": {"kolibri-1": {"name": "kolibri-1",
                                                                  "limit": {"context": 152000, "output": 32768}}}}}}
        self.run_block({"lane-before-kolibri": "qwen38-sglang.service"}, opencode=conf)
        want = int(subprocess.run([str(REPO / "oc-limits.sh"), "--preserve", "529000"],
                                  capture_output=True, text=True, check=True).stdout)
        self.assertNotEqual(want, 38000)
        for path in (self.cfg / "opencode.json", self.home / ".config/opencode/opencode.json"):
            left = json.loads(path.read_text())
            self.assertEqual(left["model"], "qwen38/qwen3.8-27b", path)
            self.assertEqual(left["compaction"]["preserve_recent_tokens"], want, path)


if __name__ == "__main__":
    unittest.main()
