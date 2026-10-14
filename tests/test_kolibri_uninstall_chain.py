#!/usr/bin/env python3
"""./install-kolibri.sh --uninstall keeps the other side lanes' way back to the text lane.

A switch to a side lane (image, video, Kolibri) writes down the lane it took the boot from,
in lane-before-<lane>: that lane's own uninstall gives the boot back to it, and install.sh
follows those links on an update to find the text lane it brings up to date. The image lane
switched to from Kolibri names Kolibri there. Removing the Kolibri lane deleted its own link
and left that one naming a unit that was gone: the chain stopped at it, the image lane's
uninstall found nothing to give the boot back to, and an update fell back to the 27B lane on
a box that had been serving the flash lane.

The script's own uninstall block runs here, whole, against a fake unit path and a fake
config directory; sudo, systemctl and docker are stubs that write down every call.
"""
import os
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "install-kolibri.sh").read_text()
START = TEXT.index('if [ "$ACTION" = uninstall ]; then')
END = TEXT.index("\n  exit 0\nfi\n", START) + len("\n  exit 0\nfi\n")
BLOCK = TEXT[START:END]


class TheUninstallKeepsTheChain(unittest.TestCase):
    def run_block(self, files, kolibri_at_boot=False):
        d = pathlib.Path(tempfile.mkdtemp(prefix="kol-uninstall-"))
        cfg = d / "cfg"
        cfg.mkdir()
        for name, value in files.items():
            (cfg / name).write_text(value + "\n")
        bin_ = d / "bin"
        bin_.mkdir()
        log = d / "calls.log"
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
            "ACTION=uninstall",
            "UNIT=qwen38-kolibri.service",
            f'INSTALLED="{unit}"',
            "CONTAINER=qwen38-kolibri",
            f'CONFIG_DIR="{cfg}"',
            f'HF_CACHE="{d}/hf"',
            "KOLIBRI_BASE_IMAGE=vllm/vllm-openai@sha256:0",
            "KOLIBRI_FP8_REPO=Aleph-Alpha/Kolibri-1",
            "KOLIBRI_NVFP4_REPO=Quantinine/Kolibri-1-NVFP4A16-FP8",
            BLOCK,
        ])
        env = dict(os.environ, PATH=f"{bin_}:/usr/bin:/bin")
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=30)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return {p.name: p.read_text().strip() for p in cfg.iterdir() if p.name.startswith("lane-before-")}

    def test_a_lane_that_named_kolibri_now_names_where_kolibri_came_from(self):
        # the reference box on 2026-10-14: flash, then image, then Kolibri, then image again
        left = self.run_block({"lane-before-image": "qwen38-kolibri.service",
                               "lane-before-kolibri": "qwen38-flash.service"})
        self.assertEqual(left, {"lane-before-image": "qwen38-flash.service"})

    def test_the_video_lane_too(self):
        left = self.run_block({"lane-before-video": "qwen38-kolibri.service",
                               "lane-before-kolibri": "qwen38-sglang.service"})
        self.assertEqual(left, {"lane-before-video": "qwen38-sglang.service"})

    def test_a_link_never_comes_back_to_its_own_lane(self):
        # image and Kolibri naming each other: the image lane's origin is unknown, not itself
        left = self.run_block({"lane-before-image": "qwen38-kolibri.service",
                               "lane-before-kolibri": "qwen38-image.service"})
        self.assertEqual(left, {})

    def test_a_link_to_kolibri_with_nothing_behind_it_goes(self):
        left = self.run_block({"lane-before-video": "qwen38-kolibri.service"})
        self.assertEqual(left, {})

    def test_links_that_never_named_kolibri_stay_as_they_are(self):
        left = self.run_block({"lane-before-image": "qwen38-flash.service",
                               "lane-before-video": "qwen38-image.service",
                               "lane-before-kolibri": "qwen38-sglang.service"})
        self.assertEqual(left, {"lane-before-image": "qwen38-flash.service",
                                "lane-before-video": "qwen38-image.service"})

    def test_kolibri_at_boot_still_gives_the_boot_back_to_its_own_lane(self):
        # the handover reads Kolibri's link before anything else touches it
        d_files = {"lane-before-kolibri": "qwen38-flash.service",
                   "lane-before-image": "qwen38-kolibri.service"}
        self.run_block(d_files, kolibri_at_boot=True)
        at = BLOCK.index('BACK="$(cat "$CONFIG_DIR/lane-before-kolibri"')
        self.assertLess(at, BLOCK.index('rm -f "$CONFIG_DIR/lane-before-kolibri"'))


if __name__ == "__main__":
    unittest.main()
