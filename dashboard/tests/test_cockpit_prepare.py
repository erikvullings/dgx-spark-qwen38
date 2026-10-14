#!/usr/bin/env python3
"""Every lane but the one that serves is installed on demand: its units at install, what it
runs at its first Load from the cockpit, as the user, by prepare-lane.sh. The cockpit says
which are prepared, as install.sh reads it (the venv and its wheel's note, the image the
Kolibri unit names, the image the flash launcher or the 27B unit runs with what that lane keeps
beside it), and runs the preparation as a job of its own, with no sudo."""
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
DASH = HERE.parents[1]
REPO = HERE.parents[2]
sys.path.insert(0, str(DASH))
ENV_BEFORE = dict(os.environ)


def tearDownModule():
    os.environ.clear()
    os.environ.update(ENV_BEFORE)


class Prepare(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cockpit-prepare-"))
        (cls.tmp / "config").mkdir()
        (cls.tmp / "config" / "api-key").write_text("k\n")
        closed = "http://127.0.0.1:1"
        os.environ.update(HOME=str(cls.tmp), COCKPIT_DRY_RUN="1", COCKPIT_CONFIG_DIR=str(cls.tmp / "config"),
                          COCKPIT_REPO_DIR=str(REPO), COCKPIT_PORT="0", COCKPIT_AGENT_PORT="0",
                          COCKPIT_AUTOHEAL="0", COCKPIT_ENGINE=closed, COCKPIT_PROXY=closed, COCKPIT_IMAGE=closed)
        spec = importlib.util.spec_from_file_location("cockpit_prepare_under_test", DASH / "cockpit.py")
        cls.cp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cp)
        cls.units = cls.tmp / "units"
        cls.units.mkdir()
        cls.cp.IMAGE_UNIT_PATH = cls.units / "qwen38-image.service"
        cls.cp.VIDEO_UNIT_PATH = cls.units / "qwen38-video.service"
        cls.cp.KOLIBRI_UNIT_PATH = cls.units / "qwen38-kolibri.service"
        cls.cp.UNIT_PATHS = {"qwen38-sglang.service": cls.units / "qwen38-sglang.service",
                             "qwen38-flash.service": cls.units / "launch-flash.sh"}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        for f in self.units.iterdir():
            f.unlink()
        self.images = set()
        self.cp.run = lambda argv, timeout=5.0, merge_err=False: (
            "sha256:abc\n" if argv[:3] == ["docker", "image", "inspect"] and argv[-1] in self.images else "")

    def lane(self, name, built):
        d = self.tmp / f"lane-{name}"
        shutil.rmtree(d, ignore_errors=True)
        (self.units / f"qwen38-{name}.service").write_text(f"[Service]\nWorkingDirectory={d}\n")
        if built:
            (d / "venv" / "bin").mkdir(parents=True)
            (d / "venv" / "bin" / "python").write_text("#!/bin/sh\n")
            (d / "sglang-wheel").write_text("0.5.21\n")

    def test_which_lanes_are_prepared(self):
        self.lane("image", built=True)
        self.lane("video", built=False)
        self.assertEqual(self.cp.collect_lanes_ready(), {"image": True, "video": False, "kolibri": None, "flash": None, "27b": None})
        (self.units / "qwen38-kolibri.service").write_text("ExecStart=/usr/bin/docker run qwen38-kolibri:1.0\n")
        self.assertIs(self.cp.collect_lanes_ready()["kolibri"], False)
        self.images.add("qwen38-kolibri:1.0")
        self.assertIs(self.cp.collect_lanes_ready()["kolibri"], True)

    @staticmethod
    def runs(image, *rest):
        """A launcher or unit as their templates write it: the image alone on the line before
        the server command."""
        return ("exec /usr/bin/docker run --rm \\\n  -e HF_HUB_OFFLINE=1 \\\n"
                f"  {image} \\\n  python3 -m sglang.launch_server \\\n    --model-path x \\\n" + "".join(rest))

    def test_the_flash_lane_beside_the_serving_one(self):
        # its image, whatever its form, and the draft vocabulary its launcher waits for when it waits for one
        vocab = self.tmp / "token-map-65536.pt"
        for img in ("lmsysorg/sglang@sha256:" + "d" * 64, "lmsysorg/sglang:dev-qwen38-next-local"):
            with self.subTest(img=img):
                self.images.clear()
                vocab.unlink(missing_ok=True)
                launch = self.units / "launch-flash.sh"
                launch.write_text(self.runs(img, f'[ -s "{vocab}" ] && TIER+=(--speculative-token-map /out/token-map-65536.pt)\n'))
                self.assertIs(self.cp.lane_prepared("flash"), False)          # no image
                self.images.add(img)
                self.assertIs(self.cp.lane_prepared("flash"), False)          # the vocabulary it waits for
                vocab.write_text("map")
                self.assertIs(self.cp.lane_prepared("flash"), True)
                launch.write_text(self.runs(img, "TIER+=(--speculative-token-map /out/token-map-65536.pt)\n"))
                vocab.unlink()
                self.assertIs(self.cp.lane_prepared("flash"), True)           # built before: the line it always had

    def test_the_27b_beside_a_serving_flash(self):
        # its image and the DFlash2 draft its unit names, in the cache it mounts, at its revision
        img, hf, rev = "lmsysorg/sglang@sha256:" + "f" * 64, self.tmp / "hf27", "a" * 40
        shutil.rmtree(hf, ignore_errors=True)
        unit = self.units / "qwen38-sglang.service"
        unit.write_text(self.runs(img, f"  -v {hf}:/root/.cache/huggingface \\\n",
                                  "    --speculative-draft-model-path org/draft \\\n",
                                  f"    --speculative-draft-model-revision {rev} \\\n"))
        self.assertIs(self.cp.lane_prepared("27b"), False)            # no image
        self.images.add(img)
        self.assertIs(self.cp.lane_prepared("27b"), False)            # no draft
        snap = hf / "hub" / "models--org--draft" / "snapshots" / rev
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")
        self.assertIs(self.cp.lane_prepared("27b"), False)            # its config, not its weights yet
        (snap / "model.safetensors").write_text("w")
        self.assertIs(self.cp.lane_prepared("27b"), True)
        unit.write_text(unit.read_text().replace(rev, "main"))        # a ref, read through refs/
        self.assertIs(self.cp.lane_prepared("27b"), False)
        (hf / "hub" / "models--org--draft" / "refs").mkdir()
        (hf / "hub" / "models--org--draft" / "refs" / "main").write_text(rev)
        self.assertIs(self.cp.lane_prepared("27b"), True)
        self.assertIs(self.cp.collect_lanes_ready()["27b"], True)

    def test_a_venv_without_its_wheels_note_is_not_prepared(self):
        # the venv is made first and the wheel's note written once the wheel is in
        self.lane("image", built=True)
        (self.tmp / "lane-image" / "sglang-wheel").unlink()
        self.assertIs(self.cp.collect_lanes_ready()["image"], False)

    def test_the_action_runs_the_installers_prepare_and_takes_no_other_lane(self):
        code, body = self.cp.start_action("prepare", {"lane": "kolibri"})
        self.assertEqual(code, 202, body)
        self.assertEqual(body["argv"], ["bash", str(REPO / "prepare-lane.sh"), "kolibri"])
        for _ in range(100):
            if self.cp.JOBS[body["job"]].status != "running":
                break
            __import__("time").sleep(0.05)
        for lane in ("27B", "stock", "../install", "", "flash "):
            with self.subTest(lane=lane):
                self.assertEqual(self.cp.start_action("prepare", {"lane": lane})[0], 400)


if __name__ == "__main__":
    unittest.main()
