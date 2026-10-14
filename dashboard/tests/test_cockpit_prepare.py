#!/usr/bin/env python3
"""The image, video and Kolibri lanes are installed on demand: their units at install, their
runtime at their first Load from the cockpit, built as the user by their installer's
--prepare. The cockpit says which are prepared (the runtime itself, as install.sh reads it:
the venv and its wheel's note, the image the Kolibri unit names) and runs --prepare as a job
of its own, with no sudo."""
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
        self.assertEqual(self.cp.collect_lanes_ready(), {"image": True, "video": False, "kolibri": None})
        (self.units / "qwen38-kolibri.service").write_text("ExecStart=/usr/bin/docker run qwen38-kolibri:1.0\n")
        self.assertIs(self.cp.collect_lanes_ready()["kolibri"], False)
        self.images.add("qwen38-kolibri:1.0")
        self.assertIs(self.cp.collect_lanes_ready()["kolibri"], True)

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
        for lane in ("flash", "27b", "../install", ""):
            with self.subTest(lane=lane):
                self.assertEqual(self.cp.start_action("prepare", {"lane": lane})[0], 400)


if __name__ == "__main__":
    unittest.main()
