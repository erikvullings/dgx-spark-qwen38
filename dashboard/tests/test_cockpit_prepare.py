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

    def test_an_empty_vocabulary_is_none(self):
        # [ -s ]: the launcher skips an empty one, and so does this (found in review, 2026-10-15)
        img, vocab = "lmsysorg/sglang@sha256:" + "d" * 64, self.tmp / "token-map-65536.pt"
        self.images.add(img)
        (self.units / "launch-flash.sh").write_text(
            self.runs(img, f'[ -s "{vocab}" ] && TIER+=(--speculative-token-map /out/token-map-65536.pt)\n'))
        vocab.write_text("")
        self.assertIs(self.cp.lane_prepared("flash"), False)
        vocab.write_text("map")
        self.assertIs(self.cp.lane_prepared("flash"), True)

    def test_a_draft_half_there_or_set_for_another_window_is_not_prepared(self):
        # one shard of two, a link to a blob gone, a config set for another window than its unit
        # serves: each read as prepared, and the next Load started the 1M unit on a native draft
        # (found in review, 2026-10-15)
        img, hf, rev = "lmsysorg/sglang@sha256:" + "f" * 64, self.tmp / "hf-draft", "a" * 40
        shutil.rmtree(hf, ignore_errors=True)
        self.images.add(img)
        unit = self.units / "qwen38-sglang.service"

        def write(window):
            unit.write_text(self.runs(img, f"  -v {hf}:/root/.cache/huggingface \\\n",
                                      "    --context-length 1010000 \\\n" if window == "1m" else "",
                                      "    --speculative-draft-model-path org/draft \\\n",
                                      f"    --speculative-draft-model-revision {rev} \\\n"))
        repo = hf / "hub" / "models--org--draft"
        snap, blobs = repo / "snapshots" / rev, repo / "blobs"
        snap.mkdir(parents=True)
        blobs.mkdir()
        (snap / "config.json").write_text('{"max_position_embeddings": 262144}')
        (snap / "model.safetensors.index.json").write_text(
            '{"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}')
        (snap / "model-1.safetensors").write_text("w")
        write("native")
        self.assertIs(self.cp.lane_prepared("27b"), False)              # one shard of two
        (blobs / "b2").write_text("w")
        (snap / "model-2.safetensors").symlink_to(blobs / "b2")
        self.assertIs(self.cp.lane_prepared("27b"), True)
        (blobs / "b2").unlink()
        self.assertIs(self.cp.lane_prepared("27b"), False)              # a link to a blob gone
        (blobs / "b2").write_text("w")
        write("1m")
        self.assertIs(self.cp.lane_prepared("27b"), False)              # native config, 1M unit
        (snap / "config.json").write_text('{"max_position_embeddings": 1010000}')
        self.assertIs(self.cp.lane_prepared("27b"), True)
        write("native")
        self.assertIs(self.cp.lane_prepared("27b"), False)              # 1M config, native unit

    def test_a_venv_without_its_wheels_note_is_not_prepared(self):
        # the venv is made first and the wheel's note written once the wheel is in
        self.lane("image", built=True)
        (self.tmp / "lane-image" / "sglang-wheel").unlink()
        self.assertIs(self.cp.collect_lanes_ready()["image"], False)

    def wait(self, job):
        for _ in range(200):
            if self.cp.JOBS[job].status != "running":
                return
            __import__("time").sleep(0.05)

    def test_the_action_prepares_the_target_a_load_loads_and_takes_nothing_else(self):
        # the lane from the target, and the target too for the two lanes whose preparation fetches
        # a checkpoint: it fetched the default, then the switch the one loaded (2026-10-15)
        for target, args in (("kolibri", ["kolibri"]), ("image-turbo", ["image"]), ("video", ["video"]),
                             ("flash-nvda", ["flash", "flash-nvda"]), ("fp8", ["27b", "fp8"])):
            with self.subTest(target=target):
                code, body = self.cp.start_action("prepare", {"target": target})
                self.assertEqual(code, 202, body)
                self.assertEqual(body["argv"], ["bash", str(REPO / "prepare-lane.sh"), *args])
                self.wait(body["job"])
        for params in ({"target": "27B"}, {"target": "../install"}, {"target": ""}, {"target": "flash "},
                       {"lane": "kolibri"}, {}):
            with self.subTest(params=params):
                self.assertEqual(self.cp.start_action("prepare", params)[0], 400)
        self.assertGreaterEqual(self.cp.ACTIONS["prepare"]["timeout"], self.cp.ACTIONS["switch"]["timeout"])


    def test_a_preparation_waits_for_a_lane_that_is_starting(self):
        # a first Load's pip and docker pull beside a lane loading its weights, on the same unified
        # memory: refused as a switch is (found in review, 2026-10-15)
        with self.cp.LIFE_LOCK:
            before = dict(self.cp.LIFE.get("states", {}))
            self.cp.LIFE["states"] = {**before, "qwen38-sglang.service": "starting"}
        try:
            code, body = self.cp.start_action("prepare", {"target": "kolibri"})
            self.assertEqual(code, 409, body)
            self.assertEqual(body["error"], "blocked")
        finally:
            with self.cp.LIFE_LOCK:
                self.cp.LIFE["states"] = before

    def test_a_preparation_goes_beside_a_lane_that_is_steady(self):
        # only a lane starting or stopping holds it back: ready, degraded, wedged, orphan or failed,
        # a Load's first step goes (a gate on any of them refused every Load)
        for state in ("ready", "degraded", "wedged", "orphan", "failed"):
            with self.subTest(state=state):
                with self.cp.LIFE_LOCK:
                    before = dict(self.cp.LIFE.get("states", {}))
                    self.cp.LIFE["states"] = {**before, "qwen38-sglang.service": state}
                try:
                    code, body = self.cp.start_action("prepare", {"target": "kolibri"})
                    self.assertEqual(code, 202, body)
                    self.wait(body["job"])
                finally:
                    with self.cp.LIFE_LOCK:
                        self.cp.LIFE["states"] = before


if __name__ == "__main__":
    unittest.main()
