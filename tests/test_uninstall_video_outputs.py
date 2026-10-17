#!/usr/bin/env python3
"""uninstall.sh removes the video lane's empty outputs/ folder, and says what it keeps of it.

Every install makes ~/.local/share/qwen38-video/outputs, a video lane installed on demand and
never loaded included (install-video.sh's service step). uninstall.sh removed the runtime under
it, then kept the directory for that empty folder, with a line that called it files the video
lane did not put there (found by the install matrix, 2026-10-17: a plain install, then
./uninstall.sh --yes, left both). The block runs here as uninstall.sh has it.
"""
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "uninstall.sh").read_text()
START = TEXT.index("# The video lane's runtime, under the same rule")
BLOCK = TEXT[START:TEXT.index("\nfi\n", START) + 4]


def run(files):
    d = pathlib.Path(tempfile.mkdtemp(prefix="uninst-video-")) / "qwen38-video"
    for rel in files:
        p = d / rel
        if rel.endswith("/"):
            p.mkdir(parents=True, exist_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")
    d.mkdir(parents=True, exist_ok=True)
    script = f'set -euo pipefail\nPRIV=0; PRIV_DONE=0; VIDEO_LANE_DIR="{d}"\n' + BLOCK + "echo BLOCK-END\n"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
    return d, r.stdout + r.stderr


class TheVideoLanesFolderGoesWithIt(unittest.TestCase):
    def test_a_lane_never_loaded_leaves_nothing(self):
        d, out = run(["outputs/"])
        self.assertFalse(d.exists(), out)
        self.assertNotIn("kept", out)
        self.assertIn("BLOCK-END", out)

    def test_a_lane_used_and_removed_leaves_nothing_of_its_own(self):
        d, out = run(["outputs/", "venv/bin/python", "sglang/setup.py", "sglang-wheel"])
        self.assertFalse(d.exists(), out)

    def test_the_videos_it_made_stay_and_are_said_to(self):
        d, out = run(["outputs/clip.mp4", "venv/bin/python"])
        self.assertTrue((d / "outputs/clip.mp4").exists())
        self.assertFalse((d / "venv").exists())
        self.assertIn(f"kept {d}/outputs: the videos the lane generated (delete them by hand: rm -r {d}/outputs)", out)
        self.assertNotIn("did not put there", out)

    def test_a_file_of_someone_else_keeps_the_folder_said_as_such(self):
        d, out = run(["outputs/", "notes.txt"])
        self.assertTrue((d / "notes.txt").exists())
        self.assertFalse((d / "outputs").exists())
        self.assertIn(f"kept {d}: it holds files the video lane did not put there", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
