#!/usr/bin/env python3
"""A round trip between side lanes keeps the way back to the text lane.

A switch to a side lane (image, video, Kolibri) writes the unit it took the boot from in
lane-before-<lane>; install.sh follows those links to the text lane it brings up to date,
and a side lane's uninstall gives the boot back along them. Coming back to a lane already
in the chain wrote two files naming each other: on the reference box, flash, then image,
Kolibri and image again (Load from the cockpit, 2026-10-14) left image and Kolibri naming
each other, and an update took the 27B for the text lane of a box that served the flash.
switch-model.sh's close_lane_loop runs before a lane's own link is written, and these
replay switches with it and install.sh's own resolution lines.
"""
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
SWITCH = (REPO / "switch-model.sh").read_text()
INSTALL = (REPO / "install.sh").read_text()
_at = SWITCH.index("close_lane_loop(){")
HELPER = SWITCH[_at:SWITCH.index("\n}\n", _at) + 3]
_r = INSTALL.index('LANE_BEFORE_IMAGE="$(cat "$CONFIG_DIR/lane-before-image"')
RESOLVE = INSTALL[_r:INSTALL.index('if [ "$FLASH_READABLE" -eq 1 ] && [ "$FLASH_ENABLED" -eq 1 ] && [ "$SGL_ENABLED" -eq 1 ]', _r)]
UNIT = {"flash": "qwen38-flash.service", "27b": "qwen38-sglang.service", "image": "qwen38-image.service",
        "video": "qwen38-video.service", "kolibri": "qwen38-kolibri.service"}


def replay(*lanes):
    """Switch through the lanes in order, as switch-model.sh writes the links, then read the
    text lane install.sh resolves for the side lane the box ends on."""
    d = pathlib.Path(tempfile.mkdtemp(prefix="lane-loops-"))
    steps = []
    for prev, nxt in zip(lanes, lanes[1:]):
        if nxt in ("image", "video", "kolibri"):
            steps.append(f'close_lane_loop {nxt} "{UNIT[prev]}"\n'
                         f'printf \'%s\\n\' "{UNIT[prev]}" > "$CONFIG_DIR/lane-before-{nxt}"\n')
    last = lanes[-1]
    script = (f'set -euo pipefail\nCONFIG_DIR="{d}"\n' + HELPER + "".join(steps) + RESOLVE
              + f'case "{last}" in image) echo "$LANE_BEFORE_IMAGE";; video) echo "$LANE_BEFORE_VIDEO";; '
                f'kolibri) echo "$LANE_BEFORE_KOLIBRI";; esac\n')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stdout + r.stderr
    files = {p.name: p.read_text().strip() for p in d.iterdir()}
    return r.stdout.strip(), files


class ARoundTripKeepsTheTextLane(unittest.TestCase):
    def test_image_kolibri_image_from_the_cockpit(self):
        # the reference box, 2026-10-14
        lane, files = replay("flash", "image", "kolibri", "image")
        self.assertEqual(lane, "qwen38-flash.service")
        self.assertEqual(files["lane-before-image"], "qwen38-kolibri.service", "its uninstall goes back to Kolibri")
        self.assertEqual(files["lane-before-kolibri"], "qwen38-flash.service")

    def test_kolibri_image_kolibri(self):
        self.assertEqual(replay("27b", "kolibri", "image", "kolibri")[0], "qwen38-sglang.service")

    def test_image_video_image_the_old_case(self):
        self.assertEqual(replay("flash", "image", "video", "image")[0], "qwen38-flash.service")

    def test_round_trips_over_and_over(self):
        self.assertEqual(replay("flash", "image", "kolibri", "image", "kolibri", "image", "kolibri")[0],
                         "qwen38-flash.service")
        self.assertEqual(replay("flash", "kolibri", "video", "image", "kolibri", "image")[0],
                         "qwen38-flash.service")

    def test_a_chain_without_a_loop_is_left_as_it_was(self):
        lane, files = replay("flash", "image", "video", "kolibri")
        self.assertEqual(lane, "qwen38-flash.service")
        self.assertEqual(files, {"lane-before-image": "qwen38-flash.service",
                                 "lane-before-video": "qwen38-image.service",
                                 "lane-before-kolibri": "qwen38-video.service"})


if __name__ == "__main__":
    unittest.main()
