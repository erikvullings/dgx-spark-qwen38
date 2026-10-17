#!/usr/bin/env python3
"""The side lanes' checkpoints in the list (v1.36.0, 2026-10-17).

The Kolibri, image and video lanes keep the checkpoint their unit names across updates (an
operator switched to kolibri-nvfp4 or image-turbo, or installed another video variant), and
the list said nothing of them (Hasan, 2026-10-17: record them from v1.36.0). They are compared
like the text lanes' targets: read back from their units the way their installers read them,
recommended as each installer installs a new box, and, since no release installed another one
by default, a different one is always a choice.
"""
import os
import pathlib
import re
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_conformity_classify import REC, box, classes  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import conformity  # noqa: E402

NVFP4 = "Quantinine/Kolibri-1-NVFP4A16-FP8"
TURBO = "Qwen/Qwen-Image-2.1-Turbo"


def bash(script):
    r = subprocess.run(["bash", "-c", "set -eu\n" + script], capture_output=True, text=True, timeout=30,
                       env={"PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def line(script, pattern):
    """The one line of an installer that matches pattern."""
    found = [ln for ln in (REPO / script).read_text().splitlines() if re.search(pattern, ln)]
    assert len(found) == 1, (script, pattern, found)
    return found[0]


class TheRecommendedOnesAreWhatTheInstallersInstall(unittest.TestCase):
    """Each installer's own lines, run with no variable and no unit, as on a new box."""

    def test_kolibri_installs_its_fp8_checkpoint(self):
        text = (REPO / "install-kolibri.sh").read_text()
        start = text.index('case "${KOLIBRI_VARIANT:-}" in')
        block = text[start:text.index("esac", start) + 4]
        got = bash(f'INSTALLED_MODEL=""; KOLIBRI_NVFP4_REPO={NVFP4}\ndie(){{ exit 3; }}\n{block}\necho "$VARIANT"')
        self.assertEqual(got, "fp8")
        self.assertEqual(REC["KOLIBRI_VARIANT"], got)

    def test_the_image_lane_installs_its_pinned_model(self):
        got = bash('unit_flag(){ :; }\nkept_flag(){ :; }\n' + line("install-image.sh", r'^MODEL="\$\{IMAGE_MODEL:-') + '\necho "$MODEL"')
        self.assertEqual(got, "Qwen/Qwen-Image-2.1")
        self.assertEqual(REC["IMAGE_MODEL"], got)

    def test_the_video_lane_installs_fl2va(self):
        got = bash('unit_flag(){ :; }\nkept_flag(){ :; }\n' + line("install-video.sh", r'^VARIANT="\$\{VIDEO_VARIANT:-') + '\necho "$VARIANT"')
        self.assertEqual(got, "fl2va")
        self.assertEqual(REC["VIDEO_VARIANT"], got)


class EachOneIsReadBackFromItsUnit(unittest.TestCase):
    def test_another_checkpoint_is_a_choice_no_release_made_the_default(self):
        for kw, knob, value in ((dict(kolibri=NVFP4), "KOLIBRI_VARIANT", "nvfp4"),
                                (dict(image_model=TURBO), "IMAGE_MODEL", TURBO),
                                (dict(video_variant="ref2va"), "VIDEO_VARIANT", "ref2va")):
            with self.subTest(knob=knob):
                units, cfg = box(**kw)
                self.assertEqual(conformity.observe(units, cfg)[knob], value)
                self.assertEqual(classes(units, cfg), {knob: ("choice", "never-default")})

    def test_a_box_without_the_side_lanes_says_nothing_of_them(self):
        units, cfg = box(side=False)
        seen = conformity.observe(units, cfg)
        for knob in ("KOLIBRI_VARIANT", "IMAGE_MODEL", "VIDEO_VARIANT"):
            self.assertNotIn(knob, seen)
        self.assertEqual(classes(units, cfg), {})


if __name__ == "__main__":
    unittest.main()
