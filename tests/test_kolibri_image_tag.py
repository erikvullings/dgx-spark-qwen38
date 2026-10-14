#!/usr/bin/env python3
"""The Kolibri lane's local image is named by both of its pins: the vLLM base and the plugin's wheel.

install-kolibri.sh builds the image only when its tag is missing. The tag named the base by its
digest but carried the plugin's version as a literal, and the build wrote that version into the
wheel's file name too: a release moving the wheel's URL and sha256 to a newer
aleph-alpha-inference found the old tag on every box that had built it, skipped the build, and
kept serving the old plugin (found in review, 2026-10-14). The script's pin block runs here as
the script has it, then with the plugin moved to another version.
"""
import pathlib
import re
import subprocess
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "install-kolibri.sh").read_text()
START = TEXT.index("KOLIBRI_BASE_IMAGE=")
END = TEXT.index('LOCAL_IMAGE="${LOCAL_IMAGE:0:52}"') + len('LOCAL_IMAGE="${LOCAL_IMAGE:0:52}"')
PINS = TEXT[START:END]


def tag(pins):
    r = subprocess.run(["bash", "-c", 'set -euo pipefail\ndie(){ echo "DIE: $*"; exit 1; }\n' + pins
                        + '\necho "$LOCAL_IMAGE"'], capture_output=True, text=True, timeout=30)
    return r.returncode, r.stdout.strip()


def moved(pins, version):
    """The plugin pinned at another version: its URL names that version's wheel."""
    out = re.sub(r'(?m)^KOLIBRI_PLUGIN_VERSION="[^"]*"', f'KOLIBRI_PLUGIN_VERSION="{version}"', pins)
    return re.sub(r"aleph_alpha_inference-[0-9][^/\"]*-py3-none-any\.whl",
                  f"aleph_alpha_inference-{version}-py3-none-any.whl", out)


class TheImageIsNamedByBothPins(unittest.TestCase):
    def test_moving_the_plugin_is_a_new_image(self):
        rc, now = tag(PINS)
        self.assertEqual(rc, 0, now)
        rc, then = tag(moved(PINS, "1.0.1"))
        self.assertEqual(rc, 0, then)
        self.assertNotEqual(now, then, "a box with the old build would keep it")
        self.assertIn("aai1.0.1", then)

    def test_a_url_that_is_not_the_pinned_version_is_refused(self):
        pins = re.sub(r'(?m)^KOLIBRI_PLUGIN_VERSION="[^"]*"', 'KOLIBRI_PLUGIN_VERSION="9.9.9"', PINS)
        rc, out = tag(pins)
        self.assertNotEqual(rc, 0)
        self.assertIn("not the aleph_alpha_inference-9.9.9-py3-none-any.whl", out)

    def test_the_build_takes_the_wheel_by_the_pinned_version(self):
        build = TEXT[TEXT.index('step "2/5'):TEXT.index('step "3/5')]
        self.assertNotRegex(build, r"aleph_alpha_inference-[0-9]", "a version written in the build itself")


if __name__ == "__main__":
    unittest.main()
