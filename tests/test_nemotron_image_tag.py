#!/usr/bin/env python3
"""The Nemotron lane's local image is named by both of its pins: the vLLM nightly and b12x's wheel.

install-nemotron.sh builds the image only when its tag is missing, as install-kolibri.sh does, so
the tag must change when either pin does: a tag that kept b12x's version as a literal would have
let a box keep serving an image built from the older wheel (the Kolibri lane's own finding,
2026-10-14). The script's pin block runs here as the script has it, then with the wheel moved.
"""
import pathlib
import re
import subprocess
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "install-nemotron.sh").read_text()
START = TEXT.index("NEMOTRON_BASE_IMAGE=")
END = TEXT.index('LOCAL_IMAGE="${LOCAL_IMAGE:0:52}"') + len('LOCAL_IMAGE="${LOCAL_IMAGE:0:52}"')
PINS = TEXT[START:END]


def tag(pins):
    r = subprocess.run(["bash", "-c", 'set -euo pipefail\ndie(){ echo "DIE: $*"; exit 1; }\n' + pins
                        + '\necho "$LOCAL_IMAGE"'], capture_output=True, text=True, timeout=30)
    return r.returncode, r.stdout.strip()


def moved(pins, version):
    """b12x pinned at another version: its URL names that version's wheel."""
    out = re.sub(r'(?m)^NEMOTRON_B12X_VERSION="[^"]*"', f'NEMOTRON_B12X_VERSION="{version}"', pins)
    return re.sub(r"b12x-[0-9][^/\"]*-py3-none-any\.whl", f"b12x-{version}-py3-none-any.whl", out)


class TheImageIsNamedByBothPins(unittest.TestCase):
    def test_moving_the_wheel_is_a_new_image(self):
        rc, now = tag(PINS)
        self.assertEqual(rc, 0, now)
        self.assertTrue(now.startswith("qwen38-nemotron:vllm77154ca6-b12x1.5.1-1ee85ca80e0f"), now)
        rc, then = tag(moved(PINS, "1.5.2"))
        self.assertEqual(rc, 0, then)
        self.assertNotEqual(now, then, "a box with the old build would keep it")
        self.assertIn("b12x1.5.2", then)

    def test_moving_the_base_is_a_new_image(self):
        rc, now = tag(PINS)
        rc2, then = tag(re.sub(r"(?m)^(NEMOTRON_BASE_IMAGE=\"vllm/vllm-openai@sha256:)[0-9a-f]{64}\"",
                               r'\g<1>' + "ab" * 32 + '"', PINS))
        self.assertEqual((rc, rc2), (0, 0))
        self.assertNotEqual(now, then)

    def test_a_url_that_is_not_the_pinned_version_is_refused(self):
        pins = re.sub(r'(?m)^NEMOTRON_B12X_VERSION="[^"]*"', 'NEMOTRON_B12X_VERSION="9.9.9"', PINS)
        rc, out = tag(pins)
        self.assertNotEqual(rc, 0)
        self.assertIn("not the b12x-9.9.9-py3-none-any.whl", out)

    def test_the_build_takes_the_wheel_by_the_pinned_version(self):
        build = TEXT[TEXT.index('step "2/5'):TEXT.index('step "3/5')]
        self.assertNotRegex(build, r"b12x-[0-9]", "a version written in the build itself")
        self.assertIn('--build-arg B12X_VERSION="$NEMOTRON_B12X_VERSION"', build)
        self.assertIn("pip install --no-deps", build)
        self.assertIn('sha256sum -c --quiet', build)


class TheLaneLeavesNothingOfRootsInTheUsersCache(unittest.TestCase):
    """vLLM runs as root in the lane's container, which mounts the user's HF cache: it left the
    reasoning parser's __pycache__ in the snapshot and transformers' copy of the checkpoint's code
    (--trust-remote-code) in the cache's modules folder, both root-owned, which the user cannot
    remove (found on the reference box, 2026-10-15). Neither is written there any more."""

    def test_no_bytecode_and_the_code_copy_inside_the_container(self):
        unit = (REPO / "qwen38-nemotron.service.template").read_text()
        run = unit[unit.index("ExecStart="):]
        self.assertIn("-e PYTHONDONTWRITEBYTECODE=1", run)
        m = re.search(r"-e HF_MODULES_CACHE=(\S+)", run)
        self.assertTrue(m, "transformers would copy the checkpoint's code into the user's HF cache")
        self.assertFalse(m.group(1).startswith("/root/.cache/huggingface"), m.group(1))
        self.assertIn(":/root/.cache/huggingface ", run, "the cache it must stay out of is the one mounted there")


if __name__ == "__main__":
    unittest.main()
