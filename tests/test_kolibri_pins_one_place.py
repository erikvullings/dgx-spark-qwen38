#!/usr/bin/env python3
"""Each number of a Kolibri checkpoint is declared once, in install-kolibri.sh's pin block.

switch-model.sh read the repos and the revisions from there, but carried each checkpoint's
share of the GPU memory and its download size as copies, and rewrote the unit's
--gpu-memory-utilization from those: a share retuned in install-kolibri.sh, where it is
measured, came back to the old value at the next switch. install.sh named the NVFP4 checkpoint
a second time to tell which one a unit serves (found in review, 2026-10-14). The scripts' own
lines run here against a copy of install-kolibri.sh whose pins have moved.
"""
import pathlib
import re
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
KOL = (REPO / "install-kolibri.sh").read_text()
SWITCH = (REPO / "switch-model.sh").read_text()
INSTALL = (REPO / "install.sh").read_text()

_k = SWITCH.index('if [ "$CHOICE" = "kolibri" ] || [ "$CHOICE" = "kolibri-nvfp4" ]; then')
_s = SWITCH.index('  eval "$(grep -E', _k)
_e = SWITCH.index("\n  fi\n", SWITCH.index("KOL_MODEL=", _s)) + len("\n  fi\n")
SWITCH_PICK = SWITCH[_s:_e]
_i = INSTALL.index("  OC_KOLIBRI_TARGET=kolibri\n")
INSTALL_PICK = INSTALL[_i:INSTALL.index("&& OC_KOLIBRI_TARGET=kolibri-nvfp4", _i) + len("&& OC_KOLIBRI_TARGET=kolibri-nvfp4")]


def moved_repo():
    """A copy of the repo's install-kolibri.sh with the FP8 share, its size and the NVFP4 repo moved."""
    d = pathlib.Path(tempfile.mkdtemp(prefix="kol-pins-"))
    text = re.sub(r'(?m)^KOLIBRI_FP8_GMU="[^"]*"', 'KOLIBRI_FP8_GMU="0.78"', KOL)
    text = re.sub(r'(?m)^KOLIBRI_FP8_GB="[^"]*"', 'KOLIBRI_FP8_GB="80"', text)
    text = re.sub(r'(?m)^KOLIBRI_NVFP4_REPO="[^"]*"', 'KOLIBRI_NVFP4_REPO="someone/Kolibri-1-NVFP4-v2"', text)
    if text == KOL:   # the code before: nothing to move, the numbers were not pins
        text = KOL.replace("GMU=0.80; WEIGHTS_GB=79", "GMU=0.78; WEIGHTS_GB=80") \
                  .replace('KOLIBRI_NVFP4_REPO="Quantinine/Kolibri-1-NVFP4A16-FP8"',
                           'KOLIBRI_NVFP4_REPO="someone/Kolibri-1-NVFP4-v2"')
    (d / "install-kolibri.sh").write_text(text)
    return d


def bash(script):
    r = subprocess.run(["bash", "-c", 'set -euo pipefail\ndie(){ echo "DIE: $*"; exit 1; }\n' + script],
                       capture_output=True, text=True, timeout=30)
    return r.returncode, r.stdout.strip()


class EachNumberIsDeclaredOnce(unittest.TestCase):
    def test_the_switch_takes_the_share_and_the_size_from_the_installer(self):
        d = moved_repo()
        rc, out = bash(f'REPO_DIR="{d}"; CHOICE=kolibri\n' + SWITCH_PICK + '\necho "$KOL_GMU $KOL_NEED_GB"')
        self.assertEqual((rc, out), (0, "0.78 80"), "a share tuned in install-kolibri.sh comes back at the next switch")

    def test_today_both_scripts_give_each_checkpoint_the_same_numbers(self):
        for choice, variant in (("kolibri", "fp8"), ("kolibri-nvfp4", "nvfp4")):
            with self.subTest(choice=choice):
                rc, switch = bash(f'REPO_DIR="{REPO}"; CHOICE={choice}\n' + SWITCH_PICK
                                  + '\necho "$KOL_MODEL $KOL_REV $KOL_GMU $KOL_NEED_GB"')
                self.assertEqual(rc, 0, switch)
                pins = KOL[KOL.index("KOLIBRI_BASE_IMAGE="):KOL.index("IMAGE_GB=")]
                pins = pins.replace('[ "$INSTALLED_MODEL" = "$KOLIBRI_NVFP4_REPO" ] && VARIANT=nvfp4', ":")
                rc, inst = bash(f'INSTALLED=/nonexistent; KOLIBRI_VARIANT={variant}\n' + pins
                                + '\necho "$MODEL $MODEL_REV $GMU $WEIGHTS_GB"')
                self.assertEqual(rc, 0, inst)
                self.assertEqual(switch, inst)

    def test_install_tells_the_nvfp4_unit_by_the_pinned_repo(self):
        d = moved_repo()
        unit = d / "qwen38-kolibri.service"
        unit.write_text("ExecStart=/usr/bin/docker run img --model someone/Kolibri-1-NVFP4-v2 --revision 4a84\n")
        rc, out = bash(f'REPO_DIR="{d}"; KOLIBRI_UNIT_PATH="{unit}"\n' + INSTALL_PICK + '\necho "$OC_KOLIBRI_TARGET"')
        self.assertEqual((rc, out), (0, "kolibri-nvfp4"))


if __name__ == "__main__":
    unittest.main()
