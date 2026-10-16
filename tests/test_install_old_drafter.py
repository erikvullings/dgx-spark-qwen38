#!/usr/bin/env python3
"""A box updated from before v1.9 is told how to reach v1.9's draft.

install.sh keeps the drafter the installed unit serves, so that the documented rollback
to the BF16 draft survives plain re-runs. That rollback is also the default of v1.2.3 to
v1.8.6, and a unit cannot say which it is: a box that was only ever updated kept the BF16
draft at depth 8 and never got the calibrated NVFP4 draft v1.9 measured at +30% (found in
review, 2026-09-24). The rule stays, and such a box is now told, with the exact command.

Since v1.36.0 (2026-10-17) the box itself tells them apart: the BF16 draft on a box older than
v1.9.0 with nothing recording it as a choice moves to the NVFP4 draft and says how to keep
it; recorded, on a younger box, or on one whose age cannot be told, it stays with the NOTE.
"""
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import installer_wall as wall  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "install.sh").read_text()
DEFAULTS = "\n".join(ln for ln in TEXT.splitlines() if re.match(r"DRAFT2_(REPO|REV|QUANT|TOKENS)=\"\$\{DRAFT2_", ln))
START = TEXT.index('''  CUR_DRAFT="$(grep -oE -- '--speculative-draft-model-path''')
BLOCK = TEXT[START:TEXT.index('\nfi\nif [ -n "$INSTALLED_CHOICE" ]; then', START)]
# old_default_moves (install.sh), which the block calls on a value an older release installed by
# default, asking conformity.py of the repo about the config folder each case gives
_MS = TEXT.index("old_default_moves() {")
MOVES = 'MOVED_TAG=""\n' + TEXT[_MS:TEXT.index("\n}\n", _MS) + 3]


def config_dir(key_at=None, record=None):
    """The config folder of a box (tests/installer_wall.py box_home): no key and no
    installed.json by default, a box whose age cannot be told."""
    return str(pathlib.Path(wall.box_home(key_at=key_at, record=record)) / ".config/qwen38")


OLD = ("ExecStart=... --speculative-draft-model-path z-lab/Qwen3.8-27B-DFlash2 "
       "--speculative-draft-model-revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 "
       "--speculative-draft-model-quantization unquant --speculative-num-draft-tokens 8 ...\n")


def run(unit_text, config=None):
    """install.sh's drafter block on this unit, for a box whose config folder is `config` (one
    whose age cannot be told by default); the pins it goes on with are on the last line."""
    unit = pathlib.Path(tempfile.mkdtemp(prefix="old-drafter-")) / "unit"
    unit.write_text(unit_text)
    script = ("set -euo pipefail\n_ENV_DRAFT2_REPO=; _ENV_DRAFT2_REV=; _ENV_DRAFT2_QUANT=; _ENV_DRAFT2_TOKENS=\n"
              + DEFAULTS + f'\nUNIT_PATH="{unit}"; REPO_DIR="{REPO}"; CONFIG_DIR="{config or config_dir()}"\n'
              + MOVES + BLOCK + '\necho "PINS $DRAFT2_REPO $DRAFT2_REV $DRAFT2_QUANT $DRAFT2_TOKENS"\n')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
    assert "command not found" not in r.stderr, r.stderr
    return r.stdout + r.stderr


class ThePreV19DraftIsNamed(unittest.TestCase):
    def test_a_box_on_the_old_default_is_told_the_way_over(self):
        out = run(OLD)
        self.assertIn("Keeping the installed drafter: z-lab/Qwen3.8-27B-DFlash2", out)
        self.assertIn("NOTE: that is the BF16 draft", out)
        default = re.search(r'DRAFT2_REPO="\$\{DRAFT2_REPO:-([^}]+)\}"', TEXT).group(1)
        self.assertIn(f"DRAFT2_REPO={default} ", out, "the command does not name today's default")
        self.assertIn("./install.sh", out)

    def test_a_box_on_the_current_draft_hears_nothing(self):
        current = run("ExecStart=... --speculative-draft-model-path "
                      + re.search(r'DRAFT2_REPO="\$\{DRAFT2_REPO:-([^}]+)\}"', TEXT).group(1)
                      + " --speculative-num-draft-tokens 16 ...\n")
        self.assertNotIn("NOTE", current)
        self.assertNotIn("Keeping the installed drafter", current)


class TheBoxDecides(unittest.TestCase):
    """v1.9.0 moved the default on 2026-09-11 at 13:06:45 (+02:00)."""
    BEFORE, AFTER = "2026-09-10T12:00:00+02:00", "2026-09-12T12:00:00+02:00"
    DEFAULT = re.search(r'DRAFT2_REPO="\$\{DRAFT2_REPO:-([^}]+)\}"', TEXT).group(1)

    def pins(self, out):
        return out.strip().splitlines()[-1].split()[1:]

    def test_on_a_box_older_than_v1_9_it_moves_to_the_nvfp4_draft_and_says_so(self):
        out = run(OLD, config_dir(key_at=self.BEFORE))
        self.assertIn("Moving the 27B drafter from z-lab/Qwen3.8-27B-DFlash2 (D=8, unquant) to " + self.DEFAULT, out)
        self.assertIn("the default since v1.9.0", out)
        self.assertIn("To keep it:", out)
        self.assertIn("DRAFT2_REPO=z-lab/Qwen3.8-27B-DFlash2 DRAFT2_REV=50307d4c4cde6860d4eee73e2547cd786fe8e8a4 "
                      "DRAFT2_QUANT=unquant DRAFT2_TOKENS=8 ./install.sh", out)
        self.assertNotIn("Keeping the installed drafter", out)
        self.assertEqual(self.pins(out), [self.DEFAULT, "bd7a934213c47a9e7ef69eef36bb3325f47fd1f1", "modelopt_fp4", "16"])

    def test_recorded_as_a_choice_it_stays_with_its_note(self):
        rec = {"DRAFT2": "z-lab/Qwen3.8-27B-DFlash2 (unquant, D=8)"}
        out = run(OLD, config_dir(key_at=self.BEFORE, record=rec))
        self.assertIn("Keeping the installed drafter: z-lab/Qwen3.8-27B-DFlash2", out)
        self.assertIn("NOTE: that is the BF16 draft", out)
        self.assertEqual(self.pins(out)[0], "z-lab/Qwen3.8-27B-DFlash2")

    def test_on_a_box_younger_than_v1_9_it_stays(self):
        out = run(OLD, config_dir(key_at=self.AFTER))
        self.assertIn("Keeping the installed drafter: z-lab/Qwen3.8-27B-DFlash2", out)
        self.assertEqual(self.pins(out), ["z-lab/Qwen3.8-27B-DFlash2", "50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
                                          "unquant", "8"])

    def test_the_bf16_draft_at_another_depth_was_somebodys_choice(self):
        out = run(OLD.replace("--speculative-num-draft-tokens 8", "--speculative-num-draft-tokens 16"),
                  config_dir(key_at=self.BEFORE))
        self.assertIn("Keeping the installed drafter: z-lab/Qwen3.8-27B-DFlash2 (D=16, unquant)", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
