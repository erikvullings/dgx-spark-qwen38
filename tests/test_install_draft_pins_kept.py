#!/usr/bin/env python3
"""Each of the four drafter pins follows the installed unit unless it is passed.

With any one of DRAFT2_REPO, DRAFT2_REV, DRAFT2_QUANT or DRAFT2_TOKENS passed, the whole
convergence was skipped and the other three fell back to the defaults: DRAFT2_TOKENS=8 on a
box serving the BF16 draft (the documented rollback) moved it to the NVFP4 draft without a
word (found in review, 2026-09-24). The revision and the quantization belong to the repo,
so they follow the unit only together with its repo. This runs install.sh's own block, on a
box whose age cannot be told unless a case says otherwise: there the BF16 draft is kept
whatever v1.36.0 decides for an old default (tests/test_install_old_drafter.py)."""
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
ENVS = "\n".join(ln for ln in TEXT.splitlines() if ln.startswith(("_ENV_DRAFT2_REPO=", "_ENV_DRAFT2_QUANT=")))
DEFAULTS = "\n".join(ln for ln in TEXT.splitlines() if re.match(r"DRAFT2_(REPO|REV|QUANT|TOKENS)=\"\$\{DRAFT2_", ln))
START = TEXT.index("  CUR_DRAFT=\"$(grep -oE -- '--speculative-draft-model-path")
BLOCK = TEXT[START:TEXT.index('\nfi\nif [ -n "$INSTALLED_CHOICE" ]; then', START)]
BF16 = ("ExecStart=... --speculative-draft-model-path z-lab/Qwen3.8-27B-DFlash2 "
        "--speculative-draft-model-revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 "
        "--speculative-draft-model-quantization unquant --speculative-num-draft-tokens 8 ...\n")
DEFAULT_REPO = re.search(r'DRAFT2_REPO="\$\{DRAFT2_REPO:-([^}]+)\}"', TEXT).group(1)
# old_default_moves (install.sh), which the block calls on a value an older release installed by
# default, asking conformity.py of the repo about the config folder each case gives
_MS = TEXT.index("old_default_moves() {")
MOVES = 'MOVED_TAG=""\n' + TEXT[_MS:TEXT.index("\n}\n", _MS) + 3]


def config_dir(key_at=None, record=None):
    """The config folder of a box (tests/installer_wall.py box_home): no key and no
    installed.json by default, a box whose age cannot be told."""
    return str(pathlib.Path(wall.box_home(key_at=key_at, record=record)) / ".config/qwen38")


def pins(unit_text, config=None, **env):
    d = pathlib.Path(tempfile.mkdtemp(prefix="draft-pins-"))
    (d / "unit").write_text(unit_text)
    script = ("set -euo pipefail\n" + ENVS + "\n" + DEFAULTS + f'\nUNIT_PATH="{d}/unit"\n'
              + f'REPO_DIR="{REPO}"; CONFIG_DIR="{config or config_dir()}"\n' + MOVES + BLOCK
              + '\necho "PINS $DRAFT2_REPO $DRAFT2_REV $DRAFT2_QUANT $DRAFT2_TOKENS"\n')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                       env={"PATH": "/usr/bin:/bin", **env})
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("PINS ")]
    assert line, r.stdout + r.stderr
    assert "command not found" not in r.stderr, r.stderr
    return line[0].split()[1:]


class EachPinFollowsTheUnit(unittest.TestCase):
    def test_the_env_block_names_all_four(self):
        self.assertIn('_ENV_DRAFT2_REV="${DRAFT2_REV:-}"', TEXT)
        self.assertIn('_ENV_DRAFT2_TOKENS="${DRAFT2_TOKENS:-}"', TEXT)

    def test_one_pin_passed_keeps_the_other_three(self):
        repo, rev, quant, tokens = pins(BF16, DRAFT2_TOKENS="16")
        self.assertEqual((repo, rev, quant), ("z-lab/Qwen3.8-27B-DFlash2", "50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
                                              "unquant"))
        self.assertEqual(tokens, "16")
        # and on a box older than v1.9.0 too: a run that passes a drafter pin is about the
        # drafter, so its BF16 draft is not moved under it
        old_box = config_dir(key_at="2026-09-10T12:00:00+02:00")
        self.assertEqual(pins(BF16, config=old_box, DRAFT2_TOKENS="16")[0], "z-lab/Qwen3.8-27B-DFlash2")
        self.assertEqual(pins(BF16, config=old_box)[0], DEFAULT_REPO, "the fixture is not an old box")

    def test_a_revision_passed_stays_on_the_installed_repo(self):
        self.assertEqual(pins(BF16, DRAFT2_REV="main")[:3], ["z-lab/Qwen3.8-27B-DFlash2", "main", "unquant"])

    def test_nothing_passed_keeps_all_four(self):
        self.assertEqual(pins(BF16), ["z-lab/Qwen3.8-27B-DFlash2", "50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
                                      "unquant", "8"])

    def test_a_repo_passed_takes_its_own_pins(self):
        self.assertEqual(pins(BF16, DRAFT2_REPO=DEFAULT_REPO)[0], DEFAULT_REPO)
        self.assertNotEqual(pins(BF16, DRAFT2_REPO=DEFAULT_REPO)[2], "unquant", "the old repo's quantization stayed")

    def test_the_default_repo_keeps_a_depth_it_was_given(self):
        unit = f"ExecStart=... --speculative-draft-model-path {DEFAULT_REPO} --speculative-num-draft-tokens 12 ...\n"
        self.assertEqual(pins(unit)[3], "12")
        self.assertEqual(pins(unit, DRAFT2_TOKENS="16")[3], "16")


if __name__ == "__main__":
    unittest.main(verbosity=2)
