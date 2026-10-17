#!/usr/bin/env python3
"""A flash box that turned off the reduced draft vocabulary or the replayssm verify state
keeps it off across plain re-runs, as it keeps its tier.

SPEC_TOKEN_MAP_SIZE=0 and FLASH_REPLAYSSM_SPEC=0 were undone by the next ./install.sh,
which read neither back from the installed launcher (found in review, 2026-09-24). A
launcher that does not speculate carries neither, and one rendered before a knob existed
says nothing about it: both keep the defaults. This runs install.sh's own lines.

The tier is read back the same way, and context was the default until v1.19.0: on a box
older than that with nothing recording it as a choice, it moves to concurrency since v1.36.0
(2026-10-17); recorded, or on a younger box, it stays."""
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
FUNC = TEXT[TEXT.index("resolve_flash_tier_args() {"):TEXT.index("\n}\n", TEXT.index("resolve_flash_tier_args() {")) + 3]
DEFAULTS = "\n".join(ln for ln in TEXT.splitlines()
                     if re.match(r'(FLASH_TIER|FLASH_REPLAYSSM_SPEC|SPEC_TOKEN_MAP_SIZE)="\$\{', ln)
                     or ln.startswith("TOKEN_MAP_NAME="))
START = TEXT.index("    # The tier lives in the launcher as the concurrency it pins")
BLOCK = TEXT[START:TEXT.index('    if [ -z "${_ENV_PLE_DIR:-}" ] && [ -n "$CUR_PLE" ]', START)]
# old_default_moves (install.sh), which the block calls on a value an older release installed by
# default, asking conformity.py of the repo about the config folder each case gives
_MS = TEXT.index("old_default_moves() {")
MOVES = 'MOVED_TAG=""\n' + TEXT[_MS:TEXT.index("\n}\n", _MS) + 3]


def config_dir(key_at=None, record=None):
    """The config folder of a box (tests/installer_wall.py box_home): no key and no
    installed.json by default, a box whose age cannot be told."""
    return str(pathlib.Path(wall.box_home(key_at=key_at, record=record)) / ".config/qwen38")


HEAD = "# Engine since v1.8: the OFFICIAL SGLang image\n"
NEW = "# No --allow-auto-truncate. It was here undocumented until 2026-09-13\n"
CONTEXT = ("TIER=(--max-running-requests 4 --max-mamba-cache-size 20 --mamba-radix-cache-strategy extra_buffer "
           "--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 "
           "--speculative-num-draft-tokens 4{replay})\n")
MAP = "TIER+=(--speculative-token-map /out/token-map-{n}.pt)\n"


def run_block(launcher, config=None, **env):
    """(output, {size, name, replay, tier, args}) of install.sh's lines on this launcher, for
    a box whose config folder is `config` (one whose age cannot be told by default)."""
    d = pathlib.Path(tempfile.mkdtemp(prefix="flash-knobs-"))
    (d / "launch-flash.sh").write_text(launcher)
    # FLASH_TIER_ENV is taken before the defaults fill FLASH_TIER in, as install.sh takes it at
    # its top: taken after them, it was never empty here and the tier was never read back
    script = ('set -euo pipefail\nRECOMMENDED=0\nFLASH_TIER_ENV="${FLASH_TIER:-}"\n' + DEFAULTS + "\n" + FUNC
              + "\nresolve_flash_tier_args\n"
              + f'REPO_DIR="{REPO}"; CONFIG_DIR="{config or config_dir()}"\n' + MOVES
              + f'_ENV_SPEC_TOKEN_MAP_SIZE="${{SPEC_TOKEN_MAP_SIZE_ENV:-}}"\n'
              f'_ENV_FLASH_REPLAYSSM_SPEC="${{FLASH_REPLAYSSM_SPEC_ENV:-}}"\nFLASH_LAUNCH="{d}/launch-flash.sh"\n'
              + BLOCK + '\necho "KNOBS $SPEC_TOKEN_MAP_SIZE $TOKEN_MAP_NAME $FLASH_REPLAYSSM_SPEC $FLASH_TIER $FLASH_TIER_ARGS"\n')
    extra = {}
    for k, v in env.items():
        extra[k] = v
        if k in ("SPEC_TOKEN_MAP_SIZE", "FLASH_REPLAYSSM_SPEC"):
            extra[k + "_ENV"] = v
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                       env={"PATH": "/usr/bin:/bin", **extra})
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("KNOBS ")]
    assert line, r.stdout + r.stderr
    assert "command not found" not in r.stderr, r.stderr
    size, name, replay, tier, *args = line[0].split()[1:]
    return r.stdout + r.stderr, {"size": size, "name": name, "replay": replay, "tier": tier, "args": " ".join(args)}


def knobs(launcher, **env):
    _, k = run_block(launcher, **env)
    return k["size"], k["name"], k["replay"], k["args"]


class TheKnobsFollowTheLauncher(unittest.TestCase):
    def test_both_off_stay_off(self):
        size, name, replay, args = knobs(HEAD + NEW + CONTEXT.format(replay=""))
        self.assertEqual((size, name, replay), ("0", "token-map-0.pt", "0"))
        self.assertNotIn("--enable-linear-replayssm-spec", args)

    def test_a_size_of_its_own_is_kept(self):
        size, name, _, _ = knobs(HEAD + NEW + CONTEXT.format(replay=" --enable-linear-replayssm-spec")
                                 + MAP.format(n=32768))
        self.assertEqual((size, name), ("32768", "token-map-32768.pt"))

    def test_the_defaults_stay_on_a_default_launcher(self):
        size, _, replay, args = knobs(HEAD + NEW + CONTEXT.format(replay=" --enable-linear-replayssm-spec")
                                      + MAP.format(n=65536))
        self.assertEqual((size, replay), ("65536", "1"))
        self.assertIn("--enable-linear-replayssm-spec", args)

    def test_a_launcher_older_than_the_knob_gets_the_default(self):
        size, _, replay, _ = knobs(CONTEXT.format(replay=""))      # neither marker: before v1.8
        self.assertEqual((size, replay), ("65536", "1"))
        _, _, replay, _ = knobs(HEAD + CONTEXT.format(replay="") + MAP.format(n=65536))   # v1.8 to v1.10.1
        self.assertEqual(replay, "1")

    def test_a_value_passed_wins(self):
        size, _, replay, _ = knobs(HEAD + NEW + CONTEXT.format(replay=""), SPEC_TOKEN_MAP_SIZE="65536",
                                   FLASH_REPLAYSSM_SPEC="1")
        self.assertEqual((size, replay), ("65536", "1"))

    def test_a_throughput_launcher_says_nothing_of_either(self):
        size, _, replay, _ = knobs(HEAD + NEW + "TIER=(--max-running-requests 24 --max-mamba-cache-size 96 "
                                   "--mamba-radix-cache-strategy extra_buffer_lazy)\n")
        self.assertEqual((size, replay), ("65536", "1"))


class AContextTierFromBeforeV119(unittest.TestCase):
    """context, the default tier until v1.19.0: the launcher cannot say whether a box kept it
    or chose it, the record and the age of the box can."""
    LAUNCHER = HEAD + NEW + CONTEXT.format(replay=" --enable-linear-replayssm-spec") + MAP.format(n=65536)
    BEFORE, AFTER = "2026-09-30T12:00:00+02:00", "2026-10-02T12:00:00+02:00"

    def test_on_a_box_older_than_v1_19_it_moves_to_concurrency_and_says_so(self):
        out, k = run_block(self.LAUNCHER, config=config_dir(key_at=self.BEFORE))
        self.assertEqual(k["tier"], "concurrency")
        self.assertIn("--max-running-requests 8 ", k["args"])
        self.assertIn("Moving the flash tier from context to concurrency, the default since v1.19.0", out)
        self.assertIn("To keep it: FLASH_TIER=context ./install.sh", out)
        self.assertNotIn("Keeping the installed flash tier", out)

    def test_recorded_as_a_choice_it_stays_on_any_box(self):
        out, k = run_block(self.LAUNCHER, config=config_dir(key_at=self.BEFORE, record={"FLASH_TIER": "context"}))
        self.assertEqual(k["tier"], "context")
        self.assertIn("Keeping the installed flash tier: context.", out)

    def test_on_a_box_younger_than_v1_19_it_stays(self):
        out, k = run_block(self.LAUNCHER, config=config_dir(key_at=self.AFTER))
        self.assertEqual(k["tier"], "context")
        self.assertIn("--max-running-requests 4 ", k["args"])

    def test_a_box_of_unknown_age_keeps_it(self):
        self.assertEqual(run_block(self.LAUNCHER)[1]["tier"], "context")

    def test_throughput_never_was_a_default_and_never_moves(self):
        launcher = HEAD + NEW + "TIER=(--max-running-requests 24 --max-mamba-cache-size 96)\n"
        self.assertEqual(run_block(launcher, config=config_dir(key_at=self.BEFORE))[1]["tier"], "throughput")


if __name__ == "__main__":
    unittest.main(verbosity=2)
