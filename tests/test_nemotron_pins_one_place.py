#!/usr/bin/env python3
"""Each number of the Nemotron checkpoint is declared once, in install-nemotron.sh's pin block.

switch-model.sh reads the repos, the revisions, the memory share and the download size from
there, as it does the Kolibri lane's (a share copied into the switch came back at the next switch
after it was retuned in the installer: the Kolibri lane's finding, 2026-10-14). The switch's own
lines run here against a copy of install-nemotron.sh whose pins have moved.
"""
import pathlib
import re
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
NEM = (REPO / "install-nemotron.sh").read_text()
SWITCH = (REPO / "switch-model.sh").read_text()

_k = SWITCH.index('if [ "$CHOICE" = "nemotron" ]; then')
_s = SWITCH.index('  eval "$(grep -E', _k)
_e = SWITCH.index('NEM_DRAFT="$NEMOTRON_BOLT_DRAFT_REPO"; NEM_DREV="$NEMOTRON_BOLT_DRAFT_REV"\n', _s) \
    + len('NEM_DRAFT="$NEMOTRON_BOLT_DRAFT_REPO"; NEM_DREV="$NEMOTRON_BOLT_DRAFT_REV"\n')
SWITCH_PICK = SWITCH[_s:_e]
_r = SWITCH.index('  NEM_CUR="$(grep -m1', _e)
_re = SWITCH.index("    NEM_REWRITTEN=1\n  fi\n", _r) + len("    NEM_REWRITTEN=1\n  fi\n")
SWITCH_REWRITE = SWITCH[_r:_re]


def moved_repo():
    """A copy of the repo's install-nemotron.sh with its share, its size and its drafter moved."""
    d = pathlib.Path(tempfile.mkdtemp(prefix="nem-pins-"))
    text = re.sub(r'(?m)^NEMOTRON_BOLT_GMU="[^"]*"', 'NEMOTRON_BOLT_GMU="0.66"', NEM)
    text = re.sub(r'(?m)^NEMOTRON_BOLT_GB="[^"]*"', 'NEMOTRON_BOLT_GB="27"', text)
    text = re.sub(r'(?m)^NEMOTRON_BOLT_DRAFT_REPO="[^"]*"', 'NEMOTRON_BOLT_DRAFT_REPO="nvidia/another-drafter"', text)
    (d / "install-nemotron.sh").write_text(text)
    return d


def bash(script):
    r = subprocess.run(["bash", "-c", 'set -euo pipefail\ndie(){ echo "DIE: $*"; exit 1; }\n' + script],
                       capture_output=True, text=True, timeout=30)
    return r.returncode, r.stdout.strip()


class EachNumberIsDeclaredOnce(unittest.TestCase):
    def test_the_switch_takes_the_share_the_size_and_the_drafter_from_the_installer(self):
        d = moved_repo()
        rc, out = bash(f'REPO_DIR="{d}"; CHOICE=nemotron\n' + SWITCH_PICK + '\necho "$NEM_GMU $NEM_NEED_GB $NEM_DRAFT"')
        self.assertEqual((rc, out), (0, "0.66 27 nvidia/another-drafter"))

    def test_today_both_scripts_give_the_checkpoint_the_same_numbers(self):
        rc, switch = bash(f'REPO_DIR="{REPO}"; CHOICE=nemotron\n' + SWITCH_PICK
                          + '\necho "$NEM_MODEL $NEM_REV $NEM_DRAFT $NEM_DREV $NEM_GMU $NEM_NEED_GB"')
        self.assertEqual(rc, 0, switch)
        pins = NEM[NEM.index("NEMOTRON_BASE_IMAGE="):NEM.index("IMAGE_GB=")]
        rc, inst = bash(pins + '\necho "$MODEL $MODEL_REV $DRAFT $DRAFT_REV $GMU $WEIGHTS_GB"')
        self.assertEqual(rc, 0, inst)
        self.assertEqual(switch, inst)

    def test_a_pin_the_installer_no_longer_defines_is_said(self):
        d = pathlib.Path(tempfile.mkdtemp(prefix="nem-pins-"))
        (d / "install-nemotron.sh").write_text(re.sub(r'(?m)^NEMOTRON_BOLT_GMU="[^"]*"\n', "", NEM))
        rc, out = bash(f'REPO_DIR="{d}"; CHOICE=nemotron\n' + SWITCH_PICK)
        self.assertNotEqual(rc, 0)
        self.assertIn("install-nemotron.sh no longer defines NEMOTRON_BOLT_GMU", out)



def rendered(model, rev, draft, drev, gmu, d):
    """The unit as install-nemotron.sh renders its template with these pins."""
    text = (REPO / "qwen38-nemotron.service.template").read_text()
    for k, v in (("__USER__", "u"), ("__GROUP__", "g"), ("__HOME__", str(d)), ("__IMAGE__", "qwen38-nemotron:test"),
                 ("__MODEL_DIR__", model.replace("/", "--")), ("__MODEL__", model), ("__REV__", rev),
                 ("__DRAFT__", draft), ("__DREV__", drev), ("__GMU__", gmu), ("__HF_CACHE__", f"{d}/hf"),
                 ("__ENGINE_BIND__", "127.0.0.1"), ("__PORT__", "30000")):
        text = text.replace(k, v)
    assert not re.findall(r"__[A-Z][A-Z0-9_]*__", text), "a placeholder of the template is not rendered here"
    return text


class TheUnitFollowsThePinsWhole(unittest.TestCase):
    """A switch after the pins moved (a repo pulled, its installer not run) rewrites every place the
    unit names them: the checkpoint's revision after the checkpoint and in the reasoning parser's
    path in its snapshot, the drafter and its revision, the memory share. The first and the share
    were the only ones (found in review, 2026-10-16), and the engine would have loaded the new
    weights with the old snapshot's parser and a drafter the switch had not checked."""
    OLD = ("nvidia/old-bolt", "1" * 40, "nvidia/old-drafter", "2" * 40, "0.66")

    def switch(self, unit, d):
        return bash(f'REPO_DIR="{REPO}"; CHOICE=nemotron; NEMOTRON_UNIT="{unit}"; NEM_STAGE="{d}/stage"\n'
                    'sudo(){ "$@"; }\n' + SWITCH_PICK + SWITCH_REWRITE + '\necho "rewritten=$NEM_REWRITTEN"')

    def test_a_unit_from_older_pins_becomes_the_one_the_installer_renders_now(self):
        d = pathlib.Path(tempfile.mkdtemp(prefix="nem-rewrite-"))
        unit = d / "qwen38-nemotron.service"
        unit.write_text(rendered(*self.OLD, d))
        rc, out = self.switch(unit, d)
        self.assertEqual(rc, 0, out)
        self.assertTrue(out.endswith("rewritten=1"), out)
        pins = dict(re.findall(r'(?m)^(NEMOTRON_BOLT_[A-Z_]+)="([^"]*)"', NEM))
        now = rendered(pins["NEMOTRON_BOLT_REPO"], pins["NEMOTRON_BOLT_REV"], pins["NEMOTRON_BOLT_DRAFT_REPO"],
                       pins["NEMOTRON_BOLT_DRAFT_REV"], pins["NEMOTRON_BOLT_GMU"], d)
        self.assertEqual(unit.read_text(), now)
        for old in ("old-bolt", "old-drafter", "1" * 40, "2" * 40):
            self.assertNotIn(old, unit.read_text())
        self.assertFalse((d / "stage").exists())
        # switched again, the unit has nothing to change
        rc, out = self.switch(unit, d)
        self.assertEqual((rc, out.splitlines()[-1]), (0, "rewritten="), out)
        self.assertEqual(unit.read_text(), now)

    def test_a_unit_that_does_not_name_its_drafter_is_refused_untouched(self):
        d = pathlib.Path(tempfile.mkdtemp(prefix="nem-rewrite-"))
        unit = d / "qwen38-nemotron.service"
        text = re.sub(r"--speculative-config\.revision [0-9a-f]{40} ", "", rendered(*self.OLD, d))
        unit.write_text(text)
        rc, out = self.switch(unit, d)
        self.assertNotEqual(rc, 0)
        self.assertIn("names no checkpoint, drafter or memory share", out)
        self.assertEqual(unit.read_text(), text)


if __name__ == "__main__":
    unittest.main()
