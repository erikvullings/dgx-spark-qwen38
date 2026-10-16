#!/usr/bin/env python3
"""Each number of a Nemotron lane's checkpoint is declared once, in its installer's pin block.

switch-model.sh reads the repos, the revisions, the memory share and the download size from
there, as it does the Kolibri lane's (a share copied into the switch came back at the next switch
after it was retuned in the installer: the Kolibri lane's finding, 2026-10-14): Bolt's from
install-nemotron.sh, Super VL's from install-nemotron-super.sh, through the one block the two
lanes share. The switch's own lines run here against copies of the installers whose pins have
moved. And the two lanes run one image: Super VL's installer names it from Bolt's pins, as Bolt's
names it.
"""
import pathlib
import re
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
NEM = (REPO / "install-nemotron.sh").read_text()
SUPER = (REPO / "install-nemotron-super.sh").read_text()
SWITCH = (REPO / "switch-model.sh").read_text()

# the lanes' shared block, from the choice of the lane to the pins read
_s = SWITCH.index('  if [ "$CHOICE" = "nemotron" ]; then\n    NEM_UNIT=')
_m = SWITCH.index('  eval "NEM_DRAFT=', _s)
_e = SWITCH.index("\n", _m) + 1
SWITCH_PICK = SWITCH[_s:_e]
_r = SWITCH.index('  NEM_CUR="$(grep -m1', _e)
_re = SWITCH.index("    NEM_REWRITTEN=1\n  fi\n", _r) + len("    NEM_REWRITTEN=1\n  fi\n")
SWITCH_REWRITE = SWITCH[_r:_re]
LANES = {"nemotron": ("install-nemotron.sh", "NEMOTRON_BOLT", NEM, "qwen38-nemotron"),
         "nemotron-super": ("install-nemotron-super.sh", "NEMOTRON_SUPER", SUPER, "qwen38-nemotron-super")}


def moved_repo(lane):
    """A copy of the repo's installers, the lane's share, size and drafter moved."""
    installer, pin, text, _ = LANES[lane]
    d = pathlib.Path(tempfile.mkdtemp(prefix="nem-pins-"))
    text = re.sub(rf'(?m)^{pin}_GMU="[^"]*"', f'{pin}_GMU="0.66"', text)
    text = re.sub(rf'(?m)^{pin}_GB="[^"]*"', f'{pin}_GB="27"', text)
    text = re.sub(rf'(?m)^{pin}_DRAFT_REPO="[^"]*"', f'{pin}_DRAFT_REPO="nvidia/another-drafter"', text)
    for name, (_, _, body, _) in LANES.items():
        (d / LANES[name][0]).write_text(text if name == lane else body)
    return d


def bash(script):
    r = subprocess.run(["bash", "-c", 'set -euo pipefail\ndie(){ echo "DIE: $*"; exit 1; }\n' + script],
                       capture_output=True, text=True, timeout=30)
    return r.returncode, r.stdout.strip()


def units(d):
    """The variables the switch's block reads, its two units files of this test's."""
    d = pathlib.Path(d)
    for u in ("qwen38-nemotron.service", "qwen38-nemotron-super.service"):
        (d / u).touch()
    return (f'NEMOTRON_UNIT="{d}/qwen38-nemotron.service"; NEMOTRON_UNIT_NAME=qwen38-nemotron.service\n'
            f'NEMOTRON_SUPER_UNIT="{d}/qwen38-nemotron-super.service"; NEMOTRON_SUPER_UNIT_NAME=qwen38-nemotron-super.service\n'
            f'NEM_STAGE="{d}/qwen38-nemotron.service.switch-stage"; NSV_STAGE="{d}/qwen38-nemotron-super.service.switch-stage"\n')


def installer_numbers(lane):
    """What the lane's installer serves, read by running its own pin lines."""
    installer, pin, text, _ = LANES[lane]
    if lane == "nemotron":
        block = text[text.index("NEMOTRON_BASE_IMAGE="):text.index("IMAGE_GB=")]
    else:
        block = text[text.index("NEMOTRON_SUPER_REPO="):text.index("\n# The image's build at its peak")]
    return bash(f'HERE="{REPO}"\n' + block + '\necho "$MODEL $MODEL_REV $DRAFT $DRAFT_REV $GMU $WEIGHTS_GB $LOCAL_IMAGE"')


class EachNumberIsDeclaredOnce(unittest.TestCase):
    def test_the_switch_takes_the_share_the_size_and_the_drafter_from_the_installer(self):
        for lane in LANES:
            with self.subTest(lane=lane):
                d = moved_repo(lane)
                rc, out = bash(f'REPO_DIR="{d}"; CHOICE={lane}\n' + units(d) + SWITCH_PICK
                               + '\necho "$NEM_GMU $NEM_NEED_GB $NEM_DRAFT"')
                self.assertEqual((rc, out), (0, "0.66 27 nvidia/another-drafter"))

    def test_today_each_switch_and_its_installer_give_the_checkpoint_the_same_numbers(self):
        for lane in LANES:
            with self.subTest(lane=lane):
                rc, switch = bash(f'REPO_DIR="{REPO}"; CHOICE={lane}\n' + units(tempfile.mkdtemp()) + SWITCH_PICK
                                  + '\necho "$NEM_MODEL $NEM_REV $NEM_DRAFT $NEM_DREV $NEM_GMU $NEM_NEED_GB"')
                self.assertEqual(rc, 0, switch)
                rc, inst = installer_numbers(lane)
                self.assertEqual(rc, 0, inst)
                self.assertEqual(switch, inst.rsplit(" ", 1)[0])

    def test_the_two_lanes_name_one_image(self):
        images = {lane: installer_numbers(lane)[1].rsplit(" ", 1)[1] for lane in LANES}
        self.assertTrue(images["nemotron"].startswith("qwen38-nemotron:vllm"), images)
        self.assertEqual(images["nemotron-super"], images["nemotron"])

    def test_a_pin_the_installer_no_longer_defines_is_said(self):
        for lane, (installer, pin, text, _) in LANES.items():
            with self.subTest(lane=lane):
                d = pathlib.Path(tempfile.mkdtemp(prefix="nem-pins-"))
                for name, (other, _, body, _) in LANES.items():
                    (d / other).write_text(re.sub(rf'(?m)^{pin}_GMU="[^"]*"\n', "", body) if name == lane else body)
                rc, out = bash(f'REPO_DIR="{d}"; CHOICE={lane}\n' + units(d) + SWITCH_PICK)
                self.assertNotEqual(rc, 0)
                self.assertIn(f"{installer} no longer defines {pin}_GMU", out)

    def test_a_lane_not_installed_is_said_with_its_install_command(self):
        d = pathlib.Path(tempfile.mkdtemp(prefix="nem-pins-"))
        rc, out = bash(f'REPO_DIR="{REPO}"; CHOICE=nemotron-super\n' + units(d).replace(
            f'{d}/qwen38-nemotron-super.service"', f'{d}/absent.service"') + SWITCH_PICK)
        self.assertNotEqual(rc, 0)
        self.assertIn("the Nemotron 3.5 Super VL lane is not installed on this box", out)
        self.assertIn("./install.sh --with-nemotron-super (about 102 GB)", out)


def rendered(template, model, rev, draft, drev, gmu, d, seqs="4"):
    """The unit as the lane's installer renders its template with these pins."""
    text = (REPO / template).read_text()
    for k, v in (("__USER__", "u"), ("__GROUP__", "g"), ("__HOME__", str(d)), ("__IMAGE__", "qwen38-nemotron:test"),
                 ("__MODEL_DIR__", model.replace("/", "--")), ("__MODEL__", model), ("__REV__", rev),
                 ("__DRAFT__", draft), ("__DREV__", drev), ("__GMU__", gmu), ("__SEQS__", seqs),
                 ("__HF_CACHE__", f"{d}/hf"), ("__ENGINE_BIND__", "127.0.0.1"), ("__PORT__", "30000")):
        text = text.replace(k, v)
    assert not re.findall(r"__[A-Z][A-Z0-9_]*__", text), "a placeholder of the template is not rendered here"
    return text


class TheUnitFollowsThePinsWhole(unittest.TestCase):
    """A switch after the pins moved (a repo pulled, its installer not run) rewrites every place the
    unit names them: the checkpoint's revision after the checkpoint and, in Bolt's unit, in the
    reasoning parser's path in its snapshot, the drafter and its revision, the memory share. The
    first and the share were the only ones (found in review, 2026-10-16), and the engine would have
    loaded the new weights with the old snapshot's parser and a drafter the switch had not checked."""
    OLD = ("nvidia/old-bolt", "1" * 40, "nvidia/old-drafter", "2" * 40, "0.66")

    def switch(self, lane, d):
        return bash(f'REPO_DIR="{REPO}"; CHOICE={lane}\n' + units(d) + 'sudo(){ "$@"; }\n'
                    + SWITCH_PICK + SWITCH_REWRITE + '\necho "rewritten=$NEM_REWRITTEN"')

    def test_a_unit_from_older_pins_becomes_the_one_the_installer_renders_now(self):
        for lane, (installer, pin, text, unit_name) in LANES.items():
            with self.subTest(lane=lane):
                d = pathlib.Path(tempfile.mkdtemp(prefix="nem-rewrite-"))
                units(d)
                unit = d / f"{unit_name}.service"
                template = f"{unit_name}.service.template"
                unit.write_text(rendered(template, *self.OLD, d))
                rc, out = self.switch(lane, d)
                self.assertEqual(rc, 0, out)
                self.assertTrue(out.endswith("rewritten=1"), out)
                pins = dict(re.findall(rf'(?m)^({pin}_[A-Z_]+)="([^"]*)"', text))
                now = rendered(template, pins[f"{pin}_REPO"], pins[f"{pin}_REV"], pins[f"{pin}_DRAFT_REPO"],
                               pins[f"{pin}_DRAFT_REV"], pins[f"{pin}_GMU"], d)
                self.assertEqual(unit.read_text(), now)
                for old in ("old-bolt", "old-drafter", "1" * 40, "2" * 40):
                    self.assertNotIn(old, unit.read_text())
                self.assertFalse((d / f"{unit_name}.service.switch-stage").exists())
                # switched again, the unit has nothing to change
                rc, out = self.switch(lane, d)
                self.assertEqual((rc, out.splitlines()[-1]), (0, "rewritten="), out)
                self.assertEqual(unit.read_text(), now)

    def test_bolts_unit_names_its_parser_in_its_snapshot_and_super_vls_none(self):
        bolt = rendered("qwen38-nemotron.service.template", *self.OLD, "/h")
        svl = rendered("qwen38-nemotron-super.service.template", *self.OLD, "/h")
        self.assertIn(f"models--nvidia--old-bolt/snapshots/{'1' * 40}/ultra_v3_reasoning_parser.py", bolt)
        self.assertNotIn("/snapshots/", svl)
        self.assertIn("--reasoning-parser nemotron_v3", svl)

    def test_super_vls_requests_at_a_time_follow_its_pin_with_its_share(self):
        # Super VL's share was measured with four requests at a time: a switch after its pins moved
        # rewrites the two together. It rewrote the share alone, and left a pairing nobody measured
        # (found in review, 2026-10-16). Bolt's are its template's, which it renders as they are.
        d = pathlib.Path(tempfile.mkdtemp(prefix="nem-rewrite-"))
        units(d)
        unit = d / "qwen38-nemotron-super.service"
        unit.write_text(rendered("qwen38-nemotron-super.service.template", *self.OLD, d, seqs="4"))
        repo = d / "repo"
        repo.mkdir()
        (repo / "install-nemotron.sh").write_text(NEM)
        (repo / "install-nemotron-super.sh").write_text(re.sub(r'(?m)^NEMOTRON_SUPER_SEQS="[^"]*"', 'NEMOTRON_SUPER_SEQS="2"', SUPER))
        rc, out = bash(f'REPO_DIR="{repo}"; CHOICE=nemotron-super\n' + units(d) + 'sudo(){ "$@"; }\n'
                       + SWITCH_PICK + SWITCH_REWRITE + '\necho "rewritten=$NEM_REWRITTEN"')
        self.assertEqual(rc, 0, out)
        self.assertTrue(out.endswith("rewritten=1"), out)
        pins = dict(re.findall(r'(?m)^(NEMOTRON_SUPER_[A-Z_]+)="([^"]*)"', SUPER))
        self.assertEqual(unit.read_text(), rendered("qwen38-nemotron-super.service.template", pins["NEMOTRON_SUPER_REPO"],
                                                    pins["NEMOTRON_SUPER_REV"], pins["NEMOTRON_SUPER_DRAFT_REPO"],
                                                    pins["NEMOTRON_SUPER_DRAFT_REV"], pins["NEMOTRON_SUPER_GMU"], d, seqs="2"))
        self.assertIn("--max-num-seqs 2 ", unit.read_text())

    def test_a_bolt_unit_whose_parser_names_another_snapshot_is_refused_untouched(self):
        # Bolt loads its reasoning parser from its checkpoint's snapshot: a unit whose parser path
        # names another snapshot than its revision (a hand edit) is refused, as 08294c6 refused it;
        # the block shared with Super VL rewrote the revision around it, and the engine would have
        # started offline on a parser path not there (found in review, 2026-10-16)
        d = pathlib.Path(tempfile.mkdtemp(prefix="nem-rewrite-"))
        units(d)
        unit = d / "qwen38-nemotron.service"
        text = rendered("qwen38-nemotron.service.template", *self.OLD, d).replace(
            f"snapshots/{'1' * 40}/", f"snapshots/{'3' * 40}/")
        self.assertIn(f"snapshots/{'3' * 40}/ultra_v3_reasoning_parser.py", text)
        unit.write_text(text)
        rc, out = self.switch("nemotron", d)
        self.assertNotEqual(rc, 0, out)
        self.assertIn("names no checkpoint, drafter or memory share", out)
        self.assertEqual(unit.read_text(), text)

    def test_a_unit_that_does_not_name_its_drafter_is_refused_untouched(self):
        for lane, (_, _, _, unit_name) in LANES.items():
            with self.subTest(lane=lane):
                d = pathlib.Path(tempfile.mkdtemp(prefix="nem-rewrite-"))
                units(d)
                unit = d / f"{unit_name}.service"
                text = re.sub(r"--speculative-config\.revision [0-9a-f]{40} ", "",
                              rendered(f"{unit_name}.service.template", *self.OLD, d))
                unit.write_text(text)
                rc, out = self.switch(lane, d)
                self.assertNotEqual(rc, 0)
                self.assertIn("names no checkpoint, drafter or memory share", out)
                self.assertEqual(unit.read_text(), text)


if __name__ == "__main__":
    unittest.main()
