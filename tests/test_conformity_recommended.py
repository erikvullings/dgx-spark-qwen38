#!/usr/bin/env python3
"""What this release recommends, and which values only an older release made the default.

An update kept, as if the operator had chosen them, values that were nothing but the default
of the release that first installed the box (the engine on every interface, the native
window, the flash context tier, the BF16 draft at depth 8). Telling the two apart starts with
two facts, and conformity.py holds both: what a box installed today gets (read from install.sh
itself, so it cannot drift from it), and the four old defaults with the instant their release
moved them (2026-10-17). These tests hold each fact to its source: the values to what a fresh
box resolves when install.sh runs, and each old default to the CHANGELOG section of the
release that moved it.
"""
import datetime
import os
import pathlib
import re
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import installer_wall as wall  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import conformity  # noqa: E402

INSTALL = (REPO / "install.sh").read_text()
CHANGELOG = (REPO / "CHANGELOG.md").read_text()
# The first refusal below everything that resolves a default (the context mode is the last
# of them); the probe prints what the run goes on with just above it.
PROBE_AT = 'if [ "$NO_COCKPIT" -eq 1 ] && [ "$WITH_COCKPIT" -eq 1 ]; then\n'
STOP = ["--no-cockpit", "--with-cockpit"]
PROBE = ('echo "RESOLVED target.27b=$MODEL_CHOICE|DRAFT2=$DRAFT2_REPO ($DRAFT2_QUANT, D=$DRAFT2_TOKENS)'
         '|CONTEXT_MODE=$CONTEXT_MODE|ENGINE_BIND=$ENGINE_BIND|PROXY_BIND=$PROXY_BIND|PORT=$PORT'
         '|PROXY_PORT=$PROXY_PORT|HF_CACHE=$HF_CACHE|PLE_DIR=$PLE_DIR|FLASH_TIER=$FLASH_TIER'
         '|SPEC_TOKEN_MAP_SIZE=$SPEC_TOKEN_MAP_SIZE|FLASH_REPLAYSSM_SPEC=$FLASH_REPLAYSSM_SPEC"\n')


def fresh_box_resolves():
    """What a walled install.sh resolves on a box with no unit and no variable set: the
    recommended state, as the installer itself computes it."""
    script = pathlib.Path(wall.walled(units=wall.units_dir()))
    text = script.read_text()
    assert text.count(PROBE_AT) == 1, "install.sh no longer has the refusal the probe goes before"
    script.write_text(text.replace(PROBE_AT, PROBE + PROBE_AT))
    env, record = wall.fenced_env()       # a throwaway HOME, removed with the wall's root
    home = env["HOME"]
    r = subprocess.run([str(script), *STOP], capture_output=True, text=True, env=env,
                       cwd=wall.cwd(), timeout=60)
    out = r.stdout + r.stderr
    assert wall.WALL not in out, out[-800:]
    assert not wall.reached(record), wall.reached(record)
    line = next((ln for ln in out.splitlines() if ln.startswith("RESOLVED ")), None)
    assert line, f"the probe never printed:\n{out[-800:]}"
    return home, dict(kv.split("=", 1) for kv in line[len("RESOLVED "):].split("|"))


def section(tag):
    """The CHANGELOG section of one release: its heading line and everything to the next one."""
    m = re.search(rf"^## {re.escape(tag)} \((\d{{4}}-\d{{2}}-\d{{2}})\): .*?(?=^## v|\Z)", CHANGELOG, re.M | re.S)
    return (m.group(1), m.group(0)) if m else (None, "")


class TheRecommendedState(unittest.TestCase):
    def test_it_is_what_install_sh_resolves_on_a_fresh_box(self):
        home, got = fresh_box_resolves()
        want = conformity.recommended(home=home)
        self.assertEqual({k: want[k] for k in got}, got)

    def test_every_knob_has_a_value(self):
        rec = conformity.recommended(home="/home/u")
        self.assertEqual(set(rec), set(conformity.KNOBS))
        self.assertTrue(all(isinstance(v, str) and v for v in rec.values()), rec)
        # the site settings follow the home they are given, not the one running the test
        self.assertEqual(rec["HF_CACHE"], "/home/u/.cache/huggingface")
        self.assertEqual(rec["PLE_DIR"], "/home/u/flashnext-ple")
        self.assertEqual(rec["PROXY_PORT"], str(int(rec["PORT"]) + 1))

    def test_the_recommended_targets_are_the_standard_checkpoints(self):
        """The 27B target a fresh box serves and the flash target its second text lane gets:
        both standard exports, never an abliterated one (the project advises the standard
        checkpoints)."""
        rec, targets = conformity.recommended(home="/h"), conformity.targets()
        self.assertEqual(targets[rec["target.27b"]], re.search(r'^STOCK_REPO="([^"]+)"', INSTALL, re.M).group(1))
        self.assertEqual(targets[rec["target.flash"]], re.search(r'^FLASH_REPO="([^"]+)"', INSTALL, re.M).group(1))
        self.assertIn(f"SECOND_TARGET={rec['target.flash']}\n", INSTALL)
        # every target install.sh accepts is known here, to the repository it serves
        accepted = re.search(r"MODEL_CHOICE must be (.+?) \(got", INSTALL).group(1)
        self.assertEqual(set(re.findall(r'"([a-z0-9-]+)"', accepted)), set(targets))

    def test_a_missing_default_fails_loudly(self):
        """A renamed variable in install.sh must not turn into a silent empty recommendation."""
        d = wall._tmp("conformity-repo-")
        (d / "install.sh").write_text(INSTALL.replace('ENGINE_BIND="${ENGINE_BIND:-127.0.0.1}"', 'ENGINE_BIND=x'))
        with self.assertRaises(KeyError):
            conformity.recommended(repo_dir=d, home="/h")


class TheOldDefaults(unittest.TestCase):
    def test_the_four_kept_as_choices_are_listed(self):
        self.assertEqual({o.knob for o in conformity.OLD_DEFAULTS},
                         {"ENGINE_BIND", "CONTEXT_MODE", "FLASH_TIER", "DRAFT2"})

    def test_each_has_its_release_its_date_and_its_words_in_the_changelog(self):
        for old in conformity.OLD_DEFAULTS:
            with self.subTest(knob=old.knob):
                when = datetime.datetime.fromisoformat(old.instant)
                self.assertIsNotNone(when.tzinfo, "an instant without its zone compares wrong")
                day, text = section(old.tag)
                self.assertEqual(day, when.date().isoformat(), f"no CHANGELOG section {old.tag} of that day")
                self.assertIn(old.says, text, f"{old.tag} does not say it moved {old.knob}")

    def test_each_differs_from_what_the_release_recommends_now(self):
        rec = conformity.recommended(home="/h")
        for old in conformity.OLD_DEFAULTS:
            self.assertNotEqual(old.value, rec[old.knob], old.knob)
            self.assertIs(conformity.old_default(old.knob, old.value), old)
            self.assertIsNone(conformity.old_default(old.knob, rec[old.knob]))

    def test_the_drafter_is_named_the_way_install_sh_reads_it(self):
        """The BF16 draft is an old default only as a whole (repository, no quantization,
        depth 8): the same repository at depth 16 was somebody's choice."""
        self.assertEqual(conformity.drafter("z-lab/Qwen3.8-27B-DFlash2", "unquant", "8"),
                         conformity.old_default_of("DRAFT2").value)
        self.assertIsNone(conformity.old_default("DRAFT2", conformity.drafter("z-lab/Qwen3.8-27B-DFlash2", "unquant", "16")))

    def test_same_day_changes_are_told_apart_by_the_instant(self):
        """v1.16.0 and v1.17.0 shipped 18 minutes apart, v1.12.0 and v1.12.1 54: the day alone
        would class a box installed between them on the wrong side."""
        for old in conformity.OLD_DEFAULTS:
            self.assertRegex(old.instant, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")


if __name__ == "__main__":
    unittest.main(verbosity=2)
