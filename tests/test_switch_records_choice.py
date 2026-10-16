#!/usr/bin/env python3
"""A switch records the target it chose, in an installed.json that exists (2026-10-17).

A switch, the cockpit's Load among them, is the operator choosing a text lane's target, and
nothing wrote it down: until the next ./install.sh, installed.json still said what the last run
had found, and the cockpit read a stale record. switch-model.sh now has conformity.py choose
record the target (how=switch), or erase the entry when the target is the recommended one, so
the box follows the defaults again. A box no run of v1.36.0 or later recorded gets no file, and
a file that cannot be read is left as it is. It is never fatal: the switch is done either way.
"""
import json
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

CONFORMITY = str(REPO / "conformity.py")
NOW = "2026-10-20T08:12:00+00:00"
LATER = "2026-10-21T09:00:00+00:00"


def a_record(**choices):
    return {"schema": 1, "release": "v1.36.0", "commit": "d" * 40, "path": "full", "lane": "27b",
            "box_since": "2026-09-05T09:00:00+00:00", "box_since_from": "api-key",
            "choices": {k: {"value": v, "since": NOW, "how": "env"} for k, v in choices.items()}}


def config(record=None):
    cfg = wall._tmp("cfg-")
    if record is not None:
        (cfg / "installed.json").write_text(record if isinstance(record, str) else json.dumps(record))
    return cfg


def cli(cfg, knob, value):
    r = subprocess.run([sys.executable, CONFORMITY, "choose", "--config-dir", str(cfg), knob, value],
                       capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout + r.stderr


class ASwitchRecordsItsTarget(unittest.TestCase):
    def test_a_target_other_than_the_recommended_one_is_recorded(self):
        cfg = config(a_record())
        rec, changed = conformity.choose(cfg, "target.flash", "flash-nvda", now=NOW)
        self.assertEqual(changed, "recorded")
        self.assertEqual(rec["choices"]["target.flash"], {"value": "flash-nvda", "since": NOW, "how": "switch"})
        self.assertEqual(json.loads((cfg / "installed.json").read_text())["choices"],
                         {"target.flash": {"value": "flash-nvda", "since": NOW, "how": "switch"}})

    def test_the_recommended_target_erases_the_entry(self):
        rec27 = conformity.recommended()["target.27b"]
        cfg = config(a_record(**{"target.27b": "fp8", "ENGINE_BIND": "0.0.0.0"}))
        rec, changed = conformity.choose(cfg, "target.27b", rec27, now=LATER)
        self.assertEqual(changed, "erased")
        self.assertNotIn("target.27b", rec["choices"])
        self.assertIn("ENGINE_BIND", json.loads((cfg / "installed.json").read_text())["choices"],
                      "the other choices stay")

    def test_the_same_choice_again_keeps_its_date_and_how(self):
        cfg = config(a_record(**{"target.27b": "fp8"}))
        rec, changed = conformity.choose(cfg, "target.27b", "fp8", now=LATER)
        self.assertIsNone(changed)
        self.assertEqual(rec["choices"]["target.27b"], {"value": "fp8", "since": NOW, "how": "env"})

    def test_a_box_with_no_record_gets_none(self):
        cfg = config()
        self.assertEqual(conformity.choose(cfg, "target.flash", "flash-nvda"), (None, None))
        self.assertFalse((cfg / "installed.json").exists())
        rc, out = cli(cfg, "target.flash", "flash-nvda")
        self.assertEqual((rc, out), (0, ""))
        self.assertFalse((cfg / "installed.json").exists())

    def test_an_unreadable_record_is_left_as_it_is(self):
        cfg = config('{"schema": 1, "choices": ')
        rc, out = cli(cfg, "target.flash", "flash-nvda")
        self.assertEqual(rc, 0)
        self.assertIn("unreadable", out)
        self.assertEqual((cfg / "installed.json").read_text(), '{"schema": 1, "choices": ')

    def test_the_command_line_says_what_it_recorded(self):
        cfg = config(a_record())
        rc, out = cli(cfg, "target.flash", "flash-nvda")
        self.assertEqual(rc, 0)
        self.assertIn("Recorded   : target.flash=flash-nvda as your choice", out)
        rc, out = cli(cfg, "target.flash", conformity.recommended()["target.flash"])
        self.assertIn("back to the recommended", out)
        self.assertEqual(json.loads((cfg / "installed.json").read_text())["choices"], {})

    def test_the_record_stays_whole_and_readable_by_all(self):
        cfg = config(a_record())
        conformity.choose(cfg, "target.flash", "flash-nvda", now=NOW)
        self.assertEqual(oct((cfg / "installed.json").stat().st_mode & 0o777), "0o644")
        self.assertEqual([p.name for p in cfg.iterdir() if p.name.endswith(".tmp")], [])


class SwitchModelCallsIt(unittest.TestCase):
    TEXT = (REPO / "switch-model.sh").read_text()

    def test_the_text_switch_records_the_target_once_queued_and_never_fails_for_it(self):
        queued = self.TEXT.index('Switch queued: %s (%s)\\033[0m\\n\' "$TARGET_REPO" "$TARGET_UNIT_NAME"')
        call = re.search(r'python3 "\$REPO_DIR/conformity\.py" choose --config-dir "\$CONFIG_DIR" '
                         r'--repo-dir "\$REPO_DIR" \\\n  "target\.\$TARGET_LANE" "\$CHOICE" \|\| true\n',
                         self.TEXT[queued:])
        self.assertIsNotNone(call, "the call follows the queued switch, and an error is not the switch's")
        self.assertLess(call.start(), 600, "right after it, before the hints")

    def test_every_target_the_switch_takes_is_a_target_the_record_knows(self):
        cases = dict(re.findall(r'^  ([a-z0-9-]+)\)\s+TARGET_REPO="\$[A-Z0-9_]+";\s+TARGET_REV="\$[A-Z0-9_]+";\s+TARGET_LANE=(27b|flash) ;;',
                                self.TEXT, re.M))
        self.assertGreaterEqual(len(cases), 7, cases)
        known = conformity.targets()
        for choice, lane in cases.items():
            with self.subTest(choice=choice):
                self.assertIn(choice, known, "observe() reads this target back by the same name")
                self.assertIn(f"target.{lane}", conformity.KNOBS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
