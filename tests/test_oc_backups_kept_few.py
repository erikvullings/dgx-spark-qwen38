#!/usr/bin/env python3
"""A config edit keeps the newest backups of the file and lets the older ones go.

Every install and every lane switch edits ~/.config/opencode/opencode.json and the cockpit's copy,
and each edit leaves a .bak-<time> beside them; none was ever removed: 98 and 91 of them on the
reference box after three weeks (2026-10-12), each one holding the API key. A backup someone made
by hand under another name is never touched."""
import importlib.util
import os
import pathlib
import tempfile
import shutil
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
MERGE = REPO / "oc-merge-limits.py"


def load_module():
    spec = importlib.util.spec_from_file_location("ocm_backups_under_test", MERGE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class OldBackupsGo(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp(prefix="oc-bak-"))
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.path = self.dir / "opencode.json"
        self.path.write_text('{"autoupdate": "notify"}\n')

    def stamped(self):
        return sorted(p.name for p in self.dir.iterdir() if p.name.startswith("opencode.json.bak-2"))

    def test_only_the_newest_backups_stay(self):
        m = load_module()
        for day in range(1, 16):
            (self.dir / f"opencode.json.bak-202609{day:02d}-120000").write_text("old\n")
        (self.dir / "opencode.json.bak-20260915-120000-1").write_text("old\n")
        (self.dir / "opencode.json.bak-avant-maj").write_text("by hand\n")
        new = m.backup(str(self.path))
        left = self.stamped()
        self.assertEqual(len(left), 10, left)
        self.assertIn(os.path.basename(new), left)
        self.assertIn("opencode.json.bak-20260915-120000-1", left, "the newest older ones stay")
        self.assertNotIn("opencode.json.bak-20260901-120000", left, "the oldest one goes")
        self.assertTrue((self.dir / "opencode.json.bak-avant-maj").exists(), "a backup made by hand is kept")

    def test_backups_of_another_file_are_not_counted(self):
        m = load_module()
        for day in range(1, 16):
            (self.dir / f"other.json.bak-202609{day:02d}-120000").write_text("x\n")
        m.backup(str(self.path))
        self.assertEqual(len([p for p in self.dir.iterdir() if p.name.startswith("other.json.bak-")]), 15)


if __name__ == "__main__":
    unittest.main()
