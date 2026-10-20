#!/usr/bin/env python3
"""Every installer of this repository runs by its path.

install.sh runs each lane's installer so ("$REPO_DIR/install-<lane>.sh"), and the docs tell a user to
run ./install-<lane>.sh. install-flash-local.sh went in without its executable bit: every test ran it
through bash, and on the reference box install.sh's run of it said "Permission denied", the lane left
out, and its first Load from the cockpit found no unit (found on the box, 2026-10-20). Read from git's
index, which is what a checkout gives each file; outside a checkout, from the files themselves."""
import os
import pathlib
import subprocess
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
RUN_BY_PATH = ["install*.sh", "prepare-lane.sh", "switch-model.sh", "uninstall.sh", "get.sh"]


class EveryInstallerRunsByItsPath(unittest.TestCase):
    def test_each_one_is_executable(self):
        r = subprocess.run(["git", "-C", str(REPO), "ls-files", "-s", "--", *RUN_BY_PATH], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout:
            modes = {line.split("\t", 1)[1]: line.split()[0] for line in r.stdout.splitlines()}
            not_executable = sorted(f for f, m in modes.items() if m != "100755")
        else:
            files = [p for pattern in RUN_BY_PATH for p in REPO.glob(pattern)]
            modes = {p.name: "" for p in files}
            not_executable = sorted(p.name for p in files if not os.access(p, os.X_OK))
        self.assertIn("install-flash-local.sh", modes)
        self.assertEqual(not_executable, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
