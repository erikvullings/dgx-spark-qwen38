#!/usr/bin/env python3
"""Every unit the scripts render takes the account's own group, not the shell's.

`$(id -gn)` is the group of the shell that runs the script: docker after `newgrp docker`, which
NVIDIA's own DGX Spark guides have a person type to reach Docker, and the units it rendered then
said Group=docker. An update from a normal shell wrote them back with the account's group, and
restarted what they run (found in review, 2026-10-20). `$(id -gn "$(id -un)")` is the account's
own group, whatever the shell's is."""
import pathlib
import shutil
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = sorted(REPO.glob("*.sh")) + sorted((REPO / "dashboard").glob("*.sh"))


class TheUnitsGroup(unittest.TestCase):
    def test_no_script_takes_the_shells_group(self):
        for f in SCRIPTS:
            with self.subTest(f.name):
                self.assertNotIn("$(id -gn)", f.read_text())

    def test_every_group_placeholder_is_rendered_from_the_account(self):
        seen = 0
        for f in SCRIPTS:
            for line in f.read_text().splitlines():
                if "__GROUP__|" in line and "s|__GROUP__|" in line:
                    seen += 1
                    with self.subTest(f"{f.name}: {line.strip()[:60]}"):
                        self.assertIn('id -gn "$(id -un)"', line)
        self.assertGreaterEqual(seen, 11, "fewer renders of __GROUP__ than the eleven known")

    def test_the_form_answers_the_accounts_group_from_a_newgrp_shell(self):
        # a stand-in id as a shell after `newgrp docker` answers it: its own group is docker, the
        # account's is alice. The form the renders use answers alice; the old one answered docker.
        d = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        (d / "id").write_text('#!/bin/sh\ncase "$1 $2" in\n  "-un ") echo alice ;;\n'
                              '  "-gn ") echo docker ;;\n  "-gn alice") echo alice ;;\n'
                              '  *) exit 1 ;;\nesac\n')
        (d / "id").chmod(0o755)
        env = {"PATH": f"{d}:/usr/bin:/bin"}
        form = subprocess.run(["/bin/bash", "-c", 'echo "$(id -gn "$(id -un)")"'], env=env,
                              capture_output=True, text=True).stdout.strip()
        old = subprocess.run(["/bin/bash", "-c", 'echo "$(id -gn)"'], env=env,
                             capture_output=True, text=True).stdout.strip()
        self.assertEqual((form, old), ("alice", "docker"))

if __name__ == "__main__":
    unittest.main(verbosity=2)
