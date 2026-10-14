#!/usr/bin/env python3
"""install-kolibri.sh's proof says why it failed, even when the lane's journal cannot be read.

Its two failure lines print the journal through `sudo journalctl ... | tail -40`, then die
with the reason. Under `set -euo pipefail`, a sudo that could not run (no ticket, no terminal)
failed the pipeline and ended the script there: only sudo's own complaint was printed, and
the reason never was (found in review, 2026-10-14). The script's own lines run here with a
sudo that fails.
"""
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "install-kolibri.sh").read_text()
_d = TEXT.index('  if [ "$(systemctl is-active "$UNIT" || true)" = failed ]')
DIED = TEXT[_d:TEXT.index("\n  fi\n", _d) + len("\n  fi\n")]
_r = TEXT.index('[ "$READY" -eq 1 ] || {')
TIMEOUT = TEXT[_r:TEXT.index("\n", _r)]


def run(block):
    d = pathlib.Path(tempfile.mkdtemp(prefix="kol-proof-"))
    for tool, body in (("sudo", 'echo "sudo: a password is required" >&2; exit 1'),
                       ("systemctl", 'case "$1" in is-active) echo failed;; show) echo 0;; esac; exit 0')):
        (d / tool).write_text(f"#!/bin/bash\n{body}\n")
        (d / tool).chmod(0o755)
    script = ("set -euo pipefail\n" 'die(){ echo "DIE: $*"; exit 1; }\n'
              'UNIT=qwen38-kolibri.service; SINCE=now; R0=0; READY=0\n' + block)
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                       env={"PATH": f"{d}:/usr/bin:/bin"})
    return r.returncode, r.stdout + r.stderr


class TheReasonIsSaidWithoutTheJournal(unittest.TestCase):
    def test_a_lane_that_died(self):
        rc, out = run(DIED)
        self.assertEqual(rc, 1)
        self.assertIn("DIE: the lane died while starting", out)

    def test_a_lane_that_never_answered(self):
        rc, out = run(TIMEOUT)
        self.assertEqual(rc, 1)
        self.assertIn("DIE: the lane did not answer /health within 20 min", out)


if __name__ == "__main__":
    unittest.main()
