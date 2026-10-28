#!/usr/bin/env python3
"""needle.sh says the engine is down when it is, instead of exiting 7 with nothing on screen.

Without --model it asks /v1/models first, through a `curl | python3` pipe inside `$(...)`
under `set -euo pipefail`: a curl that cannot connect exits 7, the assignment carries that
status, and set -e ended the script before the line that explains the failure (found in
review, 2026-09-24). Run against a port nothing listens on, with a HOME of its own.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import localports  # noqa: E402

NEEDLE = Path(__file__).resolve().parents[1] / "needle.sh"


class AnEngineThatIsDown(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="needle-down-")
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        os.makedirs(os.path.join(self.home, ".config", "qwen38"))
        with open(os.path.join(self.home, ".config", "qwen38", "api-key"), "w") as f:
            f.write("test-key\n")
        _, self.port = localports.held(self)    # nothing listens there, and no other program can

    def run_needle(self, *args):
        r = subprocess.run(["bash", str(NEEDLE), *args, "--trials", "1", "--depths", "1000", "--no-flush"],
                           capture_output=True, text=True, timeout=120,
                           env={**os.environ, "PORT": str(self.port), "HOME": self.home})
        return r.returncode, r.stdout + r.stderr

    def test_without_a_model_it_says_why_and_exits_2(self):
        rc, out = self.run_needle()
        self.assertEqual(rc, 2, out)
        self.assertIn("no model served", out)

    def test_with_a_model_the_calibration_says_why(self):
        rc, out = self.run_needle("--model", "m")
        self.assertEqual(rc, 2, out)
        self.assertIn("calibration request failed", out)


if __name__ == "__main__":
    unittest.main()
