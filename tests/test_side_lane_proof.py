#!/usr/bin/env python3
"""A side lane that serves an older unit than the one installed is said so, and restarted for its proof.

./install-image.sh, ./install-video.sh and ./install-kolibri.sh prove their lane at the end. Run
while their lane served, they proved the engine that ran, which an update of the unit (this
run's, or an earlier update's, which leaves a serving lane serving) had not reached: the proof
said "installed and proved it serves" of a unit nothing had started, and only the Kolibri
installer said a restart was needed, and only for its own run's change (found in review,
2026-10-14). The installers' own lines run here, with a systemctl that answers when the lane
started and a unit file older or newer than that.
"""
import os
import pathlib
import re
import subprocess
import tempfile
import time
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANES = REPO / "lanes.sh"
LANES_OF = {"install-image.sh": "WAS_IMAGE", "install-video.sh": "WAS_VIDEO", "install-kolibri.sh": "WAS_KOLIBRI"}


def blocks(script):
    text = (REPO / script).read_text()
    note = re.search(r'\nif lane_serves_older_unit "\$UNIT" "\$INSTALLED"; then\n  echo "NOTE: [^\n]*\nfi\n', text)
    proof = re.search(r'\nif \[ "\$%s" -eq 1 \] && lane_serves_older_unit "\$UNIT" "\$INSTALLED"; then\n.*?\nfi\n'
                      % LANES_OF[script], text, re.S)
    assert note and proof, f"{script}: its note or its proof's start moved"
    return note.group(0), proof.group(0)


def run(block, serving, unit_newer, was):
    d = pathlib.Path(tempfile.mkdtemp(prefix="side-proof-"))
    started = int(time.time()) - 600
    unit = d / "x.service"
    unit.write_text("[Service]\n")
    t = started + (60 if unit_newer else -60)
    os.utime(unit, (t, t))
    log = d / "calls"
    (d / "systemctl").write_text(
        "#!/bin/bash\n"
        f'case "$*" in *ActiveState*) echo {"active" if serving else "inactive"};; '
        f'*ExecMainStartTimestamp*) echo "@{started}";; esac\n')
    (d / "sudo").write_text(f'#!/bin/bash\necho "sudo $*" >> "{log}"\n')
    for f in ("systemctl", "sudo"):
        (d / f).chmod(0o755)
    script = (f'set -euo pipefail\n. "{LANES}"\nUNIT=x.service; INSTALLED="{unit}"\n'
              + "".join(f"{v}={was}\n" for v in LANES_OF.values()) + block)
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                       env={"PATH": f"{d}:/usr/bin:/bin"})
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout, (log.read_text().splitlines() if log.exists() else [])


class TheProofProvesTheUnitInstalled(unittest.TestCase):
    def test_a_lane_serving_an_older_unit_restarts_for_its_proof(self):
        for script in LANES_OF:
            with self.subTest(script=script):
                out, calls = run(blocks(script)[1], serving=True, unit_newer=True, was=1)
                self.assertEqual(calls, ["sudo systemctl restart x.service"], out)
                self.assertIn("restarts to prove this one", out)

    def test_a_lane_serving_its_unit_is_proved_as_it_runs(self):
        for script in LANES_OF:
            with self.subTest(script=script):
                _, calls = run(blocks(script)[1], serving=True, unit_newer=False, was=1)
                self.assertNotIn("sudo systemctl restart x.service", calls)

    def test_a_lane_that_did_not_serve_starts(self):
        for script in LANES_OF:
            with self.subTest(script=script):
                _, calls = run(blocks(script)[1], serving=False, unit_newer=True, was=0)
                self.assertEqual(calls, ["sudo systemctl start x.service"])


class AnUpdateSaysTheServingLaneNeedsARestart(unittest.TestCase):
    def test_said_when_the_unit_is_newer_than_the_lane(self):
        for script in LANES_OF:
            with self.subTest(script=script):
                out, _ = run(blocks(script)[0], serving=True, unit_newer=True, was=1)
                self.assertIn("serves the unit it started with", out)
                out, _ = run(blocks(script)[0], serving=True, unit_newer=False, was=1)
                self.assertEqual(out, "")
                out, _ = run(blocks(script)[0], serving=False, unit_newer=True, was=0)
                self.assertEqual(out, "", "a lane that does not serve needs no restart")


if __name__ == "__main__":
    unittest.main()
