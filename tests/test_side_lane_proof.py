#!/usr/bin/env python3
"""A side lane that serves older files than the ones installed is said so, and restarted for its proof.

./install-image.sh, ./install-video.sh and ./install-kolibri.sh prove their lane at the end. Run
while their lane served, they proved the engine that ran, which an update of the unit (this
run's, or an earlier update's, which leaves a serving lane serving) had not reached: the proof
said "installed and proved it serves" of a unit nothing had started, and only the Kolibri
installer said a restart was needed, and only for its own run's change (found in review,
2026-10-14). The same holds for what the unit runs: the image and video lanes' venv and source
(and the notes of their wheel and source, which said a change before v1.30), Kolibri's key
middleware, and the scripts every start runs first (the reviews of the fix, the same day). The installers' own lines run here, with a systemctl that answers when the lane started
and files older or newer than that.
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
# what each lane's start reads besides its unit, as the installers name them
FILES_OF = {"install-image.sh": ("runtime-changed", "sglang-wheel", "cfg/engine-preflight.sh"),
            "install-video.sh": ("runtime-changed", "sglang-wheel", "sglang-source", "cfg/engine-preflight.sh"),
            "install-kolibri.sh": ("cfg/vllm-auth/vllm_engine_auth.py", "cfg/engine-preflight.sh",
                                   "cfg/engine-secrets.sh")}


def blocks(script):
    text = (REPO / script).read_text()
    files = re.search(r'\nLANE_FILES=\([^\n]*\)\n', text)
    note = re.search(r'\nif \[ "\$SMOKE" -eq 0 \] && lane_serves_older "\$UNIT" "\$\{LANE_FILES\[@\]\}"; then\n'
                     r'  echo "NOTE: [^\n]*\nfi\n', text)
    proof = re.search(r'\nif \[ "\$%s" -eq 1 \] && lane_serves_older "\$UNIT" "\$\{LANE_FILES\[@\]\}"; then\n.*?\nfi\n'
                      % LANES_OF[script], text, re.S)
    assert files and note and proof, f"{script}: its files, its note or its proof's start moved"
    assert files.start() < note.start() < proof.start(), f"{script}: the files are named before they are read"
    return files.group(0), note.group(0), proof.group(0)


def run(script, block, serving, newer=(), was=1, smoke=1, restarts=None):
    """`newer`: the files (FILES_OF's names, or "unit") changed after the lane started."""
    d = pathlib.Path(tempfile.mkdtemp(prefix="side-proof-"))
    started = int(time.time()) - 600
    unit = d / "x.service"
    unit.write_text("[Service]\n")
    paths = {"unit": unit}
    for name in FILES_OF[script]:
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
        paths[name] = p
    for name, p in paths.items():
        t = started + (60 if name in newer else -60)
        os.utime(p, (t, t))
    log = d / "calls"
    count = d / "nrestarts"
    count.write_text(f"{restarts or 0}\n")
    (d / "systemctl").write_text(
        "#!/bin/bash\n"
        f'case "$*" in *ActiveState*) echo {"active" if serving else "inactive"};; '
        f'*ExecMainStartTimestamp*) echo "@{started}";; *NRestarts*) cat "{count}";; esac\n')
    # a start or a restart resets the count of automatic restarts, as systemd does
    (d / "sudo").write_text(f'#!/bin/bash\necho "sudo $*" >> "{log}"\necho 0 > "{count}"\n')
    for f in ("systemctl", "sudo"):
        (d / f).chmod(0o755)
    head = (f'set -euo pipefail\n. "{LANES}"\nUNIT=x.service; INSTALLED="{unit}"; '
            f'RUNTIME_NOTE="{d}/runtime-changed"; WHEEL_NOTE="{d}/sglang-wheel"; SOURCE_NOTE="{d}/sglang-source"; '
            f'CONFIG_DIR="{d}/cfg"; SMOKE={smoke}\n'
            + "".join(f"{v}={was}\n" for v in LANES_OF.values()))
    r = subprocess.run(["bash", "-c", head + blocks(script)[0] + block + 'echo "R0=${R0:-unset}"\n'],
                       capture_output=True, text=True, timeout=30, env={"PATH": f"{d}:/usr/bin:/bin"})
    assert r.returncode == 0, r.stdout + r.stderr
    out = "\n".join(ln for ln in r.stdout.splitlines() if not ln.startswith("R0="))
    r0 = [ln[3:] for ln in r.stdout.splitlines() if ln.startswith("R0=")][-1]
    return out, (log.read_text().splitlines() if log.exists() else []), r0


class TheProofProvesWhatWasInstalled(unittest.TestCase):
    def test_a_lane_serving_an_older_file_restarts_for_its_proof(self):
        for script in LANES_OF:
            for name in ("unit",) + FILES_OF[script]:
                with self.subTest(script=script, newer=name):
                    out, calls, _ = run(script, blocks(script)[2], serving=True, newer=(name,))
                    self.assertEqual(calls, ["sudo systemctl restart x.service"], out)
                    self.assertIn("it restarts to prove these", out)

    def test_a_lane_serving_what_is_installed_is_proved_as_it_runs(self):
        for script in LANES_OF:
            with self.subTest(script=script):
                _, calls, _ = run(script, blocks(script)[2], serving=True)
                self.assertNotIn("sudo systemctl restart x.service", calls)

    def test_a_lane_that_did_not_serve_starts(self):
        for script in LANES_OF:
            with self.subTest(script=script):
                _, calls, _ = run(script, blocks(script)[2], serving=False, newer=("unit",), was=0)
                self.assertEqual(calls, ["sudo systemctl start x.service"])


class KolibrisCrashCountIsReadAfterItsStart(unittest.TestCase):
    """The proof waits for health and gives up when the unit restarts on its own: the count it
    compares with was read before the start or restart, which resets it, so a crash loop of the
    new unit went unseen until it passed the old count (found in review, 2026-10-14)."""

    def test_the_count_compared_is_the_one_after_the_restart(self):
        text = (REPO / "install-kolibri.sh").read_text()
        m = re.search(r'\nif \[ "\$WAS_KOLIBRI" -eq 1 \] && lane_serves_older .*?\nfi\n(?:#[^\n]*\n)*'
                      r'R0="\$\(systemctl show -p NRestarts --value "\$UNIT" 2>/dev/null \| tr -dc \'0-9\'\)"\n',
                      text, re.S)
        self.assertIsNotNone(m, "R0 is read right after the start or restart")
        self.assertEqual(len(re.findall(r'\nR0=', text)), 1, "and nowhere else")
        for serving, newer, was in ((True, ("unit",), 1), (False, (), 0), (True, (), 1)):
            with self.subTest(serving=serving, newer=newer):
                _, _, r0 = run("install-kolibri.sh", m.group(0), serving=serving, newer=newer, was=was,
                               restarts=4)
                self.assertEqual(r0, "4" if (serving and not newer) else "0")


class AnUpdateSaysTheServingLaneNeedsARestart(unittest.TestCase):
    def test_said_when_no_proof_follows_and_something_is_newer_than_the_lane(self):
        for script in LANES_OF:
            note = blocks(script)[1]
            with self.subTest(script=script):
                out, _, _ = run(script, note, serving=True, newer=("unit",), smoke=0)
                self.assertIn("runs older files than the ones installed now", out)
                out, _, _ = run(script, note, serving=True, newer=(FILES_OF[script][0],), smoke=0)
                self.assertIn("runs older files than the ones installed now", out)
                out, _, _ = run(script, note, serving=True, smoke=0)
                self.assertEqual(out, "")
                out, _, _ = run(script, note, serving=False, newer=("unit",), was=0, smoke=0)
                self.assertEqual(out, "", "a lane that does not serve needs no restart")
                out, _, _ = run(script, note, serving=True, newer=("unit",), smoke=1)
                self.assertEqual(out, "", "the proof that follows restarts it")


if __name__ == "__main__":
    unittest.main()
