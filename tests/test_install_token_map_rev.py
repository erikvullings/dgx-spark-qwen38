#!/usr/bin/env python3
"""The flash draft vocabulary is built for a branch revision too, once, and only from a snapshot.

install.sh looked for the checkpoint at snapshots/$MODEL_REV, and with MODEL_REV=main (a
documented setting) that folder never exists: snapshots are named by the commit that
refs/main holds, so the reduced draft vocabulary was skipped on every run, with a note
saying the checkpoint was not in place yet (found in review, 2026-09-24).

The build lives in lanes.sh (flash_token_map) since a lane installed on demand builds it at its
first Load too (prepare-lane.sh); it runs here as written, with a fake docker that records its
arguments.
"""
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]


def build(rev, with_ref=True, built=False, snapshot=True, docker_rc=0):
    t = pathlib.Path(tempfile.mkdtemp(prefix="map-rev-"))
    hf = t / "hf"
    repo = hf / "hub" / "models--org--flash"
    if snapshot:
        (repo / "snapshots" / "abc123").mkdir(parents=True)
    if with_ref:
        (repo / "refs").mkdir(parents=True)
        (repo / "refs" / "main").write_text("abc123")
    cfg = t / "cfg"
    cfg.mkdir()
    if built:
        (cfg / "token-map-65536.pt").write_text("map")
    (t / "bin").mkdir()
    (t / "bin" / "docker").write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{t}/docker.log"\nexit {docker_rc}\n')
    (t / "bin" / "docker").chmod(0o755)
    script = (f'set -euo pipefail\nCONFIG_DIR="{cfg}"; REPO_DIR="{REPO}"\n. "{REPO}/lanes.sh"\n'
              f'flash_token_map img "{hf}" org/flash {rev} 65536 "the advice"\n')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                       env={"PATH": f"{t}/bin:/usr/bin:/bin"})
    runs = (t / "docker.log").read_text().splitlines() if (t / "docker.log").exists() else []
    return r.returncode, r.stdout + r.stderr, runs


class TheSnapshotOfABranch(unittest.TestCase):
    def test_main_leads_to_the_commit_it_names(self):
        rc, out, runs = build("main")
        self.assertEqual(rc, 0, out)
        self.assertEqual(len(runs), 1, out)
        self.assertIn("--snapshot /root/.cache/huggingface/hub/models--org--flash/snapshots/abc123 ", runs[0])
        self.assertIn("--out /out/token-map-65536.pt --size 65536", runs[0])

    def test_a_commit_still_leads_to_itself(self):
        rc, out, runs = build("abc123", with_ref=False)
        self.assertEqual(rc, 0, out)
        self.assertIn("/snapshots/abc123 ", runs[0])


class OnceAndOnlyFromASnapshot(unittest.TestCase):
    def test_a_map_already_built_is_kept(self):
        rc, out, runs = build("main", built=True)
        self.assertEqual((rc, runs), (0, []))
        self.assertIn("already built", out)

    def test_no_snapshot_yet_is_a_note_with_the_callers_advice(self):
        rc, out, runs = build("main", snapshot=False)
        self.assertEqual((rc, runs), (0, []))
        self.assertIn("no snapshot at the pinned revision yet", out)
        self.assertIn("the advice", out)

    def test_a_build_that_fails_is_a_failure(self):
        rc, out, runs = build("main", docker_rc=3)
        self.assertEqual(rc, 1, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
