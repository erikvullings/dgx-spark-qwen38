#!/usr/bin/env python3
"""ci-local.sh runs the workflow's jobs the way GitHub does.

A job that needs others runs once they have ended, and only when every one of them passed:
checks, the job the run's status goes by, ran beside the others and said "Every job passed"
before any of them had ended (2026-10-28), and said it as well after one of them failed. The
directory the jobs' worktrees go in goes with the run. Each case runs ci-local.sh on a
repository of its own, made here, whose workflow has three small jobs."""
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "ci-local.sh"
# ci-local.sh reads the workflow with PyYAML, in the python3 it finds first on PATH
HAVE_YAML = subprocess.run(["python3", "-c", "import yaml"], capture_output=True).returncode == 0

WORKFLOW = """\
name: ci
on: push
jobs:
  slow:
    runs-on: ubuntu-24.04-arm
    steps:
      - name: The slow one
        run: sleep 0.5 && {slow}
  quick:
    runs-on: ubuntu-24.04-arm
    steps:
      - name: The quick one
        run: "true"
  checks:
    needs: [slow, quick]
    runs-on: ubuntu-24.04-arm
    steps:
      - name: Every job passed
        run: echo slow and quick passed
"""


@unittest.skipUnless(HAVE_YAML, "ci-local.sh reads the workflow with PyYAML, which python3 lacks here")
class TheJobThatNeedsTheOthers(unittest.TestCase):
    def run_ci(self, slow="true", serial=False):
        root = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        repo, tmp = root / "repo", root / "tmp"
        (repo / ".github" / "workflows").mkdir(parents=True)
        tmp.mkdir()
        (repo / ".github" / "workflows" / "ci.yml").write_text(WORKFLOW.format(slow=slow))
        git = ["git", "-C", str(repo), "-c", "user.name=ci", "-c", "user.email=ci@example.invalid",
               "-c", "commit.gpgsign=false"]
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(git + ["add", "."], check=True)
        subprocess.run(git + ["commit", "-qm", "ci"], check=True)
        env = {k: v for k, v in os.environ.items() if k != "CI_LOCAL_SERIAL"}
        env["TMPDIR"] = str(tmp)
        if serial:
            env["CI_LOCAL_SERIAL"] = "1"
        r = subprocess.run(["bash", str(SCRIPT)], cwd=repo, env=env, capture_output=True, text=True,
                           timeout=120)
        return r, tmp

    def line(self, out, name):
        """The index of the line that says how the step named name ended."""
        hits = [i for i, ln in enumerate(out.splitlines()) if name in ln]
        self.assertEqual(len(hits), 1, out)
        return hits[0]

    def test_it_runs_once_they_have_ended(self):
        for serial in (False, True):
            with self.subTest(serial=serial):
                r, _ = self.run_ci(serial=serial)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                last = self.line(r.stdout, "ok   Every job passed")
                self.assertGreater(last, self.line(r.stdout, "ok   The slow one"), r.stdout)
                self.assertGreater(last, self.line(r.stdout, "ok   The quick one"), r.stdout)
                self.assertIn("CI local: 3 ok, 0 echecs, 0 sautes", r.stdout)

    def test_it_is_not_run_when_one_of_them_failed(self):
        for serial in (False, True):
            with self.subTest(serial=serial):
                r, _ = self.run_ci(slow="false", serial=serial)
                self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
                self.assertNotIn("ok   Every job passed", r.stdout)
                self.assertIn("--   Every job passed (not run: slow did not pass)", r.stdout)
                self.assertIn("CI local: 1 ok, 1 echecs, 0 sautes", r.stdout)

    def test_the_directory_of_the_worktrees_goes_with_the_run(self):
        r, tmp = self.run_ci()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(sorted(p.name for p in tmp.iterdir()), [], r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
