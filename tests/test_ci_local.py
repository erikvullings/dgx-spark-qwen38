#!/usr/bin/env python3
"""ci-local.sh runs the workflow's jobs the way GitHub does.

A job that needs others runs once they have ended, and only when every one of them passed:
checks, the job the run's status goes by, ran beside the others and said "Every job passed"
before any of them had ended (2026-10-28), and said it as well after one of them failed. The
directory the jobs' worktrees go in goes with the run. A fragment runs in the checkout as it
is, a full run hands its steps the checkout's test venv, and a stop stops every job (code
review, 2026-10-28). Each case runs ci-local.sh on a repository of its own, made here."""
import os
import pathlib
import shutil
import signal
import subprocess
import tempfile
import time
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
    def make_repo(self, workflow, files=()):
        """A repository with this workflow and these files, all committed, and an empty TMPDIR."""
        root = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        repo, tmp = root / "repo", root / "tmp"
        (repo / ".github" / "workflows").mkdir(parents=True)
        tmp.mkdir()
        (repo / ".github" / "workflows" / "ci.yml").write_text(workflow)
        for rel, text in dict(files).items():
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (repo / rel).write_text(text)
            if rel.endswith(".sh"):
                os.chmod(repo / rel, 0o755)
        self.git = ["git", "-C", str(repo), "-c", "user.name=ci", "-c", "user.email=ci@example.invalid",
                    "-c", "commit.gpgsign=false"]
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(self.git + ["add", "."], check=True)
        subprocess.run(self.git + ["commit", "-qm", "ci"], check=True)
        return repo, tmp

    def env(self, tmp, serial=False, **extra):
        env = {k: v for k, v in os.environ.items() if k not in ("CI_LOCAL_SERIAL", "QWEN38_TEST_PYTHON")}
        env.update(TMPDIR=str(tmp), **extra)
        if serial:
            env["CI_LOCAL_SERIAL"] = "1"
        return env

    def run_ci(self, slow="true", serial=False):
        repo, tmp = self.make_repo(WORKFLOW.format(slow=slow))
        r = subprocess.run(["bash", str(SCRIPT)], cwd=repo, env=self.env(tmp, serial), capture_output=True,
                           text=True, timeout=120)
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
                # and said again next to the count, where a reader of the run looks
                self.assertIn("not run, a job they need did not pass: Every job passed\n\nCI local: 1 ok, "
                              "1 echecs, 0 sautes", r.stdout)

    def test_the_directory_of_the_worktrees_goes_with_the_run(self):
        r, tmp = self.run_ci()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(sorted(p.name for p in tmp.iterdir()), [], r.stdout)

    def test_a_fragment_runs_in_the_checkout_as_it_is(self):
        # what one runs while editing: the worktrees of HEAD made it refuse every tree with a change
        repo, tmp = self.make_repo(WORKFLOW.format(slow="true"), {"notes.txt": "one\n"})
        (repo / "notes.txt").write_text("two\n")
        r = subprocess.run(["bash", str(SCRIPT), "The quick one"], cwd=repo, env=self.env(tmp),
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("  ok   The quick one", r.stdout)
        self.assertNotIn("worktree", r.stdout)

    def test_a_full_run_hands_its_steps_the_checkouts_test_venv(self):
        # .venv-test is not in git, so not in a worktree of HEAD, where tests/testpy.sh looks for it
        workflow = ("name: ci\non: push\njobs:\n  venv:\n    runs-on: ubuntu-24.04-arm\n    steps:\n"
                    "      - name: The checkout's test venv\n        run: |\n"
                    "          p=$(./tests/testpy.sh fakepkg)\n"
                    "          case \"$p\" in */.venv-test/bin/python) ;; *) echo \"found $p\"; exit 1 ;; esac\n")
        repo, tmp = self.make_repo(workflow, {".gitignore": ".venv-test/\n",
                                              "tests/testpy.sh": (REPO / "tests" / "testpy.sh").read_text()})
        venv_python = repo / ".venv-test" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text("#!/bin/sh\nexit 0\n")    # it "has" any package it is asked for
        os.chmod(venv_python, 0o755)
        r = subprocess.run(["bash", str(SCRIPT)], cwd=repo, env=self.env(tmp), capture_output=True,
                           text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("ok   The checkout's test venv", r.stdout)

    def test_a_stop_stops_every_job(self):
        # a stop through the process group killed the steps running, and each job went on to its next
        workflow = ("name: ci\non: push\njobs:\n"
                    "  long:\n    runs-on: ubuntu-24.04-arm\n    steps:\n"
                    "      - name: The long one\n        run: sleep 30\n"
                    "      - name: After the long one\n        run: sleep 2 && touch \"$MARK\"\n"
                    "  other:\n    runs-on: ubuntu-24.04-arm\n    steps:\n"
                    "      - name: The other one\n        run: sleep 30\n")
        repo, tmp = self.make_repo(workflow)
        mark = tmp.parent / "after-the-stop"
        p = subprocess.Popen(["bash", str(SCRIPT)], cwd=repo, env=self.env(tmp, MARK=str(mark)),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)

        def kill_group():
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.addCleanup(kill_group)
        self.assertIn("2 jobs side by side", p.stdout.readline())
        time.sleep(1)
        os.killpg(p.pid, signal.SIGTERM)
        t0 = time.monotonic()
        out = p.communicate(timeout=60)[0]      # to the end of python's output, bash being gone
        time.sleep(3)                           # longer than the step after the long one takes
        self.assertFalse(mark.exists(), "a step started after the stop:\n" + out)
        self.assertLess(time.monotonic() - t0, 30, out)
        self.assertIn("[long] --   The long one (stopped)", out)
        self.assertIn("[other] --   The other one (stopped)", out)
        self.assertEqual(subprocess.run(self.git + ["worktree", "list"], capture_output=True,
                                        text=True).stdout.count("\n"), 1, "a worktree was left")
        self.assertEqual(sorted(p_.name for p_ in tmp.iterdir()), [], out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
