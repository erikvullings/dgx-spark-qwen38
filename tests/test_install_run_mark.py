#!/usr/bin/env python3
"""While ./install.sh runs, the cockpit knows it runs.

An update of 1 h 45 on an ASUS Ascent GX10 (2026-10-21, 144 GB of video to fetch) ran under the
orange banner that asked to run the installer: the clone had moved, the services not yet. The run
now keeps a file while it lasts (lanes.sh run_mark), which the cockpit believes while that very
process lives (dashboard/cockpit.py install_running); dashboard/tests/update-banner-check.mjs
reads what the page says then.
"""
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import installer_wall as wall  # noqa: E402
from test_fresh_machine import box_for_step_1, walled_with_memory  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
MARK_LINE = 'run_mark "$CONFIG_DIR"; trap forget_run_mark EXIT\n'
FIRST = re.search(r"^## (v\d+\.\d+(?:\.\d+)?)\b", (REPO / "CHANGELOG.md").read_text(), re.M).group(1)


def start_of(pid):
    """A process's start time as /proc/<pid>/stat has it (its 22nd field)."""
    stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
    return stat[stat.rindex(")") + 2:].split()[19]


def lanes(script, *args, repo=None):
    """`script` run by bash with lanes.sh sourced, under set -euo pipefail as the installers run it."""
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(wall._tmp("home-"))}
    if repo:
        env["REPO_DIR"] = str(repo)
    return subprocess.run(["bash", "-c", f'set -euo pipefail\n. "{REPO}/lanes.sh"\n{script}', "lanes", *args],
                          capture_output=True, text=True, env=env, timeout=60)


class TheRunMark(unittest.TestCase):
    """lanes.sh run_mark and forget_run_mark."""

    def setUp(self):
        self.d = wall._tmp("mark-")
        self.repo = wall._tmp("clone-")
        (self.repo / "CHANGELOG.md").write_text("# Changelog\n\n## v9.8.7 (2026-10-21): a release\n")

    def marks(self):
        return sorted(p.name for p in self.d.glob("install-running*"))

    def test_it_names_this_process_and_goes_with_it(self):
        r = lanes('echo "pid $$ start $(cut -d" " -f22 /proc/$$/stat)"\nrun_mark "$1"\ncat "$RUN_MARK"; echo\n'
                  'forget_run_mark\nls -A "$1"', str(self.d), repo=self.repo)
        self.assertEqual(r.returncode, 0, r.stderr)
        head, mark, rest = r.stdout.split("\n", 2)
        pid, start = re.fullmatch(r"pid (\d+) start (\d+)", head).groups()
        m = json.loads(mark)
        self.assertEqual(m["pid"], int(pid))
        self.assertEqual(str(m["start"]), start)
        self.assertEqual(m["to"], "v9.8.7")
        self.assertLessEqual(abs(m["started"] - time.time()), 60)
        self.assertEqual(rest.strip(), "", "forget_run_mark left a file behind")

    def test_a_place_it_cannot_write_leaves_the_run_going(self):
        blocker = self.d / "a-file"
        blocker.write_text("")
        r = lanes('run_mark "$1"\necho "went on, mark=[$RUN_MARK]"\nforget_run_mark\necho done',
                  str(blocker / "qwen38"), repo=self.repo)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("went on, mark=[]", r.stdout)
        self.assertIn("done", r.stdout)

    def test_a_clone_with_no_release_says_none(self):
        r = lanes('run_mark "$1"\ncat "$RUN_MARK"', str(self.d), repo=wall._tmp("bare-"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIsNone(json.loads(r.stdout)["to"])

    def test_it_removes_the_marks_of_runs_gone_and_keeps_a_live_one(self):
        me = os.getpid()
        live = self.d / f"install-running.{me}.json"
        live.write_text(json.dumps({"pid": me, "start": start_of(me), "started": 1, "to": "v1.0.0"}))
        # a pid used again: init's, which started at 0
        (self.d / "install-running.1.json").write_text(json.dumps({"pid": 1, "start": "-1", "started": 1}))
        (self.d / "install-running.4194999.json").write_text(json.dumps({"pid": 4194999, "start": "5"}))
        (self.d / "install-running.77.json").write_text("{half a file")
        r = lanes('run_mark "$1"\necho "$RUN_MARK"', str(self.d), repo=self.repo)
        self.assertEqual(r.returncode, 0, r.stderr)
        own = pathlib.Path(r.stdout.strip()).name
        self.assertEqual(self.marks(), sorted([live.name, own]))


class TheInstaller(unittest.TestCase):
    """install.sh, run as written up to a wall past its mark."""

    def test_the_run_keeps_its_mark_and_takes_it_away(self):
        box = box_for_step_1(docker_err="", docker_rc=0, member=True, in_session=True)
        copy = pathlib.Path(walled_with_memory(at=MARK_LINE, after=True))
        text = copy.read_text()
        # the wall shows the mark it finds, then exits as a die would
        copy.write_text(text.replace(f'echo "{wall.WALL}"', 'cat "$HOME"/.config/qwen38/install-running.*.json; echo; '
                                     f'echo "{wall.WALL}"', 1))
        home = wall._tmp("home-")
        env, record = wall.fenced_env(home=home, first=str(box.bin))
        r = subprocess.run(["bash", str(copy)], capture_output=True, text=True, env=env, cwd=wall.cwd(),
                           timeout=60, stdin=subprocess.DEVNULL, start_new_session=True)
        out = r.stdout + r.stderr
        self.assertIn(wall.WALL, out, out[-800:])
        mark = json.loads(re.search(r'^\{"pid".*\}$', r.stdout, re.M).group(0))
        self.assertEqual(mark["to"], FIRST)
        self.assertEqual(r.returncode, 97, out[-800:])
        self.assertEqual(list((home / ".config/qwen38").glob("install-running*")), [],
                         "the run ended and its mark stayed")

    def test_every_exit_trap_after_the_mark_takes_it_away(self):
        """bash keeps one EXIT trap: a later `trap ... EXIT` that forgot forget_run_mark would
        leave the mark of every run that got that far."""
        text = (REPO / "install.sh").read_text()
        self.assertEqual(text.count(MARK_LINE), 1)
        after = text[text.index(MARK_LINE) + len(MARK_LINE):]
        traps = [ln for ln in after.splitlines() if re.search(r"^\s*trap .* EXIT\s*$", ln)]
        self.assertTrue(traps, "no EXIT trap after the mark: the test reads the wrong lines")
        for ln in traps:
            self.assertIn("forget_run_mark", ln)

    def test_the_mark_comes_after_every_refusal_and_the_docker_step(self):
        """--help, --plan, a refusal or a run started again under the docker group must leave no
        mark of a process that is not the one that runs the steps."""
        text = (REPO / "install.sh").read_text()
        at = text.index(MARK_LINE)
        for line in ('[ -z "$DOCKER_SAID" ] || exec 1>&3 3>&-\n', 'step "1/10 Preflight checks"\n',
                     '"--plan: nothing was changed. The same command without --plan applies it."'):
            self.assertLess(text.index(line), at, line)


class TheCockpit(unittest.TestCase):
    """dashboard/cockpit.py install_running, and the update state that carries it."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = pathlib.Path(tempfile.mkdtemp(prefix="cockpit-run-"))
        (cls.cfg / "api-key").write_text("k\n")
        cls.env = dict(os.environ)
        os.environ.update(COCKPIT_DRY_RUN="1", COCKPIT_CONFIG_DIR=str(cls.cfg), COCKPIT_REPO_DIR=str(REPO),
                          COCKPIT_PORT="0", COCKPIT_AGENT_PORT="0", COCKPIT_AUTOHEAL="0", COCKPIT_UPDATE_CHECK="0")
        sys.path.insert(0, str(REPO / "dashboard"))
        spec = importlib.util.spec_from_file_location("cockpit_run_mark", REPO / "dashboard" / "cockpit.py")
        cls.cp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cp)

    @classmethod
    def tearDownClass(cls):
        os.environ.clear()
        os.environ.update(cls.env)
        shutil.rmtree(cls.cfg, True)

    def tearDown(self):
        for p in self.cfg.glob("install-running*"):
            p.unlink()

    def mark(self, pid, start, **more):
        (self.cfg / f"install-running.{pid}.json").write_text(
            json.dumps({"pid": pid, "start": start, "started": 1761040000, "to": "v9.8.7", **more}))

    def test_no_mark_no_run(self):
        self.assertIsNone(self.cp.install_running())

    def test_a_mark_of_a_live_run(self):
        self.mark(os.getpid(), start_of(os.getpid()))
        self.assertEqual(self.cp.install_running(), {"started": 1761040000, "to": "v9.8.7"})

    def test_a_mark_whose_pid_was_used_again(self):
        self.mark(os.getpid(), str(int(start_of(os.getpid())) + 1))
        self.assertIsNone(self.cp.install_running())

    def test_a_mark_of_a_process_gone_or_half_written(self):
        self.mark(4194999, "5")
        (self.cfg / "install-running.12.json").write_text("{half")
        (self.cfg / "install-running.13.json").write_text(json.dumps({"pid": "x", "start": 1}))
        (self.cfg / "install-running.14.json").write_text(json.dumps([1, 2]))
        self.assertIsNone(self.cp.install_running())

    def test_the_update_state_carries_it_and_only_then(self):
        self.assertNotIn("install_running", self.cp.collect_update())
        self.mark(os.getpid(), start_of(os.getpid()))
        self.assertEqual(self.cp.collect_update()["install_running"]["to"], "v9.8.7")


if __name__ == "__main__":
    unittest.main(verbosity=2)
