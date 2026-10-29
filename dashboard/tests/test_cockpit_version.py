"""The cockpit says which release it is, and which release the services run.

VERSION was a constant, "1.1.2", from v1.7.2 on: the badge, the Server header and the
User-Agent of the update check all named a release two years of changes old (found in
review, 2026-09-24). It is read from CHANGELOG.md's first heading now: the code on disk.

The release the services run is another fact (2026-10-17): a git pull moves the CHANGELOG
with no ./install.sh at all, and the cockpit then said "up to date" for services that were
not. It is the one the last run that ended well recorded in installed.json; a box that no
run of v1.36.0 or later recorded runs a release before v1.36.0."""
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
DASH = HERE.parents[1]

# The environment this module sets for the cockpit it loads is handed back when it ends,
# so the next module in the same process starts from what this one found.
ENV_BEFORE = {}


def setUpModule():
    ENV_BEFORE.update(os.environ)


def tearDownModule():
    for name in set(os.environ) - set(ENV_BEFORE):
        del os.environ[name]
    os.environ.update(ENV_BEFORE)


class TheVersion(unittest.TestCase):
    def test_it_is_the_changelogs_newest_release(self):
        tmp = Path(tempfile.mkdtemp(prefix="cockpit-ver-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "api-key").write_text("k\n")
        os.environ.update(COCKPIT_DRY_RUN="1", COCKPIT_CONFIG_DIR=str(tmp), COCKPIT_REPO_DIR=str(DASH.parent),
                          COCKPIT_PORT="0", COCKPIT_AGENT_PORT="0", COCKPIT_AUTOHEAL="0")
        sys.path.insert(0, str(DASH))
        spec = importlib.util.spec_from_file_location("cockpit_version", DASH / "cockpit.py")
        cp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cp)
        first = re.search(r"^## v(\d+\.\d+(?:\.\d+)?)\b", (DASH.parent / "CHANGELOG.md").read_text(), re.M).group(1)
        self.assertEqual(cp.VERSION, first)
        self.assertEqual(cp.Handler.server_version, "SparkCockpit/" + first)


HEAD = "f" * 40


class TheInstalledRelease(unittest.TestCase):
    """A clone at v1.36.1 (its CHANGELOG), whose services the record says are what it says."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cockpit-installed-"))
        (cls.tmp / "api-key").write_text("k\n")
        os.environ.update(COCKPIT_DRY_RUN="1", COCKPIT_CONFIG_DIR=str(cls.tmp), COCKPIT_REPO_DIR=str(DASH.parent),
                          COCKPIT_PORT="0", COCKPIT_AGENT_PORT="0", COCKPIT_AUTOHEAL="0")
        sys.path.insert(0, str(DASH))
        spec = importlib.util.spec_from_file_location("cockpit_installed", DASH / "cockpit.py")
        cls.cp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cp)
        cls.cp._release = lambda: "1.36.1"
        cls.cp.run = lambda argv, **k: HEAD + "\n" if "rev-parse" in argv else ""
        cls.latest = None
        cls.cp._get_json = lambda url, timeout=5.0: {"tag_name": cls.latest}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, True)

    def setUp(self):
        self.cp._RELEASE.update(latest=None, ts=0.0, fails=0, answered=False)
        self.cp.UPDATE_CHECK = True
        (self.tmp / "installed.json").unlink(missing_ok=True)

    def record(self, release="v1.36.0", commit="e" * 40, path="full", raw=None):
        (self.tmp / "installed.json").write_text(raw if raw is not None else json.dumps(
            {"schema": 1, "release": release, "commit": commit, "path": path, "choices": {}}))

    def update(self, latest=None):
        """collect_update with GitHub answering `latest`, asked afresh (the check caches its
        answer for six hours)."""
        type(self).latest = latest
        self.cp._RELEASE.update(latest=None, ts=0.0, fails=0, answered=False)
        return self.cp.collect_update()

    def test_it_is_the_recorded_release_not_the_clones(self):
        self.record()
        out = self.update()
        self.assertEqual((out["installed"], out["clone"]), ("v1.36.0", "v1.36.1"))
        self.assertEqual((out["behind_clone"], out["behind_clone_why"]), (True, "commit"))

    def test_a_box_no_run_recorded_runs_a_release_before_the_first_that_records(self):
        out = self.update()
        self.assertEqual(out["installed"], "before v1.36.0", "it said the clone's v1.36.1 before")
        self.assertEqual((out["behind_clone"], out["behind_clone_why"]), (True, "no-record"))
        self.record(raw='{"schema": 1, "rel')                 # cut short: no record either
        self.assertEqual(self.update()["installed"], "before v1.36.0")

    def test_a_run_of_this_very_clone_is_not_behind_it(self):
        self.record(release="v1.36.1", commit=HEAD)
        out = self.update()
        self.assertFalse(out["behind_clone"])
        self.assertNotIn("behind_clone_why", out)

    def test_a_run_stopped_at_no_start_is_behind_its_clone(self):
        self.record(release="v1.36.1", commit=HEAD, path="no-start")
        self.assertEqual(self.update()["behind_clone_why"], "no-start")

    def test_a_published_release_is_compared_to_what_the_services_run(self):
        self.record()
        self.assertTrue(self.update("v1.36.1")["behind"])
        self.record(release="v1.36.1", commit=HEAD)
        self.assertFalse(self.update("v1.36.1")["behind"])
        (self.tmp / "installed.json").unlink()
        self.assertTrue(self.update("v1.36.0")["behind"], "anything before v1.36.0 is older than it")
        self.assertFalse(self.update("v1.35.0")["behind"], "before v1.36.0 may be v1.35.0: no claim")

    def test_the_same_release_on_both_sides_says_the_commits_between(self):
        # "v1.40.4" on both sides said nothing of what moved (Hasan's capture, 2026-10-29): the banner
        # names the two commits and the clone's commits past the recorded one, newest first
        self.record(release="v1.36.1")
        argvs, real = [], self.cp.run
        def git(argv, **k):
            argvs.append(argv)
            if "rev-parse" in argv:
                return HEAD + "\n"
            if "rev-list" in argv:
                return "2\n"
            if "log" in argv:
                return "fffffff tests: the second\neeeeee1 docs: the first\n"
            return ""
        self.cp.run = git
        self.addCleanup(setattr, self.cp, "run", real)
        out = self.update()
        self.assertEqual((out["installed"], out["clone"], out["behind_clone_why"]), ("v1.36.1", "v1.36.1", "commit"))
        self.assertEqual((out["installed_commit"], out["clone_commit"]), ("eeeeeee", "fffffff"))
        self.assertEqual(out["commits"], ["fffffff tests: the second", "eeeeee1 docs: the first"])
        self.assertEqual(out["commits_count"], 2)
        self.assertIn(["git", "-C", str(self.cp.REPO_DIR), "log", "-n", "5", "--format=%h %s", "e" * 40 + ".." + HEAD], argvs)

    def test_a_recorded_commit_git_does_not_know_still_names_both(self):
        self.record(release="v1.36.1")
        out = self.update()   # the class's git answers nothing past rev-parse: a commit this clone lost
        self.assertEqual((out["installed_commit"], out["clone_commit"]), ("eeeeeee", "fffffff"))
        self.assertNotIn("commits", out)

    def test_a_recorded_commit_that_is_no_hash_never_reaches_git(self):
        argvs, real = [], self.cp.run
        self.cp.run = lambda argv, **k: argvs.append(argv) or (HEAD + "\n" if "rev-parse" in argv else "")
        self.addCleanup(setattr, self.cp, "run", real)
        self.record(release="v1.36.1", commit="--output=/tmp/x")
        out = self.update()
        self.assertEqual(out["behind_clone_why"], "commit")
        self.assertNotIn("installed_commit", out)
        self.assertEqual([a for a in argvs if any("--output" in x for x in a)], [])

    def test_the_library_line_says_the_same(self):
        self.record()
        rel = self.cp.upstream_snapshot(max_age=0.0)["release"]
        self.assertEqual(rel["local"], "v1.36.0")
        (self.tmp / "installed.json").unlink()
        type(self).latest = "v1.36.0"
        rel = self.cp.upstream_snapshot(max_age=0.0)["release"]
        self.assertEqual((rel["local"], rel["behind"], rel["ahead"]), ("before v1.36.0", True, False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
