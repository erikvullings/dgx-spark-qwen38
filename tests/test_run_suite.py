#!/usr/bin/env python3
"""tests/run_suite.py catches each defect the suite's gates exist for, from one run of each file.

It replaces running the suite about four times per CI run (the offline step, then
tests/test_suite_isolation.py re-running every module to compare its environment, in two steps):
each gate it takes over is proven here on a planted defect, in a throwaway repository."""
import importlib.util
import pathlib
import shutil
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_suite", REPO / "tests" / "run_suite.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)

MAIN = '\n\nif __name__ == "__main__":\n    unittest.main()\n'
# the fake files' tests are spelled TEST_ here and test_ on disk: the CI counts that string in this
# file to know how many tests it declares
TEST_DEF = "def " + "test_"
FILES = {
    "tests/test_ok.py": "import unittest\n\nclass T(unittest.TestCase):\n    def TEST_a(self):\n        pass\n\n"
                        "    def TEST_b(self):\n        pass\n" + MAIN,
    "tests/test_fails.py": "import unittest\n\nclass T(unittest.TestCase):\n    def TEST_a(self):\n"
                           "        self.assertEqual(1, 2)\n" + MAIN,
    # declares a test and runs none when invoked directly
    "tests/test_no_main.py": "import unittest\n\nclass T(unittest.TestCase):\n    def TEST_a(self):\n        pass\n",
    "tests/test_leaks.py": "import os, unittest\n\nclass T(unittest.TestCase):\n    def TEST_a(self):\n"
                           "        os.environ['LEAKED_BY_A_TEST'] = '1'\n" + MAIN,
    "tests/test_restores.py": "import os, unittest\n\nclass T(unittest.TestCase):\n    def TEST_a(self):\n"
                              "        os.environ['SET_THEN_GONE'] = '1'\n        del os.environ['SET_THEN_GONE']\n" + MAIN,
    "tests/test_contacts.py": "import socket, unittest\n\nclass T(unittest.TestCase):\n    def TEST_a(self):\n"
                              "        s = socket.socket()\n        try:\n            s.connect(('127.0.0.1', 1))\n"
                              "        except OSError:\n            pass\n        finally:\n            s.close()\n" + MAIN,
    "tests/test_takes_a_port.py": "import socket, unittest\n\nclass T(unittest.TestCase):\n    def TEST_a(self):\n"
                                  "        s = socket.socket(); s.bind(('127.0.0.1', 0)); s.getsockname()[1]; s.close()\n" + MAIN,
    # loads a module as another script, as the proxy's tests do, and leaves the directory it started in
    "tests/test_argv_and_cwd.py": "import os, sys, unittest\n\nclass T(unittest.TestCase):\n    def TEST_a(self):\n"
                                  "        sys.argv = ['keepalive-proxy.py']\n        os.chdir('/')\n" + MAIN,
    "dashboard/tests/test_cockpit_side.py": "import unittest\n\nclass T(unittest.TestCase):\n    def TEST_a(self):\n"
                                            "        pass\n" + MAIN,
    # says where it is on stderr, as unittest does, then hangs
    "tests/test_hangs.py": "import sys, time, unittest\n\nclass T(unittest.TestCase):\n    def TEST_a(self):\n"
                           "        sys.stderr.write('hung in TEST_a\\n'); sys.stderr.flush(); time.sleep(60)\n" + MAIN,
}


class OnPlantedDefects(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = pathlib.Path(tempfile.mkdtemp(prefix="run-suite-repo-"))
        for rel, text in FILES.items():
            (cls.repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (cls.repo / rel).write_text(text.replace("def TEST_", TEST_DEF))
        cls.probe = pathlib.Path(tempfile.mkdtemp(prefix="run-suite-probe-"))
        (cls.probe / "sitecustomize.py").write_text(rs.SITECUSTOMIZE)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.repo, ignore_errors=True)
        shutil.rmtree(cls.probe, ignore_errors=True)

    def verdict(self, rel, **env):
        old = {k: rs.os.environ.get(k) for k in env}
        rs.os.environ.update(env)
        try:
            return rs.run_one(rel, self.probe, repo=self.repo, timeout=120)
        finally:
            for k, v in old.items():
                if v is None:
                    rs.os.environ.pop(k, None)
                else:
                    rs.os.environ[k] = v

    def test_a_clean_file_has_no_problem(self):
        v = self.verdict("tests/test_ok.py")
        self.assertEqual((v["rc"], v["problems"]), (0, []))

    def test_a_failing_test_is_named_with_its_output(self):
        v = self.verdict("tests/test_fails.py")
        self.assertNotEqual(v["rc"], 0)
        self.assertIn("AssertionError", v["problems"][0])

    def test_a_file_that_runs_none_of_its_tests_is_caught(self):
        v = self.verdict("tests/test_no_main.py")
        self.assertEqual(v["problems"], ["ran 0 of its 1 declared tests when invoked directly"])

    def test_a_variable_left_changed_is_named(self):
        v = self.verdict("tests/test_leaks.py")
        self.assertEqual(v["problems"], ["left these environment variables changed: LEAKED_BY_A_TEST"])

    def test_a_variable_set_and_removed_is_not(self):
        self.assertEqual(self.verdict("tests/test_restores.py")["problems"], [])

    def test_a_file_that_replaces_its_argv_and_directory_still_reports(self):
        # the proxy's tests set sys.argv to keepalive-proxy.py: their processes went unreported (2026-10-24)
        self.assertEqual(self.verdict("tests/test_argv_and_cwd.py")["problems"], [])

    def test_a_file_that_hangs_is_named_with_what_it_said(self):
        # the verdict kept the bytes of its stdout and none of its stderr, where unittest writes
        v = rs.run_one("tests/test_hangs.py", self.probe, repo=self.repo, timeout=2)
        self.assertIsNone(v["rc"])
        self.assertIn("timed out after 2 s", v["problems"][0])
        self.assertIn("hung in TEST_a", v["problems"][0])
        self.assertNotIn("b'", v["problems"][0])

    def test_a_cockpit_file_runs_from_the_dashboard(self):
        self.assertEqual(self.verdict("dashboard/tests/test_cockpit_side.py")["problems"], [])

    def test_a_contact_with_a_serving_port_is_recorded_with_its_line(self):
        log = self.probe / "contacts"
        log.unlink(missing_ok=True)
        v = self.verdict("tests/test_contacts.py", QWEN38_SERVING_PORTS="1", QWEN38_CONTACT_LOG=str(log))
        self.assertEqual(v["problems"], [])
        self.assertIn("127.0.0.1:1 from test_contacts.py:", log.read_text())

    def test_a_file_that_takes_a_port_runs_alone(self):
        together, alone = rs.plan(rs.files(self.repo), repo=self.repo, first=[])
        self.assertEqual(alone, ["tests/test_takes_a_port.py"])
        self.assertNotIn("tests/test_takes_a_port.py", together)
        self.assertEqual(len(together) + len(alone), len(FILES))

    def test_a_longest_file_that_is_gone_stops_the_run(self):
        with self.assertRaisesRegex(SystemExit, "named as the longest, are gone"):
            rs.plan(rs.files(self.repo), repo=self.repo, first=["tests/test_gone.py"])

    def test_the_real_plan_names_files_that_exist(self):
        together, alone = rs.plan(rs.files())
        self.assertEqual(sorted(together + alone), rs.files())


if __name__ == "__main__":
    unittest.main()
