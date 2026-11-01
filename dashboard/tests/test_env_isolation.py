#!/usr/bin/env python3
"""Every test module hands the next one the environment it found.

unittest runs all of these files in one process, one after the other, so whatever a module
leaves in os.environ is what the modules after it start from. The cockpit reads its
configuration from the environment when it is loaded: test_cockpit_jobs.py set
COCKPIT_DRY_RUN=0, pointed HOME and COCKPIT_CONFIG_DIR at directories it then deleted, and
left all of it behind, so every module after it ran with a HOME that did not exist and
passed or failed on that rather than on the code (found in review, 2026-09-24).

One test of each class of each module is run, which runs all of its fixtures (module, then
class), and the environment is compared before and after, in a process of its own. One test of
each module was run until 2026-10-24, and a module's other classes never ran their fixtures:
test_cockpit_logs_masked.py's second class left six COCKPIT_ variables behind since 2026-09-28,
seen by tests/run_suite.py, which compares each test file's environment at its start and its exit.
"""
import os
import pathlib
import subprocess
import sys
import unittest

TESTS = pathlib.Path(__file__).resolve().parent
DASH = TESTS.parent

PROBE = """
import os, pathlib, sys, unittest
sys.path.insert(0, os.getcwd())
for path in sorted(pathlib.Path("tests").glob("test_*.py")):
    if path.stem == "test_env_isolation":
        continue
    name = "tests." + path.stem
    suite = unittest.defaultTestLoader.loadTestsFromName(name)
    stack, firsts = [suite], {}
    while stack:
        item = stack.pop(0)
        if isinstance(item, unittest.TestSuite):
            stack[:0] = list(item)
        else:
            firsts.setdefault(type(item), item)     # one test of each class: its fixtures run
    if not firsts:
        continue
    before = dict(os.environ)
    unittest.TestSuite(list(firsts.values())).run(unittest.TestResult())
    after = dict(os.environ)
    for var in sorted(set(before) | set(after)):
        if before.get(var) != after.get(var):
            print("LEAK " + name + " " + var + ": " + repr(before.get(var)) + " -> " + repr(after.get(var)))
print("DONE")
"""


class EveryModuleRestoresTheEnvironment(unittest.TestCase):
    def test_no_module_leaves_the_environment_changed(self):
        proc = subprocess.run([sys.executable, "-c", PROBE], cwd=DASH, capture_output=True,
                              text=True, timeout=600, env=dict(os.environ))
        self.assertIn("DONE", proc.stdout,
                      f"the probe did not finish:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        leaks = [line for line in proc.stdout.splitlines() if line.startswith("LEAK ")]
        self.assertEqual(leaks, [], "these modules changed the environment the next one gets")


if __name__ == "__main__":
    unittest.main(verbosity=2)
