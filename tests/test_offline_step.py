#!/usr/bin/env python3
"""The offline step runs the test files up to four at a time, and the ones that free a port alone.

One after the other, the files of dashboard/tests and tests/ took 20 minutes of the offline
step on the reference box (2026-10-18). They now run up to four at a time, the longest first,
except the files that take a port number out of a socket they close, to start a server on it
later or to have a port where nothing listens: a file running beside one of them can be handed
that port meanwhile, so they run one at a time once the others are over.

The step's own script runs here, as ci.yml writes it, over a tree of files whose tests record
when they ran: every file still runs and has its count checked, every failure is named with
the end of its output, and nothing runs beside a file that frees a port."""
import os
import pathlib
import subprocess
import tempfile
import textwrap
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
# spelled in parts: the CI counts the first string to know how many tests a file declares,
# and takes a file holding the second for one that frees a port
TEST_DEF = "def " + "test_"
FREES_A_PORT = "getsockname" + "()[1]"
LONGEST = ("tests/test_suite_isolation.py", "tests/test_text_lane.py", "tests/test_check_pins.py")

FAKE = """import os, sys, time, unittest
# {mark}
start = time.time()
class Fake(unittest.TestCase):
{tests}
{main}"""
MAIN = """if __name__ == "__main__":
    ok = unittest.main(exit=False).result.wasSuccessful()
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write(f"{{start}} {{time.time()}} {path}\\n")
    sys.exit(0 if ok else 1)
"""


def suite_script():
    """The script the offline step writes to suite.sh, as ci.yml spells it."""
    ci = (REPO / ".github/workflows/ci.yml").read_text()
    body = ci.split("cat > \"$probe/suite.sh\" <<'SH'\n", 1)[1]
    lines = []
    for ln in body.splitlines():
        if ln.strip() == "SH":
            break
        lines.append(ln)
    return textwrap.dedent("\n".join(lines)) + "\n"


class TheOfflineStep(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name) / "repo"
        self.log = pathlib.Path(self.tmp.name) / "ran.log"
        self.files = {}
        for path in LONGEST:
            self.fake(path)
        for path in ("dashboard/tests/test_one.py", "dashboard/tests/test_two.py", "tests/test_a.py",
                     "tests/test_b.py", "tests/test_c.py", "tests/test_d.py"):
            self.fake(path)
        for path in ("tests/test_port_e.py", "tests/test_port_f.py"):
            self.fake(path, frees_a_port=True)

    def fake(self, path, frees_a_port=False, fails=None, main=True):
        """A test file of one test that sleeps, then records when it ran."""
        test = f"    {TEST_DEF}it(self):\n        time.sleep(0.8)\n"
        if fails:
            test += f"        self.fail({fails!r})\n"
        self.files[path] = FAKE.format(mark=FREES_A_PORT if frees_a_port else "", tests=test,
                                       main=MAIN.format(path=path) if main else "")

    def run_step(self):
        for path, text in self.files.items():
            (self.root / path).parent.mkdir(parents=True, exist_ok=True)
            (self.root / path).write_text(text)
        script = pathlib.Path(self.tmp.name) / "suite.sh"
        script.write_text(suite_script())
        r = subprocess.run(["bash", "-e", str(script)], cwd=self.root, capture_output=True, text=True,
                           timeout=120, env={**os.environ, "FAKE_LOG": str(self.log)})
        ran = {}
        if self.log.exists():
            for ln in self.log.read_text().splitlines():
                start, end, path = ln.split()
                ran[path] = (float(start), float(end))
        return r.returncode, r.stdout + r.stderr, ran

    def test_every_file_runs_and_the_ones_that_free_a_port_run_alone(self):
        rc, out, ran = self.run_step()
        self.assertEqual(rc, 0, out)
        self.assertEqual(set(ran), set(self.files), out)
        self.assertIn("ran its declared tests on its own: 9 of them", out)
        self.assertIn("then 2 one at a time", out)
        self.assertRegex(out, r"the longest: \S+ \d+ s, \S+ \d+ s, \S+ \d+ s\n")
        alone = {p: ran[p] for p in ("tests/test_port_e.py", "tests/test_port_f.py")}
        for path, (start, end) in alone.items():
            for other, (s, e) in ran.items():
                if other != path:
                    self.assertTrue(e <= start or s >= end, f"{other} ran beside {path}")
        together = [span for p, span in ran.items() if p not in alone]
        most = max(sum(s <= t < e for s, e in together) for t, _ in together)
        self.assertLessEqual(most, 4)
        if (os.cpu_count() or 1) >= 2:
            self.assertGreaterEqual(most, 2, "the files ran one after the other")
        first_four = sorted(together)[:4]
        for path in LONGEST:
            self.assertIn(ran[path], first_four, f"{path} did not start among the first")

    def test_every_failure_is_named_with_the_end_of_its_output(self):
        self.fake("tests/test_b.py", fails="b went wrong")
        self.fake("tests/test_port_f.py", frees_a_port=True, fails="f went wrong")
        rc, out, ran = self.run_step()
        self.assertEqual(rc, 1, out)
        self.assertEqual(set(ran), set(self.files), "a failure kept the other files from running")
        for path, why in (("tests/test_b.py", "b went wrong"), ("tests/test_port_f.py", "f went wrong")):
            said = out.split(f"{path} failed, the end of its output:", 1)
            self.assertEqual(len(said), 2, out)
            self.assertIn(why, said[1], out)

    def test_a_file_that_runs_fewer_tests_than_it_declares_fails(self):
        self.fake("tests/test_c.py", main=False)    # exits 0 having run nothing
        rc, out, _ = self.run_step()
        self.assertEqual(rc, 1, out)
        self.assertIn("tests/test_c.py ran 0 of its 1 declared tests when invoked directly", out)

    def test_a_longest_file_that_is_gone_is_named(self):
        del self.files["tests/test_check_pins.py"]
        rc, out, ran = self.run_step()
        self.assertEqual(rc, 1, out)
        self.assertIn("tests/test_check_pins.py, named as one of the longest, is gone", out)
        self.assertEqual(ran, {}, "the step ran files with a stale list")


if __name__ == "__main__":
    unittest.main()
