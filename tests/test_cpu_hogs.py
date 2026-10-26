#!/usr/bin/env python3
"""cpu-hogs.py names the processes that hold a CPU core outside the serving engine.

On another DGX Spark, in October 2026, three command-line calls that old sessions had left
behind each held a core for four to six days, seen only in the load average: on a GB10 the CPU
and the GPU share one power budget, so such a process lowers what the engine measures. bench.sh
names them before it measures."""
import importlib.util
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cpu_hogs", REPO / "cpu-hogs.py")
hogs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hogs)


class FromTwoReadings(unittest.TestCase):
    def test_a_process_at_the_floor_is_named_and_one_below_is_not(self):
        before = {1: (0, "spin"), 2: (0, "idle"), 3: (0, "half")}
        after = {1: (300, "spin"), 2: (10, "idle"), 3: (150, "half")}    # ticks over 3 s at 100 Hz
        self.assertEqual([p for p, _, _ in hogs.busy(before, after, 3.0, 100, 0.5)], [1, 3])

    def test_the_busiest_comes_first_with_its_share_of_a_core(self):
        before = {1: (0, "a"), 2: (0, "b")}
        after = {1: (120, "a"), 2: (600, "b")}                           # 2 s: 0.6 and 3 cores
        found = hogs.busy(before, after, 2.0, 100, 0.5)
        self.assertEqual([(p, round(s, 2)) for p, s, _ in found], [(2, 3.0), (1, 0.6)])

    def test_a_process_seen_in_one_reading_only_is_left_out(self):
        before = {1: (0, "gone")}
        after = {2: (900, "new")}
        self.assertEqual(hogs.busy(before, after, 3.0, 100, 0.5), [])

    def test_what_the_caller_skips_is_not_named(self):
        before = {1: (0, "engine"), 2: (0, "ghost")}
        after = {1: (300, "engine"), 2: (300, "ghost")}
        found = hogs.busy(before, after, 3.0, 100, 0.5, skip=lambda pid: pid == 1)
        self.assertEqual([p for p, _, _ in found], [2])


class TheProcFiles(unittest.TestCase):
    def test_a_name_with_spaces_and_parentheses_is_read_whole(self):
        d = pathlib.Path(tempfile.mkdtemp(prefix="hogs-proc-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(d, True))
        (d / "42").mkdir()
        # pid (comm) state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt utime stime ...
        (d / "42" / "stat").write_text("42 (a (b) c) R 1 42 42 0 -1 0 0 0 0 0 700 50 0 0 20 0 1 0\n")
        (d / "self").mkdir()                     # not a pid: skipped
        self.assertEqual(hogs.ticks(str(d)), {42: (750, "a (b) c")})


def stop(p):
    p.kill()
    p.wait()


class OnThisMachine(unittest.TestCase):
    def test_a_spinning_process_is_named_and_a_sleeping_one_is_not(self):
        spin = subprocess.Popen([sys.executable, "-c", "while True: pass"])
        nap = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(stop, nap)
        self.addCleanup(stop, spin)
        time.sleep(0.3)
        found = {pid: share for pid, share, _, _ in hogs.hogs(window=1.5, floor=0.5)}
        self.assertIn(spin.pid, found)
        self.assertGreaterEqual(found[spin.pid], 0.5)
        self.assertNotIn(nap.pid, found)

    def test_the_command_line_prints_one_tab_separated_line_per_process(self):
        spin = subprocess.Popen([sys.executable, "-c", "while True: pass"])
        self.addCleanup(stop, spin)
        time.sleep(0.3)
        r = subprocess.run([sys.executable, str(REPO / "cpu-hogs.py"), "--window", "1.5"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        rows = [line.split("\t") for line in r.stdout.splitlines()]
        self.assertTrue(all(len(row) == 4 for row in rows), rows)
        mine = [row for row in rows if row[0] == str(spin.pid)]
        self.assertEqual(len(mine), 1, rows)
        self.assertGreaterEqual(float(mine[0][1]), 0.5)
        self.assertIn("while True", mine[0][3])

    def test_a_tab_or_an_escape_in_a_command_line_never_splits_or_reaches_the_terminal(self):
        # a busy process whose argv held a tab vanished from bench.sh's report, which keeps lines of four
        # fields, and an escape sequence in an argv reached the terminal (verification, 2026-10-26)
        spin = subprocess.Popen([sys.executable, "-c", "while True: pass", "a\tb", "\x1b]0;title\x07"])
        self.addCleanup(stop, spin)
        time.sleep(0.3)
        r = subprocess.run([sys.executable, str(REPO / "cpu-hogs.py"), "--window", "1.5"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        mine = [line.split("\t") for line in r.stdout.splitlines() if line.startswith(f"{spin.pid}\t")]
        self.assertEqual(len(mine), 1, r.stdout)
        self.assertEqual(len(mine[0]), 4, mine)
        self.assertNotIn("\x1b", r.stdout)


if __name__ == "__main__":
    unittest.main()
