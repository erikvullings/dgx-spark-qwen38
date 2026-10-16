#!/usr/bin/env python3
"""Each lane installer's --help prints its whole header, and nothing else.

The help printed fixed line ranges (sed -n '2,32p') that drifted as the headers grew: the Nemotron
lane's printed two lines of its code (v1.34.0), the image lane's stopped in the middle of a sentence,
and the Kolibri and video lanes' left their last paragraph out (found in review, 2026-10-16). Each
now prints its comment lines from the second to the first line of code, whatever their number. The
help's own line runs here on the installer's file; the installer itself is never run.
"""
import pathlib
import re
import subprocess
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
INSTALLERS = ("install-image.sh", "install-video.sh", "install-kolibri.sh", "install-nemotron.sh",
              "install-nemotron-super.sh", "install-speech.sh")
# What the header's last paragraph opens with. The GPU lanes end on the rule that one engine runs
# at a time; the speech lane runs on the CPU beside any of them, and ends on why the one command
# leaves it out.
LAST_PARAGRAPH = {"install-speech.sh": "NOT IN THE ONE COMMAND YET"}


class TheHelpIsTheHeader(unittest.TestCase):
    def test_each_prints_its_whole_header_and_no_code(self):
        for script in INSTALLERS:
            path = REPO / script
            text = path.read_text()
            lines = text.splitlines()
            end = next(i for i, ln in enumerate(lines) if i > 0 and not ln.startswith("#"))
            with self.subTest(script=script):
                m = re.search(r"\n  -h\|--help\) (awk '[^']*') \"\$0\"; exit 0 ;;\n", text)
                self.assertIsNotNone(m, "its --help prints the header by its shape, not by line numbers")
                out = subprocess.run(["bash", "-c", f'{m.group(1)} "$1"', "help", str(path)],
                                     capture_output=True, text=True, check=True, timeout=10).stdout.splitlines()
                self.assertEqual(out, lines[1:end])
                self.assertGreater(len(out), 10)
                self.assertTrue(all(ln.startswith("#") for ln in out), "no line of code")
                self.assertIn(LAST_PARAGRAPH.get(script, "ONE ENGINE AT A TIME"), "\n".join(out),
                              "the header's last paragraph is there")


if __name__ == "__main__":
    unittest.main()
