#!/usr/bin/env python3
"""A switch to what a 27B unit or the flash launch script already loads rewrites nothing.

Rewritten, the file got a new date, which reads as an update the serving engine has not taken:
the cockpit then offered a restart of 8 to 13 minutes for nothing (found in review, 2026-10-17).
Kolibri, Nemotron and the image lane already compared before rewriting; the 27B unit and the
flash launch script now do too. The switch's own lines run here with sudo and install recorded.
"""
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "switch-model.sh").read_text()


def block(first):
    start = TEXT.index(first)
    return TEXT[start:TEXT.index("\n  fi\n", start) + 6]


UNIT_27B = block('  if cmp -s "$STAGE_UNIT" "$TARGET_UNIT"; then')
FLASH = block('  if cmp -s "$TMP_UNIT" "$FLASH_LAUNCH"; then')


def run(snippet, staged, installed):
    d = pathlib.Path(tempfile.mkdtemp(prefix="sw-same-"))
    (d / "staged").write_text(staged)
    (d / "installed").write_text(installed)
    script = ("set -euo pipefail\n"
              f'log="{d}/calls"; sudo(){{ echo "sudo $*" >> "$log"; }}; install(){{ echo "install $*" >> "$log"; }}\n'
              f'CHOICE=stock; STAGE_UNIT="{d}/staged"; TARGET_UNIT="{d}/installed"; '
              f'TMP_UNIT="{d}/staged"; FLASH_LAUNCH="{d}/installed"\n' + snippet + "echo REACHED\n")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
    calls = (d / "calls").read_text() if (d / "calls").exists() else ""
    return r.stdout + r.stderr, calls


class ASwitchToWhatIsInstalledRewritesNothing(unittest.TestCase):
    def test_the_27b_unit_unchanged_keeps_its_date(self):
        out, calls = run(UNIT_27B, "ExecStart=x --revision a\n", "ExecStart=x --revision a\n")
        self.assertEqual(calls, "", out)
        self.assertIn("unit unchanged: it already loads stock", out)
        self.assertIn("REACHED", out)

    def test_the_27b_unit_changed_is_installed(self):
        out, calls = run(UNIT_27B, "ExecStart=x --revision b\n", "ExecStart=x --revision a\n")
        self.assertIn("sudo install -m 644", calls, out)
        self.assertIn("REACHED", out)

    def test_the_flash_launcher_unchanged_keeps_its_date(self):
        out, calls = run(FLASH, "docker run x\n", "docker run x\n")
        self.assertEqual(calls, "", out)
        self.assertIn("launch script unchanged", out)

    def test_the_flash_launcher_changed_is_installed(self):
        out, calls = run(FLASH, "docker run y\n", "docker run x\n")
        self.assertIn("install -m 755", calls, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
