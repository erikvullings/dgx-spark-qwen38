#!/usr/bin/env python3
"""A home given with a trailing slash ("/home/nvidia/" in /etc/passwd, on a second ASUS Ascent GX10) put
a "//" in every path the scripts built on it, while systemd hands the cockpit the
same home without the slash. The cockpit's sudoers allowlist, rendered from a login shell, named
"/home/nvidia//.config/qwen38/...switch-stage", the cockpit's own switch called
"/home/nvidia/.config/qwen38/...switch-stage", and sudo asked for a password at every Load that installs a
unit (found on that box, 2026-10-22). Every script that renders or compares a path built on the home
now spells it one way, without the slash, before it uses it."""
import os
import pathlib
import re
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
LINE = 'while [ -n "${HOME:-}" ] && [ "$HOME" != / ] && [ "${HOME%/}" != "$HOME" ]; do HOME="${HOME%/}"; done; export HOME'
# the entry points that render a unit, the sudoers or a config from the home, or compare one with it
ENTRY = ("install.sh", "switch-model.sh", "prepare-lane.sh", "uninstall.sh", "get.sh", "install-image.sh",
         "install-video.sh", "install-kolibri.sh", "install-nemotron.sh", "install-nemotron-super.sh",
         "install-flash-local.sh", "install-ornith-122b.sh", "install-speech.sh", "dashboard/install-dashboard.sh",
         "dashboard/install-agent.sh")


def canonical(home):
    env = {"PATH": "/usr/bin:/bin"} if home is None else {"PATH": "/usr/bin:/bin", "HOME": home}
    r = subprocess.run(["bash", "-c", "set -euo pipefail\n" + LINE + '\nprintf "%s" "${HOME-unset}"'],
                       capture_output=True, text=True, env=env, timeout=10)
    return r.returncode, r.stdout


class OneSpellingOfTheHome(unittest.TestCase):
    def test_the_line_takes_every_trailing_slash_and_nothing_else(self):
        for given, want in (("/home/nvidia/", "/home/nvidia"), ("/home/nvidia//", "/home/nvidia"),
                            ("/home/nvidia", "/home/nvidia"), ("/", "/"), ("/x y/", "/x y")):
            with self.subTest(given=given):
                self.assertEqual(canonical(given), (0, want))
        self.assertEqual(canonical(None), (0, "unset"), "no home: nothing made up, and set -u does not stop it")

    def test_every_entry_point_says_it_once_before_it_uses_the_home(self):
        for f in ENTRY:
            with self.subTest(f):
                lines = (REPO / f).read_text().splitlines()
                at = [i for i, ln in enumerate(lines) if ln == LINE]
                self.assertEqual(len(at), 1, "the line, once")
                uses = [i for i, ln in enumerate(lines) if re.search(r"\$\{?HOME\b", ln)
                        and not ln.lstrip().startswith("#") and ln != LINE]
                self.assertTrue(uses, "it uses the home")
                self.assertLess(at[0], uses[0], f"line {uses[0] + 1} uses the home before it is spelled one way")

    def test_the_sudoers_a_login_shell_renders_is_what_the_cockpit_calls(self):
        # install-dashboard.sh's own lines, from its first to the one before HERE=, then its sed of the
        # allowlist, run from a login shell's home with the slash
        text = (REPO / "dashboard/install-dashboard.sh").read_text().splitlines()
        head = text[:next(i for i, ln in enumerate(text) if ln.startswith('HERE="$(cd'))]
        sed = next(ln for ln in text if "sudoers-cockpit.template" in ln and ln.startswith("sed "))
        with tempfile.TemporaryDirectory() as t:
            home = pathlib.Path(t) / "home"
            out = pathlib.Path(t) / "sudoers"
            script = "\n".join(head) + f'\nHERE="{REPO / "dashboard"}"; TMP_SUDO="{out}"\n' + sed + "\n"
            r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10,
                               env={"PATH": "/usr/bin:/bin", "HOME": f"{home}/"})
            self.assertEqual(r.returncode, 0, r.stderr)
            sudoers = out.read_text()
        self.assertNotIn(f"{home}//", sudoers)
        # the cockpit's switch, under systemd, with the same home spelled without the slash
        for lane in ("sglang", "image", "kolibri", "nemotron", "nemotron-super", "flash-local", "ornith-122b"):
            with self.subTest(lane=lane):
                call = (f"/usr/bin/install -m 644 {home}/.config/qwen38/qwen38-{lane}.service.switch-stage "
                        f"/etc/systemd/system/qwen38-{lane}.service")
                self.assertIn(f"NOPASSWD: {call}\n", sudoers)

    def test_a_switch_from_a_login_shell_stages_where_the_sudoers_says(self):
        # switch-model.sh's own lines up to its first use of the terminal (DL_NO_BARS), then its config
        # directory as it builds it, from a home with the slash
        text = (REPO / "switch-model.sh").read_text().splitlines()
        head = text[:next(i for i, ln in enumerate(text) if ln.startswith("DL_NO_BARS="))]
        with tempfile.TemporaryDirectory() as t:
            r = subprocess.run(["bash", "-c", "\n".join(head) + '\nCONFIG_DIR="$HOME/.config/qwen38"\n'
                                'printf "%s" "$CONFIG_DIR/qwen38-ornith-122b.service.switch-stage"'],
                               capture_output=True, text=True, timeout=10, env={"PATH": "/usr/bin:/bin", "HOME": f"{t}/"})
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout, f"{t}/.config/qwen38/qwen38-ornith-122b.service.switch-stage")

if __name__ == "__main__":
    unittest.main()
