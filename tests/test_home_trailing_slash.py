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
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_conformity_classify import HOME, box, classes  # noqa: E402
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


class WhatWasWrittenWithTheSlash(unittest.TestCase):
    """A box installed before this release, from a home with the slash, has units that say "//". Read back,
    such a path is the same folder: kept as the operator's choice it was never mended, and uninstall.sh
    listed the cache twice (found in review, 2026-10-22)."""

    def test_the_cache_a_unit_mounts_reads_back_one_slash_at_a_time(self):
        with tempfile.TemporaryDirectory() as t:
            unit = pathlib.Path(t) / "u.service"
            unit.write_text("ExecStart=docker run -v /home/nvidia//.cache/huggingface:/root/.cache/huggingface img\n")
            r = subprocess.run(["bash", "-c", f'. "{REPO}/lanes.sh"; unit_hf_cache "{unit}"'], capture_output=True,
                               text=True, timeout=10, env={"PATH": "/usr/bin:/bin", "HOME": t})
            self.assertEqual(r.stdout, "/home/nvidia/.cache/huggingface\n", r.stderr)

    def test_the_ple_folder_install_sh_reads_back_too(self):
        line = next(ln for ln in (REPO / "install.sh").read_text().splitlines() if ln.strip().startswith("CUR_PLE="))
        with tempfile.TemporaryDirectory() as t:
            launch = pathlib.Path(t) / "launch-flash.sh"
            launch.write_text("docker run -v /home/nvidia//flashnext-ple:/ple img\n")
            r = subprocess.run(["bash", "-c", f'set -euo pipefail\nFLASH_LAUNCH="{launch}"\n{line.strip()}\nprintf "%s" "$CUR_PLE"'],
                               capture_output=True, text=True, timeout=10, env={"PATH": "/usr/bin:/bin"})
            self.assertEqual(r.stdout, "/home/nvidia/flashnext-ple", r.stderr)

    def test_uninstall_lists_the_cache_once(self):
        text = (REPO / "uninstall.sh").read_text()
        block = text[text.index("unit_mount(){"):text.index("PLE_DIRS=")]
        with tempfile.TemporaryDirectory() as t:
            units = pathlib.Path(t) / "units"
            units.mkdir()
            (units / "qwen38-sglang.service").write_text(
                "ExecStart=docker run -v /home/nvidia//.cache/huggingface:/root/.cache/huggingface img\n")
            # the image and the video lanes name it in their HF_HOME (v1.38.0 listed it twice from them)
            for lane in ("image", "video"):
                (units / f"qwen38-{lane}.service").write_text("Environment=HF_HOME=/home/nvidia//.cache/huggingface\n")
            r = subprocess.run(["bash", "-c", f'set -euo pipefail\nSYSTEMD_DIR="{units}"; CONFIG_DIR="{t}/cfg"; HF_CACHE=""\n'
                                + block + '\nprintf "%s\\n" "$HF_CACHES"'],
                               capture_output=True, text=True, timeout=10, env={"PATH": "/usr/bin:/bin", "HOME": "/home/nvidia"})
            self.assertEqual(r.stdout.split(), ["/home/nvidia/.cache/huggingface"], r.stderr)

    def test_the_side_lanes_read_their_folder_and_cache_back_one_slash_at_a_time(self):
        # each installer's own read-back lines, from a unit it rendered from a home with the slash: the
        # unit an update renders is then the one the home says, not the one it found (v1.38.0 kept theirs)
        for script, lane in (("install-image.sh", "image"), ("install-video.sh", "video"), ("install-speech.sh", "speech")):
            with self.subTest(script):
                lines = (REPO / script).read_text().splitlines()
                take = [ln for ln in lines if re.match(r"(installed|installed_env)\(\)\{|LANE_DIR=|HF_CACHE=\"\$\{HF_CACHE:-\$\(installed_env", ln)]
                self.assertEqual(len(take), 4 if lane != "speech" else 2, take)
                with tempfile.TemporaryDirectory() as t:
                    unit = pathlib.Path(t) / f"qwen38-{lane}.service"
                    unit.write_text(f"[Service]\nWorkingDirectory=/home/nvidia//.local/share/qwen38-{lane}\n"
                                    "Environment=HF_HOME=/home/nvidia//.cache/huggingface\n")
                    r = subprocess.run(["bash", "-c", f'set -euo pipefail\nINSTALLED="{unit}"\ntext_cache(){{ echo /elsewhere; }}\n'
                                        + "\n".join(take) + '\nprintf "%s|%s" "$LANE_DIR" "${HF_CACHE:-}"'],
                                       capture_output=True, text=True, timeout=10, env={"PATH": "/usr/bin:/bin", "HOME": "/home/nvidia"})
                want = f"/home/nvidia/.local/share/qwen38-{lane}|" + ("/home/nvidia/.cache/huggingface" if lane != "speech" else "")
                self.assertEqual((r.returncode, r.stdout), (0, want), r.stderr)

    def test_conformity_classes_the_cache_and_the_ple_folder_as_the_default(self):
        units, cfg = box(hf=f"{HOME}//.cache/huggingface", ple=f"{HOME}//flashnext-ple")
        seen = classes(units, cfg)
        self.assertNotIn("HF_CACHE", seen)
        self.assertNotIn("PLE_DIR", seen)

if __name__ == "__main__":
    unittest.main()
