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
import test_install_agent_keeps as keeps  # noqa: E402
from test_install_agent_keeps import DASH_CUSTOM, oc_unit, run_in  # noqa: E402
LINE = 'while [ -n "${HOME:-}" ] && [ "$HOME" != / ] && [ "${HOME%/}" != "$HOME" ]; do HOME="${HOME%/}"; done; export HOME'
PATH_LINE = 'PATH="$(printf \'%s\' "$PATH" | tr -s /)"; export PATH'
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

    def test_install_sh_finds_local_bin_on_the_path_such_a_login_gives(self):
        # its own first lines, through the PATH line, then its note from the case before it to the esac, from a
        # login on a home with the slash: v1.38.0 said "/home/nvidia//.local/bin" was not on the PATH
        lines = (REPO / "install.sh").read_text().splitlines()
        head = lines[:lines.index(PATH_LINE) + 1]
        note = next(i for i, ln in enumerate(lines) if "is not in this shell's PATH" in ln)
        start = max(i for i in range(note) if lines[i].lstrip().startswith("case "))
        stop = next(i for i in range(note, len(lines)) if lines[i].strip() == "esac")
        for path, says in (("/home/nvidia//.local/bin:/usr/bin:/bin", False), ("/usr/bin:/bin", True)):
            with self.subTest(path=path):
                r = subprocess.run(["bash", "-c", "\n".join(head) + '\nOC_BIN="$HOME/.local/bin/oc"\n'
                                    + "\n".join(lines[start:stop + 1])],
                                   capture_output=True, text=True, timeout=10, env={"PATH": path, "HOME": "/home/nvidia/"})
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual("NOTE" in r.stdout, says, r.stdout)

    def test_the_scripts_that_read_the_path_spell_it_right_after_the_home(self):
        # install.sh compares what the PATH finds with the folders it builds on the home (its note, its prepend of
        # opencode's folder, the way it says to start oc), install-agent.sh writes what it finds into a unit
        for f in ("install.sh", "dashboard/install-agent.sh"):
            with self.subTest(f):
                lines = (REPO / f).read_text().splitlines()
                self.assertEqual(lines.count(PATH_LINE), 1)
                self.assertEqual(lines.index(PATH_LINE), lines.index(LINE) + 3, "the PATH line follows the home's")

    def test_install_agent_finds_opencode_as_the_home_spells_it(self):
        # install-agent.sh's own lines up to the render, run alone from a login on a home with the slash, whose PATH
        # says "<home>//bin": the opencode it puts in opencode-web's ExecStart, which install.sh's runs spelled
        # without the slash, so that each run after the other restarted the Agent view (found in review)
        with tempfile.TemporaryDirectory(prefix="agent-slash-") as t:
            t = pathlib.Path(t)
            (t / "qwen38-dashboard.service").write_text(DASH_CUSTOM)
            fake = t / "bin"
            fake.mkdir()
            (fake / "opencode").write_text(keeps.OPENCODE)
            (fake / "tailscale").write_text("#!/bin/sh\nexit 1\n")
            for f in fake.iterdir():
                f.chmod(0o755)
            r = subprocess.run(["bash", "-c", keeps.HEAD.replace("/etc/systemd/system", str(t)) + 'echo "RESULT|$OPENCODE_BIN"\n'],
                               capture_output=True, text=True, timeout=60, cwd=REPO / "dashboard",
                               env={"PATH": f"{t}//bin:/usr/bin:/bin", "HOME": f"{t}/"})
            got = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT|")), r.stdout + r.stderr)
            self.assertEqual(got, f"RESULT|{t}/bin/opencode")

    def test_opencode_webs_path_names_each_folder_once(self):
        # the PATH install-agent.sh writes into opencode-web's unit, from such a login and the unit v1.38.0
        # wrote there, which named opencode's folder in both spellings
        home = "/home/nvidia"
        installed = [f"{home}/.opencode/bin", f"{home}//.local/bin", "/usr/bin", f"{home}//.opencode/bin"]
        with tempfile.TemporaryDirectory() as t:
            (_, _, _, path), _ = run_in(pathlib.Path(t), DASH_CUSTOM, oc_unit(installed),
                                        caller=(f"{home}//.local/bin", f"{home}//.opencode/bin"))
        dirs = path.split(":")
        self.assertEqual([d for d in dirs if "//" in d], [], dirs)
        self.assertEqual((dirs.count(f"{home}/.opencode/bin"), dirs.count(f"{home}/.local/bin")), (1, 1), dirs)

    def test_uninstall_lists_each_folder_once(self):
        # the folders given from a shell on such a home (PLE_DIR=~/...) beside their defaults, and the side lanes'
        # folders their units name, as uninstall.sh lists them before it removes anything
        text = (REPO / "uninstall.sh").read_text()
        block = text[text.index("unit_mount(){"):text.index("# The diffusion runtime's own caches")]
        dirs = [ln for ln in text.splitlines() if re.match(r'(IMAGE|VIDEO|SPEECH)_LANE_DIR="\$\{\1_LANE_DIR:-\$\(', ln)]
        self.assertEqual(len(dirs), 3, dirs)
        with tempfile.TemporaryDirectory() as t:
            units = pathlib.Path(t) / "units"
            units.mkdir()
            for lane in ("image", "video", "speech"):
                (units / f"qwen38-{lane}.service").write_text(f"WorkingDirectory=/home/nvidia//.local/share/qwen38-{lane}\n")
            r = subprocess.run(["bash", "-c", f'set -euo pipefail\nSYSTEMD_DIR="{units}"; CONFIG_DIR="{t}/cfg"\n' + block
                                + "\n".join(dirs) + '\nprintf "%s|" "$PLE_DIRS" "$FL_TABLE_DIRS" "$IMAGE_LANE_DIR" "$VIDEO_LANE_DIR" "$SPEECH_LANE_DIR"'],
                               capture_output=True, text=True, timeout=10,
                               env={"PATH": "/usr/bin:/bin", "HOME": "/home/nvidia", "PLE_DIR": "/home/nvidia//flashnext-ple",
                                    "FLASH_LOCAL_PLE_DIR": "/home/nvidia//flash-local-ple"})
        self.assertEqual(r.stdout.split("|")[:5], ["/home/nvidia/flashnext-ple", "/home/nvidia/flash-local-ple"]
                         + [f"/home/nvidia/.local/share/qwen38-{lane}" for lane in ("image", "video", "speech")], r.stderr)

    def test_flash_local_lists_its_table_folders_once(self):
        # install-flash-local.sh --uninstall: the folder its installer recorded as given (FLASH_LOCAL_PLE_DIR=~/...
        # from such a shell) and the default folder are one
        lines = (REPO / "install-flash-local.sh").read_text().splitlines()
        start = next(i for i, ln in enumerate(lines) if ln.strip().startswith('TABLE_DIRS="$('))
        stop = next(i for i in range(start, len(lines)) if "!seen[$0]++')\"" in lines[i])
        with tempfile.TemporaryDirectory() as t:
            cfg = pathlib.Path(t) / "cfg"
            cfg.mkdir()
            (cfg / "flash-local-ple-dir").write_text("/home/nvidia//flash-local-ple\n")
            r = subprocess.run(["bash", "-c", f'set -euo pipefail\nINSTALLED="{t}/none.service"; CONFIG_DIR="{cfg}"\n'
                                + "\n".join(lines[start:stop + 1]) + '\nprintf "%s" "$TABLE_DIRS"'],
                               capture_output=True, text=True, timeout=10, env={"PATH": "/usr/bin:/bin", "HOME": "/home/nvidia"})
        self.assertEqual(r.stdout.split(), ["/home/nvidia/flash-local-ple"], r.stderr)

    def test_conformity_classes_the_cache_and_the_ple_folder_as_the_default(self):
        units, cfg = box(hf=f"{HOME}//.cache/huggingface", ple=f"{HOME}//flashnext-ple")
        seen = classes(units, cfg)
        self.assertNotIn("HF_CACHE", seen)
        self.assertNotIn("PLE_DIR", seen)

if __name__ == "__main__":
    unittest.main()
