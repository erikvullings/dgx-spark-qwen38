#!/usr/bin/env python3
"""switch-model.sh <target> --fetch: the target's checkpoint, fetched as a switch fetches it
(the cache first, resumable, its room checked), and nothing else: no unit rewritten, no boot
moved, no engine touched, no record written. Library's "Download" and a lane installed
without its weights (the on-demand install) get their weights this way before a Load.

The script runs as written on a copy whose unit paths point at this test's files, with a
fake docker whose download succeeds and a fake systemctl that records every call.
"""
import pathlib
import re
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
SWITCH = (REPO / "switch-model.sh").read_text()

FAKE_DOCKER = """#!/bin/sh
case "$1" in
  image) exit 0 ;;
  run) printf '%s\\n' "$*" >> "$DOCKER_LOG"; exit 0 ;;   # the download succeeds
esac
exit 0
"""


class Box:
    def __init__(self):
        self.t = pathlib.Path(tempfile.mkdtemp(prefix="switch-fetch-"))
        self.home = self.t / "home"
        (self.home / ".config/qwen38").mkdir(parents=True)
        units = self.t / "units"
        units.mkdir()
        self.unit = units / "qwen38-sglang.service"
        self.unit.write_text(
            "[Service]\nExecStart=/bin/bash -c 'exec docker run --rm --name qwen38-sglang \\\n"
            f"  -v {self.home}/.cache/huggingface:/root/.cache/huggingface \\\n"
            "  lmsysorg/sglang@sha256:" + "a" * 64 + " \\\n"
            "  python3 -m sglang.launch_server --model-path RadixArk/Qwen3.8-27B-NVFP4 \\\n"
            "  --host 127.0.0.1 --port 30000'\n")
        repo = self.t / "repo"
        repo.mkdir()
        (repo / "install.sh").write_text((REPO / "install.sh").read_text())
        text = SWITCH
        for unit in ("qwen38-sglang.service", "qwen38-flash.service"):
            old = f'"/etc/systemd/system/{unit}"'
            assert old in text, f"switch-model.sh no longer names {unit} that way"
            text = text.replace(old, f'"{units}/{unit}"')
        (repo / "switch-model.sh").write_text(text)
        for f in ("lanes.sh", "ckpt-fetch.py"):
            (repo / f).write_text((REPO / f).read_text())
        (repo / "switch-model.sh").chmod(0o755)
        self.bin = self.t / "bin"
        self.bin.mkdir()
        (self.bin / "docker").write_text(FAKE_DOCKER)
        (self.bin / "systemctl").write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$SYSTEMCTL_LOG"\nexit 1\n')
        (self.bin / "sudo").write_text('#!/bin/sh\nprintf "sudo %s\\n" "$*" >> "$SYSTEMCTL_LOG"\nexit 1\n')
        for n in ("docker", "systemctl", "sudo"):
            (self.bin / n).chmod(0o755)
        self.docker_log = self.t / "docker.log"
        self.sysctl_log = self.t / "systemctl.log"
        self.script = repo / "switch-model.sh"

    def run(self, *args):
        e = {"PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(self.home),
             "DOCKER_LOG": str(self.docker_log), "SYSTEMCTL_LOG": str(self.sysctl_log)}
        r = subprocess.run([str(self.script), *args], capture_output=True, text=True, env=e, timeout=60)
        runs = self.docker_log.read_text().splitlines() if self.docker_log.exists() else []
        calls = self.sysctl_log.read_text().splitlines() if self.sysctl_log.exists() else []
        return r.returncode, r.stdout + r.stderr, runs, calls


class FetchOnly(unittest.TestCase):
    def test_a_text_target_is_fetched_and_nothing_else_changes(self):
        box = Box()
        unit_before = box.unit.read_bytes()
        rc, out, runs, calls = box.run("stock", "--fetch")
        self.assertEqual(rc, 0, out)
        self.assertIn("nothing else was changed", out)
        self.assertEqual(len(runs), 1, runs)
        self.assertIn("/opt/qwen38/ckpt-fetch.py RadixArk/Qwen3.8-27B-NVFP4 ", runs[0])
        self.assertEqual(box.unit.read_bytes(), unit_before)
        self.assertFalse([c for c in calls if re.search(r"\b(enable|disable|daemon-reload|restart|start|stop)\b", c)], calls)
        self.assertFalse((box.home / ".config/qwen38/text-lane").exists(), "--fetch wrote the text-lane record")

    def test_without_it_the_switch_goes_on_past_the_download(self):
        rc, out, runs, _ = Box().run("stock")
        self.assertEqual(len(runs), 1, out)
        self.assertNotIn("nothing else was changed", out)

    def test_an_unknown_option_is_refused_before_anything_runs(self):
        rc, out, runs, calls = Box().run("stock", "--fetchh")
        self.assertNotEqual(rc, 0)
        self.assertIn("unknown option", out)
        self.assertEqual((runs, calls), ([], []))

    def test_every_lane_stops_right_after_its_own_download(self):
        # the four branches that fetch (Kolibri, image, video, the text lanes): each one's
        # fetch command is followed at once by the --fetch stop, before anything it writes
        calls = [m.start() for m in re.finditer(r"ckpt-fetch\.py\"? \"\$", SWITCH)]
        self.assertEqual(len(calls), 4)
        for at in calls:
            rest = SWITCH[at:]
            die = rest.index("|| die ")
            after = rest[rest.index("\n", die) + 1:]
            self.assertTrue(after.lstrip().startswith("fetched "), after[:120])


if __name__ == "__main__":
    unittest.main()
