#!/usr/bin/env python3
"""A vLLM lane that starts at boot keeps its unit when the new one mounts a file not written yet.

The Ornith lane's unit mounts its chat template since v1.40.2. An update that found the lane's
checkpoint out of reach (its cache on a disk absent then) runs the lane's installer --units-only,
which writes no template, and put the new unit: docker refuses an absent source, so a box booting
that lane would not have served at its next boot (found in review, 2026-10-24). vllm-lane.sh's own
vlane_put_unit runs here against an installed unit, a new one, and a systemctl that says whether the
lane starts at boot."""
import os
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]


class TheUnitOfALaneThatStartsAtBoot(unittest.TestCase):
    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp(prefix="vlane-boot-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.d, True))
        (self.d / "cfg").mkdir()
        self.bin = self.d / "bin"
        self.bin.mkdir()
        (self.bin / "systemctl").write_text(
            '#!/bin/bash\ncase "$*" in *is-enabled*) [ "${ENABLED:-0}" = 1 ] ;; *) exit 0 ;; esac\n')
        # and a sudo that runs the stub: the real one reached the real systemctl through its secure_path,
        # and the cockpit's sudoers let each run reload the box's systemd (code review, 2026-10-24)
        (self.bin / "sudo").write_text('#!/bin/bash\necho "sudo $*" >> "$SUDO_LOG"\nexec "$@"\n')
        for stub in ("systemctl", "sudo"):
            self.bin.joinpath(stub).chmod(0o755)
        self.sudo_log = self.d / "sudo.log"
        self.installed = self.d / "installed.service"
        self.installed.write_text("[Service]\nExecStart=/bin/bash -c 'exec docker run old'\n")
        self.template = self.d / "cfg" / "chat-template.jinja"

    def put(self, enabled, new=None):
        new_unit = self.d / "new.service"
        new_unit.write_text(new or ("[Service]\nExecStart=/bin/bash -c 'exec docker run \\\n"
                                    f"  --mount type=bind,source={self.template},target=/out/t.jinja,readonly \\\n"
                                    f"  --mount type=bind,source={self.d}/cfg,target=/out/cfg,readonly new'\n"))
        script = "\n".join([
            "set -euo pipefail", f'. "{REPO}/vllm-lane.sh"',
            'die(){ echo "die: $*"; exit 1; }', 'unit_put(){ cp "$1" "$2"; echo installed; }',
            f'CONFIG_DIR="{self.d}/cfg"', f'INSTALLED="{self.installed}"', "UNIT=qwen38-x.service",
            "VLANE_SHORT=X", "SMOKE=1", f'vlane_put_unit "{new_unit}"'])
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                           env=dict(os.environ, PATH=f"{self.bin}:/usr/bin:/bin", ENABLED="1" if enabled else "0",
                                    SUDO_LOG=str(self.sudo_log)))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout, self.installed.read_text()

    def test_a_boot_lane_keeps_its_unit_when_the_new_one_mounts_a_missing_file(self):
        out, installed = self.put(enabled=True)
        self.assertIn(f"NOTE: X starts at boot, and its new unit mounts what is not there yet: {self.template}.", out)
        self.assertIn("docker run old", installed)

    def test_a_boot_lane_takes_the_new_unit_once_the_file_is_there(self):
        self.template.write_text("{{ messages }}\n")
        out, installed = self.put(enabled=True)
        self.assertNotIn("NOTE", out)
        self.assertIn(" new'", installed)

    def test_a_lane_that_does_not_start_at_boot_takes_the_new_unit(self):
        out, installed = self.put(enabled=False)
        self.assertNotIn("NOTE", out)
        self.assertIn(" new'", installed)
        # the reload went to the stubs, never to the box's systemd
        self.assertEqual(self.sudo_log.read_text(), "sudo systemctl daemon-reload\n")

    def test_an_unchanged_unit_is_not_held_back(self):
        out, installed = self.put(enabled=True, new=self.installed.read_text())
        self.assertNotIn("NOTE", out)


if __name__ == "__main__":
    unittest.main()
