#!/usr/bin/env python3
"""The other text lane's run, installed on demand (QWEN38_SECONDARY + QWEN38_UNITS_ONLY=1), run as
written to its last line. Its guards are tested one by one elsewhere; this runs the whole path,
so a step that trips over what the guards skipped (a variable only a skipped step set, a file
only a skipped step wrote) shows up here, and not as a lane missing from every new box.

The copy ends at a wall just past its last line (installer_wall); the commands that act on the
box are stubs that answer as a box with no lane image would (docker has none, nvidia-smi sees
the GPU), sudo runs what it is given in the test's own folders, and the units are written to a
folder the test owns. The machine is a GB10 whatever runs the suite: uname says aarch64 and the
copy reads the reference box's MemTotal, as the preflight refuses a box below the 64 GB class
(GitHub's x86_64 runner, 16 GB, stopped it there: found by the CI run of v1.33.0, 2026-10-15)."""
import pathlib
import shutil
import subprocess
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import installer_wall as wall  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
DONE = '    step "Done: installed as $UNIT_NAME ($MODEL_CHOICE), not enabled at boot, not prepared yet"\n'
# where the run of a box that starts nothing begins: a side lane at boot, or a flash lane its GPU
# cannot start (issue #54)
NO_START = '  # Updated, not switched to. Starting it would stop the lane that is serving, and\n'


def run_units_only(lane, gpu="ATS", main=False):
    """The run for the other text lane, or with main=True the run of a box whose own text lane is
    the flash lane, installed and enabled at boot, walled where that run takes the path that starts
    nothing. gpu: the addressing mode nvidia-smi reports."""
    units = wall.units_dir()
    script = pathlib.Path(wall.walled(at=NO_START if main else DONE, after=not main, units=units))
    # the reference box's memory, as install.sh reads it, by QWEN38_MEMINFO (its class at the top
    # and its GiB at step 1, boxmem.py): rewriting the fallback in the copy was dead code under the
    # wall's own file (found in review, 2026-10-23)
    meminfo = wall._tmp("meminfo-") / "meminfo"
    meminfo.write_text("MemTotal:       127535272 kB\n")
    for f in REPO.iterdir():      # what install.sh reads beside itself (its templates, its helpers)
        if f.is_file() and not (script.parent / f.name).exists():
            shutil.copy2(f, script.parent / f.name)
    shutil.copytree(REPO / "dashboard", script.parent / "dashboard", dirs_exist_ok=True)
    home = wall._tmp("home-")
    (home / ".config/qwen38").mkdir(parents=True)
    (home / ".config/qwen38/api-key").write_text("k\n")
    if main:   # a flash lane this box already had: its unit and the launcher of v1.36.3
        (pathlib.Path(units) / "qwen38-flash.service").write_text(
            f"[Service]\nExecStart=/bin/bash {home}/.config/qwen38/launch-flash.sh\n")
        (home / ".config/qwen38/launch-flash.sh").write_text(
            "exec /usr/bin/docker run --rm --name qwen38-flash lmsysorg/sglang@sha256:" + "a" * 64
            + " python3 -m sglang.launch_server --model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 --tp-size 1\n")
    stubs, docker_root = wall._tmp("stubs-"), wall._tmp("docker-root-")
    calls = stubs / "calls"
    calls.touch()
    for name, body in (
            ("nvidia-smi", 'case "$*" in *driver_version*) echo 580.173.02 ;; *memory.total*) echo "[N/A]" ;; '
                           f'*addressing_mode*) echo {gpu} ;; *) echo "NVIDIA GB10" ;; esac\n'),
            ("uname", 'echo aarch64\n'),
            ("docker", f'case "$1 $2" in "info "*) echo "{docker_root}" ;; "image inspect") exit 1 ;; esac\nexit 0\n'),
            ("sudo", 'while [ -n "${1:-}" ] && [ "${1#-}" != "$1" ]; do shift; done\n'
                     '[ "$#" -eq 0 ] || [ "$1" = true ] && exit 0\n"$@"\n'),
            ("systemctl", ('[ "$1 $2" = "is-enabled qwen38-flash.service" ] && exit 0\n' if main else "")
                          + 'case "$1" in is-enabled|is-active) exit 1 ;; esac\nexit 0\n'),
            # the preflight has to find it, and a lane's run then asks it nothing (the parent
            # checked the ports): the host's was used, and a host without iproute2 died there,
            # "ss not found" (a cloud container, 2026-10-18)
            ("ss", 'exit 0\n')):
        (stubs / name).write_text(f'#!/bin/bash\necho "{name} $*" >> "{calls}"\n' + body)
        (stubs / name).chmod(0o755)
    run = {} if main else {"QWEN38_SECONDARY": lane, "QWEN38_UNITS_ONLY": "1",
                           "MODEL_CHOICE": "flash" if lane == "flash" else "stock"}
    env, fenced = wall.fenced_env(home=home, first=str(stubs), HF_CACHE=str(home / "hf"), PORT="30000",
                                  QWEN38_MEMINFO=str(meminfo),
                                  PROXY_PORT="30001", **run)
    r = subprocess.run([str(script)] + (["--no-opencode"] if main else []), capture_output=True, text=True,
                       env=env, cwd=wall.cwd(), timeout=300)
    return (r.returncode, r.stdout + r.stderr, calls.read_text().splitlines(), wall.reached(fenced),
            pathlib.Path(units), home)


class TheUnitsOnlyRunToItsEnd(unittest.TestCase):
    def check(self, lane, unit):
        rc, out, calls, fenced, units, home = run_units_only(lane)
        self.assertIn(wall.WALL, out, out[-1500:])      # it reached its last line
        self.assertEqual(rc, 97, out[-1500:])
        self.assertEqual(fenced, [])                      # nothing past the stubs
        # no image pulled or run, no checkpoint fetched: docker was only asked what it has
        self.assertEqual(sorted({" ".join(c.split()[:3]) for c in calls if c.startswith("docker ")}),
                         ["docker image inspect", "docker info", "docker info --format"], calls)
        self.assertFalse((home / "hf" / "hub").exists() and any((home / "hf" / "hub").iterdir()))
        # root for the unit and nothing else: not enabled, not started
        sudo = [c for c in calls if c.startswith("sudo ") and c not in ("sudo -n true", "sudo -v")]
        self.assertEqual([c.split()[1] for c in sudo], ["install", "systemctl"], sudo)
        self.assertEqual(sudo[-1], "sudo systemctl daemon-reload")
        self.assertTrue((units / unit).is_file())
        self.assertIn("not prepared yet", out)
        return out, units, home

    def test_the_flash_lane(self):
        out, units, home = self.check("flash", "qwen38-flash.service")
        launcher = (home / ".config/qwen38/launch-flash.sh").read_text()
        self.assertIn('[ -s "', launcher)                  # the vocabulary waited for, not required
        self.assertIn("&& TIER+=(--speculative-token-map /out/token-map-", launcher)
        # the run an update gives a box outside ATS (issue #54): the launcher asks the GPU's mode
        # at every start, from the copy this run installed, and the unit does not start it again
        # on the code it then ends with
        self.assertIn(f'bash {home}/.config/qwen38/flash-ats-check.sh)"', launcher)
        self.assertTrue((home / ".config/qwen38/flash-ats-check.sh").is_file())
        self.assertIn("RestartPreventExitStatus=78", (units / "qwen38-flash.service").read_text())

    def test_the_flash_lane_on_a_gpu_outside_ats_promises_no_first_load(self):
        """Given its units only because this GPU cannot start it (issue #54), the run said its
        image and checkpoint came at its first Load, which is refused there (review, 2026-10-19)."""
        rc, out, calls, fenced, units, home = run_units_only("flash", gpu="None")
        # it ends on its own line, before the wall of a run that is not prepared yet
        self.assertEqual(rc, 0, out[-1500:])
        self.assertEqual(fenced, [])
        self.assertIn("2/10 The Flash-Next image: not pulled, as this GPU cannot start the flash lane", out)
        self.assertIn("3/10 The GPU from a container: not asked, as this GPU cannot start the flash lane", out)
        self.assertIn("4/10 The Flash-Next checkpoint: not fetched, as this GPU cannot start the flash lane", out)
        self.assertIn("not enabled at boot: this GPU cannot start it", out)
        for promise in ("first Load", "not prepared yet", "comes later"):
            self.assertNotIn(promise, out)
        launcher = (home / ".config/qwen38/launch-flash.sh").read_text()
        self.assertIn(f'bash {home}/.config/qwen38/flash-ats-check.sh)"', launcher)

    def test_the_27b_lane(self):
        out, units, home = self.check("27b", "qwen38-sglang.service")
        unit = (units / "qwen38-sglang.service").read_text()
        self.assertIn("--speculative-draft-model-path", unit)
        self.assertIn("checkpoints' configs for the 1m window: set at the lane's first Load", out)


class TheBoxsOwnFlashLaneOnAGpuOutsideAts(unittest.TestCase):
    """A box whose text lane is the flash lane, at boot, on a GPU outside its ATS mode (issue #54).
    Its update stopped at step 1 and kept the launcher that starts again every 15 s, and a box
    booting a side lane with the flash lane as its text lane was never updated again (found in
    review, 2026-10-19). Run whole up to the path that starts nothing: its new units are written,
    nothing is fetched, enabled or started."""

    def test_its_update_writes_its_units_and_starts_nothing(self):
        rc, out, calls, fenced, units, home = run_units_only("flash", gpu="None", main=True)
        self.assertIn(wall.WALL, out, out[-2500:])
        self.assertEqual(rc, 97, out[-2500:])
        self.assertEqual(fenced, [])
        self.assertIn("NOTE: the flash lane cannot start on this box: the GPU's addressing mode is None", out)
        self.assertIn("this run starts nothing", out)
        self.assertIn("2/10 The Flash-Next image: not pulled, as this GPU cannot start the flash lane", out)
        self.assertEqual(sorted({" ".join(c.split()[:3]) for c in calls if c.startswith("docker ")}
                                - {"docker image inspect", "docker info", "docker info --format"}), [], calls)
        flash = [c for c in calls if c.startswith(("sudo systemctl", "systemctl")) and "qwen38-flash.service" in c
                 and any(v in c.split() for v in ("enable", "start", "restart"))]
        self.assertEqual(flash, [], "the flash lane enabled or started")
        launcher = (home / ".config/qwen38/launch-flash.sh").read_text()
        self.assertIn(f'bash {home}/.config/qwen38/flash-ats-check.sh)"', launcher)
        self.assertTrue((home / ".config/qwen38/flash-ats-check.sh").is_file())
        self.assertIn("RestartPreventExitStatus=78", (units / "qwen38-flash.service").read_text())


if __name__ == "__main__":
    unittest.main()
